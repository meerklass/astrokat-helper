

import os
import numpy as np
import katpoint
import astrokat
import ephem
from datetime import datetime, timedelta
from astropy.coordinates import SkyCoord
from astropy import units
from scripts.yaml_offseter import get_coordinate_strings
import argparse
from typing import Optional
from matplotlib import pyplot as plt


SECONDS_IN_ONE_DAY = 24*60*60


class WhenIsPatchObservable:
    """ Class to explicitly check drift-scannability of a patch for a set of dates. """

    def __init__(self,
                 point_list: list[str],
                 buffer: int,
                 from_date: Optional[datetime] = None,
                 min_elevation: float = 35,
                 max_elevation: float = 50,
                 days_from_now: int = 365,
                 time_res: float = 10,  # in sec
                 plot_dir: str | None = None):
        """
        Initialise
        :param point_list: `list` of `str` point coordinates (hourangle). must have length 4
        :param buffer: integer buffer from sunrise and sunset in minutes
        :param from_date: first date to consider, if `None`, today is used
        :param min_elevation: minimum elevation of target patch
        :param max_elevation: maximum elevation of target patch
        :param days_from_now: number of days to consider starting from `from_date`
        :param plot_dir: directory to store plots, if `None`, they are not created
        """
        date_format = '%Y-%m-%d'
        self.corners = point_list
        self.point_list = self.get_point_list(point_list=point_list)
        if from_date is not None:
            self.from_date = datetime.strptime(from_date, date_format)
        else:
            self.from_date = datetime.today()
        self.min_elevation = min_elevation
        self.max_elevation = max_elevation

        location = astrokat.Observatory().location
        self.ref_antenna = katpoint.Antenna(location)
        self.ref_antenna.observer.date = ephem.Date(self.from_date.strftime(date_format))
        self.ref_antenna.observer.horizon = ephem.degrees(str(0)) # sun horizon = 0 deg elevation

        self.sun_threshold = buffer
        self.days_from_now = days_from_now
        
        self.time_res = time_res
#        print(time_res)

        self.plot_dir = plot_dir

    def print(self):
        ra_1, ra_2, dec_1, dec_2 = self.corners
        ra = (float(ra_1) + float(ra_2))/2
        dec = (float(dec_1) + float(dec_2))/2
        p = SkyCoord(ra, dec, unit=units.deg)
        ra, dec = get_coordinate_strings(coordinate=p)
        print("\nParameters:",
              self.corners,
              ra, dec,
              self.point_list,
              self.from_date,
              self.sun_threshold,
              self.min_elevation,
              self.max_elevation,
              self.days_from_now,
              self.plot_dir, "\n", flush = True)
        


    @staticmethod
    def get_point_list(point_list: list[float]) -> list[str]:
        """
        Returns a `list` of `str` hourangle corners corresponding to the degree right
        ascension and declination min and max values defined in `point_list`.
        :param point_list: needs to be length 4, `[ra_min, ra_max, dec_min, dec_max]`
        :raise ValueError: if `point_list` is not exactly of length 4
        """
        if (len_point_list := len(point_list)) != 4:
            raise ValueError(f'`point_list` must have exactly 4 entries, got {len_point_list}.')
        ra_1, ra_2, dec_1, dec_2 = point_list
        p1 = SkyCoord(ra_1, dec_1, unit=units.deg)
        p2 = SkyCoord(ra_2, dec_1, unit=units.deg)
        p3 = SkyCoord(ra_2, dec_2, unit=units.deg)
        p4 = SkyCoord(ra_1, dec_2, unit=units.deg)
        result = []
        for point in [p1, p2, p3, p4]:
            ra, dec = get_coordinate_strings(coordinate=point)
            result.append(f'{ra} {dec}')
        return result



    def target_body_list(self) -> list[ephem.FixedBody]:
        """ Return a `list` of `ephem.FixedBody`s constructed from the points defined in `self.point_list`. """
        target_body_list = []
        for i, target in enumerate(self.point_list):
            ra, dec = target.split(' ')
            x = ephem.FixedBody()
            x._ra = ephem.hours(ra)
            x._dec = ephem.degrees(dec)
            x.name = f'p{i}'
            target_body_list.append(x)
        return target_body_list



    def days_list(self) -> list[datetime]:
        """
        Return a `list` of `datetime` objects 
        for each day between `self.from_date` and `self.days_from_now` later.
        """
        return [self.from_date + timedelta(days=d) for d in range(self.days_from_now)]


    def rise_set(self, start: datetime, end: datetime, target: ephem.FixedBody):
        """
        Return start/end positions for target to be observed between a given time interval:
            start rising time and elevation
            end rising time and elevation
            start setting time and elevation
            end setting time and elevation
            times in datetime format and elevations in degrees
        """
        
        observer = self.ref_antenna.observer
        el_min = self.min_elevation
        el_max = self.max_elevation
        time_res = timedelta(seconds = self.time_res) # resolution in seconds
        rise_start = None
        rise_end = None
        set_start = None
        set_end = None
        time=start
        el1 = self.elevation_at(time, observer, target)
        
        rise_time_list=[]
        rise_elevation_list=[]
        set_time_list=[]
        set_elevation_list=[]
        rise_start=set_start=rise_end=set_end=False
#        print(el1)
        while time <= end and (not rise_end or not set_end):   # we can only have one rising and setting in 24h
            el2 = self.elevation_at(time+time_res, observer, target)
            if rise_start and not rise_end: # we're rising
                rise_time_list.append(time)
                rise_elevation_list.append(el1)
                if el1 >= el_max or el2 <= el1: # rise ends
                    rise_end = True
            elif set_start and not set_end:   # we're setting
                set_time_list.append(time)
                set_elevation_list.append(el1)
                if el1 <= el_min or el2 >= el1:  # set ends
                    set_end = True
            elif el1 >= el_min and el1 <= el_max: # start of rising or setting
                # ignore the unusual situation where target only goes above el_min for time_res
                if el2 > el1:    # we're rising
                    rise_time_list.append(time)
                    rise_elevation_list.append(el1)
                    rise_start = True
                elif el2 < el1:  # we're setting
                    set_time_list.append(time)
                    set_elevation_list.append(el1)
                    set_start = True
#                    print('Test', flush=True)
            el1=el2
            time = time + time_res
            
        return [rise_time_list, rise_elevation_list, set_time_list, set_elevation_list]
            
                        
                    
    def elevation_curve(self, day: datetime, observer: ephem.Observer, target: ephem.FixedBody):
        """
        Return a `` `np.ndarray` with elevation curve for day specified 
        """    
        
        minutes_per_day = 24*60
        elevation_list = []
        for minutes in range(minutes_per_day):
            now = day + timedelta(minutes=minutes)
            elevation_list.append(self.elevation_at(now, observer, target))
            
        elevation_list = np.asarray(elevation_list)
        return elevation_list 
    
    
    def get_lst(self, observer: ephem.Observer, date: datetime):
        """"
        returns lst in hour:min:sec pyephem format
        """
        date_backup = observer.date
        observer.date = ephem.Date(date)      # don't we need to use ephem.Date(date) since date is in datetime?
        lst = observer.sidereal_time()
        observer.date = date_backup
        return lst
    

    def run(self):
        """
        Print all drift-scannable days in `self.days_list()` 
        along with the hours that the patch defined in `self.point_list` is observable on each day rising and setting. 
        """
        target_body_list = self.target_body_list()
        patch_observed = False
        el_res = self.time_res*360/24/60/60/2
        print(f'{"day":<11}{"rise/set":<11}{"min_start_time":<17}{"min_start_lst":<16}{"elevation":<12}{"duration":<11}{"max_start_time":<17}{"max_start_lst":<16}{"elevation":<12}{"duration":8}')
        for day in self.days_list():
            ephem_day = ephem.Date(day)
            self.ref_antenna.observer.date = ephem_day
        
            sun = ephem.Sun()
            sunset = self.ref_antenna.observer.next_setting(sun)
            self.ref_antenna.observer.date = sunset
            sunrise = self.ref_antenna.observer.next_rising(sun)
            sunset_cut = ephem.Date(sunset + self.sun_threshold*ephem.minute)
            sunrise_cut = ephem.Date(sunrise - self.sun_threshold*ephem.minute)
#            print('Sunset: {}, sunset_cut: {}, sunrise: {}, sunrise_cut: {}'.format(sunset, sunset_cut, sunrise, sunrise_cut))
            
            if sunrise_cut > sunset_cut:      
                rise_set_curve_list=[]
#                n=0        
                for target in target_body_list:
#                    print(target, flush = True)
                    rise_set_curve = self.rise_set(sunset_cut.datetime(), sunrise_cut.datetime(), target)
                    rise_set_curve_list.append(rise_set_curve)
#                    if rise_set_curve[0]:  # there was a rise
#                        print('rise  {}  {:<16} {:<7.2f} {:<16} {:<7.2f}'.format(n, rise_set_curve[0][0].strftime('%d/%m/%y %H:%M'), rise_set_curve[1][0], rise_set_curve[0][-1].strftime('%d/%m/%y %H:%M'), rise_set_curve[1][-1]))
#                    if rise_set_curve[2]:  # there was a set
#                        print('set   {}  {:<16} {:<7.2f} {:<16} {:<7.2f}'.format(n, rise_set_curve[2][0].strftime('%d/%m/%y %H:%M'), rise_set_curve[3][0], rise_set_curve[2][-1].strftime('%d/%m/%y %H:%M'), rise_set_curve[3][-1]))
#                    n=n+1
 
                if all(x[0] for x in rise_set_curve_list):   # all points rise
                    start_el = [x[1][0] for x in rise_set_curve_list]
                    rise_min_el = max(start_el)
                    time=[]
                    rise_min_el_time=rise_min_el_duration=-1
                    for x in rise_set_curve_list:
                        idx = [i for i,f in enumerate(x[1]) if abs(f-rise_min_el) <= el_res]
                        if idx != []: time.append(x[0][idx[0]])
                    if len(time) == len(target_body_list):
                        patch_observed = True 
                        rise_min_el_time = min(time)
                        end_time = max(time)
                        rise_min_el_duration = end_time - rise_min_el_time
                    end_el = [x[1][-1] for x in rise_set_curve_list]
                    rise_max_el = min(end_el)
                    time=[]
                    rise_max_el_time=rise_max_el_duration=-1
                    for x in rise_set_curve_list:
                        idx = [i for i,f in enumerate(x[1]) if abs(f-rise_max_el) <= el_res]
                        if idx != []: time.append(x[0][idx[0]])
                    if len(time) == len(target_body_list):  #note: if there is a min el then there should be a max el and vice versa
                        patch_observed = True 
                        rise_max_el_time = min(time)
                        end_time = max(time)
                        rise_max_el_duration = end_time - rise_max_el_time
                        rise_min_el_lst = self.get_lst(self.ref_antenna.observer, rise_min_el_time)
                        rise_max_el_lst = self.get_lst(self.ref_antenna.observer, rise_max_el_time)
                        print(f'{day.strftime("%d/%m/%y"):<11}{"rise":<11}{rise_min_el_time.strftime("%d/%m/%y %H:%M"):<17}{str(rise_min_el_lst):<16}{rise_min_el:<12.2f}{str(rise_min_el_duration):<11}{rise_max_el_time.strftime("%d/%m/%y %H:%M"):<17}{str(rise_max_el_lst):<16}{rise_max_el:<12.2f}{str(rise_max_el_duration):8}')
#                        for target in target_body_list:
#                            print(self.azimuth_at(rise_min_el_time, self.ref_antenna.observer, target))
#                            print(self.azimuth_at(rise_max_el_time, self.ref_antenna.observer, target))
                
                if all(x[2] for x in rise_set_curve_list):   # all points set
                    start_el = [x[3][0] for x in rise_set_curve_list]
                    set_max_el = min(start_el)  # it is going down...
                    time=[]
                    set_max_el_time = set_max_el_duration = -1
                    for x in rise_set_curve_list:
                        idx = [i for i,f in enumerate(x[3]) if abs(f-set_max_el) <= el_res]
                        if idx != []: time.append(x[2][idx[0]])
                    if len(time) == len(target_body_list):
                        patch_observed = True 
                        set_max_el_time = min(time)
                        end_time = max(time)
                        set_max_el_duration = end_time - set_max_el_time
                    end_el = [x[3][-1] for x in rise_set_curve_list]
                    set_min_el = max(end_el)
                    time=[]
                    set_min_el_time = set_min_el_duration = -1
                    for x in rise_set_curve_list:
                        idx = [i for i,f in enumerate(x[3]) if abs(f-set_min_el) <= el_res]
                        if idx != []: time.append(x[2][idx[0]])
                    if len(time) == len(target_body_list):
                        patch_observed = True 
                        set_min_el_time = min(time)
                        end_time = max(time)
                        set_min_el_duration = end_time - set_min_el_time
                        set_min_el_lst = self.get_lst(self.ref_antenna.observer, set_min_el_time)
                        set_max_el_lst = self.get_lst(self.ref_antenna.observer, set_max_el_time)
                        print(f'{day.strftime("%d/%m/%y"):<11}{"set":<11}{set_max_el_time.strftime("%d/%m/%y %H:%M"):<17}{str(set_max_el_lst):<16}{set_max_el:<12.2f}{str(set_max_el_duration):<11}{set_min_el_time.strftime("%d/%m/%y %H:%M"):<17}{str(set_min_el_lst):<16}{set_min_el:<12.2f}{str(set_min_el_duration):8}')
#                        for target in target_body_list:
#                            print(self.azimuth_at(set_max_el_time, self.ref_antenna.observer, target))
#                            print(self.azimuth_at(set_min_el_time, self.ref_antenna.observer, target))
                            
                            
                if self.plot_dir is not None:
                    plt.figure(figsize=(20, 5))
                    for i, x in enumerate(rise_set_curve_list):
                        plt.plot(x[0],x[1], label=f'rise {i}')
                        plt.plot(x[2],x[3], label=f'set {i}')
                    plt.xlabel('time of day UTC [hours]')
                    plt.legend()
                    plt.savefig(os.path.join(self.plot_dir, f'riseset_curve_{day.strftime("%d_%m_%y")}.png'))
#                    plt.savefig(f'corner_{day.strftime("%d/%m/%y")}.png')
                    plt.close()
                
        if not patch_observed: print('Not visible')
                        
                    
   

    @staticmethod
    def elevation_at(date: datetime, observer: ephem.Observer, body: ephem.FixedBody) -> float:
        """ Return the `elevation` in degrees of `body` at `date` as seen by `observer`. """
        date_backup = observer.date
        observer.date = ephem.Date(date)      # don't we need to use ephem.Date(date) since date is in datetime?
        body.compute(observer)
        result = float(body.alt)*180/np.pi
        observer.date = date_backup
        return result
    
    @staticmethod
    def azimuth_at(date: datetime, observer: ephem.Observer, body: ephem.FixedBody) -> float:
        """ Return the `elevation` in degrees of `body` at `date` as seen by `observer`. """
        date_backup = observer.date
        observer.date = ephem.Date(date)      # don't we need to use ephem.Date(date) since date is in datetime?
        body.compute(observer)
        result = float(body.az)*180/np.pi
        observer.date = date_backup
        return result


def main():
    """ Run the `run` method with arguments from the command line. """
    
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--azel",
        nargs=2,  # 4 more values expected => creates a list
        type=float,
        help='provide start azimuth and elevation as degrees',
        required=True
    )
    cli.add_argument(
        "--duration",
        nargs=1,
        type=int,
        help='provide duration of drift scan observation in seconds',
        required=True
    )
    args = cli.parse_args()



    if args.date is None:
        from_date = None
    else:
        from_date = args.date[0]
        
    
        when_is_patch_observable = WhenIsPatchObservable(
        point_list=args.corners,
        from_date=from_date,
        buffer=args.buffer[0],
        min_elevation=args.min_elevation[0],
        max_elevation=args.max_elevation[0],
        days_from_now=args.days[0],
        plot_dir=plot_dir
    )
    when_is_patch_observable.print()
    when_is_patch_observable.run()


if __name__ == '__main__':
    main()

import numpy as np
import katpoint
from scipy import ndimage
import astrokat
import ephem
from datetime import datetime, timedelta
from astropy.coordinates import SkyCoord
from astropy import units
import sys
from scripts.yaml_offseter import get_coordinate_strings
import argparse
from typing import Optional


SECONDS_IN_ONE_DAY = 24*60*60


class WhenIsPatchObservable:
    """ DOC """

    def __init__(self,
                 point_list: list[str],
                 buffer: int,
                 from_date: Optional[datetime] = None,
                 min_elevation: float = 35,
                 max_elevation: float = 50,
                 days_from_now: int = 365):
        """
        Initialise
        :param point_list: `list` of `str` point coordinates (hourangle). must have length 4
        :param buffer: integer buffer from sunrise and sunset in minutes
        :param from_date: first date to consider, if `None`, today is used
        :param min_elevation: minimum elevation of target patch
        :param max_elevation: maximum elevation of target patch
        :param days_from_now: number of days to consider starting from `from_date`
        """
        self.point_list = self.get_point_list(point_list=point_list)
        if from_date is not None:
            self.from_date = datetime.strptime(from_date, '%Y-%m-%d')
        else:
            self.from_date = datetime.today()
        self.min_elevation = min_elevation
        self.max_elevation = max_elevation

        location = astrokat.Observatory().location
        self.ref_antenna = katpoint.Antenna(location)
        self.ref_antenna.observer.date = ephem.Date(from_date)
        self.ref_antenna.observer.horizon = ephem.degrees(str(self.min_elevation))

        self.ref_antenna_sun = katpoint.Antenna(location)
        self.ref_antenna_sun.observer.date = ephem.Date(from_date)
        self.ref_antenna_sun.observer.horizon = ephem.degrees(str(0))

        self.sun = ephem.Sun()
        self.sun_threshold = buffer  # mintues

        self.timedelta = timedelta(minutes=self.sun_threshold)
        self.days_from_now = days_from_now

    @staticmethod
    def get_point_list(point_list: list[float]) -> list[str]:
        """ DOC """
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

    @staticmethod
    def up_array(day: datetime, target: ephem.FixedBody, observer: ephem.Observer) -> np.ndarray[bool]:
        """
        Return a `boolean` `np.ndarray` with entries for each `minutes` for `day` which is 1 where `target` is observable
        for `observer` and 0 otherwise.
        """
        minutes_per_day = 24*60
        try:
            rising = observer.next_rising(target).datetime()
        except ephem.NeverUpError:
            print(f'Target {target.name} never up on {day}.')
            return np.ndarray([0 for _ in range(minutes_per_day)], dtype=bool)
        setting = observer.next_setting(target).datetime()
        result = []
        minutes_list = []
        for minutes in range(minutes_per_day):
            now = day + timedelta(minutes=minutes)
            minutes_list.append(now)
            if setting < rising:
                is_up = now <= setting or now >= rising
            else:
                is_up = rising <= now <= setting
            result.append(is_up)
        result = np.asarray(result, dtype=bool)
        minutes_list = np.asarray(minutes_list)
        transit_index = np.argmin(abs(minutes_list - observer.next_transit(target).datetime()))
        return result, transit_index

    def run(self):
        """
        Print all days in `self.days_list()` 
        along with the hours that the patch defined in `self.point_list` is observable on each day. 
        """
        target_body_list = self.target_body_list()
        for day in self.days_list():
            ephem_day = ephem.Date(day)
            self.ref_antenna.observer.date = ephem_day
            self.ref_antenna_sun.observer.date = ephem_day
            sun_up_array, _ = self.up_array(day, self.sun, self.ref_antenna_sun.observer)
            sun_up_array = ndimage.binary_dilation(sun_up_array, iterations=self.sun_threshold)
            is_observable_array_list = []
            transit_index_list = []
            for target in target_body_list:
                target_up_array, transit_index = self.up_array(day=day,
                                                               target=target,
                                                               observer=self.ref_antenna.observer)
                is_observable_array = target_up_array * np.logical_not(sun_up_array)
                is_observable_array_list.append(is_observable_array)
                transit_index_list.append(transit_index)

            if not self.patch_is_observable(target_body_list=target_body_list,
                                            day=day,
                                            is_observable_array_list=is_observable_array_list,
                                            transit_index_list=transit_index_list,
                                            observer=self.ref_antenna.observer):
                continue
            rising_hours = np.min([np.sum(array_[:transit_index]) for array_ in is_observable_array_list])/60
            setting_hours = np.min([np.sum(array_[transit_index:]) for array_ in is_observable_array_list])/60
            print(f'{day.date()} {rising_hours:.1f} hours rising {setting_hours:.1f} hours setting')

    def patch_is_observable(self,
                            target_body_list,
                            day,
                            is_observable_array_list,
                            transit_index_list,
                            observer):
        max_alt_list = []
        alt_list_rising = []
        alt_list_setting = []
        for i, (body, transit_index) in enumerate(zip(target_body_list, transit_index_list)):
            transit_date = day + timedelta(minutes=float(transit_index))
            max_alt_list.append(self.elevation_at(date=transit_date,
                                                  observer=observer,
                                                  body=body))
            rise_index, set_index = self.changing_indices(array=is_observable_array_list[i], index=transit_index)
            if rise_index is None:
                return False
            rising_date = day + timedelta(minutes=float(rise_index))
            setting_date = day + timedelta(minutes=float(set_index))
            alt_list_rising.append(self.elevation_at(date=rising_date,
                                                     observer=observer,
                                                     body=body))
            alt_list_setting.append(self.elevation_at(date=setting_date,
                                                      observer=observer,
                                                      body=body))

        max_at_rise = max(alt_list_rising)
        max_at_set = max(alt_list_setting)
        return all(np.asarray(max_alt_list) > max(max_at_set, max_at_rise))

    @staticmethod
    def changing_indices(array, index):
        if array[index]:
            labelled, _ = ndimage.label(array)
            label = labelled[index]
            where = np.where(labelled == label)[0]
            first = where[0]
            last = where[-1]
            return first, last
        else:
            return None, None

    @staticmethod
    def elevation_at(date, observer, body):
        date_backup = observer.date
        observer.date = date
        body.compute(observer)
        result = float(body.alt)*180/np.pi
        observer.date = date_backup
        return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--corners",
        nargs=4,  # 4 more values expected => creates a list
        type=float,
        help='provide right ascension min, max and declination min, max values as degree floats',
        required=True
    )
    cli.add_argument(
        "--date",
        nargs=1,
        type=str,
        help='provide starting date as `string`, e.g. "2023-12-24"',
        default=None
    )
    cli.add_argument(
        "--buffer",
        nargs=1,
        type=int,
        help='provide integer buffer from sunrise and sunset in minutes',
        required=True
    )
    cli.add_argument(
        "--min_elevation",
        nargs=1,
        type=float,
        help='provide min elevation in degrees',
        default=[35]
    )
    cli.add_argument(
        "--max_elevation",
        nargs=1,
        type=float,
        help='provide max elevation in degrees',
        default=[50]
    )
    cli.add_argument(
        "--days",
        nargs=1,
        type=int,
        help='provide number of days from input date to consider',
        default=[365]
    )
    args = cli.parse_args()

    when_is_patch_observable = WhenIsPatchObservable(
        point_list=args.corners,
        from_date=args.date[0],
        buffer=args.buffer[0],
        min_elevation=args.min_elevation[0],
        max_elevation=args.max_elevation[0],
        days_from_now=args.days[0]
    )
    when_is_patch_observable.run()

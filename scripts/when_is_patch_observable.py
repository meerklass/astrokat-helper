

import os
import numpy as np
import katpoint
from scipy import ndimage
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
        self.ref_antenna.observer.horizon = ephem.degrees(str(self.min_elevation))

        self.ref_antenna_sun = katpoint.Antenna(location)
        self.ref_antenna_sun.observer.date = ephem.Date(self.from_date.strftime(date_format))
        self.ref_antenna_sun.observer.horizon = ephem.degrees(str(0))

        self.sun = ephem.Sun()
        self.sun_threshold = buffer  # mintues

        self.timedelta = timedelta(minutes=self.sun_threshold)
        self.days_from_now = days_from_now

        self.plot_dir = plot_dir

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

    @staticmethod
    def up_array(day: datetime, target: ephem.FixedBody, observer: ephem.Observer) -> np.ndarray[bool]:
        """
        Return a `boolean` `np.ndarray` with entries for each `minutes` for `day` which is 1 where `target` 
        is observable for `observer` and 0 otherwise.
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
            condition_a = now >= rising
            condition_b = now <= setting
            if setting < rising:
                is_up = condition_a or condition_b
            else:
                is_up = condition_a and condition_b
            result.append(is_up)
            if is_up:
                minutes_list.append(now)
            else:
                minutes_list.append(day+timedelta(days=10))  # arbitrary date in the future
        result = np.asarray(result, dtype=bool)
        minutes_list = np.asarray(minutes_list)
        transit_index = np.argmin(abs(minutes_list - observer.next_transit(target).datetime()))
        return result, transit_index

    def run(self):
        """
        Print all drift-scannable days in `self.days_list()` 
        along with the hours that the patch defined in `self.point_list` is observable on each day rising and setting. 
        """
        target_body_list = self.target_body_list()
        for day in self.days_list():
            ephem_day = ephem.Date(day)
            self.ref_antenna.observer.date = ephem_day
            self.ref_antenna_sun.observer.date = ephem_day
            sun_up_array, _ = self.up_array(day, self.sun, self.ref_antenna_sun.observer)
            sun_up_array = ndimage.binary_dilation(sun_up_array, iterations=self.sun_threshold)

            if self.plot_dir is not None:
                plt.figure(figsize=(20, 5))
                plt.plot(np.arange(len(sun_up_array))/60, sun_up_array)
                plt.xlabel('time of day UTC [hours]')
                plt.title('binary observability')
                plt.savefig(os.path.join(self.plot_dir, 'sun.png'))
                plt.close()

            is_observable_array_list = []
            transit_index_list = []
            for target in target_body_list:
                target_up_array, transit_index = self.up_array(day=day,
                                                               target=target,
                                                               observer=self.ref_antenna.observer)
                is_observable_array = target_up_array * np.logical_not(sun_up_array)
                is_observable_array_list.append(is_observable_array)
                transit_index_list.append(transit_index)

            is_observable = self.patch_is_observable(target_body_list=target_body_list,
                                                     day=day,
                                                     is_observable_array_list=is_observable_array_list,
                                                     transit_index_list=transit_index_list,
                                                     observer=self.ref_antenna.observer)
            if is_observable == (False, False):
                print_str = 'NO'
            elif is_observable == (True, False):
                print_str = 'only setting'
            elif is_observable == (False, True):
                print_str = 'only rising'
            else:
                print_str = 'both rising and setting'
            print(f'{day.date()} {print_str}')

    def patch_is_observable(self,
                            target_body_list: list[ephem.FixedBody],
                            day: datetime,
                            is_observable_array_list: list[np.ndarray[bool]],
                            transit_index_list: list[int],
                            observer: ephem.Observer) -> bool:
        """
        Return `True` if the patch defined in `target_body_list` is observable on `day` and `False` otherwise.
        :param target_body_list: `list` of `ephem` bodies defining the patch
        :param day: date in question
        :param is_observable_array_list: one `boolean` array for each body in `target_body_list`, each of length
                                         24*60, i.e. one entry per minute. `True` means the target is up on the sky
        :param transit_index_list: `list` with entries for each target body, defining the index of the minute of
                                  maximum elevation of that target
        :param observer: ephem observer
        :return: `bool` `True` if the patch is observable
        """
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
                return False, False

            minutes = [day + timedelta(minutes=float(x)) for x in range(len(is_observable_array_list[i]))]
            if self.plot_dir is not None:
                plt.figure(figsize=(20, 5))
                plt.plot(np.arange(len(is_observable_array_list[i]))/60, is_observable_array_list[i])
                plt.axvline(transit_index/60, color='red', label='max elevation')
                plt.axvline(rise_index/60, color='blue', label='rise')
                plt.axvline(set_index/60, color='green', label='set')
                plt.xlabel('time of day UTC [hours]')
                plt.title('binary observability')
                plt.legend()
                plt.savefig(f'corner_{i}.png')
                plt.close()
            rising_date = day + timedelta(minutes=float(rise_index))
            setting_date = day + timedelta(minutes=float(set_index))
            alt_list_rising.append(self.elevation_at(date=rising_date,
                                                     observer=observer,
                                                     body=body))
            alt_list_setting.append(self.elevation_at(date=setting_date,
                                                      observer=observer,
                                                      body=body))

        max_at_rise = np.asarray(max(alt_list_rising))
        max_at_set = np.asarray(max(alt_list_setting))
        return all(max_alt_list > max_at_set), all(max_alt_list > max_at_rise)

    @staticmethod
    def first_last(array: np.ndarray[int], index: int):
        """ Return the first and last index of the blob in `array` that contains entry at `index`. """
        labelled, _ = ndimage.label(array)
        label = labelled[index]
        where = np.where(labelled == label)[0]
        first = where[0]
        last = where[-1]
        return first, last

    def changing_indices(self, array: np.ndarray[bool], index: int) -> tuple[int | None, int | None]:
        """ Return optional start and end indices of the blob in `array` containing `index`. """
        last_index = len(array) - 1
        if array[index]:
            is_true = np.where(array)[0]
            if 0 in is_true and last_index in is_true:  # blob spans end to start
                _, last = self.first_last(array=array, index=0)
                first, _ = self.first_last(array=array, index=last_index)
            else:
                first, last = self.first_last(array=array, index=index)
            return first, last
        else:
            return None, None

    @staticmethod
    def elevation_at(date: datetime, observer: ephem.Observer, body: ephem.FixedBody) -> float:
        """ Return the `elevation` of `body` at `date` as seen by `observer`. """
        date_backup = observer.date
        observer.date = date
        body.compute(observer)
        result = float(body.alt)*180/np.pi
        observer.date = date_backup
        return result


def main():
    """ Run the `run` method with arguments from the command line. """
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
    cli.add_argument(
        "--plot_dir",
        nargs=1,
        type=str,
        help='',
        default=None,
    )
    args = cli.parse_args()

    if args.date is None:
        from_date = None
    else:
        from_date = args.date[0]
    if args.plot_dir is None:
        plot_dir = None
    else:
        plot_dir = args.plot_dir[0]
    when_is_patch_observable = WhenIsPatchObservable(
        point_list=args.corners,
        from_date=from_date,
        buffer=args.buffer[0],
        min_elevation=args.min_elevation[0],
        max_elevation=args.max_elevation[0],
        days_from_now=args.days[0],
        plot_dir=plot_dir
    )
    when_is_patch_observable.run()


if __name__ == '__main__':
    main()

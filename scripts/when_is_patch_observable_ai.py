import os
import numpy as np
import katpoint
import astrokat
import ephem
from datetime import datetime, timedelta
from astropy.coordinates import SkyCoord
from astropy import units
from yaml_offseter import get_coordinate_strings
import argparse
from typing import Optional, List, Tuple, Any, Union
import logging
from matplotlib import pyplot as plt
import yaml
import sys


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SECONDS_IN_ONE_DAY = 24*60*60


class WhenIsPatchObservable:
    """ Class to explicitly check drift-scannability of a patch for a set of dates. """

    def __init__(self,
                 point_list: list[float],
                 buffer: int,
                 from_date: Optional[str] = None,
                 min_elevation: float = 35,
                 max_elevation: float = 50,
                 days_from_now: int = 365,
                 time_res: float = 10,  # in sec
                 plot_dir: str | None = None,
                 csv_output: bool = False,
                 moon_buffer_ra: float = 0.0,  # degrees on either side of RA range
                 moon_buffer_dec: float = 0.0):  # degrees above/below Dec range
        """
        Initialise
        :param point_list: `list` of `str` point coordinates (hourangle). must have length 4
        :param buffer: integer buffer from sunrise and sunset in minutes
        :param from_date: first date to consider, if `None`, today is used
        :param min_elevation: minimum elevation of target patch
        :param max_elevation: maximum elevation of target patch
        :param days_from_now: number of days to consider starting from `from_date`
        :param plot_dir: directory to store plots, if `None`, they are not created
        :param csv_output: whether to output in CSV format for spreadsheet import
        :param moon_buffer_ra: buffer in degrees on either side of RA range for moon detection
        :param moon_buffer_dec: buffer in degrees above/below Dec range for moon detection
        :raises ValueError: if arguments are invalid
        """
        # Validate buffer is non-negative
        if buffer < 0:
            raise ValueError(f"Buffer time must be non-negative, got {buffer}")
            
        # Validate min_elevation < max_elevation
        if min_elevation >= max_elevation:
            raise ValueError(f"Minimum elevation ({min_elevation}) must be less than maximum elevation ({max_elevation})")
            
        # Validate time_res is positive
        if time_res <= 0:
            raise ValueError(f"Time resolution must be positive, got {time_res}")
            
        # Validate days_from_now is positive
        if days_from_now <= 0:
            raise ValueError(f"Days from now must be positive, got {days_from_now}")
        
        # Continue with existing initialization
        date_format = '%Y-%m-%d'
        self.corners = point_list
        self.point_list = self.get_point_list(point_list=point_list)
        if from_date is not None:
            try:
                self.from_date = datetime.strptime(from_date, date_format)
            except ValueError:
                raise ValueError(f"Invalid date format. Expected {date_format}, got {from_date}")
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
        logger.debug(f"Time resolution set to {time_res} seconds")

        self.plot_dir = plot_dir
        self.csv_output = csv_output
        
        # Store moon buffer values
        self.moon_buffer_ra = moon_buffer_ra
        self.moon_buffer_dec = moon_buffer_dec
        
        # Create plot_dir if it doesn't exist
        if plot_dir is not None:
            if not os.path.exists(plot_dir):
                try:
                    os.makedirs(plot_dir)
                    logger.info(f"Created plot directory: {plot_dir}")
                except OSError as e:
                    raise ValueError(f"Failed to create plot directory '{plot_dir}': {e}")
            elif not os.path.isdir(plot_dir):
                raise ValueError(f"'{plot_dir}' exists but is not a directory")
            
    def print(self):
        """Print detailed information about the patch and observation parameters."""
        # Calculate center RA/Dec of the patch
        ra_1, ra_2, dec_1, dec_2 = self.corners
        ra_center = (float(ra_1) + float(ra_2)) / 2
        dec_center = (float(dec_1) + float(dec_2)) / 2
        patch_size_ra = abs(float(ra_2) - float(ra_1))
        patch_size_dec = abs(float(dec_2) - float(dec_1))
        
        # Convert center to SkyCoord for string representation
        p = SkyCoord(ra_center, dec_center, unit=units.deg)
        ra_str, dec_str = get_coordinate_strings(coordinate=p)
        
        # Create formatted output
        print("\n=== PATCH OBSERVABILITY CALCULATION PARAMETERS ===")
        print(f"Corners (RA min, RA max, Dec min, Dec max): {self.corners}")
        print(f"Patch Center: RA = {ra_str} ({ra_center:.4f}°), Dec = {dec_str} ({dec_center:.4f}°)")
        print(f"Patch Size: {patch_size_ra:.2f}° × {patch_size_dec:.2f}°")
        print(f"Corner Points:")
        for i, point in enumerate(self.point_list):
            print(f"  Corner {i+1}: {point}")
        
        print("\n=== OBSERVATION PARAMETERS ===")
        print(f"Start Date: {self.from_date.strftime('%Y-%m-%d')}")
        print(f"Days to Check: {self.days_from_now}")
        print(f"End Date: {(self.from_date + timedelta(days=self.days_from_now-1)).strftime('%Y-%m-%d')}")
        print(f"Sun Threshold (buffer): {self.sun_threshold} minutes after sunset/before sunrise")
        print(f"Elevation Range: {self.min_elevation}° to {self.max_elevation}°")
        print(f"Moon Detection Buffer: RA = ±{self.moon_buffer_ra}°, Dec = ±{self.moon_buffer_dec}°")
        
        if self.plot_dir:
            print(f"Plotting Enabled: Yes (saving to '{self.plot_dir}')")
        else:
            print("Plotting Enabled: No")
        
        print(f"Output Format: {'CSV' if self.csv_output else 'Formatted Text'}")
        print("=" * 50)
        print("")  # Empty line before results

    @staticmethod
    def get_point_list(point_list: List[float]) -> List[str]:
        """
        Returns a `list` of `str` hourangle corners corresponding to the degree right
        ascension and declination min and max values defined in `point_list`.
        :param point_list: needs to be length 4, `[ra_min, ra_max, dec_min, dec_max]`
        :raise ValueError: if `point_list` is not exactly of length 4 or if coordinates are invalid
        :return: List of strings representing points in hourangle format
        """
        if (len_point_list := len(point_list)) != 4:
            raise ValueError(f'`point_list` must have exactly 4 entries, got {len_point_list}.')
            
        ra_1, ra_2, dec_1, dec_2 = point_list
        
        # Validate RA values are in correct range
        for ra in [ra_1, ra_2]:
            if not 0 <= ra < 360:
                raise ValueError(f'Right ascension must be between 0 and 360 degrees, got {ra}')
                
        # Validate DEC values are in correct range
        for dec in [dec_1, dec_2]:
            if not -90 <= dec <= 90:
                raise ValueError(f'Declination must be between -90 and 90 degrees, got {dec}')
        
        # Validate that RA and DEC ranges make sense
        if ra_1 >= ra_2:
            raise ValueError(f'RA min ({ra_1}) must be less than RA max ({ra_2})')
        if dec_1 >= dec_2:
            raise ValueError(f'DEC min ({dec_1}) must be less than DEC max ({dec_2})')
        
        p1 = SkyCoord(ra_1, dec_1, unit=units.deg)
        p2 = SkyCoord(ra_2, dec_1, unit=units.deg)
        p3 = SkyCoord(ra_2, dec_2, unit=units.deg)
        p4 = SkyCoord(ra_1, dec_2, unit=units.deg)
        result = []
        for point in [p1, p2, p3, p4]:
            ra, dec = get_coordinate_strings(coordinate=point)
            result.append(f'{ra} {dec}')
        return result

    def target_body_list(self) -> List[Any]:
        """ 
        Return a `list` of `ephem.FixedBody`s constructed from the points defined in `self.point_list`.
        Each FixedBody is properly initialized with compute() called on it.
        
        :raises ValueError: if coordinates cannot be parsed correctly
        :return: List of ephem.FixedBody objects representing the corners of the patch
        """
        target_body_list = []
        for i, target in enumerate(self.point_list):
            try:
                ra, dec = target.split(' ')
                x = ephem.FixedBody()
                # Use the public API instead of private attributes when possible
                # But PyEphem's API sometimes requires setting private attributes
                x._ra = ephem.hours(ra)
                x._dec = ephem.degrees(dec)
                x.name = f'p{i}'
                
                # Compute the body's initial position with the current observer
                # This ensures the body is properly initialized
                x.compute(self.ref_antenna.observer)
                
                target_body_list.append(x)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Failed to parse coordinates for point {i}: {target}") from e
        return target_body_list

    def days_list(self) -> List[datetime]:
        """
        Return a `list` of `datetime` objects 
        for each day between `self.from_date` and `self.days_from_now` later.
        :raises OverflowError: if days_from_now is too large for datetime calculations
        :return: List of datetime objects for each day to check
        """
        try:
            return [self.from_date + timedelta(days=d) for d in range(self.days_from_now)]
        except OverflowError as e:
            raise OverflowError(f"days_from_now ({self.days_from_now}) is too large") from e

    def rise_set(self, start: datetime, end: datetime, target: Any) -> List[List]:
        """
        Return start/end positions for target to be observed between a given time interval.
        
        :param start: Start time for the observation window
        :param end: End time for the observation window
        :param target: The celestial object to observe
        :return: A list containing:
            [0] - List of rising times
            [1] - List of elevations at rising times
            [2] - List of setting times
            [3] - List of elevations at setting times
        :raises ValueError: if start time is after end time
        """
        # Validate start and end times
        if start >= end:
            raise ValueError(f"Start time ({start}) must be before end time ({end})")
        
        observer = self.ref_antenna.observer
        el_min = self.min_elevation
        el_max = self.max_elevation
        time_res = timedelta(seconds=self.time_res) # resolution in seconds
        
        # Initialize all flags as False directly (rather than None then False)
        rise_start = False
        rise_end = False
        set_start = False
        set_end = False
        
        time = start
        
        # Add try-except for potential pyephem errors
        try:
            el1 = self.elevation_at(time, observer, target)
        except Exception as e:
            raise ValueError(f"Failed to calculate elevation for {target.name} at {time}: {e}")
        
        rise_time_list = []
        rise_elevation_list = []
        set_time_list = []
        set_elevation_list = []
        
        logger.debug(f"Starting rise/set calculation for {target.name} from {start} to {end}")
        
        while time <= end and (not rise_end or not set_end):   # we can only have one rising and setting in 24h
            try:
                el2 = self.elevation_at(time+time_res, observer, target)
            except Exception as e:
                raise ValueError(f"Failed to calculate elevation for {target.name} at {time+time_res}: {e}")
                
            if rise_start and not rise_end: # we're rising
                rise_time_list.append(time)
                rise_elevation_list.append(el1)
                if el1 >= el_max or el2 <= el1: # rise ends
                    rise_end = True
                    logger.debug(f"Rise ends for {target.name} at {time}, elevation: {el1}")
            elif set_start and not set_end:   # we're setting
                set_time_list.append(time)
                set_elevation_list.append(el1)
                if el1 <= el_min or el2 >= el1:  # set ends
                    set_end = True
                    logger.debug(f"Set ends for {target.name} at {time}, elevation: {el1}")
            elif el1 >= el_min and el1 <= el_max: # start of rising or setting
                # ignore the unusual situation where target only goes above el_min for time_res
                if el2 > el1:    # we're rising
                    rise_time_list.append(time)
                    rise_elevation_list.append(el1)
                    rise_start = True
                    logger.debug(f"Rise starts for {target.name} at {time}, elevation: {el1}")
                elif el2 < el1:  # we're setting
                    set_time_list.append(time)
                    set_elevation_list.append(el1)
                    set_start = True
                    logger.debug(f"Set starts for {target.name} at {time}, elevation: {el1}")
            el1 = el2
            time = time + time_res
            
        return [rise_time_list, rise_elevation_list, set_time_list, set_elevation_list]

    def elevation_curve(self, day: datetime, observer: ephem.Observer, target: Any) -> np.ndarray:
        """
        Return a numpy array with elevation curve for day specified.
        
        :param day: The day to calculate the elevation curve for
        :param observer: The observer (telescope/antenna)
        :param target: The celestial object
        :return: Numpy array of elevations at 1-minute intervals
        """    
        minutes_per_day = 24*60
        elevation_list = []
        for minutes in range(minutes_per_day):
            now = day + timedelta(minutes=minutes)
            elevation_list.append(self.elevation_at(now, observer, target))
            
        elevation_list = np.asarray(elevation_list)
        return elevation_list 
    
    def get_lst(self, observer: Any, date: datetime) -> Any:
        """
        Returns lst in hour:min:sec pyephem format.
        
        :param observer: The observer (telescope/antenna)
        :param date: The date to calculate LST for
        :return: The Local Sidereal Time as a pyephem angle
        """
        date_backup = observer.date
        try:
            observer.date = ephem.Date(date)
            lst = observer.sidereal_time()
            return lst
        finally:  # Ensure observer date is restored even if an exception occurs
            observer.date = date_backup
    
    def is_moon_in_patch(self, start_time: datetime, end_time: datetime) -> bool:
        """
        Check if the moon is within the patch of sky (plus buffer) during the observation window.
        
        :param start_time: Start time of the observation window
        :param end_time: End time of the observation window
        :return: True if the moon is in the patch at any time during the window, False otherwise
        """
        if start_time is None or end_time is None:
            return False
            
        # Extract the RA/Dec bounds from corners and apply buffers
        ra_min, ra_max, dec_min, dec_max = self.corners
        
        # Apply buffers to the boundaries
        ra_min_buffered = max(0, ra_min - self.moon_buffer_ra)
        ra_max_buffered = min(360, ra_max + self.moon_buffer_ra)
        dec_min_buffered = max(-90, dec_min - self.moon_buffer_dec)
        dec_max_buffered = min(90, dec_max + self.moon_buffer_dec)
        
        # Number of points to check (every 20 minutes)
        time_delta = timedelta(minutes=20)
        
        # Create moon object
        moon = ephem.Moon()
        observer = self.ref_antenna.observer
        
        # Check moon position at regular intervals
        current_time = start_time
        while current_time <= end_time:
            try:
                # Calculate moon position
                date_backup = observer.date
                observer.date = ephem.Date(current_time)
                moon.compute(observer)
                
                # Get moon RA and Dec in degrees
                moon_ra = float(moon.ra) * 180.0 / np.pi
                moon_dec = float(moon.dec) * 180.0 / np.pi
                
                # Restore observer date
                observer.date = date_backup
                
                # Check if moon is within the buffered patch bounds
                # Handle RA wrap-around (0/360 degrees)
                in_ra_range = False
                if ra_min_buffered < ra_max_buffered:
                    # Normal case
                    in_ra_range = ra_min_buffered <= moon_ra <= ra_max_buffered
                else:
                    # RA wraps around 0/360
                    in_ra_range = moon_ra >= ra_min_buffered or moon_ra <= ra_max_buffered
                    
                in_dec_range = dec_min_buffered <= moon_dec <= dec_max_buffered
                
                if in_ra_range and in_dec_range:
                    logger.info(f"Moon in patch (with buffer) at {current_time} (RA: {moon_ra:.2f}°, Dec: {moon_dec:.2f}°)")
                    return True
                    
            except Exception as e:
                logger.error(f"Error checking moon position: {e}")
                
            current_time += time_delta
            
        return False

    def print_observation_data(self, day, obs_type, start_time, start_lst, start_el, 
                              start_duration, end_time, end_lst, end_el, end_duration, moon_marker):
        """
        Print observation data in either formatted or CSV format.
        
        :param day: The observation day
        :param obs_type: Either "rise" or "set"
        :param start_time: Start time of observation
        :param start_lst: LST at start time
        :param start_el: Elevation at start time
        :param start_duration: Duration from start
        :param end_time: End time of observation
        :param end_lst: LST at end time
        :param end_el: Elevation at end time
        :param end_duration: Duration from end
        :param moon_marker: "Yes" or "No" indicating moon presence
        """
        day_str = day.strftime("%d/%m/%y")
        start_time_str = start_time.strftime("%d/%m/%y %H:%M")
        end_time_str = end_time.strftime("%d/%m/%y %H:%M")
        
        if self.csv_output:
            print(f'{day_str},{obs_type},{start_time_str},{str(start_lst)},{start_el:.2f},'
                  f'{str(start_duration)},{end_time_str},{str(end_lst)},{end_el:.2f},'
                  f'{str(end_duration)},{moon_marker}')
        else:
            print(f'{day_str:<11}{obs_type:<11}{start_time_str:<17}{str(start_lst):<16}'
                  f'{start_el:<12.2f}{str(start_duration):<11}{end_time_str:<17}{str(end_lst):<16}'
                  f'{end_el:<12.2f}{str(end_duration):<11}{moon_marker:<5}')

    def print_header(self):
        """Print the appropriate header based on output format"""
        if self.csv_output:
            print("day,rise_set,min_start_time,min_start_lst,min_elevation,min_duration,"
                  "max_start_time,max_start_lst,max_elevation,max_duration,moon")
        else:
            columns = [
                ("day", 11),
                ("rise/set", 11),
                ("min_start_time", 17),
                ("min_start_lst", 16),
                ("elevation", 12),
                ("duration", 11),
                ("max_start_time", 17),
                ("max_start_lst", 16),
                ("elevation", 12),
                ("duration", 11),
                ("Moon", 5)
            ]
            header = ""
            for name, width in columns:
                header += f"{name:<{width}}"
            print(header)

    def run(self):
        """
        Calculate and print observable times for the patch of sky.
        Also checks if the moon appears in the patch during each observation window.
        """
        target_body_list = self.target_body_list()
        patch_observed = False
        
        # Calculate elevation resolution without logging it
        el_res = self._calculate_elevation_resolution()
        
        # Print header for results
        self.print_header()
        
        # Process each day
        for day in self.days_list():
            # Get night-time window (between sunset and sunrise)
            sunset_cut, sunrise_cut = self._get_night_window(day)
            
            if sunrise_cut <= sunset_cut:
                # No observable period (sunrise before sunset or equal)
                continue
                
            # Calculate rise/set curves for all targets
            rise_set_curve_list = self._calculate_rise_set_curves(sunset_cut, sunrise_cut, target_body_list)
            
            # Process rising points (if all targets have rising data)
            if all(x[0] for x in rise_set_curve_list):
                patch_observed |= self._process_rising_points(day, rise_set_curve_list, target_body_list, el_res)
            
            # Process setting points (if all targets have setting data)
            if all(x[2] for x in rise_set_curve_list):
                patch_observed |= self._process_setting_points(day, rise_set_curve_list, target_body_list, el_res)
                
            # Create plots if requested
            if self.plot_dir is not None:
                self._create_elevation_plot(day, rise_set_curve_list)
        
        # Report if patch was never observable
        if not patch_observed: 
            print('Not visible')
            logger.info("Patch is not visible within specified parameters")

    def _calculate_elevation_resolution(self):
        """Calculate the elevation resolution based on time resolution."""
        seconds_per_day = 24 * 60 * 60
        degrees_per_second = 360 / seconds_per_day
        return self.time_res * degrees_per_second / 2

    def _get_night_window(self, day):
        """Get the observable window between sunset and sunrise."""
        ephem_day = ephem.Date(day)
        self.ref_antenna.observer.date = ephem_day

        sun = ephem.Sun()
        sunset = self.ref_antenna.observer.next_setting(sun)
        self.ref_antenna.observer.date = sunset
        sunrise = self.ref_antenna.observer.next_rising(sun)
        
        # Apply buffer to sunset/sunrise times
        sunset_cut = ephem.Date(sunset + self.sun_threshold*ephem.minute)
        sunrise_cut = ephem.Date(sunrise - self.sun_threshold*ephem.minute)
        
        logger.debug(f"Day: {day.strftime('%Y-%m-%d')}, Sunset: {sunset}, Sunrise: {sunrise}")
        logger.debug(f"Observation window: {sunset_cut} to {sunrise_cut}")
        
        return sunset_cut, sunrise_cut

    def _calculate_rise_set_curves(self, sunset_cut, sunrise_cut, target_body_list):
        """Calculate rise/set curves for all targets during the night window."""
        rise_set_curve_list = []
        
        for target in target_body_list:
            logger.debug(f"Calculating rise/set for target {target.name}")
            rise_set_curve = self.rise_set(sunset_cut.datetime(), sunrise_cut.datetime(), target)
            rise_set_curve_list.append(rise_set_curve)
            
        return rise_set_curve_list

    def _find_matching_times(self, data_list, target_value, el_res):
        """Find times when elevation is close to target value."""
        times = []
        for x in data_list:
            if len(x[1]) > 0:  # Make sure there are points
                # Find the elevation closest to target_value
                closest_idx = min(range(len(x[1])), key=lambda i: abs(x[1][i]-target_value))
                # Check if it's within our tolerance
                if abs(x[1][closest_idx] - target_value) <= el_res:
                    times.append(x[0][closest_idx])
        return times

    def _process_rising_points(self, day, rise_set_curve_list, target_body_list, el_res):
        """Process rising points and print results if observable."""
        # Find initial and final elevations for rising
        start_el = [x[1][0] for x in rise_set_curve_list]
        rise_min_el = max(start_el)
        
        # Find times matching the minimum elevation
        time = self._find_matching_times(
            [(x[0], x[1]) for x in rise_set_curve_list],  # Times, elevations for rising
            rise_min_el,
            el_res
        )
        
        if len(time) < len(target_body_list):
            return False  # Not all points reach this elevation
            
        rise_min_el_time = min(time)
        end_time = max(time)
        rise_min_el_duration = end_time - rise_min_el_time
        
        # Find times matching the maximum elevation
        end_el = [x[1][-1] for x in rise_set_curve_list]
        rise_max_el = min(end_el)
        
        time = self._find_matching_times(
            [(x[0], x[1]) for x in rise_set_curve_list],  # Times, elevations for rising
            rise_max_el,
            el_res
        )
        
        if len(time) < len(target_body_list):
            return False  # Not all points reach this elevation
            
        rise_max_el_time = min(time)
        end_time = max(time)
        rise_max_el_duration = end_time - rise_max_el_time
        
        # Get LST values
        rise_min_el_lst = self.get_lst(self.ref_antenna.observer, rise_min_el_time)
        rise_max_el_lst = self.get_lst(self.ref_antenna.observer, rise_max_el_time)
        
        # Check for moon in patch
        total_end_time = rise_max_el_time + rise_max_el_duration
        moon_in_patch = self.is_moon_in_patch(rise_min_el_time, total_end_time)
        moon_marker = "Yes" if moon_in_patch else "No"
        
        # Print the results
        self.print_observation_data(
            day=day,
            obs_type="rise",
            start_time=rise_min_el_time,
            start_lst=rise_min_el_lst,
            start_el=rise_min_el,
            start_duration=rise_min_el_duration,
            end_time=rise_max_el_time,
            end_lst=rise_max_el_lst,
            end_el=rise_max_el,
            end_duration=rise_max_el_duration,
            moon_marker=moon_marker
        )
        
        return True  # Patch was observable

    def _process_setting_points(self, day, rise_set_curve_list, target_body_list, el_res):
        """Process setting points and print results if observable."""
        # Find initial and final elevations for setting
        start_el = [x[3][0] for x in rise_set_curve_list]
        set_max_el = min(start_el)  # Going down
        
        # Find times matching the maximum elevation
        time = self._find_matching_times(
            [(x[2], x[3]) for x in rise_set_curve_list],  # Times, elevations for setting
            set_max_el,
            el_res
        )
        
        if len(time) < len(target_body_list):
            return False  # Not all points reach this elevation
            
        set_max_el_time = min(time)
        end_time = max(time)
        set_max_el_duration = end_time - set_max_el_time
        
        # Find times matching the minimum elevation
        end_el = [x[3][-1] for x in rise_set_curve_list]
        set_min_el = max(end_el)
        
        time = self._find_matching_times(
            [(x[2], x[3]) for x in rise_set_curve_list],  # Times, elevations for setting
            set_min_el,
            el_res
        )
        
        if len(time) < len(target_body_list):
            return False  # Not all points reach this elevation
            
        set_min_el_time = min(time)
        end_time = max(time)
        set_min_el_duration = end_time - set_min_el_time
        
        # Get LST values
        set_min_el_lst = self.get_lst(self.ref_antenna.observer, set_min_el_time)
        set_max_el_lst = self.get_lst(self.ref_antenna.observer, set_max_el_time)
        
        # Check for moon in patch
        total_end_time = set_min_el_time + set_min_el_duration
        moon_in_patch = self.is_moon_in_patch(set_max_el_time, total_end_time)
        moon_marker = "Yes" if moon_in_patch else "No"
        
        # Print the results
        self.print_observation_data(
            day=day,
            obs_type="set",
            start_time=set_max_el_time,
            start_lst=set_max_el_lst,
            start_el=set_max_el,
            start_duration=set_max_el_duration,
            end_time=set_min_el_time,
            end_lst=set_min_el_lst,
            end_el=set_min_el,
            end_duration=set_min_el_duration,
            moon_marker=moon_marker
        )
        
        return True  # Patch was observable

    def _create_elevation_plot(self, day, rise_set_curve_list):
        """Create and save elevation plots if plot_dir is specified."""
        plt.figure(figsize=(20, 5))
        for i, x in enumerate(rise_set_curve_list):
            plt.plot(x[0], x[1], label=f'rise {i}')
            plt.plot(x[2], x[3], label=f'set {i}')
        plt.xlabel('time of day UTC [hours]')
        plt.legend()
        plot_path = os.path.join(self.plot_dir, f'riseset_curve_{day.strftime("%d_%m_%y")}.png')
        plt.savefig(plot_path)
        logger.debug(f"Saved plot to {plot_path}")
        plt.close()

    @staticmethod
    def elevation_at(date: datetime, observer: Any, body: Any) -> float:
        """ 
        Return the elevation in degrees of a celestial body at a specific date and time
        as seen by the observer.
        
        This method handles conversions between datetime and PyEphem date formats
        and ensures proper state preservation of the observer object.
        
        :param date: The date and time (in UTC) to calculate the elevation for
        :param observer: The PyEphem observer object (telescope/antenna)
        :param body: The PyEphem celestial body object
        :return: Elevation in degrees (positive above horizon, negative below)
        :raises Exception: If computation fails for any reason
        """
        # Save all observer state that might be modified
        date_backup = observer.date
        # PyEphem doesn't provide a clean way to clone all observer state
        # so we focus on the date which is most commonly changed
        
        try:
            # Convert datetime to PyEphem date - PyEphem expects UTC dates
            observer.date = ephem.Date(date)
            
            # Compute body position - this modifies the body's internal state
            body.compute(observer)
            
            # Convert altitude from radians to degrees
            # PyEphem provides degrees() but it returns a string, so we use manual conversion
            elevation_deg = float(body.alt) * 180.0 / np.pi
            
            # Round to a reasonable precision for astronomical calculations
            # (typically arcminutes or arcseconds are sufficient)
            return round(elevation_deg, 6)
        except Exception as e:
            # Log the specific error for debugging
            logger.error(f"Error calculating elevation: {e}")
            raise
        finally:
            # Always restore the observer's date, even if an exception occurs
            observer.date = date_backup
    
    @staticmethod
    def azimuth_at(date: datetime, observer: Any, body: Any) -> float:
        """ 
        Return the azimuth in degrees of a celestial body at a specific date and time
        as seen by the observer.
        
        This method handles conversions between datetime and PyEphem date formats
        and ensures proper state preservation of the observer object.
        
        :param date: The date and time (in UTC) to calculate the azimuth for
        :param observer: The PyEphem observer object (telescope/antenna)
        :param body: The PyEphem celestial body object
        :return: Azimuth in degrees (0-360, with 0 at North, increasing clockwise)
        :raises Exception: If computation fails for any reason
        """
        date_backup = observer.date
        try:
            observer.date = ephem.Date(date)
            body.compute(observer)
            # Convert azimuth from radians to degrees
            azimuth_deg = float(body.az) * 180.0 / np.pi
            return round(azimuth_deg, 6)
        except Exception as e:
            logger.error(f"Error calculating azimuth: {e}")
            raise
        finally:
            # Always restore the observer's date, even if an exception occurs
            observer.date = date_backup

def load_yaml_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration parameters
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML file is invalid
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"Loaded configuration from {config_path}")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file '{config_path}' not found")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file: {e}")
        raise

def get_config_from_args():
    """
    Parse command line arguments and load configuration from YAML if specified.
    
    Command line arguments override configuration file values.
    
    Returns:
        dict: Complete configuration parameters
    """
    cli = argparse.ArgumentParser(
        description="Calculate when a patch of sky is observable based on elevation constraints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
# Check when a patch is observable using a YAML config file
python when_is_patch_observable_ai.py --config observation_config.yaml

# Override specific config values from command line
python when_is_patch_observable_ai.py --config observation_config.yaml --min_elevation 30 --max_elevation 60

# Use command line arguments without a config file (original mode)
python when_is_patch_observable_ai.py --corners 10 20 -30 -20 --buffer 30 --days 30

# Generate plots in addition to console output
python when_is_patch_observable_ai.py --corners 230 240 10 20 --buffer 30 --plot_dir ./elevation_plots

# Enable detailed logging
python when_is_patch_observable_ai.py --config observation_config.yaml --verbose

# Output in CSV format for spreadsheet import
python when_is_patch_observable_ai.py --config observation_config.yaml --csv > results.csv
        """
    )
    
    # Add config file argument
    cli.add_argument(
        "--config",
        type=str,
        help='path to YAML configuration file',
        required=False
    )
    
    # Keep all existing arguments
    cli.add_argument(
        "--corners",
        nargs=4,
        type=float,
        help='provide right ascension min, max and declination min, max values as degree floats',
        required=False
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
        required=False
    )
    cli.add_argument(
        "--min_elevation",
        nargs=1,
        type=float,
        help='provide min elevation in degrees',
        default=None
    )
    cli.add_argument(
        "--max_elevation",
        nargs=1,
        type=float,
        help='provide max elevation in degrees',
        default=None
    )
    cli.add_argument(
        "--days",
        nargs=1,
        type=int,
        help='provide number of days from input date to consider',
        default=None
    )
    cli.add_argument(
        "--plot_dir",
        nargs=1,
        type=str,
        help='directory to store elevation plots',
        default=None,
    )
    cli.add_argument(
        "--verbose",
        action="store_true",
        help='enable debug logging',
    )
    cli.add_argument(
        "--csv",
        action="store_true",
        help='output in CSV format for easy import into spreadsheets',
    )
    cli.add_argument(
        "--moon-buffer-ra",
        type=float,
        help='buffer in degrees on either side of RA range for moon detection',
        default=None
    )
    cli.add_argument(
        "--moon-buffer-dec",
        type=float,
        help='buffer in degrees above/below Dec range for moon detection',
        default=None
    )
    cli.add_argument(
        "--time-res",
        type=float,
        help='time resolution in seconds for elevation calculations',
        default=None
    )
    
    # Parse arguments
    args = cli.parse_args()
    
    # Set up default configuration
    config = {
        'corners': None,
        'date': None,
        'buffer': None,
        'min_elevation': 35,
        'max_elevation': 50,
        'days': 366,
        'plot_dir': None,
        'verbose': False,
        'csv': False,
        'moon_buffer_ra': 0.0,
        'moon_buffer_dec': 0.0,
        'time_res': 10
    }
    
    # Load from config file if provided
    if args.config:
        try:
            file_config = load_yaml_config(args.config)
            
            # Map YAML keys to config dictionary
            # Handle special case for corners which is a list
            if 'ra_min' in file_config and 'ra_max' in file_config and 'dec_min' in file_config and 'dec_max' in file_config:
                config['corners'] = [
                    file_config['ra_min'],
                    file_config['ra_max'],
                    file_config['dec_min'],
                    file_config['dec_max']
                ]
            
            # Handle date
            if 'start_date' in file_config:
                config['date'] = file_config['start_date']
            
            # Handle days
            if 'days' in file_config:
                config['days'] = file_config['days']
            elif 'end_date' in file_config and config['date']:
                # Calculate days from start_date to end_date
                from datetime import datetime
                start = datetime.strptime(config['date'], '%Y-%m-%d')
                end = datetime.strptime(file_config['end_date'], '%Y-%m-%d')
                config['days'] = (end - start).days + 1
            
            # Map other parameters directly
            mapping = {
                'buffer': 'sunset_buffer',  # or sunrise_buffer
                'min_elevation': 'min_elevation',
                'max_elevation': 'max_elevation',
                'plot_dir': 'plot_dir',
                'verbose': 'verbose',
                'csv': 'csv_output',
                'moon_buffer_ra': 'moon_buffer_ra',
                'moon_buffer_dec': 'moon_buffer_dec',
                'time_res': 'time_res'
            }
            
            for config_key, yaml_key in mapping.items():
                if yaml_key in file_config:
                    config[config_key] = file_config[yaml_key]
                    
            # Handle special case for buffer (can be either sunrise_buffer or sunset_buffer)
            if 'sunrise_buffer' in file_config and 'sunset_buffer' in file_config:
                # Use the average if both are specified
                config['buffer'] = (file_config['sunrise_buffer'] + file_config['sunset_buffer']) // 2
            elif 'sunrise_buffer' in file_config:
                config['buffer'] = file_config['sunrise_buffer']
            elif 'sunset_buffer' in file_config:
                config['buffer'] = file_config['sunset_buffer']
                
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Error with configuration file: {e}")
            sys.exit(1)
    
    # Command line arguments override config file values
    if args.verbose:
        config['verbose'] = True
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
        
    if args.csv:
        config['csv'] = True
        
    if args.corners:
        config['corners'] = args.corners
        
    if args.date:
        config['date'] = args.date[0]
        
    if args.buffer:
        config['buffer'] = args.buffer[0]
        
    if args.min_elevation:
        config['min_elevation'] = args.min_elevation[0]
        
    if args.max_elevation:
        config['max_elevation'] = args.max_elevation[0]
        
    if args.days:
        config['days'] = args.days[0]
        
    if args.plot_dir:
        config['plot_dir'] = args.plot_dir[0]
        
    if args.moon_buffer_ra is not None:
        config['moon_buffer_ra'] = args.moon_buffer_ra
        
    if args.moon_buffer_dec is not None:
        config['moon_buffer_dec'] = args.moon_buffer_dec
        
    if args.time_res is not None:
        config['time_res'] = args.time_res
    
    # Validate required parameters
    if config['corners'] is None:
        if args.config:
            print(f"Error: The 'corners' parameter (or ra_min, ra_max, dec_min, dec_max) must be specified in the config file or with --corners")
        else:
            print("\nError: The --corners argument is required unless using a config file.")
        print("Here's how to use this script:\n")
        cli.print_help()
        sys.exit(1)
        
    if config['buffer'] is None:
        if args.config:
            print(f"Error: The 'buffer' parameter (or sunset_buffer/sunrise_buffer) must be specified in the config file or with --buffer")
        else:
            print("\nError: The --buffer argument is required unless using a config file.")
        print("Here's how to use this script:\n")
        cli.print_help()
        sys.exit(1)
    
    return config


def main():
    """ Run the `run` method with arguments from the command line or config file. """
    
    logger.info("Starting patch observability calculation")

    # Get configuration from arguments and/or config file
    config = get_config_from_args()
    
    # Create the WhenIsPatchObservable instance
    when_is_patch_observable = WhenIsPatchObservable(
        point_list=config['corners'],
        from_date=config['date'],
        buffer=config['buffer'],
        min_elevation=config['min_elevation'],
        max_elevation=config['max_elevation'],
        days_from_now=config['days'],
        plot_dir=config['plot_dir'],
        csv_output=config['csv'],
        moon_buffer_ra=config['moon_buffer_ra'],
        moon_buffer_dec=config['moon_buffer_dec'],
        time_res=config['time_res']
    )
    
    # Run the calculation
    when_is_patch_observable.print()
    when_is_patch_observable.run()
    
if __name__ == '__main__':
    main()

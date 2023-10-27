import re

import cv2
import scipy.ndimage
import sys
from datetime import timedelta

import katpoint
import numpy as np
from astropy import units
from astropy.coordinates import AltAz, ITRS
from astropy.coordinates import EarthLocation, SkyCoord, ICRS
from astropy.time import Time
from katpoint import Antenna
from matplotlib import pyplot as plt

from museek.data_element import DataElement
from museek.time_ordered_data_mapper import TimeOrderedDataMapper

MEERKAT_REFERENCE_LOCATION = "ref, -30:42:39.8, 21:26:38.0, 1035.0, 0.0, , , 1.15"


def get_tod_from_simulation_output(simulation_output: list[str], reference_antenna: Antenna):
    first_time_plotting = True
    first_time = True
    ra = []
    dec = []
    duration = None
    scan_extent_a = None
    scan_extent_b = None
    for line_ in simulation_output:
        # print(line_)
        if "Scan duration is" in line_:
            duration = float(re.split("Scan duration is|and scan speed is", line_)[1])
        elif 'Azimuth scan extent ' in line_:
            extent_str = line_.split('extent ')[1][:-1]
            scan_extent_a, scan_extent_b = np.asarray(extent_str.strip('[').strip(']').split(', '), float)
        elif 'Slewed to scan_azel_with_nd_trigger' in line_ and duration and scan_extent_a and scan_extent_b:
            # if first_time:
            #     first_time = False
            #     continue
            time_ = line_.split(' - ')[0]
            azel_str = line_.split('azel ')[1][:-5]
            az, el = np.asarray(azel_str.strip('[(').strip(')]').split(', '), float)

            label = ''
            ra_swing = []
            dec_swing = []
            observing_time = Time(time_, scale='utc')
            # alt_az_frame = AltAz(location=location_m000, obstime=observing_time)
            for azz, off_time in zip(np.linspace(az + scan_extent_a, az + scan_extent_b),
                                     np.linspace(0, duration)):
                target = katpoint.Target("point, azel , %f ,%f" % (azz, el))
                radec = np.degrees(target.radec(timestamp=katpoint.Timestamp(time_.replace('Z', '')) + off_time,
                                                antenna=reference_antenna))
                ra_swing.append(radec[0])
                dec_swing.append(radec[1])

                # coord_alt_az = SkyCoord(el, azz, unit=units.deg, frame=alt_az_frame)
                # coord_icrs = coord_alt_az.transform_to(ICRS)
                # ra.append(coord_icrs.ra.deg)
                # dec.append(coord_icrs.dec.deg)
            ra.extend(ra_swing)
            dec.extend(dec_swing)
            # plt.plot(ra_swing, dec_swing, color='black', label=label)
    return ra, dec


def main(output_file_paths: list[str],
         labels: list[str]):
    desi_1_rising_corners = dict(
        ra_min=140.23,
        ra_max=154.00,
        dec_min=-8.85,
        dec_max=5.16,
    )

    desi_1_setting_corners = dict(
        ra_min=141.99,
        ra_max=155.99,
        dec_min=-8.83,
        dec_max=5.53,
    )

    desi_2_rising_corners = dict(
        ra_min=174.01,
        ra_max=160.22,
        dec_min=-8.85,
        dec_max=5.45,
    )

    desi_2_setting_corners = dict(
        ra_max=175.98,
        ra_min=161.99,
        dec_min=-8.85,
        dec_max=5.54,
    )
    default_colours = ['#1f77b4',
                       '#ff7f0e',
                       '#2ca02c',
                       '#d62728',
                       '#9467bd',
                       '#8c564b',
                       '#e377c2',
                       '#7f7f7f',
                       '#bcbd22',
                       '#17becf']

    ref_antenna = katpoint.Antenna(MEERKAT_REFERENCE_LOCATION)

    # # convert utc to lst
    # start_time = '2023-01-20 23:00:00'
    # end_time = '2023-01-20 04:00:00'
    # print(ref_antenna.local_sidereal_time(timestamp=start_time))
    # print(ref_antenna.local_sidereal_time(timestamp=end_time))
    all_ra = []
    all_dec = []
    all_to_map = []
    map_value = 1

    for output_file_path, label in zip(output_file_paths, labels):
        with open(output_file_path, 'r') as output_file:
            ra, dec = get_tod_from_simulation_output(simulation_output=output_file.readlines(),
                                                     reference_antenna=ref_antenna)
        # plt.scatter(ra[0], dec[0], color='blue', marker='o')
        plt.plot(ra, dec, label=label)
        all_ra.extend(ra)
        all_dec.extend(dec)
        all_to_map.extend([map_value for _ in ra])
        map_value += 1

    for corners, corner_color in zip([desi_1_rising_corners,
                                      desi_1_setting_corners,
                                      desi_2_rising_corners,
                                      desi_2_setting_corners],
                                     default_colours):
        p1 = SkyCoord(corners['ra_max'], corners['dec_min'], unit=units.deg)
        p2 = SkyCoord(corners['ra_min'], corners['dec_min'], unit=units.deg)
        p3 = SkyCoord(corners['ra_min'], corners['dec_max'], unit=units.deg)
        p4 = SkyCoord(corners['ra_max'], corners['dec_max'], unit=units.deg)
        for point, label, marker in zip([p1, p2, p3, p4], ['p1', 'p2', 'p3', 'p4'], ['x', '+', '<', 'o']):
            plt.scatter(point.ra.deg, point.dec.deg, marker=marker, color=corner_color)

    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.legend()
    plt.show()

    right_ascension = DataElement(array=np.asarray(all_ra)[:, np.newaxis, np.newaxis])
    declination = DataElement(array=np.asarray(all_dec)[:, np.newaxis, np.newaxis])
    to_map = DataElement(array=np.asarray(all_to_map)[:, np.newaxis, np.newaxis])
    maps, _ = TimeOrderedDataMapper(right_ascension=right_ascension,
                                    declination=declination,
                                    to_map=to_map).grid(grid_size=(60, 60), method='nearest')
    maps = maps[0]
    convolution_kernel = np.array([[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1]])/9
    filtered = cv2.filter2D(np.asarray(maps, float), -1, convolution_kernel)
    mask = np.ones_like(maps)
    mask[abs(maps-filtered)<1e-3] = 0
    mask = scipy.ndimage.binary_closing(mask)
    mask = scipy.ndimage.binary_erosion(mask, iterations=2)
    mask = scipy.ndimage.binary_dilation(mask, iterations=2)

    ra_range = max(right_ascension) - min(right_ascension)
    dec_range = max(declination) - min(declination)
    area_per_pixel = ra_range * dec_range / 60**2

    mask_area = area_per_pixel * np.sum(mask)
    print(f'the mask area is {mask_area} square degrees.')

    plt.imshow(mask*maps)
    plt.show()




if __name__ == '__main__':
    # main(['/home/amadeus/git/astrokat-helper/output/desi_1_rising_observe.txt',
    #       '/home/amadeus/git/astrokat-helper/output/desi_1_setting_observe.txt',
    #       '/home/amadeus/git/astrokat-helper/output/desi_2_rising_observe.txt',
    #       '/home/amadeus/git/astrokat-helper/output/desi_2_setting_observe.txt'],
    #      ['desi 1 rising', 'desi 1 setting', 'desi 2 rising', 'desi 2 setting'])
    main(['/home/amadeus/git/astrokat-helper/output/desi_2_rising_no_initial_calibrators_observe.txt'],
         ['desi 2 rising no initial calibrators'])
    # main(['/home/amadeus/git/astrokat-helper/output/desi_2_rising_observe.txt',
    #       '/home/amadeus/git/astrokat-helper/output/desi_2_setting_observe.txt'],
    #      ['desi 2 rising', 'desi 2 setting'])

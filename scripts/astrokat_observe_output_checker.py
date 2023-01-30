import re
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

MEERKAT_REFERENCE_LOCATION = "ref, -30:42:39.8, 21:26:38.0, 1035.0, 0.0, , , 1.15"


def get_tod_from_simulation_output(simulation_output: list[str], reference_antenna: Antenna):
    first_time_plotting = True
    first_time = True
    ra = []
    dec = []
    for line_ in simulation_output:
        # print(line_)
        if "Scan duration is" in line_:
            duration = float(re.split("Scan duration is|and scan speed is", line_)[1])
        elif 'Azimuth scan extent ' in line_:
            extent_str = line_.split('extent ')[1][:-1]
            scan_extent_a, scan_extent_b = np.asarray(extent_str.strip('[').strip(']').split(', '), float)
        elif 'Slewed to scan_azel_with_nd_trigger' in line_:
            if first_time:
                first_time = False
                continue
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

    for output_file_path, label in zip(output_file_paths, labels):
        with open(output_file_path, 'r') as output_file:
            ra, dec = get_tod_from_simulation_output(simulation_output=output_file.readlines(),
                                                     reference_antenna=ref_antenna)
        # plt.scatter(ra[0], dec[0], color='blue', marker='o')
        plt.plot(ra, dec, label=label)

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


if __name__ == '__main__':
    main(['/home/amadeus/git/astrokat-helper/output/desi_1_rising_observe.txt',
          '/home/amadeus/git/astrokat-helper/output/desi_1_setting_observe.txt',
          '/home/amadeus/git/astrokat-helper/output/desi_2_rising_observe.txt',
          '/home/amadeus/git/astrokat-helper/output/desi_2_setting_observe.txt'],
         ['desi 1 rising', 'desi 1 setting', 'desi 2 rising', 'desi 2 setting'])

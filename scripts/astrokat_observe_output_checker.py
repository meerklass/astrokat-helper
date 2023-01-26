import re
import sys
from datetime import timedelta

import katpoint
import numpy as np
from astropy import units
from astropy.coordinates import AltAz, ITRS
from astropy.coordinates import EarthLocation, SkyCoord, ICRS
from astropy.time import Time
from matplotlib import pyplot as plt

MEERKAT_REFERENCE_LOCATION = "ref, -30:42:39.8, 21:26:38.0, 1035.0, 0.0, , , 1.15"


def main(output_file_path: str):
    # must be as in yaml:
    # # DESI 1 rising
    # ra_min = 140.23
    # ra_max = 154.00
    # dec_min = -8.85
    # dec_max = 5.16

    # # DESI 1 setting
    # ra_min = 141.99
    # ra_max = 155.99
    # dec_min = -8.83
    # dec_max = 5.53

    # # DESI 2 rising
    # ra_min = 174.01
    # ra_max = 160.22
    # dec_min = -8.85
    # dec_max = 5.45

    # DESI 2 setting
    ra_max = 175.98
    ra_min = 161.99
    dec_min = -8.85
    dec_max = 5.54

    p1 = SkyCoord(ra_max, dec_min, unit=units.deg)
    p2 = SkyCoord(ra_min, dec_min, unit=units.deg)
    p3 = SkyCoord(ra_min, dec_max, unit=units.deg)
    p4 = SkyCoord(ra_max, dec_max, unit=units.deg)

    ref_antenna = katpoint.Antenna(MEERKAT_REFERENCE_LOCATION)

    # convert utc to lst

    # start_time = Time('2023-01-20 23:00:00', scale='utc')
    # end_time = Time('2023-01-20 04:00:00', scale='utc')
    start_time = '2023-01-20 23:00:00'
    end_time = '2023-01-20 04:00:00'
    print(ref_antenna.local_sidereal_time(timestamp=start_time))
    print(ref_antenna.local_sidereal_time(timestamp=end_time))

    with open(output_file_path, 'r') as output_file:
        first_time_plotting = True
        first_time = True
        for line_ in output_file.readlines():
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
                ra = []
                dec = []
                observing_time = Time(time_, scale='utc')
                # alt_az_frame = AltAz(location=location_m000, obstime=observing_time)
                for azz, off_time in zip(np.linspace(az + scan_extent_a, az + scan_extent_b),
                                         np.linspace(0, duration)):
                    target = katpoint.Target("point, azel , %f ,%f" % (azz, el))
                    radec = np.degrees(target.radec(timestamp=katpoint.Timestamp(time_.replace('Z', '')) + off_time,
                                                    antenna=ref_antenna))
                    ra.append(radec[0])
                    dec.append(radec[1])

                    # coord_alt_az = SkyCoord(el, azz, unit=units.deg, frame=alt_az_frame)
                    # coord_icrs = coord_alt_az.transform_to(ICRS)
                    # ra.append(coord_icrs.ra.deg)
                    # dec.append(coord_icrs.dec.deg)
                if first_time_plotting:
                    label = 'astrokat simulation scan lines'
                    plt.scatter(ra[0], dec[0], color='blue', marker='o', label='start')
                    first_time_plotting = False
                plt.plot(ra, dec, color='black', label=label)

    # ra = np.asarray(ra)
    # dec = np.asarray(dec)
    #
    # plt.scatter(ra, dec)
    for point, label, marker in zip([p1, p2, p3, p4], ['p1', 'p2', 'p3', 'p4'], ['x', '+', '<', 'o']):
        plt.scatter(point.ra.deg, point.dec.deg, marker=marker, color='red', label=label)

    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.legend()
    plt.show()

    # azel = (51.6, 47.6)  # directly from output
    # alt = azel[1]
    # azz = azel[0]
    # scan_extent = [-15.9, 15.9]
    # print('This should match the outer corners of the scan:')
    # for observing_time, time_label in zip([start_time, end_time], ['start_time', 'end_time']):
    #     print(time_label)
    #     for az in [azz + scan_extent[0], azz + scan_extent[1]]:
    #         alt_az_frame = AltAz(location=location_m000, obstime=observing_time)
    #         coord_alt_az = SkyCoord(alt, az, unit=units.deg, frame=alt_az_frame)
    #         coord_icrs = coord_alt_az.transform_to(ICRS)
    #         # print(coord_alt_az)
    #         print(coord_icrs)


if __name__ == '__main__':
    main(sys.argv[1])

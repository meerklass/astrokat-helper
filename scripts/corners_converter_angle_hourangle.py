from astropy import units
from astropy.coordinates import SkyCoord

from scripts.yaml_offseter import get_coordinate_strings


def main():
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
    # ra_max = 174.01
    # ra_min = 160.22
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

    for point in [p1, p2, p3, p4]:
        ra, dec = get_coordinate_strings(coordinate=point)
        print(f'{ra}, {dec}')


if __name__ == '__main__':
    main()

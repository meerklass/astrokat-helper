import sys

from astropy import units
from astropy.coordinates import SkyCoord, Angle

from astrokat import read_yaml
from scripts.yaml_offseter import get_coordinate_strings


CALIBRATOR_TARGETS = {  # degrees
    'PictorA': (79.95717083, -45.77882781),
    '3C273': (187.277915, 2.052388),
    'HydraA': (139.523620, -12.095543),
    '3C353': (260.117326, -0.979617),
    'CentaurusA': (201.365063, -43.019113)
}


def check_calibrator_pointing(true: SkyCoord, from_yaml: SkyCoord, offset: str):
    tolerance = 1e-5

    if offset:
        offset_value = float(offset[1:])
        offset_angle = Angle(offset_value, unit=units.deg)
        offsets = {
            'u0.8': (Angle(0, unit=units.deg), offset_angle),
            'd0.8': (Angle(180, unit=units.deg), offset_angle),
            'r0.8': (Angle(90, unit=units.deg), offset_angle),
            'l0.8': (Angle(270, unit=units.deg), offset_angle)
        }
        true = true.directional_offset_by(position_angle=offsets[offset][0],
                                          separation=offsets[offset][1])

    separation = true.separation(from_yaml).to(units.deg)
    true_string = '{:02.0f}:{:02.0f}:{:02.5f} {:02.0f}:{:02.0f}:{:02.5f}'.format(*true.ra.hms, *true.dec.dms)
    if separation.deg > tolerance:
        position_angle = true.position_angle(from_yaml).to(units.deg)
        print(f'Warning! Calibrator pointing is off by {separation} in direction of {position_angle}.\n'
              f'True position is {true.ra.hms} {true.dec.dms} {true_string}')
    else:
        print('Calibrator pointing okay.')


def extract_target_offset(target: str) -> str:
    try:
        return target.split(',')[0].split('=')[1].split('_')[1]
    except IndexError:
        return ''


def extract_target_name(target: str) -> str:
    return target.split(',')[0].split('=')[1].split('_')[0]


def extract_ra_dec(target: str) -> SkyCoord:
    ra_dec_hourangle = target.split(',')[1].lstrip('radec= ')
    ra_dec = SkyCoord(ra_dec_hourangle, unit=(units.hourangle, units.deg))
    return ra_dec


def get_calibrator_target_names(target_list: list[str]) -> list[str]:
    target_names: list[str] = []
    for target in target_list:
        if not target.startswith('name=scan_azel_with_nd_trigger'):
            target_names.append(extract_target_name(target=target))
    target_names = list(set(target_names))
    return target_names


def check_calibrator_targets(target_list: list[str]):
    calibrator_target_names = get_calibrator_target_names(target_list=target_list)
    for target in target_list:
        target_name = extract_target_name(target=target)
        if target_name in calibrator_target_names:
            if target_name not in CALIBRATOR_TARGETS:
                print(f'WARNING: target_list contains unknown targets. add {target_name} to TARGETS.')
                continue
            offset = extract_target_offset(target=target)
            true = SkyCoord(*CALIBRATOR_TARGETS[target_name], unit=units.deg)
            from_yaml = extract_ra_dec(target=target)
            print(target_name + ' ' + offset)
            check_calibrator_pointing(true=true, from_yaml=from_yaml, offset=offset)
            print('*' * 20)


def get_scan_target(target_list: list[str]) -> str | None:
    for target in target_list:
        target_name = extract_target_name(target=target)
        if target_name == 'scan':
            return target


def get_scan_center_from_corners(p1: SkyCoord, p2: SkyCoord, p3: SkyCoord, p4: SkyCoord) -> SkyCoord:
    mean = [0, 0]  # ra, dec

    for point in [p1, p2, p3, p4]:
        ra = point.ra
        dec = point.dec
        if point.ra > 180 * units.deg:
            ra = point.ra - 360 * units.deg
        if point.dec > 180 * units.deg:
            dec = point.dec - 360 * units.deg
        mean[0] += ra
        mean[1] += dec

    mean[0] /= 4
    mean[1] /= 4

    return SkyCoord(mean[0], mean[1])


def check_scan_pointing_center(center_from_corners: SkyCoord, center: SkyCoord):
    tolerance = 1e-5
    if (separation := center_from_corners.separation(center).deg) > tolerance:
        print(f'Warning! Scan pointing center is off by {separation}.'
              f' The true center should be {center_from_corners.ra.hms} {center_from_corners.dec.dms}.')
    else:
        print('Scan pointing okay')


def main(yaml_file_path: str):
    data = read_yaml(yaml_file_path)

    target_list = data['observation_loop'][0]['target_list']
    check_calibrator_targets(target_list=target_list)

    # now the scanning part
    p1 = SkyCoord(data['scan']['radec_p1'].replace(',', ''), unit=(units.hourangle, units.deg))
    p2 = SkyCoord(data['scan']['radec_p2'].replace(',', ''), unit=(units.hourangle, units.deg))
    p3 = SkyCoord(data['scan']['radec_p3'].replace(',', ''), unit=(units.hourangle, units.deg))
    p4 = SkyCoord(data['scan']['radec_p4'].replace(',', ''), unit=(units.hourangle, units.deg))
    scan_center_from_corners = get_scan_center_from_corners(p1=p1, p2=p2, p3=p3, p4=p4)
    scan_target = get_scan_target(target_list=target_list)
    scan_pointing_center = SkyCoord(scan_target.split(',')[1].lstrip(' radec= '), unit=(units.hourangle, units.deg))

    check_scan_pointing_center(center_from_corners=scan_center_from_corners, center=scan_pointing_center)


if __name__ == '__main__':
    for target_name, degrees in CALIBRATOR_TARGETS.items():
        coordinate_strings = get_coordinate_strings(SkyCoord(*degrees, unit=units.deg))
        print(target_name, coordinate_strings[0], coordinate_strings[1])

    main(sys.argv[1])
    print('Checks completed')
    print('*' * 20)

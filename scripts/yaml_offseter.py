import sys

from astropy import units
from astropy.coordinates import SkyCoord


def contains_scan_coordinates(line_: str) -> bool:
    if ('radec_p1' in line_
            or 'radec_p2' in line_
            or 'radec_p3' in line_
            or 'radec_p4' in line_
            or 'scan_azel_with_nd_trigger' in line_):
        return True
    return False


def get_coordinate_strings(coordinate: SkyCoord) -> tuple[str, str]:
    new_ra_string = ''
    new_ra_h, new_ra_m, new_ra_s = coordinate.ra.hms
    if coordinate.ra < 0:
        new_ra_string += '-'
        new_ra_h *= -1
        new_ra_m *= -1
        new_ra_s *= -1
    new_ra_string += f'{new_ra_h:02.0f}' \
                     f':{new_ra_m:02.0f}' \
                     f':{new_ra_s:05.2f}'

    new_dec_string = ''
    new_dec_d, new_dec_m, new_dec_s = coordinate.dec.dms
    if coordinate.dec < 0:
        new_dec_string += '-'
        new_dec_d *= -1
        new_dec_m *= -1
        new_dec_s *= -1
    new_dec_string += f'{new_dec_d:02.0f}' \
                      f':{new_dec_m:02.0f}' \
                      f':{new_dec_s:05.2f}'
    return new_ra_string, new_dec_string


def offset_coordinate(coordinate: SkyCoord, offset_angle: float, offset_direction: str) -> SkyCoord:
    new_ra = coordinate.ra.deg
    new_dec = coordinate.dec.deg
    if offset_direction == 'ra':
        new_ra += offset_angle
    elif offset_direction == 'dec':
        new_dec += offset_angle
    return SkyCoord(new_ra, new_dec, unit=units.deg)


def offset_coordinates_line(line_: str, offset_angle: float, offset_direction: str) -> str:
    if 'name=scan_azel_with_nd_trigger' in line_:
        parts = line_.split(', ')
        ra_dec_string = parts[1].split('= ')[1]
        sky_coordinate = SkyCoord(ra_dec_string, unit=(units.hourangle, units.deg))
        new_sky_coordinate = offset_coordinate(coordinate=sky_coordinate,
                                               offset_angle=offset_angle,
                                               offset_direction=offset_direction)
        new_ra_string, new_dec_string = get_coordinate_strings(coordinate=new_sky_coordinate)
        new_line = f'{parts[0]}, radec= {new_ra_string} {new_dec_string}, {parts[2]}, {parts[3]}, {parts[4]}'
        return new_line
    no_comment, comment = line_.split('#')
    corner_name = no_comment.split(':')[0]
    ra, dec = no_comment[len(corner_name) + 1:].strip(' ').split(',')

    sky_coordinate = SkyCoord(ra + dec, unit=(units.hourangle, units.deg))
    new_sky_coordinate = offset_coordinate(coordinate=sky_coordinate,
                                           offset_angle=offset_angle,
                                           offset_direction=offset_direction)

    new_ra_string, new_dec_string = get_coordinate_strings(coordinate=new_sky_coordinate)

    new_line = f'{corner_name}: {new_ra_string}, {new_dec_string}  #{comment}'
    return new_line


def main(offset_angle: str, offset_direction: str, yaml_file_path: str, output_file_path: str):
    """ offset direction needs to be either 'ra' or 'dec'. """
    offset_angle = float(offset_angle)
    offset_yaml = ''
    with open(yaml_file_path) as yaml_file:
        for line_ in yaml_file.readlines():
            if contains_scan_coordinates(line_):
                offset_yaml += offset_coordinates_line(line_, offset_angle, offset_direction)
            else:
                offset_yaml += line_

    with open(output_file_path, 'w') as output_file:
        output_file.write(offset_yaml)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

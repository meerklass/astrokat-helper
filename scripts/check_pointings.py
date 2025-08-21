from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy import units as u
import numpy as np

def find_same_elevation_points(ra_deg, dec_deg,
                                deltat_sec=80, angular_sep_deg=0.8, elev_tol_deg=0.003,
                                n_samples=1000, t0=None):
    # MeerKAT telescope location
    meerkat_lat = -30.7130    # degrees
    meerkat_lon = 21.4430     # degrees
    meerkat_elev = 1050       # meters

    # Observer's location (MeerKAT)
    location = EarthLocation(lat=meerkat_lat*u.deg, lon=meerkat_lon*u.deg, height=meerkat_elev*u.m)

    # Observation times
    if t0 is None:
        t0 = Time.now()
    t1 = t0 + deltat_sec * u.second

    # Original sky position
    original_coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')

    # AltAz frames
    altaz0 = AltAz(obstime=t0, location=location)
    altaz1 = AltAz(obstime=t1, location=location)

    # Compute original elevation
    orig_altaz = original_coord.transform_to(altaz0)
    original_elevation = orig_altaz.alt.deg

    # Generate circle of points around original_coord with 0.8° angular separation
    thetas = np.linspace(0, 2*np.pi, n_samples)
    circle_points = original_coord.directional_offset_by(position_angle=thetas*u.rad,
                                                         separation=angular_sep_deg*u.deg)

    # Transform circle points to AltAz at t1
    altaz_circle = circle_points.transform_to(altaz1)

    # Filter by elevation closeness
    good_points = []
    for coord, altaz in zip(circle_points, altaz_circle):
        if abs(altaz.alt.deg - original_elevation) <= elev_tol_deg:
            good_points.append((coord.ra.deg, coord.dec.deg, altaz.alt.deg))

    return good_points, original_elevation, t0

# Example usage
if __name__ == "__main__":
    from astropy.time import Time

#    Pictor A
    ra0 = 79.957171      # degrees
    dec0 = -45.778828      # degrees

    # Explicit time (optional)
    custom_t0 = Time("2025-10-16T07:20:00", scale="utc")

    matches, original_alt, used_t0 = find_same_elevation_points(
        ra0, dec0, t0=custom_t0)

    print(f"Time used (UTC): {used_t0.iso}")
    print(f"Original Elevation at t0: {original_alt:.4f}°")
    print(f"Matching points at t+80s (RA, Dec, Elevation):")
    for ra, dec, alt in matches:
        print(f"RA: {ra-ra0:.4f}°, Dec: {dec-dec0:.4f}°, Elev: {alt:.4f}°")

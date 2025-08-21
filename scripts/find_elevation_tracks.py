import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from scipy.spatial import cKDTree

def find_points_across_elevations(target_ra, target_dec, elevation_range, 
                               dt=80*u.second, elev_error=0.01*u.deg,
                               angular_separation=0.8*u.deg, angular_sep_error=0.01*u.deg,
                               coord_error=0.01*u.deg, n_points=100, n_radii=5):
    """
    Find sky points that:
    1. Are at the same elevation as the target point after time dt
    2. Are separated from the target by the specified angular separation (within error)
    3. Have consistent positions across the entire elevation range
    
    Parameters:
    -----------
    target_ra, target_dec : float or Quantity
        RA and Dec of target point (in degrees if float)
    elevation_range : list or tuple of Quantity
        Range of elevations to check [min_elevation, max_elevation]
    dt : Quantity
        Time delay after which to check elevation
    elev_error : Quantity
        Allowed error in elevation matching
    angular_separation : Quantity
        Desired angular separation between target and output points
    angular_sep_error : Quantity
        Allowed error in angular separation
    coord_error : Quantity
        Maximum error in RA and Dec to consider two points the same
    n_points : int
        Number of test points to generate around each circle
    n_radii : int
        Number of radii to test within the allowed error range
    
    Returns:
    --------
    Dictionary with 'rising' and 'setting' lists of SkyCoord objects meeting the criteria
    """
    # MeerKAT location
    meerkat = EarthLocation.from_geodetic(
        lon=21.4439*u.deg, 
        lat=-30.7112*u.deg, 
        height=1086*u.m
    )
    
    # Ensure quantities have proper units
    if not isinstance(target_ra, u.Quantity):
        target_ra = target_ra * u.deg
    if not isinstance(target_dec, u.Quantity):
        target_dec = target_dec * u.deg
    if not isinstance(dt, u.Quantity):
        dt = dt * u.second
    
    # Make sure elevation_range is a list/tuple of quantities
    if not isinstance(elevation_range[0], u.Quantity):
        elevation_range = [el * u.deg for el in elevation_range]
    
    # Create target SkyCoord
    target = SkyCoord(ra=target_ra, dec=target_dec, frame='icrs')
    
    # Sample elevations within the range
    n_elevations = 10  # Number of elevation samples
    elevations = np.linspace(elevation_range[0].value, 
                           elevation_range[1].value, 
                           n_elevations) * u.deg
    
    # Lists to store matching points at each elevation
    rising_matches_by_elev = []
    setting_matches_by_elev = []
    
    # Generate test points in the annular region around the target
    test_points = []
    
    # Test multiple radii within the allowed error range
    min_sep = angular_separation - angular_sep_error
    max_sep = angular_separation + angular_sep_error
    
    # Create array of radii to test
    separations = np.linspace(min_sep.value, max_sep.value, n_radii) * angular_separation.unit
    
    # For each radius, create points around the circle
    for sep in separations:
        position_angles = np.linspace(0, 360, n_points) * u.deg
        for pa in position_angles:
            # Create a new point at the specified separation and position angle
            sep_point = target.directional_offset_by(pa, sep)
            test_points.append(sep_point)
    
    # Simplified function to convert LST to UTC time
    def lst_to_time(lst):
        # Use a reference time and iterate to find the correct time for the given LST
        t = Time('2000-01-01T00:00:00', scale='utc')
        
        # Calculate initial LST at reference time
        initial_lst = t.sidereal_time('apparent', meerkat.lon)
        
        # Calculate hour angle difference
        lst_diff = (lst - initial_lst).wrap_at(24*u.hourangle)
        
        # Convert to time (approximation)
        # Sidereal day is about 23h 56m 4.1s = 0.99727 solar days
        t = t + (lst_diff.hour * 0.99727 * u.hour)
        
        # Fine tune with a couple of iterations
        for _ in range(2):
            current_lst = t.sidereal_time('apparent', meerkat.lon)
            lst_diff = (lst - current_lst).wrap_at(24*u.hourangle)
            t = t + (lst_diff.hour * 0.99727 * u.hour)
        
        return t
    
    # Process each elevation in the range (minimal output)
    print(f"Processing {n_elevations} elevations from {elevation_range[0]} to {elevation_range[1]}...")
    
    for target_elevation in elevations:
        # Calculate hour angle for this elevation
        lat = meerkat.lat.rad
        dec = target.dec.rad
        alt = target_elevation.to(u.rad).value
        
        # Calculate the cosine of the hour angle
        cos_ha = (np.sin(alt) - np.sin(dec) * np.sin(lat)) / (np.cos(dec) * np.cos(lat))
        
        # Skip if target cannot reach this elevation
        if abs(cos_ha) > 1:
            continue
        
        # Calculate hour angles (in radians, then convert to angle)
        ha_rising = np.arccos(cos_ha) * u.rad  # When the object is rising (positive HA)
        ha_setting = -np.arccos(cos_ha) * u.rad  # When the object is setting (negative HA)
        
        # Calculate LST when target is at the specified elevation
        lst_rising = (target.ra - ha_rising).wrap_at(360 * u.deg).to(u.hourangle)
        lst_setting = (target.ra - ha_setting).wrap_at(360 * u.deg).to(u.hourangle)
        
        # Convert LST to actual time
        time_rising = lst_to_time(lst_rising)
        time_setting = lst_to_time(lst_setting)
        
        # Add the time delay
        time_rising_later = time_rising + dt
        time_setting_later = time_setting + dt
        
        # Create AltAz frames for the original and delayed times
        frame_rising = AltAz(obstime=time_rising, location=meerkat)
        frame_setting = AltAz(obstime=time_setting, location=meerkat)
        frame_rising_later = AltAz(obstime=time_rising_later, location=meerkat)
        frame_setting_later = AltAz(obstime=time_setting_later, location=meerkat)
        
        # Get test points' elevations after time delay
        test_points_rising_altaz = [p.transform_to(frame_rising_later) for p in test_points]
        test_points_setting_altaz = [p.transform_to(frame_setting_later) for p in test_points]
        
        # Find matching points
        rising_matches = []
        setting_matches = []
        
        for i, point_altaz in enumerate(test_points_rising_altaz):
            if abs(point_altaz.alt - target_elevation) <= elev_error:
                rising_matches.append(test_points[i])
        
        for i, point_altaz in enumerate(test_points_setting_altaz):
            if abs(point_altaz.alt - target_elevation) <= elev_error:
                setting_matches.append(test_points[i])
        
        # Store matches for this elevation
        rising_matches_by_elev.append(rising_matches)
        setting_matches_by_elev.append(setting_matches)
    
    # Find points common across all elevations using a distance threshold
    def find_common_points(matches_by_elev, coord_error):
        if not matches_by_elev or not all(matches_by_elev):
            return []
        
        # List of points that appear consistently across elevations
        common_points = []
        
        # Convert coord_error to degrees for comparison
        error_deg = coord_error.to(u.deg).value
        
        # Treat the first elevation's points as candidates
        for candidate in matches_by_elev[0]:
            # Check if this candidate appears in all other elevations
            consistent = True
            
            for elev_matches in matches_by_elev[1:]:
                # Convert candidate to (ra, dec) in degrees for comparison
                candidate_coords = np.array([candidate.ra.deg, candidate.dec.deg])
                
                # Convert all matches at this elevation to (ra, dec) arrays
                elev_coords = np.array([[p.ra.deg, p.dec.deg] for p in elev_matches])
                
                # Use KDTree for efficient nearest neighbor search
                if len(elev_coords) > 0:
                    tree = cKDTree(elev_coords)
                    dist, idx = tree.query(candidate_coords)
                    
                    # Check if nearest match is within tolerance
                    if dist > error_deg:
                        consistent = False
                        break
                else:
                    # No matches at this elevation
                    consistent = False
                    break
            
            if consistent:
                common_points.append(candidate)
        
        return common_points
    
    # Find common points for rising and setting
    common_rising = find_common_points(rising_matches_by_elev, coord_error)
    common_setting = find_common_points(setting_matches_by_elev, coord_error)
    
    # Remove duplicate points (points with nearly identical RA, Dec)
    def remove_duplicates(points, coord_error):
        if not points:
            return []
        
        # Convert coord_error to degrees
        error_deg = coord_error.to(u.deg).value
        
        # List to store unique points
        unique_points = []
        
        # Convert all points to (ra, dec) arrays
        points_coords = np.array([[p.ra.deg, p.dec.deg] for p in points])
        
        # Keep track of which points have been added
        used = np.zeros(len(points), dtype=bool)
        
        # For each point
        for i in range(len(points)):
            if used[i]:
                continue
                
            # Mark this point as used
            used[i] = True
            unique_points.append(points[i])
            
            # Mark all similar points as used
            for j in range(i+1, len(points)):
                if not used[j]:
                    dist = np.sqrt((points_coords[i,0] - points_coords[j,0])**2 + 
                                   (points_coords[i,1] - points_coords[j,1])**2)
                    if dist <= error_deg:
                        used[j] = True
        
        return unique_points
    
    # Remove duplicates from the results
    unique_rising = remove_duplicates(common_rising, coord_error)
    unique_setting = remove_duplicates(common_setting, coord_error)
    
    # Return results
    return {
        'rising_points': unique_rising,
        'setting_points': unique_setting,
        'target': target,
        'elevation_range': elevation_range
    }

def main():
    # Example usage
    #    Pictor A
    target_ra = 79.957171 * u.deg 
    target_dec = -45.778828 * u.deg
    elevation_range = [30 * u.deg, 50 * u.deg]  # Example elevation range
    
    try:
        # Find matching points
        results = find_points_across_elevations(
            target_ra, target_dec, elevation_range,
            dt=80*u.second,
            elev_error=0.1*u.deg,
            angular_separation=0.8*u.deg,
            angular_sep_error=0.1*u.deg,
            coord_error=0.01*u.deg  # 0.01 degree tolerance for RA and Dec
        )
        
        # Extract results
        rising_points = results['rising_points']
        setting_points = results['setting_points']
        target = results['target']
        elevation_range = results['elevation_range']
        
        # Print minimal summary
        print(f"\nSummary:")
        print(f"Target: RA={target.ra.deg:.5f}°, Dec={target.dec.deg:.5f}°")
        print(f"Elevation range: {elevation_range[0]} to {elevation_range[1]}")
        print(f"Found {len(rising_points)} unique rising points and {len(setting_points)} unique setting points")
        
        # Print rising points with delta RA, delta Dec
        if rising_points:
            print("\nRising points (dRA, dDec, Angular Separation):")
            for i, p in enumerate(rising_points):
                delta_ra = p.ra.deg - target.ra.deg
                delta_dec = p.dec.deg - target.dec.deg
                sep = target.separation(p).deg
                print(f"{i+1}: dRA={delta_ra:.5f}°, dDec={delta_dec:.5f}°, Sep={sep:.5f}°")
        
        # Print setting points with delta RA, delta Dec
        if setting_points:
            print("\nSetting points (dRA, dDec, Angular Separation):")
            for i, p in enumerate(setting_points):
                delta_ra = p.ra.deg - target.ra.deg
                delta_dec = p.dec.deg - target.dec.deg
                sep = target.separation(p).deg
                print(f"{i+1}: dRA={delta_ra:.5f}°, dDec={delta_dec:.5f}°, Sep={sep:.5f}°")
        
        # Plot results if any points found
        if rising_points or setting_points:
            plt.figure(figsize=(12, 8))
            
            # Plot rising points
            if rising_points:
                ra_vals = [p.ra.deg for p in rising_points]
                dec_vals = [p.dec.deg for p in rising_points]
                plt.scatter(ra_vals, dec_vals, alpha=0.7, color='blue', label='Rising Points')
            
            # Plot setting points
            if setting_points:
                ra_vals = [p.ra.deg for p in setting_points]
                dec_vals = [p.dec.deg for p in setting_points]
                plt.scatter(ra_vals, dec_vals, alpha=0.7, color='green', label='Setting Points')
            
            # Plot target
            plt.scatter(target_ra.value, target_dec.value, color='red', s=100, marker='*', label='Target')
            
            plt.xlabel('RA (deg)')
            plt.ylabel('Dec (deg)')
            plt.title(f'Points common across elevation range {elevation_range[0]} to {elevation_range[1]}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot angular separations histograms if we have points
            if rising_points or setting_points:
                plt.figure(figsize=(10, 4))
                
                if rising_points and setting_points:
                    # Rising points histogram
                    plt.subplot(1, 2, 1)
                    seps_rising = [target.separation(p).deg for p in rising_points]
                    plt.hist(seps_rising, bins=20)
                    plt.xlabel('Angular Separation (deg)')
                    plt.ylabel('Count')
                    plt.title('Distribution - Rising Points')
                    plt.axvline(0.8, color='red', linestyle='--', label='Target Sep')
                    plt.legend()
                    
                    # Setting points histogram
                    plt.subplot(1, 2, 2)
                    seps_setting = [target.separation(p).deg for p in setting_points]
                    plt.hist(seps_setting, bins=20)
                    plt.xlabel('Angular Separation (deg)')
                    plt.title('Distribution - Setting Points')
                    plt.axvline(0.8, color='red', linestyle='--', label='Target Sep')
                    plt.legend()
                elif rising_points:
                    seps_rising = [target.separation(p).deg for p in rising_points]
                    plt.hist(seps_rising, bins=20)
                    plt.xlabel('Angular Separation (deg)')
                    plt.ylabel('Count')
                    plt.title('Distribution - Rising Points')
                    plt.axvline(0.8, color='red', linestyle='--', label='Target Sep')
                    plt.legend()
                elif setting_points:
                    seps_setting = [target.separation(p).deg for p in setting_points]
                    plt.hist(seps_setting, bins=20)
                    plt.xlabel('Angular Separation (deg)')
                    plt.ylabel('Count')
                    plt.title('Distribution - Setting Points')
                    plt.axvline(0.8, color='red', linestyle='--', label='Target Sep')
                    plt.legend()
            
            plt.tight_layout()
            plt.show()
    
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
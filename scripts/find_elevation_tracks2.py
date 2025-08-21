import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from scipy.spatial import cKDTree
import argparse
import csv
import os

def find_points_across_elevations(target_ra, target_dec, elevation_range, 
                               dt=80*u.second, elev_error=0.01*u.deg,
                               angular_separation=0.8*u.deg, angular_sep_error=0.01*u.deg,
                               coord_error=0.01*u.deg, n_points=100, n_radii=10):
    """
    Find sky points that:
    1. Are at the same elevation as the target point after time dt AND before time dt (-dt)
    2. Are separated from the target by the specified angular separation (within error)
    3. Have consistent positions across the entire elevation range
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
    
    # Create negative dt for the "before" calculation
    neg_dt = -dt
    
    # Make sure elevation_range is a list/tuple of quantities
    if not isinstance(elevation_range[0], u.Quantity):
        elevation_range = [el * u.deg for el in elevation_range]
    
    # Create target SkyCoord
    target = SkyCoord(ra=target_ra, dec=target_dec, frame='icrs')
    
    # Calculate number of elevation samples based on range width
    # Use at least 3 samples, and approximately one sample per 5 degrees in the range
    elev_range_width = (elevation_range[1] - elevation_range[0]).value
    n_elevations = max(3, int(np.ceil(elev_range_width)/5))
    
    # Sample elevations within the range
    elevations = np.linspace(elevation_range[0].value, 
                           elevation_range[1].value, 
                           n_elevations) * u.deg
    
    # Lists to store matching points at each elevation
    rising_matches_by_elev = []
    setting_matches_by_elev = []
    rising_neg_matches_by_elev = []  # For negative dt (before)
    setting_neg_matches_by_elev = []  # For negative dt (before)
    
    # Generate test points in the annular region around the target
    test_points = []
        
    # Test multiple radii within the allowed error range
    min_sep = angular_separation - angular_sep_error
    max_sep = angular_separation + angular_sep_error

    # Make sure we include the endpoints precisely
    separations = np.linspace(min_sep.value, max_sep.value, n_radii, endpoint=True) * angular_separation.unit

    # Debug: Print the actual values being used
    # print(f"Testing angular separations: {[sep.value for sep in separations]}")
    
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
    
    # Process each elevation in the range
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
        
        # Add the time delay for positive dt (after)
        time_rising_later = time_rising + dt
        time_setting_later = time_setting + dt
        
        # Add the time delay for negative dt (before)
        time_rising_earlier = time_rising + neg_dt
        time_setting_earlier = time_setting + neg_dt
        
        # Create AltAz frames for the original, positive dt and negative dt times
        frame_rising = AltAz(obstime=time_rising, location=meerkat)
        frame_setting = AltAz(obstime=time_setting, location=meerkat)
        frame_rising_later = AltAz(obstime=time_rising_later, location=meerkat)
        frame_setting_later = AltAz(obstime=time_setting_later, location=meerkat)
        frame_rising_earlier = AltAz(obstime=time_rising_earlier, location=meerkat)
        frame_setting_earlier = AltAz(obstime=time_setting_earlier, location=meerkat)
        
        # Get test points' elevations after time delay (positive dt)
        test_points_rising_altaz = [p.transform_to(frame_rising_later) for p in test_points]
        test_points_setting_altaz = [p.transform_to(frame_setting_later) for p in test_points]
        
        # Get test points' elevations before time delay (negative dt)
        test_points_rising_neg_altaz = [p.transform_to(frame_rising_earlier) for p in test_points]
        test_points_setting_neg_altaz = [p.transform_to(frame_setting_earlier) for p in test_points]
        
        # Find matching points for positive dt
        rising_matches = []
        setting_matches = []
        
        for i, point_altaz in enumerate(test_points_rising_altaz):
            if abs(point_altaz.alt - target_elevation) <= elev_error:
                rising_matches.append(test_points[i])
        
        for i, point_altaz in enumerate(test_points_setting_altaz):
            if abs(point_altaz.alt - target_elevation) <= elev_error:
                setting_matches.append(test_points[i])
        
        # Find matching points for negative dt
        rising_neg_matches = []
        setting_neg_matches = []
        
        for i, point_altaz in enumerate(test_points_rising_neg_altaz):
            if abs(point_altaz.alt - target_elevation) <= elev_error:
                rising_neg_matches.append(test_points[i])
        
        for i, point_altaz in enumerate(test_points_setting_neg_altaz):
            if abs(point_altaz.alt - target_elevation) <= elev_error:
                setting_neg_matches.append(test_points[i])
        
        # Store matches for this elevation
        rising_matches_by_elev.append(rising_matches)
        setting_matches_by_elev.append(setting_matches)
        rising_neg_matches_by_elev.append(rising_neg_matches)
        setting_neg_matches_by_elev.append(setting_neg_matches)
    
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
    
    # Find common points for positive and negative dt
    common_rising = find_common_points(rising_matches_by_elev, coord_error)
    common_setting = find_common_points(setting_matches_by_elev, coord_error)
    common_rising_neg = find_common_points(rising_neg_matches_by_elev, coord_error)
    common_setting_neg = find_common_points(setting_neg_matches_by_elev, coord_error)
    
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
    unique_rising_neg = remove_duplicates(common_rising_neg, coord_error)
    unique_setting_neg = remove_duplicates(common_setting_neg, coord_error)
    
    # Return results
    return {
        'rising_points': unique_rising,
        'setting_points': unique_setting,
        'rising_neg_points': unique_rising_neg,
        'setting_neg_points': unique_setting_neg,
        'target': target,
        'elevation_range': elevation_range,
        'dt': dt
    }

def find_common_points_between_sets(set1, set2, coord_error):
    """
    Find points that appear in both set1 and set2 within the coordinate error
    """
    if not set1 or not set2:
        return []
    
    # Convert coord_error to degrees
    error_deg = coord_error.to(u.deg).value
    
    # List to store common points (will use set1's points)
    common_points = []
    
    # Convert all points from set2 to (ra, dec) arrays for KDTree
    set2_coords = np.array([[p.ra.deg, p.dec.deg] for p in set2])
    
    # Create KDTree for efficient lookup
    tree = cKDTree(set2_coords)
    
    # For each point in set1
    for point in set1:
        # Convert point to (ra, dec)
        point_coords = np.array([point.ra.deg, point.dec.deg])
        
        # Find closest match in set2
        dist, idx = tree.query(point_coords)
        
        # If closest match is within error, consider it the same point
        if dist <= error_deg:
            common_points.append(point)
    
    return common_points
       
                
def save_points_to_csv(points, filename, target=None):
    """Save a list of SkyCoord points to a CSV file"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        if target:
            # Include angular separation from target
            writer.writerow(['RA (deg)', 'Dec (deg)', 'RA (HMS)', 'Dec (DMS)', 
                            'dRA (deg)', 'dDec (deg)', 'Angular Separation (deg)'])
            
            # Write data
            for p in points:
                ra_str = p.ra.to_string(unit=u.hour, sep=':')
                dec_str = p.dec.to_string(unit=u.deg, sep=':')
                delta_ra = p.ra.deg - target.ra.deg
                delta_dec = p.dec.deg - target.dec.deg
                sep = target.separation(p).deg
                writer.writerow([p.ra.deg, p.dec.deg, ra_str, dec_str, 
                               delta_ra, delta_dec, sep])
        else:
            # Without target (no angular separation or deltas)
            writer.writerow(['RA (deg)', 'Dec (deg)', 'RA (HMS)', 'Dec (DMS)'])
            
            # Write data
            for p in points:
                ra_str = p.ra.to_string(unit=u.hour, sep=':')
                dec_str = p.dec.to_string(unit=u.deg, sep=':')
                writer.writerow([p.ra.deg, p.dec.deg, ra_str, dec_str])
                

def create_plots(target, rising_points, setting_points, rising_neg_points, setting_neg_points, 
                common_rise_set_pos, common_rise_set_neg, common_pos_neg_rise, common_pos_neg_set,
                common_all,
                elevation_range, dt, output_file=None):
    """Create plots of the results and save to file if specified"""
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Define a colormap for the points
    colors = {
        'rising_pos': 'blue',
        'setting_pos': 'green',
        'rising_neg': 'cyan',
        'setting_neg': 'magenta',
        'common_rise_set_pos': 'orange',
        'common_rise_set_neg': 'purple',
        'common_pos_neg_rise': 'brown',
        'common_pos_neg_set': 'olive',
        'common_all': 'red'
    }
    
    # First subplot: All points
    ax1 = plt.subplot(2, 2, 1)
    
    # Plot all sets of points
    if rising_points:
        ra_vals = [p.ra.deg for p in rising_points]
        dec_vals = [p.dec.deg for p in rising_points]
        ax1.scatter(ra_vals, dec_vals, alpha=0.7, color=colors['rising_pos'], label=f'Rising dt={dt.value}s')
    
    if setting_points:
        ra_vals = [p.ra.deg for p in setting_points]
        dec_vals = [p.dec.deg for p in setting_points]
        ax1.scatter(ra_vals, dec_vals, alpha=0.7, color=colors['setting_pos'], label=f'Setting dt={dt.value}s')
    
    if rising_neg_points:
        ra_vals = [p.ra.deg for p in rising_neg_points]
        dec_vals = [p.dec.deg for p in rising_neg_points]
        ax1.scatter(ra_vals, dec_vals, alpha=0.7, color=colors['rising_neg'], label=f'Rising dt={-dt.value}s')
    
    if setting_neg_points:
        ra_vals = [p.ra.deg for p in setting_neg_points]
        dec_vals = [p.dec.deg for p in setting_neg_points]
        ax1.scatter(ra_vals, dec_vals, alpha=0.7, color=colors['setting_neg'], label=f'Setting dt={-dt.value}s')
    
    # Plot target
    ax1.scatter(target.ra.deg, target.dec.deg, color='black', s=100, marker='*', label='Target')
    
    ax1.set_xlabel('RA (deg)')
    ax1.set_ylabel('Dec (deg)')
    ax1.set_title(f'All Points - Elevation range {elevation_range[0]} to {elevation_range[1]}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize='small')
    
    # Second subplot: Common points between rising and setting
    ax2 = plt.subplot(2, 2, 2)
    
    if common_rise_set_pos:
        ra_vals = [p.ra.deg for p in common_rise_set_pos]
        dec_vals = [p.dec.deg for p in common_rise_set_pos]
        ax2.scatter(ra_vals, dec_vals, alpha=0.7, color=colors['common_rise_set_pos'], label=f'Common Rise/Set dt={dt.value}s')
    
    if common_rise_set_neg:
        ra_vals = [p.ra.deg for p in common_rise_set_neg]
        dec_vals = [p.dec.deg for p in common_rise_set_neg]
        ax2.scatter(ra_vals, dec_vals, alpha=0.7, color=colors['common_rise_set_neg'], label=f'Common Rise/Set dt={-dt.value}s')
    
    # Plot target
    ax2.scatter(target.ra.deg, target.dec.deg, color='black', s=100, marker='*', label='Target')
    
    ax2.set_xlabel('RA (deg)')
    ax2.set_ylabel('Dec (deg)')
    ax2.set_title('Common Points Between Rising and Setting')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize='small')
    
    # Third subplot: Common points between positive and negative dt
    ax3 = plt.subplot(2, 2, 3)
    
    if common_pos_neg_rise:
        ra_vals = [p.ra.deg for p in common_pos_neg_rise]
        dec_vals = [p.dec.deg for p in common_pos_neg_rise]
        ax3.scatter(ra_vals, dec_vals, alpha=0.7, color=colors['common_pos_neg_rise'], label='Common +/-dt Rising')
    
    if common_pos_neg_set:
        ra_vals = [p.ra.deg for p in common_pos_neg_set]
        dec_vals = [p.dec.deg for p in common_pos_neg_set]
        ax3.scatter(ra_vals, dec_vals, alpha=0.7, color=colors['common_pos_neg_set'], label='Common +/-dt Setting')
    
    # Plot target
    ax3.scatter(target.ra.deg, target.dec.deg, color='black', s=100, marker='*', label='Target')
    
    ax3.set_xlabel('RA (deg)')
    ax3.set_ylabel('Dec (deg)')
    ax3.set_title('Common Points Between +dt and -dt')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize='small')
    
    # Fourth subplot: Points common to everything
    ax4 = plt.subplot(2, 2, 4)
    
    if common_all:
        ra_vals = [p.ra.deg for p in common_all]
        dec_vals = [p.dec.deg for p in common_all]
        ax4.scatter(ra_vals, dec_vals, alpha=0.7, color=colors['common_all'], label='Common to All Categories')
    
    # Plot target
    ax4.scatter(target.ra.deg, target.dec.deg, color='black', s=100, marker='*', label='Target')
    
    ax4.set_xlabel('RA (deg)')
    ax4.set_ylabel('Dec (deg)')
    ax4.set_title('Points Common to All Categories')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    
    # Save to file if specified, otherwise display
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Find points at constant elevation with respect to a target")
    parser.add_argument("target_ra", type=float, help="Target RA in degrees")
    parser.add_argument("target_dec", type=float, help="Target Dec in degrees")
    parser.add_argument("min_elevation", type=float, help="Minimum elevation in degrees")
    parser.add_argument("max_elevation", type=float, help="Maximum elevation in degrees")
    parser.add_argument("--dt", type=float, default=80, help="Time delay in seconds (default: 80)")
    parser.add_argument("--elev-error", type=float, default=0.15, help="Elevation error in degrees (default: 0.01)")
    parser.add_argument("--angular-separation", type=float, default=0.85, 
                        help="Angular separation in degrees (default: 0.8)")
    parser.add_argument("--angular-sep-error", type=float, default=0.15, 
                        help="Angular separation error in degrees (default: 0.01)")
    parser.add_argument("--coord-error", type=float, default=0.01, 
                        help="Coordinate error to consider points the same (default: 0.01)")
    parser.add_argument("--plot", type=str, help="Generate plots and save to the specified file")
    parser.add_argument("--csv", type=str, help="Save results to CSV files with the specified prefix")
    
    args = parser.parse_args()
    
    
    try:
        # Convert command line arguments to proper units
        target_ra = args.target_ra * u.deg
        target_dec = args.target_dec * u.deg
        elevation_range = [args.min_elevation * u.deg, args.max_elevation * u.deg]
        dt = args.dt * u.second
        elev_error = args.elev_error * u.deg
        angular_separation = args.angular_separation * u.deg
        angular_sep_error = args.angular_sep_error * u.deg
        coord_error = args.coord_error * u.deg
        
        # Find matching points
        results = find_points_across_elevations(
            target_ra, target_dec, elevation_range,
            dt=dt,
            elev_error=elev_error,
            angular_separation=angular_separation,
            angular_sep_error=angular_sep_error,
            coord_error=coord_error
        )
        
        # Extract results
        rising_points = results['rising_points']
        setting_points = results['setting_points']
        rising_neg_points = results['rising_neg_points']
        setting_neg_points = results['setting_neg_points']
        target = results['target']
        elevation_range = results['elevation_range']
        
        # Find common points between different sets
        common_rise_set_pos = find_common_points_between_sets(rising_points, setting_points, coord_error)
        common_rise_set_neg = find_common_points_between_sets(rising_neg_points, setting_neg_points, coord_error)
        common_pos_neg_rise = find_common_points_between_sets(rising_points, rising_neg_points, coord_error)
        common_pos_neg_set = find_common_points_between_sets(setting_points, setting_neg_points, coord_error)
        
        # Find points common to all four categories
        common_rise_set = find_common_points_between_sets(common_rise_set_pos, common_rise_set_neg, coord_error)
        common_pos_neg = find_common_points_between_sets(common_pos_neg_rise, common_pos_neg_set, coord_error)
        common_all = find_common_points_between_sets(common_rise_set, common_pos_neg, coord_error)
        
        # Print summary
        print(f"\nSummary:")
        print(f"Target: RA={target.ra.to_string(unit=u.hour, sep=':')} Dec={target.dec.to_string(unit=u.deg, sep=':')}")
        print(f"Elevation range: {elevation_range[0]} to {elevation_range[1]}")
        print(f"Time delay: dt = {dt.value} seconds")
        print(f"Angular separation: {angular_separation} ± {angular_sep_error}")
        
        # Print count of points in each category
        print("\nNumber of matching points in each category:")
        print(f"  Rising, dt={dt.value}s: {len(rising_points)}")
        print(f"  Setting, dt={dt.value}s: {len(setting_points)}")
        print(f"  Rising, dt={-dt.value}s: {len(rising_neg_points)}")
        print(f"  Setting, dt={-dt.value}s: {len(setting_neg_points)}")
        print(f"  Common Rise/Set, dt={dt.value}s: {len(common_rise_set_pos)}")
        print(f"  Common Rise/Set, dt={-dt.value}s: {len(common_rise_set_neg)}")
        print(f"  Common +/-dt, Rising: {len(common_pos_neg_rise)}")
        print(f"  Common +/-dt, Setting: {len(common_pos_neg_set)}")
        print(f"  Common to all categories: {len(common_all)}")
        
        # If we have points common to all categories, print them
        if common_all:
            print("\nPoints common to all categories (RA, Dec, Angular Separation):")
            for i, p in enumerate(common_all):
                ra_str = p.ra.to_string(unit=u.hour, sep=':')
                dec_str = p.dec.to_string(unit=u.deg, sep=':')
                sep = target.separation(p).deg
                print(f"{i+1}: RA={ra_str}, Dec={dec_str}, Sep={sep:.5f}°")
        else:
            # If no common points, give a helpful suggestion
            print("\nNo points common to all categories. Consider checking the individual categories.")
            
            # Check which combination has the most points
            combinations = [
                ("Common Rise/Set, dt=+dt", common_rise_set_pos),
                ("Common Rise/Set, dt=-dt", common_rise_set_neg),
                ("Common +/-dt, Rising", common_pos_neg_rise),
                ("Common +/-dt, Setting", common_pos_neg_set)
            ]
            
            most_points = max(combinations, key=lambda x: len(x[1]))
            if most_points[1]:
                print(f"\nThe most promising category is '{most_points[0]}' with {len(most_points[1])} points.")
                print("Here are the first 5 points from this category:")
                for i, p in enumerate(most_points[1][:5]):
                    ra_str = p.ra.to_string(unit=u.hour, sep=':')
                    dec_str = p.dec.to_string(unit=u.deg, sep=':')
                    sep = target.separation(p).deg
                    print(f"{i+1}: RA={ra_str}, Dec={dec_str}, Sep={sep:.5f}°")
        
        # Save results to CSV if requested
        if args.csv:
            # Create directory if it doesn't exist
            csv_dir = os.path.dirname(args.csv)
            if csv_dir and not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
            
            # Save each category to a separate CSV file
            categories = {
                "rising_pos": rising_points,
                "setting_pos": setting_points,
                "rising_neg": rising_neg_points,
                "setting_neg": setting_neg_points,
                "common_rise_set_pos": common_rise_set_pos,
                "common_rise_set_neg": common_rise_set_neg,
                "common_pos_neg_rise": common_pos_neg_rise,
                "common_pos_neg_set": common_pos_neg_set,
                "common_all": common_all
            }
            
            for category, points in categories.items():
                if points:  # Only save if there are points
                    filename = f"{args.csv}_{category}.csv"
                    save_points_to_csv(points, filename, target)
                    print(f"Saved {len(points)} points to {filename}")
            
            # Also save a summary CSV with counts from each category
            summary_filename = f"{args.csv}_summary.csv"
            with open(summary_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Category', 'Count'])
                for category, points in categories.items():
                    writer.writerow([category, len(points)])
                print(f"Saved summary information to {summary_filename}")

        # Generate plots if requested
        if args.plot:
            create_plots(
                target, 
                rising_points, setting_points, 
                rising_neg_points, setting_neg_points,
                common_rise_set_pos, common_rise_set_neg,
                common_pos_neg_rise, common_pos_neg_set,
                common_all,
                elevation_range, 
                dt, 
                args.plot
            )

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
                                        
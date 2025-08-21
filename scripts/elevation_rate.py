import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from datetime import datetime, timedelta
from astropy.time import TimeDelta

def calculate_for_lst_range(ra, dec, observer_location=None, lst_start=0, lst_end=24, lst_step=0.25):
    """
    Calculate elevation and elevation rate as a function of Local Sidereal Time (LST).
    
    Parameters:
    -----------
    ra : str or float
        Right Ascension of the object (in hours if float, or formatted string like '05h34m31.94s')
    dec : str or float
        Declination of the object (in degrees if float, or formatted string like '-05d27m10.18s')
    observer_location : EarthLocation, optional
        Observer's location. Default is MeerKAT radio telescope in South Africa.
    lst_start : float, optional
        Starting LST in hours (0-24). Default is 0.
    lst_end : float, optional
        Ending LST in hours (0-24). Default is 24.
    lst_step : float, optional
        LST step size in hours. Default is 0.25 (15 minutes).
    
    Returns:
    --------
    lst_hours : ndarray
        Array of LST values in hours
    elevations : ndarray
        Array of elevation values in degrees
    elevation_rates : ndarray
        Array of elevation rate values in degrees per hour
    """
    # Set default observer location to MeerKAT if not provided
    if observer_location is None:
        # MeerKAT coordinates: 30°43′16″S 21°24′40″E, elevation 1086m
        observer_location = EarthLocation(
            lat=-30.721111 * u.deg,
            lon=21.411111 * u.deg,
            height=1086 * u.m
        )
    
    # Create a SkyCoord object for the target
    if isinstance(ra, str) and isinstance(dec, str):
        target = SkyCoord(ra, dec, frame='icrs')
    else:
        # Assume ra in hours and dec in degrees if they're not strings
        target = SkyCoord(ra=ra*u.hourangle, dec=dec*u.deg, frame='icrs')
    
    # Create LST array
    lst_hours = np.arange(lst_start, lst_end, lst_step)
    
    # We'll use the current date but set times that correspond to the desired LST values
    # This is a simplification but works for our purpose since we only care about LST
    base_time = Time(datetime.utcnow())
    
    # Calculate the current LST at the observer location
    current_lst = base_time.sidereal_time('apparent', observer_location).hour
    
    # Create an array of times that correspond to the desired LST values
    times = []
    for lst in lst_hours:
        # Calculate how many hours to add to the current time to reach the desired LST
        # This handles the LST wrapping around 24 hours
        if lst >= current_lst:
            delta_hours = lst - current_lst
        else:
            delta_hours = 24 - (current_lst - lst)
        
        # Add the calculated delta to the base time
        times.append(base_time + TimeDelta(delta_hours * 3600, format='sec'))
    
    times = Time(times)
    
    # Verify the LST values (useful for debugging)
    calculated_lst = [t.sidereal_time('apparent', observer_location).hour for t in times]
    
    # Calculate altitude and azimuth for each time
    altaz_frames = AltAz(obstime=times, location=observer_location)
    altaz = target.transform_to(altaz_frames)
    
    # Extract elevation (altitude) values in degrees
    elevations = altaz.alt.deg
    
    # Calculate the rate of change of elevation (degrees per hour)
    # For the rate calculations, we need to consider the time step in actual hours, not just LST
    # LST advances at approximately 1.0027 times the rate of UTC
    real_time_step = lst_step * 0.9973  # Convert LST difference to approximate real time difference
    
    elevation_rates = np.diff(elevations) / real_time_step
    
    # For plotting, we'll use the midpoint LST values for the rates
    midpoint_lst = (lst_hours[:-1] + lst_hours[1:]) / 2
    
    return lst_hours, elevations, midpoint_lst, elevation_rates

def plot_vs_lst(ra, dec, observer_location=None, lst_start=0, lst_end=24, lst_step=0.25, save_path=None):
    """
    Plot elevation and elevation rate as a function of LST.
    
    Parameters are the same as calculate_for_lst_range function with addition of:
    save_path : str, optional
        If provided, save the plot to this file path instead of displaying it.
    """
    lst_hours, elevations, midpoint_lst, elevation_rates = calculate_for_lst_range(
        ra, dec, observer_location, lst_start, lst_end, lst_step
    )
    
    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot elevation vs LST
    ax1.plot(lst_hours, elevations, 'b-', linewidth=2)
    ax1.set_ylabel('Elevation (degrees)')
    ax1.set_title(f'Elevation vs LST for RA={ra}, Dec={dec}')
    ax1.grid(True, alpha=0.3)
    
    # Add horizontal line at 0 degrees elevation
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    
    # Add horizontal line at 15 degrees elevation (typical minimum for observations)
    ax1.axhline(y=15, color='g', linestyle='--', alpha=0.7)
    ax1.text(lst_start + 0.5, 16, 'Minimum elevation (15°)', color='green')
    
    # Plot elevation rate vs LST
    ax2.plot(midpoint_lst, elevation_rates, 'r-', linewidth=2)
    ax2.set_xlabel('Local Sidereal Time (hours)')
    ax2.set_ylabel('Elevation Rate (degrees/hour)')
    ax2.set_title('Elevation Rate vs LST')
    ax2.grid(True, alpha=0.3)
    
    # Add horizontal line at 0 for rate
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    # Add annotations for rising and setting
    y_min, y_max = ax2.get_ylim()
    mid_y = (y_max + y_min) / 4
    ax2.text(lst_start + 0.5, mid_y, 'RISING', fontsize=12, color='blue')
    ax2.text(lst_start + 0.5, -mid_y, 'SETTING', fontsize=12, color='red')
    
    # Format x-axis ticks to show LST hours
    ax2.set_xticks(np.arange(lst_start, lst_end + 1, 2))
    ax2.set_xlim(lst_start, lst_end)
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    return plt

def calculate_elevation_rate_vs_elevation(ra, dec, observer_location=None, lst_start=0, lst_end=24, lst_step=0.25):
    """
    Calculate elevation rate vs elevation (as in the original function, but using LST range).
    
    Parameters are the same as calculate_for_lst_range function.
    
    Returns:
    --------
    elevations : ndarray
        Array of elevation values in degrees
    elevation_rates : ndarray
        Array of elevation rate values in degrees per hour
    """
    lst_hours, elevations, midpoint_lst, elevation_rates = calculate_for_lst_range(
        ra, dec, observer_location, lst_start, lst_end, lst_step
    )
    
    # For plotting elevation rate vs elevation, we need to match the arrays
    # We'll use the midpoint elevations
    midpoint_elevations = (elevations[:-1] + elevations[1:]) / 2
    
    return midpoint_elevations, elevation_rates

def plot_elevation_rate_vs_elevation(ra, dec, observer_location=None, lst_start=0, lst_end=24, lst_step=0.25, save_path=None):
    """
    Plot the rate of change of elevation versus elevation.
    
    Parameters are the same as calculate_for_lst_range function with addition of:
    save_path : str, optional
        If provided, save the plot to this file path instead of displaying it.
    """
    elevations, elevation_rates = calculate_elevation_rate_vs_elevation(
        ra, dec, observer_location, lst_start, lst_end, lst_step
    )
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Create a colormap based on LST
    lst_range = np.linspace(0, 1, len(elevations))
    
    # Plot with color gradient to show progression
    sc = plt.scatter(elevations, elevation_rates, c=lst_range, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='LST Progression')
    
    # Also plot the line connecting the points
    plt.plot(elevations, elevation_rates, 'r-', alpha=0.3)
    
    plt.title(f'Elevation Rate vs Elevation for RA={ra}, Dec={dec}')
    plt.xlabel('Elevation (degrees)')
    plt.ylabel('Elevation Rate (degrees/hour)')
    plt.grid(True, alpha=0.3)
    
    # Add a horizontal line at y=0 to indicate when the object is rising vs setting
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add annotations for rising and setting regions
    y_min, y_max = plt.ylim()
    plt.text(min(elevations) + 5, (y_max - y_min) * 0.1, 'RISING', fontsize=12, color='blue')
    plt.text(min(elevations) + 5, -(y_max - y_min) * 0.1, 'SETTING', fontsize=12, color='red')
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    return plt

# Example usage
if __name__ == "__main__":
    
    # Pictor A
    ra='05h19m50s'
    dec='-45d46m44s'

    # Plot elevation and elevation rate vs LST
    plot_vs_lst(ra, dec, lst_start=0, lst_end=24, lst_step=0.25, save_path="elevation_vs_lst.png")
    
    # Plot elevation rate vs elevation
    plot_elevation_rate_vs_elevation(ra, dec, lst_start=0, lst_end=24, lst_step=0.25, save_path="elevation_rate_vs_elevation.png")
    
    # Example with different input format (hours and degrees)
#    ra_hours = 5.575539  # Example: M42 (Orion Nebula)
#    dec_degrees = -5.45283
   
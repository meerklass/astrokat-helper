# satellite_plotter.py

import requests
import matplotlib.pyplot as plt
import numpy as np
from skyfield.api import load, EarthSatellite
from datetime import datetime, timezone
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SatellitePlotter:
    """
    A class for plotting satellite positions based on celestial coordinates.
    """
    
    def __init__(self, tle_url="https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle", satnogs_url="https://db.satnogs.org/api/transmitters/"):
        """
        Initialize the SatellitePlotter with optional frequency range for filtering.
        
        Parameters:
        -----------
        tle_url: str, optional
            url to access TLE data via Celestrak API
        satnogs_url: str, optional
            url to access frequency data via SATNOGS API
        """
        self.tle_url = tle_url
        self.satnogs_url = satnogs_url
        self.tle_list = None
        self.satnogs_data = None
        
        self.fetch_tle_data()
        self.fetch_satnogs_catalog()
    
    
    def fetch_tle_data(self, url=None):
        """
        Fetch TLE data from Celestrak.
        
        Parameters:
        -----------
        url : str, optional
            URL for the TLE data (default: Celestrak active satellites)
            GNSS: https://celestrak.org/NORAD/elements/gp.php?GROUP=gnss&FORMAT=tle
            GPS: https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle
            Galileo: https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle
            GLONASS: https://celestrak.org/NORAD/elements/gp.php?GROUP=glo-ops&FORMAT=tle
            Beidou: https://celestrak.org/NORAD/elements/gp.php?GROUP=beidou&FORMAT=tle
            100 or so brightest: https://celestrak.org/NORAD/elements/gp.php?GROUP=visual&FORMAT=tle
            GEO: https://celestrak.org/NORAD/elements/gp.php?GROUP=geo&FORMAT=tle
            
        Returns:
        --------
        list
            List of TLE data tuples (name, line1, line2)
        """
        logger.info("Fetching TLE data from Celestrak...")
        if url:
            self.tle_url=url
        response = requests.get(self.tle_url)
        if response.status_code != 200:
            logger.error("Error fetching data from Celestrak")
            return None
        lines = response.text.strip().split("\n")
        self.tle_list = [(lines[i].strip(), lines[i+1].strip(), lines[i+2].strip()) 
                         for i in range(0, len(lines), 3)]
        logger.info(f"Retrieved {len(self.tle_list)} TLEs")
        return self.tle_list


    def fetch_satnogs_catalog(self, url=None):
        """
        Fetch the full SatNOGS catalog from the API.
        
        Returns:
        --------
        dict
            The SatNOGS catalog data
        """
        logger.info("Fetching SatNOGS catalog...")
        if url:
            self.tle_url=url
        response = requests.get(self.satnogs_url)
        if response.status_code != 200:
            logger.error("Error fetching data from SatNOGS API")
            return None
        data = response.json()
        if not data:
            logger.error("No data found in SatNOGS API")
            return None
        self.satnogs_data = data
        logger.info(f"Retrieved {len(data)} SatNOGS Sats")
        return data
    
    def filter_freq_ids(self, min_freq, max_freq):
        """
        Get a set of NORAD IDs that match the frequency filter criteria.
        Parameters:
        -----------
        min_freq: float
            Minimum satellite frequency
        max_freq: float
            Maximum satellite frequency
        Returns:
        --------
        set
            Set of NORAD IDs within the frequency range
        """                     
        min_freq_hz = min_freq * 1e6
        max_freq_hz = max_freq * 1e6
        
        filtered_ids = {
            tx["norad_cat_id"]
            for tx in self.satnogs_data
            if tx.get("downlink_low") and min_freq_hz <= tx["downlink_low"] <= max_freq_hz 
               and tx.get("norad_cat_id") and tx.get("status") == "active"
        }
        
        logger.info(f"Found {len(filtered_ids)} satellites in frequency range {min_freq}-{max_freq} MHz")
        return filtered_ids
        
    def compute_positions(self, time=None, min_freq=None, max_freq=None):
        """
        Compute equatorial positions (RA/DEC) of satellites.
        
        Parameters:
        -----------
        time : datetime, optional
            Time for position calculation (default: current UTC time)
        min_freq: float, optional
            Minimum satellite frequency
        max_freq: float, optional
            Maximum satellite frequency     
                   
        Returns:
        --------
        list
            Positions of satellites with RA/DEC in degrees
        """
        if (min_freq is None and max_freq is not None) or (min_freq is not None and max_freq is None):
            raise TypeError("Function must have both minimum and maximum frequency or none")

        if time is None:
            time = datetime.now(timezone.utc)
        elif time.tzinfo is None:
            # If time has no timezone, assume UTC
            time = time.replace(tzinfo=timezone.utc)
                        
        # Get filtered NORAD IDs if needed
        if min_freq:
            filtered_norad_ids = self.filter_freq_ids(min_freq, max_freq)
        
        logger.info(f"Computing positions for satellites at {time}")
        ts = load.timescale()
        time_obj = ts.utc(time.year, time.month, time.day, 
                          time.hour, time.minute, time.second)
        
        positions = []
        for name, line1, line2 in self.tle_list:
            try:
                # Extract NORAD ID
                try:
                    norad_id = int(line2.split()[1])
                except (IndexError, ValueError):
                    norad_id = None
                
                # Skip if we're filtering by frequency and this satellite doesn't match
                if min_freq and filtered_norad_ids and norad_id not in filtered_norad_ids:
                    continue
                
                # Compute position
                sat = EarthSatellite(line1, line2, name, ts)
                ra, dec, _ = sat.at(time_obj).radec()
                
                # Get frequency info if available
                frequency = None
                if self.satnogs_data and norad_id:
                    for tx in self.satnogs_data:
                        if tx.get("norad_cat_id") == norad_id and tx.get("downlink_low"):
                            frequency = tx.get("downlink_low") / 1e6  # Convert to MHz
                            break
                
                positions.append({
                    "name": name, 
                    "RA": ra.hours * 15,  # Convert to degrees
                    "Dec": dec.degrees,
                    "norad_id": norad_id,
                    "frequency": frequency
                })
            except Exception as e:
                logger.warning(f"Error computing position for {name}: {e}")
                
        if min_freq:
            logger.info(f"Computed positions for {len(positions)} frequency-filtered satellites at {time}")
        else:
            logger.info(f"Computed positions for {len(positions)} satellites at {time}")
        return positions
    
    def filter_by_box(self, positions, ra_min, ra_max, dec_min, dec_max):
        """
        Filter satellites within a rectangular box defined by RA/DEC boundaries.
        
        Parameters:
        -----------
        ra_min : float
            Minimum Right Ascension in degrees (0-360)
        ra_max : float
            Maximum Right Ascension in degrees (0-360)
        dec_min : float
            Minimum Declination in degrees (-90 to 90)
        dec_max : float
            Maximum Declination in degrees (-90 to 90)
        positions : list
            List of satellite positions to filter
            
        Returns:
        --------
        list
            Filtered satellite positions
        """                            
        # Validate input coordinates
        if ra_min < 0 or ra_max > 360 or dec_min < -90 or dec_max > 90:
            logger.error("Invalid RA/DEC values. RA should be between 0-360 degrees, DEC between -90° and 90°.")
            return []
        
        # Handle RA wrap-around if crossing the 0/360 boundary
        if ra_min > ra_max:  # Box crosses the 0/360 boundary
            filtered_satellites = [
                sat for sat in positions
                if (sat["Dec"] >= dec_min and sat["Dec"] <= dec_max) and
                   ((sat["RA"] >= ra_min and sat["RA"] <= 360) or (sat["RA"] >= 0 and sat["RA"] <= ra_max))
            ]
        else:  # Normal case
            filtered_satellites = [
                sat for sat in positions
                if (sat["Dec"] >= dec_min and sat["Dec"] <= dec_max) and
                   (sat["RA"] >= ra_min and sat["RA"] <= ra_max)
            ]

        logger.info(f"Found {len(filtered_satellites)} satellites within box: RA=[{ra_min}, {ra_max}], Dec=[{dec_min}, {dec_max}]")
        return filtered_satellites
    
    def filter_by_radius(self, positions, target_ra, target_dec, radius_deg=10):
        """
        Filters satellites within a given angular radius from target RA/DEC.
        
        Parameters:
        -----------
        target_ra : float
            Target Right Ascension in degrees (0-360)
        target_dec : float
            Target Declination in degrees (-90 to 90)
        radius_deg : float
            Search radius in degrees (default: 10)
        positions : list
            List of satellite positions to filter
            
        Returns:
        --------
        list
            Filtered satellite positions
        """
            
        if target_ra < 0 or target_ra > 360 or target_dec < -90 or target_dec > 90:
            logger.error("Invalid RA/DEC values. RA should be between 0-360 degrees, DEC between -90° and 90°.")
            return []
        
        # Convert target RA/DEC to Cartesian coordinates
        target_vector = self._ra_dec_to_cartesian(target_ra, target_dec)

        filtered_satellites = []
        
        for sat in positions:
            sat_ra = sat["RA"]
            sat_dec = sat["Dec"]
            
            # Convert satellite RA/DEC to Cartesian
            sat_vector = self._ra_dec_to_cartesian(sat_ra, sat_dec)

            # Compute angular separation using the dot product
            cos_theta = np.dot(target_vector, sat_vector)
            theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
            
            # If within the search radius, keep the satellite
            if theta_deg <= radius_deg:
                filtered_satellites.append(sat)

        logger.info(f"Found {len(filtered_satellites)} satellites within {radius_deg}° of RA={target_ra}, Dec={target_dec}")
        return filtered_satellites
    
    def _ra_dec_to_cartesian(self, ra_deg, dec_deg):
        """
        Convert RA, DEC in degrees to Cartesian unit vector (x, y, z).
        
        Parameters:
        -----------
        ra_deg : float
            Right Ascension in degrees
        dec_deg : float
            Declination in degrees
            
        Returns:
        --------
        numpy.ndarray
            Cartesian unit vector
        """
        ra_rad = np.radians(ra_deg)
        dec_rad = np.radians(dec_deg)

        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)

        return np.array([x, y, z])

    def plot_equatorial(self, positions, filename=None):
        """
        Plot all satellites on a full-sky Mollweide projection.
        
        Parameters:
        -----------
        positions : list
            List of satellite positions
        filename : str, optional
            Output filename
                        
        Returns:
        --------
        """
                
        logger.info(f"Plotting {len(positions)} satellites on equatorial projection")
        
        # Create the figure with Mollweide projection
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection='mollweide')
        
        # Convert RA, Dec to plot coordinates
        ra_degrees = [(sat["RA"] + 180) % 360 - 180 for sat in positions]
        ra_radians = np.radians(-np.array(ra_degrees))
        dec_radians = np.radians([sat["Dec"] for sat in positions])
        
        ax.scatter(ra_radians, dec_radians, s=5, color='red', alpha=0.7)
        
        # Add RA/DEC labels
        ra_ticks_deg = [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]
        ra_tick_labels = ['12h', '10h', '8h', '6h', '4h', '2h', '0h', '22h', '20h', '18h', '16h', '14h']
        ax.set_xticks(np.radians(ra_ticks_deg))
        ax.set_xticklabels(ra_tick_labels)
        ax.set_yticks(np.radians([-75, -45, -15, 15, 45, 75]))
        ax.set_yticklabels(["-75°", "-45°", "-15°", "15°", "45°", "75°"])
        
        ax.grid(True, alpha=0.3)
                
        title = f"Artificial Satellites in Equatorial Coordinates\n({len(positions)} satellites)"
            
        plt.title(title, fontsize=12)

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved as {filename}")
        
        return
    
    def plot_region_box(self, positions, ra_min, ra_max, dec_min, dec_max, filename=None):
        """
        Plot satellites in a rectangular region defined by RA/DEC boundaries.
        
        Parameters:
        -----------
        ra_min : float
            Minimum Right Ascension in degrees (0-360)
        ra_max : float
            Maximum Right Ascension in degrees (0-360)
        dec_min : float
            Minimum Declination in degrees (-90 to 90)
        dec_max : float
            Maximum Declination in degrees (-90 to 90)
        positions : list
            List of satellite positions
        filename : str, optional
            Output filename (default: generates name based on coordinates)
            
        Returns:
        --------
        """
        # Count only the satellites in the box even if more satellite positions are provided
        positions = self.filter_by_box(positions, ra_min, ra_max, dec_min, dec_max)
                
        if not positions:
            logger.warning(f"No satellites found in the region box: RA=[{ra_min}, {ra_max}], Dec=[{dec_min}, {dec_max}]")
            return None
                
        # Create figure with larger size for better readability
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Extract satellite positions
        ra_values = [sat["RA"] for sat in positions]
        dec_values = [sat["Dec"] for sat in positions]
        names = [sat["name"] for sat in positions]
        norad_ids = [sat.get("norad_id", 0) for sat in positions]
        frequencies = [sat.get("frequency", "Unknown") for sat in positions]
        
        # Format frequencies as strings with units
        freq_labels = []
        for freq in frequencies:
            if freq is None or freq == "Unknown":
                freq_labels.append("Unknown")
            else:
                try:
                    freq_labels.append(f"{float(freq):.3f} MHz")
                except (ValueError, TypeError):
                    freq_labels.append(str(freq))
        
        # Handle RA wrap-around for plot limits
        # Apply a small padding for better visibility
        padding = 1.0  # 1 degree padding
        
        # Determine if the plot crosses the 0/360 boundary
        crosses_boundary = ra_min > ra_max
        
        if crosses_boundary:
            # For plots that cross the 0/360 boundary, we need special handling
            # Shift all RA values to a continuous range
            ra_values = [(ra - 360) if ra > 180 else ra for ra in ra_values]
            ra_min_plot = ra_min - 360 if ra_min > 180 else ra_min
            ra_max_plot = ra_max
        else:
            # Normal case
            ra_min_plot = ra_min - padding
            ra_max_plot = ra_max + padding
        
        # Set plot limits with padding
        ax.set_xlim(ra_min_plot, ra_max_plot)
        ax.set_ylim(dec_min - padding, dec_max + padding)
        
        colors = ['red'] * len(ra_values)
        
        # Plot satellites with slightly larger points
        scatter = ax.scatter(ra_values, dec_values, s=40, color=colors, alpha=0.7, 
                            picker=True, zorder=3)
        
        # Create tooltip for satellite names and frequency when hovering
        annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                           bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                           arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(ind):
            pos = scatter.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            idx = ind["ind"][0]
            # Show satellite name and frequency
            text = f"{names[idx]}\nFreq: {freq_labels[idx]}\nNORAD: {norad_ids[idx]}"
            annot.set_text(text)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
        
        # Add the hover event
        fig.canvas.mpl_connect("motion_notify_event", hover)
        
        # Draw the box boundary
        box_style = dict(fill=False, edgecolor='blue', linestyle='--', linewidth=2, alpha=0.7)
        if crosses_boundary:
            # Draw two boxes for crossing the boundary
            # Box from ra_min to 360
            rect1 = plt.Rectangle((ra_min, dec_min), 360-ra_min, dec_max-dec_min, **box_style)
            # Box from 0 to ra_max
            rect2 = plt.Rectangle((0, dec_min), ra_max, dec_max-dec_min, **box_style)
            ax.add_patch(rect1)
            ax.add_patch(rect2)
        else:
            # Draw a single box
            rect = plt.Rectangle((ra_min, dec_min), ra_max-ra_min, dec_max-dec_min, **box_style)
            ax.add_patch(rect)
        
        # Add grid, labels, and title
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Right Ascension (degrees)")
        ax.set_ylabel("Declination (degrees)")
        
        title = f"Satellites in region: RA=[{ra_min:.1f}°, {ra_max:.1f}°], Dec=[{dec_min:.1f}°, {dec_max:.1f}°]\n"
        ax.set_title(title)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add coordinate grid with finer lines
        grid_spacing = 5  # 5 degree grid lines
        ra_grid = np.arange(np.floor(ra_min_plot/grid_spacing)*grid_spacing, 
                           np.ceil(ra_max_plot/grid_spacing)*grid_spacing, 
                           grid_spacing)
        dec_grid = np.arange(np.floor((dec_min-padding)/grid_spacing)*grid_spacing, 
                            np.ceil((dec_max+padding)/grid_spacing)*grid_spacing, 
                            grid_spacing)
        
        for ra in ra_grid:
            ax.axvline(x=ra, color='gray', linestyle='-', linewidth=0.5, alpha=0.3, zorder=0)
        for dec in dec_grid:
            ax.axhline(y=dec, color='gray', linestyle='-', linewidth=0.5, alpha=0.3, zorder=0)
        
        # Save figure if filename provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Region plot saved as {filename}")
        
        plt.tight_layout()
        
        return
    
    def plot_region(self, positions, target_ra, target_dec, region_size=10,
                   filename=None):
        """
        Plot satellites in a rectangular region around a target RA/DEC.
        
        Parameters:
        -----------
        target_ra : float
            Target Right Ascension in degrees
        target_dec : float
            Target Declination in degrees
        region_size : float
            Half-width of the region in degrees (default: 10, making a 20×20 region)
        positions : list, 
            List of satellite positions
        filename : str, optional
            Output filename (default: generates name based on coordinates)
    
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        # Calculate box coordinates from target and region size
        ra_min = target_ra - region_size
        ra_max = target_ra + region_size
        dec_min = target_dec - region_size
        dec_max = target_dec + region_size
        
        # Normalize RA values to 0-360 range
        ra_min = ra_min % 360
        ra_max = ra_max % 360
        
        # Cap declination values to valid range
        dec_min = max(dec_min, -90)
        dec_max = min(dec_max, 90)
                
        # Use the box plotting function with the cross marker for the target
        self.plot_region_box(positions, ra_min, ra_max, dec_min, dec_max, filename)
                
        return
    
    
    
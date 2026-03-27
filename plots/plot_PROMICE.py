import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap, BoundaryNorm
import matplotlib

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 26  # Set default font size to 26

# DTU color scheme
dtu_navy = '#030F4F'
dtu_red = '#990000'
dtu_grey = '#DADADA'
white = '#ffffff'
black = '#000000'

# Color intermediates
phase1_blue = '#030F4F'
phase2_blue = '#3d4677'
phase3_blue = '#babecf'
phase1_red = '#990000'
phase2_red = '#bc5959'
phase3_red = '#e6c1c1'

# Color lists for colormaps
dtu_coolwarm = [dtu_navy, white, dtu_red]
dtu_blues = ['#030f4f', white]
dtu_reds = [white, dtu_red]

# Custom colormaps
dtu_coolwarm_cmap = LinearSegmentedColormap.from_list("dtu_coolwarm", dtu_coolwarm)
dtu_blues_cmap = LinearSegmentedColormap.from_list("dtu_blues", dtu_blues)
dtu_reds_cmap = LinearSegmentedColormap.from_list("dtu_reds", dtu_reds)

# Load the data
file_path = r"C:\Users\user\OneDrive\Desktop\DMI_project\Data\mashup_2025.csv"
df = pd.read_csv(file_path)

# Get unique stations
stations = df['station_id'].unique()
print("Stations:", stations)

# Define station coordinates (lat, lon)
station_coords = {
    'KPC_U': (79.91, -24.08),   # Kronprins Christian Land Upper
    'EGP': (75.62, -35.97),      # Egig Point (EastGRIP)
    'SCO_U': (72.22, -26.82),    # Scoresbysund Upper
    'KAN_U': (67.00, -47.03),    # Kangerlussuaq Upper (S-10)
    'QAS_U': (61.03, -46.85),    # Qassimiut Upper
    'UPE_U': (72.89, -54.30)     # Upernavik Upper
}

# Create the plot
fig = plt.figure(figsize=(12, 12))
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([-82, -10, 65, 90], crs=ccrs.PlateCarree())  # zoom on Greenland

# Add coastlines and features with DTU grey
ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor=black)
ax.add_feature(cfeature.LAND, facecolor=dtu_grey, alpha=0.5)

# Add gridlines
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, 
                  linewidth=0.5, color=dtu_grey, alpha=0.9, linestyle='-')
gl.xlabel_style = {'size':26}
gl.ylabel_style = {'size': 26}

# Plot each station with annotations
for station_id, (lat, lon) in station_coords.items():
    if station_id in stations:  # Only plot stations that exist in your data
        
        # Plot dots - all same color
        ax.scatter(lon, lat, s=260, color=dtu_red, 
                   transform=ccrs.PlateCarree(), alpha=1, 
                   marker='o', edgecolors=black, linewidths=2.5, zorder=5)
        
        # Add spaces before and after EGP
        display_name = f' {station_id} ' if station_id == 'EGP' else station_id
        
        # Annotate each point
        ax.annotate(display_name, 
                    xy=(lon, lat), 
                    xytext=(-40, 30),  # offset in points
                    textcoords='offset points',
                    transform=ccrs.PlateCarree(),
                    fontsize=26,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=white, 
                             edgecolor=black, alpha=0.95, linewidth=1.5),
                    zorder=6)
        
        print(f"{station_id}: Lat={lat}, Lon={lon}")

plt.tight_layout()
plt.show()
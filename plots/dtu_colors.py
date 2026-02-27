
from matplotlib.colors import LinearSegmentedColormap, Normalize,ListedColormap, BoundaryNorm
import matplotlib

matplotlib.rcParams['font.family'] = 'Arial'

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
dtu_blues = ['#030f4f',white]
dtu_reds = [white,dtu_red]

# Custom colormaps
dtu_coolwarm_cmap = LinearSegmentedColormap.from_list("dtu_coolwarm", dtu_coolwarm)
dtu_blues_cmap = LinearSegmentedColormap.from_list("dtu_blues", dtu_blues)
dtu_reds_cmap = LinearSegmentedColormap.from_list("dtu_reds", dtu_reds)

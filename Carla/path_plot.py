import os
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# Mirrors run_iter.py's TRACKS config (kept separate to avoid this analysis-only script
# depending on carla/torch). boundary_half_width matches the historical fixed 7m plot offset for
# town04; austin_map instead uses the real per-point corridor width from the source CSV, since a
# single constant would misrepresent a real track whose width varies 11-27.6m.
TRACK_CONFIGS = {
    'town04': {
        'centerline_file': 'town04_waypoints.txt',
        'boundary_half_width': 7.,
        'width_source_csv': None,
    },
    'austin_map': {
        'centerline_file': 'austin_waypoints.txt',
        'boundary_half_width': None,
        'width_source_csv': os.path.join(SCRIPT_DIR, 'racetrack_source', 'Austin.csv'),
    },
}

# Default number of trajectories to loop through
n_trajs = 11

# Parse command line arguments for run numbers
argparser = argparse.ArgumentParser()
argparser.add_argument(
    '-n', '--run_no',
    metavar='P',
    default=-1,
    type=int,
    help='Number of run trajectories to plot'
)
argparser.add_argument(
    '--mode',
    choices=['with_cbf', 'without_cbf'],
    default='with_cbf',
    help='Which controller_output/<mode>/ subfolder to read trajectories from'
)
argparser.add_argument(
    '--track',
    default='austin_map',
    choices=sorted(TRACK_CONFIGS.keys()),
    help='which track the runs were collected on (default: austin_map; pass --track town04 for '
         'the original Phase 1 baseline runs)'
)
args = argparser.parse_args()

if args.run_no != -1:
    n_trajs = args.run_no

track_cfg = TRACK_CONFIGS[args.track]

# Load center waypoints for the selected track
file_centre_line = track_cfg['centerline_file']
centre_line = np.loadtxt(file_centre_line, delimiter=",")

# Extract center coordinates and calculate track heading (yaw)
tx_center = centre_line[:-1, 0]
ty_center = centre_line[:-1, 1]
tyaw_center = np.arctan2(
    centre_line[1:, 1] - centre_line[:-1, 1],
    centre_line[1:, 0] - centre_line[:-1, 0]
)

# Initialize Plot Layout
plt.figure(figsize=(6.9, 10.5))

# Plot Start Line
plt.plot(
    [tx_center[0] + np.cos(tyaw_center[0] + math.pi/2), tx_center[0] - np.cos(tyaw_center[0] + math.pi/2)],
    [ty_center[0] + np.sin(tyaw_center[0] + math.pi/2), ty_center[0] - np.sin(tyaw_center[0] + math.pi/2)],
    linewidth=5.0, color='green'
)
plt.text(tx_center[0], ty_center[0], 'Start line', fontweight='bold')

# Plot Finish Line
plt.plot(
    [tx_center[-1] + np.cos(tyaw_center[-1] + math.pi/2), tx_center[-1] - np.cos(tyaw_center[-1] + math.pi/2)],
    [ty_center[-1] + np.sin(tyaw_center[-1] + math.pi/2), ty_center[-1] - np.sin(tyaw_center[-1] + math.pi/2)],
    linewidth=5.0, color='red'
)
plt.text(tx_center[-1], ty_center[-1], 'End line', fontweight='bold')

# Calculate Track Boundaries -- real per-point width for austin_map, fixed 7m offset for town04
if track_cfg['width_source_csv']:
    _widths = np.loadtxt(track_cfg['width_source_csv'], delimiter=',', skiprows=1)
    half_width = (_widths[:-1, 2] + _widths[:-1, 3]) / 2.
else:
    half_width = track_cfg['boundary_half_width']
left_boundary = np.array([tx_center - half_width * np.sin(tyaw_center), ty_center + half_width * np.cos(tyaw_center)]).T
right_boundary = np.array([tx_center + half_width * np.sin(tyaw_center), ty_center - half_width * np.cos(tyaw_center)]).T

# Plot Reference Line (Run 0 / Center Line)
ref_traj_path = f'controller_output/{args.mode}/trajectory_run0.txt'
try:
    traj_ref = np.loadtxt(ref_traj_path, delimiter=',')
    plt.plot(traj_ref[:, 0], traj_ref[:, 1], '-.', color='gray', label="Center line (ref)")
except OSError:
    print(f"Warning: Reference file {ref_traj_path} not found.")

# Plot Track Boundaries
plt.plot(left_boundary[:, 0], left_boundary[:, 1], '--', color='black', alpha=0.6, label="Track left boundary")
plt.plot(right_boundary[:, 0], right_boundary[:, 1], '--', color='black', alpha=0.6, label="Track right boundary")

# Loop through and plot individual closed-loop iteration trajectories
for i in range(1, n_trajs + 1):
    traj_path = f'controller_output/{args.mode}/trajectory_run{i}.txt'
    try:
        traj = np.loadtxt(traj_path, delimiter=',')
        plt.plot(traj[:, 0], traj[:, 1], '-', label=f"Followed trajectory (iter {i})")
    except OSError:
        # Skip missing run files gracefully without crashing the script
        continue

# Plot Formatting
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Closed-Loop Trajectory Evolution Across Iterations")
plt.legend(loc='upper right', fontsize='small')
plt.axis('equal')
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()

# Save High-Resolution Output
out_path = f'{args.mode}_all_trajs_plot.png'
plt.savefig(out_path, dpi=400)
print(f"Successfully generated and saved: {out_path}")

plt.show()
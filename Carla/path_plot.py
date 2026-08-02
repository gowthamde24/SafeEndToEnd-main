import numpy as np
import matplotlib.pyplot as plt
import math
import argparse

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
args = argparser.parse_args()

if args.run_no != -1:
    n_trajs = args.run_no

# Load optimal racing line and center waypoints
opt_racing_line = np.loadtxt('waypoints_new.csv', delimiter=',')[:100]
file_centre_line = 'town04_waypoints.txt'

if file_centre_line is not None:
    centre_line = np.loadtxt(file_centre_line, delimiter=",")
else:
    centre_line = None

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

# Calculate Track Boundaries (7-meter offset)
left_boundary = np.array([tx_center - 7 * np.sin(tyaw_center), ty_center + 7 * np.cos(tyaw_center)]).T
right_boundary = np.array([tx_center + 7 * np.sin(tyaw_center), ty_center - 7 * np.cos(tyaw_center)]).T

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
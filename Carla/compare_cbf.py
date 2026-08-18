"""
Compare a closed-loop run driven with the CBF safety gate on vs. off.

Consumes the per-mode outputs written by run_iter.py for a given --run number:
  controller_output/{with_cbf,without_cbf}/trajectory_run<N>.txt
  results/{with_cbf,without_cbf}_{x,theta,curvature,steer}_comps_run<N>.csv

Produces, under results/comparison/:
  trajectory_overlay_run<N>.png       -- both driven paths vs track boundaries
  cross_track_error_run<N>.png        -- x(t) for both modes
  heading_error_run<N>.png            -- theta(t) for both modes
  cbf_intervention_run<N>.png         -- |steer_applied - steer_raw| for the with-CBF run
  summary_run<N>.csv                  -- one row per mode: violation count, intervention stats

Run 'python run_iter.py -r <N> --cbf' and 'python run_iter.py -r <N> --no-cbf' first
(requires a live CARLA server) to generate the inputs this script reads.
"""
import argparse
import math
import os

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# Mirrors run_iter.py's TRACKS config (kept separate here to avoid this analysis-only script
# depending on carla/torch). boundary_half_width matches the historical fixed 7m plot offset for
# town04; austin_map instead uses the real per-point corridor width from the source CSV, since
# a single constant would misrepresent a real track whose width varies 11-27.6m.
TRACK_CONFIGS = {
    'town04': {
        'centerline_file': 'town04_waypoints.txt',
        'lane_width': 11.,  # matches run_iter.py's CBF barrier constant (h_left/h_right = LANE_WIDTH/2 - x)
        'boundary_half_width': 7.,
        'width_source_csv': None,
    },
    'austin_map': {
        'centerline_file': 'austin_waypoints.txt',
        'lane_width': 14.,
        'boundary_half_width': None,
        'width_source_csv': os.path.join(SCRIPT_DIR, 'racetrack_source', 'Austin.csv'),
    },
}


def load_trajectory(mode, run_no):
    path = os.path.join(SCRIPT_DIR, 'controller_output', mode, f'trajectory_run{run_no}.txt')
    if not os.path.exists(path):
        return None
    return np.loadtxt(path, delimiter=',')  # columns: x, y, v, t, steer, throttle


def load_comps(mode, run_no, name):
    path = os.path.join(SCRIPT_DIR, 'results', f'{mode}_{name}_comps_run{run_no}.csv')
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path)
    return data if data.ndim > 1 else data.reshape(-1, data.shape[-1] if data.ndim else 1)


def track_boundaries(track_cfg):
    centre_line = np.loadtxt(os.path.join(SCRIPT_DIR, track_cfg['centerline_file']), delimiter=',')
    tx = centre_line[:-1, 0]
    ty = centre_line[:-1, 1]
    tyaw = np.arctan2(centre_line[1:, 1] - centre_line[:-1, 1], centre_line[1:, 0] - centre_line[:-1, 0])
    if track_cfg['width_source_csv']:
        widths = np.loadtxt(track_cfg['width_source_csv'], delimiter=',', skiprows=1)
        half_w = (widths[:-1, 2] + widths[:-1, 3]) / 2.  # real per-point w_tr_right+w_tr_left
    else:
        half_w = track_cfg['boundary_half_width']
    left = np.array([tx - half_w * np.sin(tyaw), ty + half_w * np.cos(tyaw)]).T
    right = np.array([tx + half_w * np.sin(tyaw), ty - half_w * np.cos(tyaw)]).T
    return left, right


def plot_trajectory_overlay(run_no, out_dir, track_cfg, with_traj, without_traj):
    if with_traj is None and without_traj is None:
        print("Skipping trajectory overlay: no trajectory files found for either mode.")
        return
    left, right = track_boundaries(track_cfg)
    plt.figure(figsize=(7, 10))
    plt.plot(left[:, 0], left[:, 1], '--', color='black', alpha=0.5, label='Track left boundary')
    plt.plot(right[:, 0], right[:, 1], '--', color='black', alpha=0.5, label='Track right boundary')
    if with_traj is not None:
        plt.plot(with_traj[:, 0], with_traj[:, 1], '-', color='#1f77b4', linewidth=1.8, label='With CBF')
    if without_traj is not None:
        plt.plot(without_traj[:, 0], without_traj[:, 1], '-', color='#d62728', linewidth=1.8, label='Without CBF')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(f'Closed-Loop Trajectory: With vs. Without CBF (run {run_no})')
    plt.legend(loc='upper right', fontsize='small')
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    save_path = os.path.join(out_dir, f'trajectory_overlay_run{run_no}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Successfully saved {save_path}")


def plot_error_overlay(run_no, out_dir, name, ylabel, title, with_comps, without_comps, gt_col):
    if with_comps is None and without_comps is None:
        print(f"Skipping {name}: no comps files found for either mode.")
        return
    plt.figure(figsize=(10, 6))
    if with_comps is not None:
        plt.plot(with_comps[:, gt_col], color='#2ca02c', linewidth=1.5, label='Observed (with CBF)')
        plt.plot(with_comps[:, 0], '--', color='#1f77b4', linewidth=1.5, label='Predicted (with CBF)')
    if without_comps is not None:
        plt.plot(without_comps[:, gt_col], color='#ff7f0e', linewidth=1.5, label='Observed (without CBF)')
        plt.plot(without_comps[:, 0], '--', color='#d62728', linewidth=1.5, label='Predicted (without CBF)')
    plt.xlabel('Evaluation Frames')
    plt.ylabel(ylabel)
    plt.title(f'{title} (run {run_no})')
    plt.legend(fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_path = os.path.join(out_dir, f'{name}_run{run_no}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Successfully saved {save_path}")


def plot_cbf_intervention(run_no, out_dir, steer_comps):
    if steer_comps is None:
        print("Skipping CBF intervention plot: no with_cbf steer_comps file found.")
        return
    raw, applied = steer_comps[:, 0], steer_comps[:, 1]
    delta = np.abs(applied - raw)
    plt.figure(figsize=(10, 6))
    plt.plot(delta, color='#9467bd', linewidth=1.5, label='|steer_applied - steer_raw_nn|')
    plt.xlabel('Evaluation Frames')
    plt.ylabel('CBF Steering Correction Magnitude')
    plt.title(f'CBF Intervention Magnitude Over Time (run {run_no})')
    plt.legend(fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_path = os.path.join(out_dir, f'cbf_intervention_run{run_no}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Successfully saved {save_path}")


def summarize(run_no, out_dir, mode, lane_width, x_comps, steer_comps, epsilon=0.05):
    # epsilon=0.05 separates real CBF corrections (observed as ~1.0 magnitude, full-lock-scale)
    # from floating-point-level noise between steer_raw_nn and steer_applied (~1e-3 or smaller).
    row = {'mode': mode}
    if x_comps is not None:
        observed_x = x_comps[:, 2]
        row['n_frames'] = len(observed_x)
        row['lane_violation_count'] = int(np.sum(np.abs(observed_x) > lane_width / 2.))
        row['mean_abs_x'] = float(np.mean(np.abs(observed_x)))
        row['max_abs_x'] = float(np.max(np.abs(observed_x)))
    if steer_comps is not None:
        delta = np.abs(steer_comps[:, 1] - steer_comps[:, 0])
        row['cbf_intervention_frames'] = int(np.sum(delta > epsilon))
        row['cbf_mean_correction'] = float(np.mean(delta))
        row['cbf_max_correction'] = float(np.max(delta))
    return row


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-r', '--run', type=int, default=1, help='Run number to compare')
    argparser.add_argument('--track', default='austin_map', choices=sorted(TRACK_CONFIGS.keys()),
                            help='which track the run was collected on (default: austin_map; '
                                 'pass --track town04 for the original Phase 1 baseline runs)')
    args = argparser.parse_args()
    run_no = args.run
    track_cfg = TRACK_CONFIGS[args.track]

    out_dir = os.path.join(SCRIPT_DIR, 'results', 'comparison')
    os.makedirs(out_dir, exist_ok=True)

    with_traj = load_trajectory('with_cbf', run_no)
    without_traj = load_trajectory('without_cbf', run_no)
    with_x = load_comps('with_cbf', run_no, 'x')
    without_x = load_comps('without_cbf', run_no, 'x')
    with_theta = load_comps('with_cbf', run_no, 'theta')
    without_theta = load_comps('without_cbf', run_no, 'theta')
    with_steer = load_comps('with_cbf', run_no, 'steer')

    if all(v is None for v in (with_traj, without_traj, with_x, without_x, with_theta, without_theta)):
        raise SystemExit(
            f"No data found for run {run_no}. Run 'python run_iter.py -r {run_no} --cbf' and "
            f"'python run_iter.py -r {run_no} --no-cbf' against a live CARLA server first, then re-run this script."
        )

    plot_trajectory_overlay(run_no, out_dir, track_cfg, with_traj, without_traj)
    # x_comps columns: [pred, uncertainty, ground_truth] -> ground truth is column 2
    plot_error_overlay(run_no, out_dir, 'cross_track_error', 'Cross-Track Error (m)',
                        'Cross-Track Error: With vs. Without CBF', with_x, without_x, gt_col=2)
    # theta_comps columns: [pred, ground_truth] -> ground truth is column 1
    plot_error_overlay(run_no, out_dir, 'heading_error', 'Heading Error (rad)',
                        'Heading Error: With vs. Without CBF', with_theta, without_theta, gt_col=1)
    plot_cbf_intervention(run_no, out_dir, with_steer)

    rows = [
        summarize(run_no, out_dir, 'with_cbf', track_cfg['lane_width'], with_x, with_steer),
        summarize(run_no, out_dir, 'without_cbf', track_cfg['lane_width'], without_x, None),
    ]
    summary_path = os.path.join(out_dir, f'summary_run{run_no}.csv')
    keys = sorted({k for row in rows for k in row})
    with open(summary_path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for row in rows:
            f.write(','.join(str(row.get(k, '')) for k in keys) + '\n')
    print(f"Successfully saved {summary_path}")
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()

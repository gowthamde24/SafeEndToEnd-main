import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def _load_val_loss(run_no, n_train_steps, column):
    """Loads val_losses_run<N>.csv (one row per epoch, columns [steering, curvature, x, theta])
    and returns (x_positions, values) aligned to the same 'Training Steps' x-axis as the
    per-step training loss, so both can be plotted on one figure."""
    val_path = f'val_losses_run{run_no}.csv'
    if not os.path.exists(val_path):
        return None, None
    val_losses = np.loadtxt(val_path, skiprows=1, delimiter=',')
    if val_losses.ndim == 1:
        val_losses = val_losses.reshape(1, -1)
    n_epochs = val_losses.shape[0]
    steps_per_epoch = n_train_steps / n_epochs
    x_positions = [(i + 1) * steps_per_epoch - 1 for i in range(n_epochs)]
    return x_positions, val_losses[:, column]


def plot_convergence(mode='with_cbf', run_no=1):
    # Define the output directory (relative to this script, not a machine-specific absolute path)
    RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

    # Ensure the directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # =============================================================
    # PART 1: TRAINING LOSS CONVERGENCE (From losses_*.csv)
    # =============================================================

    # 0. Steering Convergence
    if os.path.exists('losses_0.csv'):
        losses_0 = np.loadtxt('losses_0.csv')
        trained_loss_steer = losses_0[:, 0]
        zero_baseline_steer = losses_0[:, 1]

        plt.figure(figsize=(10, 6))
        plt.plot(trained_loss_steer, label='Trained Model Loss', color='#1f77b4', linewidth=1.8)
        plt.plot(zero_baseline_steer, label='Zero-Prediction Baseline', color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8)
        val_x_pos, val_steer_vals = _load_val_loss(run_no, len(trained_loss_steer), column=0)
        if val_x_pos is not None:
            plt.plot(val_x_pos, val_steer_vals, 'o-', label='Validation Loss', color='#2ca02c', linewidth=1.8, markersize=5)
        plt.title('Steering Convergence', fontsize=14, fontweight='bold')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        save_path = os.path.join(RESULTS_DIR, 'steering_convergence.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Successfully saved {save_path}")

    # 1. Safety Cross-Track Error (X) Convergence
    if os.path.exists('losses_2.csv'):
        losses_2 = np.loadtxt('losses_2.csv')
        trained_loss_x = losses_2[:, 0]
        zero_baseline_x = losses_2[:, 1]

        plt.figure(figsize=(10, 6))
        plt.plot(trained_loss_x, label='Trained Model Loss', color='#1f77b4', linewidth=1.8)
        plt.plot(zero_baseline_x, label='Zero-Prediction Baseline', color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8)
        val_x_pos, val_x_vals = _load_val_loss(run_no, len(trained_loss_x), column=2)
        if val_x_pos is not None:
            plt.plot(val_x_pos, val_x_vals, 'o-', label='Validation Loss', color='#2ca02c', linewidth=1.8, markersize=5)
        plt.title('Safety Cross-Track Error (X) Convergence', fontsize=14, fontweight='bold')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        save_path = os.path.join(RESULTS_DIR, 'safety_cross_track_error_x_convergence.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Successfully saved {save_path}")

    # 2. Safety Curvature Convergence
    if os.path.exists('losses_1.csv'):
        losses_1 = np.loadtxt('losses_1.csv')
        trained_loss_curv = losses_1[:, 0]
        zero_baseline_curv = losses_1[:, 1]

        plt.figure(figsize=(10, 6))
        plt.plot(trained_loss_curv, label='Trained Model Loss', color='#1f77b4', linewidth=1.8)
        plt.plot(zero_baseline_curv, label='Zero-Prediction Baseline', color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8)
        val_x_pos, val_curv_vals = _load_val_loss(run_no, len(trained_loss_curv), column=1)
        if val_x_pos is not None:
            plt.plot(val_x_pos, val_curv_vals, 'o-', label='Validation Loss', color='#2ca02c', linewidth=1.8, markersize=5)
        plt.title('Safety Curvature Convergence', fontsize=14, fontweight='bold')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        save_path = os.path.join(RESULTS_DIR, 'safety_curvature_convergence.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Successfully saved {save_path}")

    # 3. Safety Heading Error (Theta) Convergence
    if os.path.exists('losses_3.csv'):
        losses_3 = np.loadtxt('losses_3.csv')
        trained_loss_theta = losses_3[:, 0]
        zero_baseline_theta = losses_3[:, 1]

        plt.figure(figsize=(10, 6))
        plt.plot(trained_loss_theta, label='Trained Model Loss', color='#1f77b4', linewidth=1.8)
        plt.plot(zero_baseline_theta, label='Zero-Prediction Baseline', color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8)
        val_x_pos, val_theta_vals = _load_val_loss(run_no, len(trained_loss_theta), column=3)
        if val_x_pos is not None:
            plt.plot(val_x_pos, val_theta_vals, 'o-', label='Validation Loss', color='#2ca02c', linewidth=1.8, markersize=5)
        plt.title('Safety Heading Error (Theta) Convergence', fontsize=14, fontweight='bold')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        save_path = os.path.join(RESULTS_DIR, 'safety_heading_error_theta_convergence.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Successfully saved {save_path}")

    # =============================================================
    # PART 1.5: TRAIN VS VALIDATION ACCURACY (From train_val_accuracy_run<N>.csv)
    # Accuracy = % of predictions within a physical-unit tolerance of ground truth
    # (see TOLERANCES in train.py); one point per epoch.
    # =============================================================
    accuracy_path = f'train_val_accuracy_run{run_no}.csv'
    if os.path.exists(accuracy_path):
        acc = np.loadtxt(accuracy_path, skiprows=1, delimiter=',')
        if acc.ndim == 1:
            acc = acc.reshape(1, -1)
        epochs = np.arange(1, acc.shape[0] + 1)
        heads = ['steering', 'curvature', 'x', 'theta']
        for idx, head in enumerate(heads):
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, acc[:, idx], 'o-', label='Train Accuracy', color='#1f77b4', linewidth=1.8)
            plt.plot(epochs, acc[:, idx + 4], 'o-', label='Validation Accuracy', color='#2ca02c', linewidth=1.8)
            plt.title(f'{head.capitalize()}: Train vs. Validation Accuracy', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Accuracy (%, within tolerance)', fontsize=12)
            plt.ylim(0, 105)
            plt.legend(fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()

            save_path = os.path.join(RESULTS_DIR, f'accuracy_{head}_run{run_no}.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Successfully saved {save_path}")

        # Overall accuracy: macro-average across all 4 heads, train vs val, per epoch.
        overall_train_acc = acc[:, 0:4].mean(axis=1)
        overall_val_acc = acc[:, 4:8].mean(axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, overall_train_acc, 'o-', label='Overall Train Accuracy', color='#1f77b4', linewidth=1.8)
        plt.plot(epochs, overall_val_acc, 'o-', label='Overall Validation Accuracy', color='#2ca02c', linewidth=1.8)
        plt.title('Overall Accuracy (macro-average across steering/curvature/x/theta)', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%, within tolerance)', fontsize=12)
        plt.ylim(0, 105)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        save_path = os.path.join(RESULTS_DIR, f'accuracy_overall_run{run_no}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Successfully saved {save_path}")

    # =============================================================
    # PART 2: LIVE INFERENCE PREDICTIONS VS GROUND TRUTH
    # (from run_iter.py's results/<mode>_{x,theta,curvature}_comps_run<N>.csv)
    # =============================================================
    prefix = os.path.join(RESULTS_DIR, f'{mode}_')
    suffix = f'_run{run_no}.csv'

    # 4. Cross-Track Error (X) Predictions
    x_comps_path = f'{prefix}x_comps{suffix}'
    if os.path.exists(x_comps_path):
        x_comps = np.loadtxt(x_comps_path)
        # Columns are [prediction, MC-dropout uncertainty, ground_truth] -- ground truth is
        # column 2, NOT column 1 (which is the near-always-zero uncertainty estimate).
        if x_comps.ndim > 1 and x_comps.shape[1] >= 3:
            pred_x = x_comps[:, 0]
            true_x = x_comps[:, 2]

            plt.figure(figsize=(10, 6))
            plt.plot(true_x, label='Observed X (Ground Truth)', color='#2ca02c', linewidth=2)
            plt.plot(pred_x, label='Predicted X', color='#ff7f0e', linestyle='--', linewidth=1.8)
            plt.title('Cross-Track Error (X): Predicted vs. Observed', fontsize=14, fontweight='bold')
            plt.xlabel('Evaluation Frames', fontsize=12)
            plt.ylabel('Cross-Track Error (meters)', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()

            save_path = os.path.join(RESULTS_DIR, f'{mode}_x_predictions_comparison_run{run_no}.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Successfully saved {save_path}")

    # 5. Curvature Predictions
    curv_comps_path = f'{prefix}curvature_comps{suffix}'
    if os.path.exists(curv_comps_path):
        curv_comps = np.loadtxt(curv_comps_path)
        if curv_comps.ndim > 1 and curv_comps.shape[1] >= 2:
            pred_curv = curv_comps[:, 0]
            true_curv = curv_comps[:, 1]

            plt.figure(figsize=(10, 6))
            plt.plot(true_curv, label='Observed Curvature (Ground Truth)', color='#2ca02c', linewidth=2)
            plt.plot(pred_curv, label='Predicted Curvature', color='#ff7f0e', linestyle='--', linewidth=1.8)
            plt.title('Road Curvature: Predicted vs. Observed', fontsize=14, fontweight='bold')
            plt.xlabel('Evaluation Frames', fontsize=12)
            plt.ylabel('Curvature', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()

            save_path = os.path.join(RESULTS_DIR, f'{mode}_curvature_predictions_comparison_run{run_no}.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Successfully saved {save_path}")

    # 6. Heading Error (Theta) Predictions & Wrapped Metric
    theta_comps_path = f'{prefix}theta_comps{suffix}'
    if os.path.exists(theta_comps_path):
        theta_comps = np.loadtxt(theta_comps_path)
        if theta_comps.ndim > 1 and theta_comps.shape[1] >= 2:
            pred_theta = theta_comps[:, 0]
            true_theta = theta_comps[:, 1]

            # Plot 6a: Direct comparison
            plt.figure(figsize=(10, 6))
            plt.plot(true_theta, label='Observed Theta (Ground Truth)', color='#2ca02c', linewidth=2)
            plt.plot(pred_theta, label='Predicted Theta', color='#ff7f0e', linestyle='--', linewidth=1.8)
            plt.title('Heading Error ($\Theta$): Predicted vs. Observed', fontsize=14, fontweight='bold')
            plt.xlabel('Evaluation Frames', fontsize=12)
            plt.ylabel('Heading Error (radians)', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()

            save_path = os.path.join(RESULTS_DIR, f'{mode}_theta_predictions_comparison_run{run_no}.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Successfully saved {save_path}")

            # Plot 6b: Wrapped Circular Loss to remove pi boundary spikes
            diff = pred_theta - true_theta
            wrapped_diff = (diff + np.pi) % (2 * np.pi) - np.pi
            wrapped_loss = wrapped_diff ** 2

            plt.figure(figsize=(10, 6))
            plt.plot(wrapped_loss, label='Trained Model Loss (Wrapped Metric)', color='#1f77b4', linewidth=1.8)
            plt.title('Safety Heading Error ($\Theta$) Evaluation (Circular Metric)', fontsize=14, fontweight='bold')
            plt.xlabel('Evaluation Frames', fontsize=12)
            plt.ylabel('Wrapped MSE Loss', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()

            save_path = os.path.join(RESULTS_DIR, f'{mode}_theta_evaluation_wrapped_run{run_no}.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Successfully saved {save_path}")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', choices=['with_cbf', 'without_cbf'], default='with_cbf',
                            help='Which results/<mode>_*_comps_run<N>.csv set (from run_iter.py) to plot')
    argparser.add_argument('-r', '--run_no', type=int, default=1, help='Run number')
    args = argparser.parse_args()
    plot_convergence(mode=args.mode, run_no=args.run_no)

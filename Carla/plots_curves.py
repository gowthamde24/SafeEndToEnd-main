import os
import numpy as np
import matplotlib.pyplot as plt

def plot_convergence():
    # Define the output directory
    RESULTS_DIR = '/home/student/SafeEndToEnd/SafeEndToEnd/Carla/results'
    
    # Ensure the directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # =============================================================
    # PART 1: TRAINING LOSS CONVERGENCE (From losses_*.csv)
    # =============================================================

    # 1. Safety Cross-Track Error (X) Convergence
    if os.path.exists('losses_2.csv'):
        losses_2 = np.loadtxt('losses_2.csv')
        trained_loss_x = losses_2[:, 0]
        zero_baseline_x = losses_2[:, 1]

        plt.figure(figsize=(10, 6))
        plt.plot(trained_loss_x, label='Trained Model Loss', color='#1f77b4', linewidth=1.8)
        plt.plot(zero_baseline_x, label='Zero-Prediction Baseline', color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8)
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
    # PART 2: LIVE INFERENCE PREDICTIONS VS GROUND TRUTH
    # =============================================================

    # 4. Cross-Track Error (X) Predictions
    if os.path.exists('x_comps.csv'):
        x_comps = np.loadtxt('x_comps.csv')
        if x_comps.ndim > 1 and x_comps.shape[1] >= 2:
            pred_x = x_comps[:, 0]
            true_x = x_comps[:, 1]
            
            plt.figure(figsize=(10, 6))
            plt.plot(true_x, label='Observed X (Ground Truth)', color='#2ca02c', linewidth=2)
            plt.plot(pred_x, label='Predicted X', color='#ff7f0e', linestyle='--', linewidth=1.8)
            plt.title('Cross-Track Error (X): Predicted vs. Observed', fontsize=14, fontweight='bold')
            plt.xlabel('Evaluation Frames', fontsize=12)
            plt.ylabel('Cross-Track Error (meters)', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            save_path = os.path.join(RESULTS_DIR, 'x_predictions_comparison.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Successfully saved {save_path}")

    # 5. Curvature Predictions
    if os.path.exists('curvature_comps.csv'):
        curv_comps = np.loadtxt('curvature_comps.csv')
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
            
            save_path = os.path.join(RESULTS_DIR, 'curvature_predictions_comparison.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Successfully saved {save_path}")

    # 6. Heading Error (Theta) Predictions & Wrapped Metric
    if os.path.exists('theta_comps.csv'):
        theta_comps = np.loadtxt('theta_comps.csv')
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
            
            save_path = os.path.join(RESULTS_DIR, 'theta_predictions_comparison.png')
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
            
            save_path = os.path.join(RESULTS_DIR, 'theta_evaluation_wrapped.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Successfully saved {save_path}")

if __name__ == '__main__':
    plot_convergence()
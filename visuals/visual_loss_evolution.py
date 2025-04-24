import pandas as pd
import matplotlib.pyplot as plt

def plot_loss_evolution(filename, component_ids=None, component_names=None):
    """
    Plots the loss evolution over epochs from a CSV file.

    Parameters:
    - filename (str): Path to the CSV file.
    - component_ids (list of int): List of component loss column ids (from 1) to plot. Optional.
    - component_names (list of str): Display name for each component
    """
    df = pd.read_csv(filename)

    if "Epoch #" not in df.columns or "Item Loss" not in df.columns:
        raise ValueError("CSV must contain 'Epoch #' and 'Item Loss' columns.")

    df = df.sort_values("Epoch #")
    plt.figure(figsize=(6.5, 6))
    plt.plot(df["Epoch #"], df["Item Loss"], label="Total Item Loss", linewidth=2)

    if component_ids:
        for comp in component_ids:
            column = f"Component #{comp}"
            if column in df.columns:
                plt.plot(df["Epoch #"], df[column], label=f"{component_names[comp - 1]} Loss", linestyle="--")
            else:
                print(f"[Warning] Component '{comp}' not found in CSV columns.")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Evolution over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# plot_loss_evolution(r"history\gtzan\vq-resnet2d_v1\20250327_114125\progress.csv", [1, 2], ["MSE Loss", "Codebook Loss"])
# plot_loss_evolution(r"history\vectors_5d\ae_deep_world\20250423_230452\progress.csv")
# plot_loss_evolution(r"history\mnist\ae_vanilla\20250301_212032\progress.csv", [1], ["MSE Loss"])
# plot_loss_evolution(r"history\mnist\vae_small_kl\20250301_212831\progress.csv", [1, 2], ["MSE Loss", "KL Divergence"])
plot_loss_evolution(r"history\mnist\vae_big_kl\20250301_213545\progress.csv", [1, 2], ["MSE Loss", "KL Divergence"])


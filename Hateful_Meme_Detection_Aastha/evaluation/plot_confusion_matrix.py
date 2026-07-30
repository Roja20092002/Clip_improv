import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def plot_cm(npy_path, title):
    cm = np.load(npy_path)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)

    out_path = npy_path.replace(".npy", ".png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_confusion_matrix.py <path.npy> <title>")
        sys.exit(1)

    plot_cm(sys.argv[1], sys.argv[2])

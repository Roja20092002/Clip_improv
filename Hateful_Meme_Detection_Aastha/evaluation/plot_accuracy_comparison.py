import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the summary CSV
df = pd.read_csv("evaluation/final_accuracy_comparison.csv")

datasets = df["Dataset"]
single = df["Single-Task Accuracy"]
joint = df["Joint MTL Accuracy"]

x = np.arange(len(datasets))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(x - width/2, single, width, label="Single-Task")
plt.bar(x + width/2, joint, width, label="Joint MTL")

plt.xticks(x, datasets, rotation=15)
plt.ylabel("Accuracy")
plt.xlabel("Dataset")
plt.title("Single-Task vs Joint MTL Accuracy Comparison")
plt.legend()

plt.tight_layout()

# Save figure
plt.savefig("evaluation/accuracy_comparison.png")
print("Saved plot to evaluation/accuracy_comparison.png")

plt.show()

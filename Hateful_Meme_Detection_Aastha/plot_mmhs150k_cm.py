import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cm = np.load("evaluation/joint_mmhs150k_confusion_matrix.npy")

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Joint MTL – MMHS150K Confusion Matrix")

plt.tight_layout()
plt.savefig("evaluation/joint_mmhs150k_confusion_matrix.png", dpi=300)
plt.close()

print("Saved confusion matrix to evaluation/joint_mmhs150k_confusion_matrix.png")

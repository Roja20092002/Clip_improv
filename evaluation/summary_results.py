import pandas as pd

# ===============================
# Final Accuracy Summary
# ===============================

data = {
    "Dataset": [
        "Hateful Memes",
        "MAMI (Label)",
        "MultiOFF",
        "MMHS150K"
    ],
    "Single-Task Accuracy": [
        0.58,    # approx HM single-task
        0.6250,
        0.5926,
        0.6585
    ],
    "Joint MTL Accuracy": [
        0.5860,
        0.6250,
        0.6148,
        0.6923
    ]
}

df = pd.DataFrame(data)

print("\n===== FINAL ACCURACY COMPARISON =====\n")
print(df)

# Save to CSV
df.to_csv("evaluation/final_accuracy_comparison.csv", index=False)
print("\nSaved to evaluation/final_accuracy_comparison.csv")

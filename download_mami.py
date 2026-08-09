from datasets import load_dataset

print("Loading MAMI dataset...")

dataset = load_dataset("scintist/MAMI-dataset")

print("\nDataset:")
print(dataset)

print("\nFirst sample:")
print(dataset["train"][0])
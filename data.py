from datasets import load_dataset

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Split the dataset into training and testing sets
train_data = dataset["train"]
test_data = dataset["test"]
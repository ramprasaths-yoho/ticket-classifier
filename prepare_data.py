import pandas as pd

from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/support_tickets.csv")

# Convert text to lowercase
df["text"] = df["text"].str.lower()

# Remove extra spaces
df["text"] = df["text"].str.strip()

# Remove duplicate tickets
df = df.drop_duplicates(subset=["text"])

# Remove missing values
df = df.dropna()

print("Total rows:", len(df))

# Category class balance
print("\nCategory counts:")
print(df["category"].value_counts())

# Priority class balance
print("\nPriority counts:")
print(df["priority"].value_counts())

# Split data into 70% training and 30% temporary
train, temp = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["category"]
)

# Split temporary data into 15% validation and 15% testing
validation, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    stratify=temp["category"]
)

# Save datasets
train.to_csv("data/train.csv", index=False)
validation.to_csv("data/validation.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("\nDataset splitting completed!")

print("Training rows:", len(train))
print("Validation rows:", len(validation))
print("Testing rows:", len(test))
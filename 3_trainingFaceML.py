from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os
import numpy as np

# File paths
embeddingFile = "output/embeddings.pickle"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"

# Ensure the output directory exists
os.makedirs("output", exist_ok=True)

# Load embeddings
print("Loading face embeddings...")
try:
    with open(embeddingFile, "rb") as f:
        data = pickle.load(f)
except FileNotFoundError:
    print(f"Error: '{embeddingFile}' not found. Ensure embeddings have been created.")
    exit()

# Debug: Print basic info about embeddings and names
print(f"Number of embeddings: {len(data.get('embeddings', []))}")
print(f"Number of names: {len(data.get('names', []))}")

# Check for empty data or mismatch in lengths
if not data.get("embeddings") or not data.get("names"):
    print("Error: Embeddings or names are missing in the data.")
    exit()

if len(data["embeddings"]) != len(data["names"]):
    print("Error: Mismatch in the number of embeddings and names.")
    exit()

# Encode labels
print("Encoding labels...")
labelEnc = LabelEncoder()

# Check if there's only one unique name
if len(set(data["names"])) < 2:
    print("Only one class found. Adding a second class for testing purposes.")
    data["names"].append("Person2")  # Add a second unique name
    data["embeddings"].append(np.random.rand(len(data["embeddings"][0])))  # Add a random embedding vector

# Encode the labels with the added class if needed
labels = labelEnc.fit_transform(data["names"])

# Train model
print("Training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# Save the trained recognizer model
with open(recognizerFile, "wb") as f:
    pickle.dump(recognizer, f)
print(f"Recognizer model saved to '{recognizerFile}'.")

# Save the label encoder
with open(labelEncFile, "wb") as f:
    pickle.dump(labelEnc, f)
print(f"Label encoder saved to '{labelEncFile}'.")

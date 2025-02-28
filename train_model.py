import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Load the census.csv data.
project_path = os.path.abspath(".")
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)
data = pd.read_csv(data_path)

# Split the data into training and testing datasets.
train, test = train_test_split(data, test_size=0.2, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process training data.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process testing data.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train the model on the training dataset.
model = train_model(X_train, y_train)

# Save the model and the encoder.
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)

# Load the model.
model = load_model(model_path)

# Run inference on the test dataset.
preds = inference(model, X_test)

# Calculate and print the metrics.
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute performance on model slices and write to file.
with open("slice_output.txt", "w") as f:
    for col in cat_features:
        for slice_value in sorted(test[col].unique()):
            count = test[test[col] == slice_value].shape[0]
            p, r, fb = performance_on_categorical_slice(
                test,
                col,
                slice_value,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                model=model
            )
            print(f"{col}: {slice_value}, Count: {count:,}", file=f)
            print(
                f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}",
                file=f
            )

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = r"C:\Cancer_ANN_Project"

MODEL_PATH = os.path.join(PROJECT_DIR, "improved_cnn.keras")

VALIDATION_CSV = os.path.join(
    PROJECT_DIR, "validation_split.csv"
)

TEST_CSV = os.path.join(
    PROJECT_DIR, "final_test_split.csv"
)

IMAGE_ROOT = os.path.join(
    PROJECT_DIR, "jpeg"
)

IMAGE_SIZE = (224, 224)

# ============================================================
# IMAGE PATH RESOLUTION
# ============================================================

print("=" * 60)
print("MODEL PREDICTION ANALYSIS")
print("=" * 60)

print("\nSearching for images...")

image_files = {}

for root, dirs, files in os.walk(IMAGE_ROOT):
    for file in files:
        if file.lower().endswith(".jpg"):
            image_files[file] = os.path.join(root, file)

print("Images found:", len(image_files))


def resolve_image_path(path):
    """
    Resolve the image using the filename contained
    in the CSV path.
    """

    filename = os.path.basename(str(path))

    if filename in image_files:
        return image_files[filename]

    return None


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading improved model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_image(path):

    image = Image.open(path).convert("L")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=-1
    )

    return image


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def generate_predictions(csv_path):

    df = pd.read_csv(csv_path)

    paths = []
    labels = []

    for _, row in df.iterrows():

        resolved = resolve_image_path(
            row["image_path"]
        )

        if resolved is not None:

            paths.append(resolved)

            labels.append(
                int(row["label"])
            )

    print(
        f"\nRecords loaded from {os.path.basename(csv_path)}:",
        len(paths)
    )

    images = np.array(
        [load_image(p) for p in paths],
        dtype=np.float32
    )

    labels = np.array(
        labels,
        dtype=np.int32
    )

    probabilities = model.predict(
        images,
        batch_size=32,
        verbose=1
    ).ravel()

    return labels, probabilities


# ============================================================
# VALIDATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION PREDICTIONS")
print("=" * 60)

val_y, val_prob = generate_predictions(
    VALIDATION_CSV
)


# ============================================================
# TEST ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST PREDICTIONS")
print("=" * 60)

test_y, test_prob = generate_predictions(
    TEST_CSV
)


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("PREDICTED PROBABILITY DISTRIBUTION")
print("=" * 60)

print("\nValidation probabilities:")
print("Minimum :", round(float(val_prob.min()), 4))
print("Maximum :", round(float(val_prob.max()), 4))
print("Mean    :", round(float(val_prob.mean()), 4))
print("Median  :", round(float(np.median(val_prob)), 4))

print("\nTest probabilities:")
print("Minimum :", round(float(test_prob.min()), 4))
print("Maximum :", round(float(test_prob.max()), 4))
print("Mean    :", round(float(test_prob.mean()), 4))
print("Median  :", round(float(np.median(test_prob)), 4))


# ============================================================
# PERCENTAGE ABOVE DIFFERENT THRESHOLDS
# ============================================================

print("\n" + "=" * 60)
print("PREDICTIONS ABOVE THRESHOLD")
print("=" * 60)

thresholds = [
    0.10,
    0.20,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.70,
    0.80,
    0.90
]

results = []

for threshold in thresholds:

    val_pred = (
        val_prob >= threshold
    ).astype(int)

    test_pred = (
        test_prob >= threshold
    ).astype(int)

    val_positive = val_pred.mean() * 100
    test_positive = test_pred.mean() * 100

    val_accuracy = accuracy_score(
        val_y,
        val_pred
    )

    val_precision = precision_score(
        val_y,
        val_pred,
        zero_division=0
    )

    val_recall = recall_score(
        val_y,
        val_pred,
        zero_division=0
    )

    val_f1 = f1_score(
        val_y,
        val_pred,
        zero_division=0
    )

    results.append({
        "threshold": threshold,
        "validation_positive_percent":
            val_positive,
        "test_positive_percent":
            test_positive,
        "validation_accuracy":
            val_accuracy,
        "validation_precision":
            val_precision,
        "validation_recall":
            val_recall,
        "validation_f1":
            val_f1
    })


results_df = pd.DataFrame(results)

print(
    results_df.to_string(index=False)
)


# ============================================================
# CLASS-SPECIFIC PROBABILITIES
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION PROBABILITY BY ACTUAL CLASS")
print("=" * 60)

non_malignant_prob = val_prob[
    val_y == 0
]

malignant_prob = val_prob[
    val_y == 1
]

print("\nNon-malignant cases:")
print(
    "Mean probability:",
    round(float(non_malignant_prob.mean()), 4)
)
print(
    "Median probability:",
    round(float(np.median(non_malignant_prob)), 4)
)

print("\nMalignant cases:")
print(
    "Mean probability:",
    round(float(malignant_prob.mean()), 4)
)
print(
    "Median probability:",
    round(float(np.median(malignant_prob)), 4)
)


# ============================================================
# ROC-AUC
# ============================================================

try:

    val_auc = roc_auc_score(
        val_y,
        val_prob
    )

    test_auc = roc_auc_score(
        test_y,
        test_prob
    )

    print("\n" + "=" * 60)
    print("ROC-AUC")
    print("=" * 60)

    print(
        "Validation ROC-AUC:",
        round(float(val_auc), 4)
    )

    print(
        "Test ROC-AUC:",
        round(float(test_auc), 4)
    )

except Exception as e:

    print(
        "\nCould not calculate ROC-AUC:",
        e
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = os.path.join(
    PROJECT_DIR,
    "prediction_diagnostic_analysis.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print(
    "\nDiagnostic analysis saved to:"
)

print(output_path)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
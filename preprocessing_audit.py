import os
import hashlib
import pandas as pd
import numpy as np
from PIL import Image

DATASET_PATH = r"C:\Cancer_ANN_Project"

FILES = {
    "Training": "train_split.csv",
    "Validation": "validation_split.csv",
    "Test": "final_test_split.csv"
}

# ---------------------------------------------------------
# 1. Load all splits
# ---------------------------------------------------------

datasets = {}

for name, filename in FILES.items():

    filepath = os.path.join(
        DATASET_PATH,
        filename
    )

    datasets[name] = pd.read_csv(filepath)

all_data = pd.concat(
    datasets.values(),
    ignore_index=True
)

print("=" * 75)
print("PREPROCESSING AND DATA-QUALITY AUDIT")
print("=" * 75)

print("Total records:", len(all_data))
print(
    "Unique patients:",
    all_data["patient_id"].nunique()
)

# ---------------------------------------------------------
# 2. Missing values
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("MISSING VALUES")
print("=" * 75)

missing = all_data.isna().sum()

missing = missing[
    missing > 0
].sort_values(
    ascending=False
)

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing.to_string())

print("\nTarget label missing:")
print(
    all_data["label"].isna().sum()
)

print("Image path missing:")
print(
    all_data["local_image_path"].isna().sum()
)

# ---------------------------------------------------------
# 3. Duplicate records
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("DUPLICATE CHECK")
print("=" * 75)

print(
    "Duplicate complete rows:",
    all_data.duplicated().sum()
)

print(
    "Duplicate patient + image paths:",
    all_data.duplicated(
        subset=["patient_id", "local_image_path"]
    ).sum()
)

print(
    "Duplicate image paths:",
    all_data["local_image_path"].duplicated().sum()
)

print(
    "Unique image paths:",
    all_data["local_image_path"].nunique()
)

# ---------------------------------------------------------
# 4. Patient leakage
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("PATIENT-LEVEL LEAKAGE CHECK")
print("=" * 75)

train_patients = set(
    datasets["Training"]["patient_id"]
)

validation_patients = set(
    datasets["Validation"]["patient_id"]
)

test_patients = set(
    datasets["Test"]["patient_id"]
)

print(
    "Training/Validation overlap:",
    len(train_patients & validation_patients)
)

print(
    "Training/Test overlap:",
    len(train_patients & test_patients)
)

print(
    "Validation/Test overlap:",
    len(validation_patients & test_patients)
)

# ---------------------------------------------------------
# 5. Image quality and dimensions
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("IMAGE QUALITY AUDIT")
print("=" * 75)

image_results = []

corrupt_images = []

for index, row in all_data.iterrows():

    path = row["local_image_path"]

    result = {
        "patient_id": row["patient_id"],
        "label": row["label"],
        "path": path,
        "width": np.nan,
        "height": np.nan,
        "mode": None,
        "pixel_min": np.nan,
        "pixel_max": np.nan,
        "pixel_mean": np.nan,
        "pixel_std": np.nan,
        "sha256": None,
        "valid": False
    }

    try:

        with Image.open(path) as image:

            image.verify()

        with Image.open(path) as image:

            image = image.convert("L")

            array = np.asarray(
                image,
                dtype=np.float32
            )

            result["width"] = image.width
            result["height"] = image.height
            result["mode"] = image.mode

            result["pixel_min"] = float(
                array.min()
            )

            result["pixel_max"] = float(
                array.max()
            )

            result["pixel_mean"] = float(
                array.mean()
            )

            result["pixel_std"] = float(
                array.std()
            )

        # SHA-256 hash for exact duplicate-image detection
        sha256 = hashlib.sha256()

        with open(path, "rb") as file:

            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b""
            ):
                sha256.update(chunk)

        result["sha256"] = sha256.hexdigest()
        result["valid"] = True

    except Exception as error:

        corrupt_images.append(
            (path, str(error))
        )

    image_results.append(result)

    if (index + 1) % 500 == 0:

        print(
            f"Checked {index + 1} / {len(all_data)} images..."
        )

image_df = pd.DataFrame(image_results)

# ---------------------------------------------------------
# 6. Corrupted/unreadable images
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("CORRUPTED IMAGE CHECK")
print("=" * 75)

print(
    "Valid images:",
    image_df["valid"].sum()
)

print(
    "Invalid/corrupted images:",
    (~image_df["valid"]).sum()
)

if corrupt_images:

    print("\nExamples of invalid images:")

    for path, error in corrupt_images[:10]:

        print(path)
        print(error)

# ---------------------------------------------------------
# 7. Image dimensions
# ---------------------------------------------------------

valid_images = image_df[
    image_df["valid"]
].copy()

print("\n" + "=" * 75)
print("ORIGINAL IMAGE DIMENSIONS")
print("=" * 75)

print(
    "Width minimum:",
    int(valid_images["width"].min())
)

print(
    "Width maximum:",
    int(valid_images["width"].max())
)

print(
    "Width median:",
    float(valid_images["width"].median())
)

print(
    "Height minimum:",
    int(valid_images["height"].min())
)

print(
    "Height maximum:",
    int(valid_images["height"].max())
)

print(
    "Height median:",
    float(valid_images["height"].median())
)

# ---------------------------------------------------------
# 8. Dimension outlier detection using IQR
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("DIMENSION OUTLIER CHECK")
print("=" * 75)


def find_iqr_outliers(series):

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)

    return lower, upper


width_lower, width_upper = find_iqr_outliers(
    valid_images["width"]
)

height_lower, height_upper = find_iqr_outliers(
    valid_images["height"]
)

width_outliers = valid_images[
    (valid_images["width"] < width_lower)
    | (valid_images["width"] > width_upper)
]

height_outliers = valid_images[
    (valid_images["height"] < height_lower)
    | (valid_images["height"] > height_upper)
]

print(
    f"Width IQR bounds: "
    f"{width_lower:.2f} to {width_upper:.2f}"
)

print(
    "Width outliers:",
    len(width_outliers)
)

print(
    f"Height IQR bounds: "
    f"{height_lower:.2f} to {height_upper:.2f}"
)

print(
    "Height outliers:",
    len(height_outliers)
)

print(
    "\nNote: Dimension outliers are flagged for review, "
    "not automatically deleted."
)

# ---------------------------------------------------------
# 9. Exact duplicate image detection
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("DUPLICATE IMAGE CONTENT CHECK")
print("=" * 75)

duplicate_hashes = valid_images[
    valid_images["sha256"].duplicated(
        keep=False
    )
]

print(
    "Images involved in exact duplicate groups:",
    len(duplicate_hashes)
)

print(
    "Unique image hashes:",
    valid_images["sha256"].nunique()
)

# ---------------------------------------------------------
# 10. Pixel statistics
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("PIXEL VALUE CHECK")
print("=" * 75)

print(
    "Original pixel minimum:",
    valid_images["pixel_min"].min()
)

print(
    "Original pixel maximum:",
    valid_images["pixel_max"].max()
)

print(
    "Mean image pixel value:",
    valid_images["pixel_mean"].mean()
)

print(
    "Average image pixel standard deviation:",
    valid_images["pixel_std"].mean()
)

# ---------------------------------------------------------
# 11. Class distribution
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("CLASS DISTRIBUTION")
print("=" * 75)

class_counts = all_data[
    "label"
].value_counts().sort_index()

for label, count in class_counts.items():

    percentage = (
        count / len(all_data)
    ) * 100

    if label == 0:
        name = "Non-malignant"
    else:
        name = "Malignant"

    print(
        f"{name}: {count} "
        f"({percentage:.2f}%)"
    )

# ---------------------------------------------------------
# 12. Save audit results
# ---------------------------------------------------------

audit_file = os.path.join(
    DATASET_PATH,
    "preprocessing_image_audit.csv"
)

image_df.to_csv(
    audit_file,
    index=False
)

print("\n" + "=" * 75)
print("AUDIT FILE SAVED")
print("=" * 75)

print(audit_file)

print("\n" + "=" * 75)
print("PREPROCESSING AUDIT COMPLETE")
print("=" * 75)
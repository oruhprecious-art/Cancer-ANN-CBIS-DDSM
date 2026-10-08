import os
import pandas as pd

DATASET_PATH = r"C:\Cancer_ANN_Project"
CSV_PATH = os.path.join(DATASET_PATH, "csv")
JPEG_PATH = os.path.join(DATASET_PATH, "jpeg")

# Load test CSV files
calc_test = pd.read_csv(
    os.path.join(CSV_PATH, "calc_case_description_test_set.csv")
)

mass_test = pd.read_csv(
    os.path.join(CSV_PATH, "mass_case_description_test_set.csv")
)

dicom = pd.read_csv(
    os.path.join(CSV_PATH, "dicom_info.csv")
)

# Identify lesion type
calc_test["lesion_type"] = "calcification"
mass_test["lesion_type"] = "mass"

# Combine test datasets
test_df = pd.concat(
    [calc_test, mass_test],
    ignore_index=True
)

# Convert pathology into binary classification
test_df["label"] = test_df["pathology"].map({
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1
})

# Extract the DICOM series identifier
test_df["cropped_series"] = (
    test_df["cropped image file path"]
    .str.split("/")
    .str[0]
)

# Keep only cropped images
cropped_dicom = dicom[
    dicom["SeriesDescription"]
    .astype(str)
    .str.lower() == "cropped images"
].copy()

cropped_dicom = cropped_dicom[
    [
        "PatientID",
        "image_path",
        "SeriesDescription",
        "Rows",
        "Columns"
    ]
]

# Merge test metadata with cropped-image information
mapping = test_df.merge(
    cropped_dicom,
    left_on="cropped_series",
    right_on="PatientID",
    how="left"
)

# Remove duplicate mappings
mapping = mapping.drop_duplicates(
    subset=[
        "patient_id",
        "cropped_series",
        "pathology"
    ]
)

print("=" * 70)
print("TEST DATA MAPPING")
print("=" * 70)

print("Test records:", len(test_df))

print(
    "Mapped records:",
    mapping["image_path"].notna().sum()
)

print(
    "Unmapped records:",
    mapping["image_path"].isna().sum()
)

# Convert dataset-relative image path to local path
def find_local_image(relative_path):

    if pd.isna(relative_path):
        return None

    relative_path = str(relative_path)

    marker = "CBIS-DDSM/jpeg/"

    if marker in relative_path:
        relative_path = relative_path.split(
            marker,
            1
        )[1]

    return os.path.join(
        JPEG_PATH,
        relative_path
    )


mapping["local_image_path"] = (
    mapping["image_path"]
    .apply(find_local_image)
)

# Check whether image exists
def image_exists(path):

    if not isinstance(path, str):
        return False

    return os.path.isfile(path)


mapping["image_exists"] = (
    mapping["local_image_path"]
    .apply(image_exists)
)

print("\n" + "=" * 70)
print("LOCAL IMAGE EXISTENCE")
print("=" * 70)

print(
    "Images found:",
    mapping["image_exists"].sum()
)

print(
    "Images NOT found:",
    (~mapping["image_exists"]).sum()
)

print("\n" + "=" * 70)
print("TEST LABEL DISTRIBUTION")
print("=" * 70)

print(mapping["label"].value_counts().sort_index())

print("\n0 = BENIGN + BENIGN_WITHOUT_CALLBACK")
print("1 = MALIGNANT")

# Keep only usable records
usable = mapping[
    mapping["image_exists"]
    & mapping["label"].notna()
].copy()

print("\n" + "=" * 70)
print("USABLE TEST RECORDS")
print("=" * 70)

print("Usable records:", len(usable))

print(
    "Malignant:",
    (usable["label"] == 1).sum()
)

print(
    "Non-malignant:",
    (usable["label"] == 0).sum()
)

# Save mapping
output_file = os.path.join(
    DATASET_PATH,
    "test_image_mapping.csv"
)

usable.to_csv(
    output_file,
    index=False
)

print("\nTest mapping saved to:")
print(output_file)

print("\n" + "=" * 70)
print("TEST MAPPING COMPLETE")
print("=" * 70)
import os
import pandas as pd


# ============================================================
# CBIS-DDSM IMAGE-LABEL MAPPING CHECK
# ============================================================

DATASET_PATH = r"C:\Cancer_ANN_Project"
CSV_PATH = os.path.join(DATASET_PATH, "csv")
JPEG_PATH = os.path.join(DATASET_PATH, "jpeg")


# ------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------

calc_train = pd.read_csv(
    os.path.join(CSV_PATH, "calc_case_description_train_set.csv")
)

mass_train = pd.read_csv(
    os.path.join(CSV_PATH, "mass_case_description_train_set.csv")
)

dicom = pd.read_csv(
    os.path.join(CSV_PATH, "dicom_info.csv")
)


# ------------------------------------------------------------
# Add lesion type
# ------------------------------------------------------------

calc_train["lesion_type"] = "calcification"
mass_train["lesion_type"] = "mass"


# ------------------------------------------------------------
# Combine training metadata
# ------------------------------------------------------------

train_df = pd.concat(
    [calc_train, mass_train],
    ignore_index=True
)


# ------------------------------------------------------------
# Convert pathology to binary classification
# ------------------------------------------------------------

train_df["label"] = train_df["pathology"].map({
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1
})


# ------------------------------------------------------------
# Extract cropped series identifier
# ------------------------------------------------------------

train_df["cropped_series"] = (
    train_df["cropped image file path"]
    .str.split("/")
    .str[0]
)


# ------------------------------------------------------------
# DICOM PatientID matching
# ------------------------------------------------------------

dicom_patient_ids = set(
    dicom["PatientID"]
    .dropna()
    .astype(str)
)

train_df["series_match"] = (
    train_df["cropped_series"]
    .astype(str)
    .isin(dicom_patient_ids)
)


print("=" * 70)
print("CSV TO DICOM SERIES MATCH")
print("=" * 70)

print("Training records:", len(train_df))

print(
    "Records with matching DICOM PatientID:",
    train_df["series_match"].sum()
)

print(
    "Records without matching DICOM PatientID:",
    (~train_df["series_match"]).sum()
)


# ------------------------------------------------------------
# Show unmatched records
# ------------------------------------------------------------

if (~train_df["series_match"]).sum() > 0:

    print("\nUnmatched examples:")

    print(
        train_df.loc[
            ~train_df["series_match"],
            [
                "patient_id",
                "cropped_series",
                "pathology"
            ]
        ]
        .to_string(index=False)
    )


# ------------------------------------------------------------
# Keep only cropped images
#
# IMPORTANT:
# Do NOT train the model using ROI mask images.
# ------------------------------------------------------------

cropped_dicom = dicom[
    dicom["SeriesDescription"].astype(str).str.lower()
    == "cropped images"
].copy()


print("\n" + "=" * 70)
print("CROPPED IMAGE FILTER")
print("=" * 70)

print(
    "Total DICOM/JPEG records:",
    len(dicom)
)

print(
    "Cropped image records:",
    len(cropped_dicom)
)

print(
    "Other records:",
    len(dicom) - len(cropped_dicom)
)


# ------------------------------------------------------------
# Select only required DICOM information
# ------------------------------------------------------------

cropped_dicom = cropped_dicom[
    [
        "PatientID",
        "image_path",
        "SeriesDescription",
        "Rows",
        "Columns"
    ]
].copy()


# ------------------------------------------------------------
# Merge training metadata with cropped JPEG information
# ------------------------------------------------------------

mapping = train_df.merge(
    cropped_dicom,
    left_on="cropped_series",
    right_on="PatientID",
    how="left"
)


# ------------------------------------------------------------
# Remove duplicate mappings if necessary
# ------------------------------------------------------------

mapping = mapping.drop_duplicates(
    subset=[
        "patient_id",
        "cropped_series",
        "pathology"
    ]
)


print("\n" + "=" * 70)
print("CROPPED IMAGE MAPPING")
print("=" * 70)

print(
    "Training records:",
    len(train_df)
)

print(
    "Mapped records:",
    mapping["image_path"].notna().sum()
)

print(
    "Unmapped records:",
    mapping["image_path"].isna().sum()
)


# ------------------------------------------------------------
# Show examples
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MAPPING EXAMPLES")
print("=" * 70)

print(
    mapping[
        [
            "patient_id",
            "cropped_series",
            "pathology",
            "label",
            "image_path",
            "SeriesDescription",
            "Rows",
            "Columns"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# Convert relative JPEG path to local path
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Safely check whether image exists
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Binary label distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BINARY LABEL DISTRIBUTION")
print("=" * 70)

print(
    mapping["label"]
    .value_counts()
    .sort_index()
)

print("\n0 = BENIGN + BENIGN_WITHOUT_CALLBACK")
print("1 = MALIGNANT")


# ------------------------------------------------------------
# Final usable records
# ------------------------------------------------------------

usable = mapping[
    mapping["image_exists"]
    &
    mapping["label"].notna()
].copy()


print("\n" + "=" * 70)
print("USABLE TRAINING RECORDS")
print("=" * 70)

print(
    "Usable records:",
    len(usable)
)

print(
    "Malignant:",
    (usable["label"] == 1).sum()
)

print(
    "Non-malignant:",
    (usable["label"] == 0).sum()
)


# ------------------------------------------------------------
# Save mapping for later use
# ------------------------------------------------------------

output_file = os.path.join(
    DATASET_PATH,
    "training_image_mapping.csv"
)

usable.to_csv(
    output_file,
    index=False
)


print("\nMapping saved to:")

print(output_file)


print("\n" + "=" * 70)
print("MAPPING CHECK COMPLETE")
print("=" * 70)
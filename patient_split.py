import os
import random
import pandas as pd

DATASET_PATH = r"C:\Cancer_ANN_Project"

TRAIN_FILE = os.path.join(
    DATASET_PATH,
    "training_image_mapping.csv"
)

TEST_FILE = os.path.join(
    DATASET_PATH,
    "test_image_mapping.csv"
)

# ---------------------------------------------------------
# 1. Load the mapped datasets
# ---------------------------------------------------------

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("=" * 70)
print("PATIENT-LEVEL DATA SPLIT")
print("=" * 70)

print("Original training records:", len(train_df))
print("Original test records:", len(test_df))

# Combine all usable records
data = pd.concat(
    [train_df, test_df],
    ignore_index=True
)

print("Combined records:", len(data))

# ---------------------------------------------------------
# 2. Check patients
# ---------------------------------------------------------

unique_patients = data["patient_id"].dropna().unique().tolist()

print("\nUnique patients:", len(unique_patients))

records_per_patient = data.groupby(
    "patient_id"
).size()

print(
    "Patients with more than one image:",
    (records_per_patient > 1).sum()
)

print(
    "Maximum images for one patient:",
    records_per_patient.max()
)

# ---------------------------------------------------------
# 3. Random patient-level split
# ---------------------------------------------------------

random.seed(42)

random.shuffle(unique_patients)

total_patients = len(unique_patients)

train_patient_count = round(total_patients * 0.70)
validation_patient_count = round(total_patients * 0.15)

train_patients = set(
    unique_patients[:train_patient_count]
)

validation_patients = set(
    unique_patients[
        train_patient_count:
        train_patient_count + validation_patient_count
    ]
)

test_patients = set(
    unique_patients[
        train_patient_count + validation_patient_count:
    ]
)

# ---------------------------------------------------------
# 4. Create datasets
# ---------------------------------------------------------

train_data = data[
    data["patient_id"].isin(train_patients)
].copy()

validation_data = data[
    data["patient_id"].isin(validation_patients)
].copy()

final_test_data = data[
    data["patient_id"].isin(test_patients)
].copy()

# ---------------------------------------------------------
# 5. Display dataset sizes
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL DATASET SIZES")
print("=" * 70)

print("Training records:", len(train_data))
print("Validation records:", len(validation_data))
print("Test records:", len(final_test_data))

print(
    "Total:",
    len(train_data)
    + len(validation_data)
    + len(final_test_data)
)

# ---------------------------------------------------------
# 6. Display patient counts
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PATIENT COUNTS")
print("=" * 70)

print(
    "Training patients:",
    train_data["patient_id"].nunique()
)

print(
    "Validation patients:",
    validation_data["patient_id"].nunique()
)

print(
    "Test patients:",
    final_test_data["patient_id"].nunique()
)

# ---------------------------------------------------------
# 7. Patient leakage check
# ---------------------------------------------------------

train_patient_check = set(
    train_data["patient_id"]
)

validation_patient_check = set(
    validation_data["patient_id"]
)

test_patient_check = set(
    final_test_data["patient_id"]
)

train_validation_overlap = (
    train_patient_check
    & validation_patient_check
)

train_test_overlap = (
    train_patient_check
    & test_patient_check
)

validation_test_overlap = (
    validation_patient_check
    & test_patient_check
)

print("\n" + "=" * 70)
print("PATIENT LEAKAGE CHECK")
print("=" * 70)

print(
    "Training/Validation overlap:",
    len(train_validation_overlap)
)

print(
    "Training/Test overlap:",
    len(train_test_overlap)
)

print(
    "Validation/Test overlap:",
    len(validation_test_overlap)
)

# ---------------------------------------------------------
# 8. Class distribution
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)


def show_distribution(name, df):

    print(f"\n{name}")

    counts = df["label"].value_counts().sort_index()

    print(counts)

    total = len(df)

    for label, count in counts.items():

        percentage = (count / total) * 100

        if label == 0:
            class_name = "Non-malignant"
        else:
            class_name = "Malignant"

        print(
            f"{class_name}: "
            f"{count} ({percentage:.2f}%)"
        )


show_distribution(
    "TRAINING SET",
    train_data
)

show_distribution(
    "VALIDATION SET",
    validation_data
)

show_distribution(
    "FINAL TEST SET",
    final_test_data
)

# ---------------------------------------------------------
# 9. Save the datasets
# ---------------------------------------------------------

train_output = os.path.join(
    DATASET_PATH,
    "train_split.csv"
)

validation_output = os.path.join(
    DATASET_PATH,
    "validation_split.csv"
)

test_output = os.path.join(
    DATASET_PATH,
    "final_test_split.csv"
)

train_data.to_csv(
    train_output,
    index=False
)

validation_data.to_csv(
    validation_output,
    index=False
)

final_test_data.to_csv(
    test_output,
    index=False
)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(train_output)
print(validation_output)
print(test_output)

print("\n" + "=" * 70)
print("PATIENT-LEVEL SPLIT COMPLETE")
print("=" * 70)
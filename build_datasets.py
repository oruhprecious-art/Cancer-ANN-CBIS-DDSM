import os
import pandas as pd
import tensorflow as tf

DATASET_PATH = r"C:\Cancer_ANN_Project"

TRAIN_FILE = os.path.join(
    DATASET_PATH,
    "train_split.csv"
)

VALIDATION_FILE = os.path.join(
    DATASET_PATH,
    "validation_split.csv"
)

TEST_FILE = os.path.join(
    DATASET_PATH,
    "final_test_split.csv"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------
# 1. Load metadata
# ---------------------------------------------------------

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

print("=" * 70)
print("TENSORFLOW DATA PIPELINE")
print("=" * 70)

print("Training records:", len(train_df))
print("Validation records:", len(validation_df))
print("Test records:", len(test_df))


# ---------------------------------------------------------
# 2. Image loading and preprocessing
# ---------------------------------------------------------

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=1
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image, label


# ---------------------------------------------------------
# 3. Create TensorFlow datasets
# ---------------------------------------------------------

def create_dataset(dataframe, training=False):

    paths = dataframe["local_image_path"].values
    labels = dataframe["label"].values.astype("float32")

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=42,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        AUTOTUNE
    )

    return dataset


train_dataset = create_dataset(
    train_df,
    training=True
)

validation_dataset = create_dataset(
    validation_df,
    training=False
)

test_dataset = create_dataset(
    test_df,
    training=False
)


# ---------------------------------------------------------
# 4. Inspect one batch
# ---------------------------------------------------------

images, labels = next(
    iter(train_dataset)
)

print("\n" + "=" * 70)
print("TRAINING BATCH CHECK")
print("=" * 70)

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Image data type:", images.dtype)
print(
    "Minimum pixel value:",
    float(tf.reduce_min(images))
)
print(
    "Maximum pixel value:",
    float(tf.reduce_max(images))
)

print("\nFirst 10 labels:")
print(labels[:10].numpy())


# ---------------------------------------------------------
# 5. Check dataset sizes
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BATCH COUNTS")
print("=" * 70)

print(
    "Training batches:",
    tf.data.experimental.cardinality(
        train_dataset
    ).numpy()
)

print(
    "Validation batches:",
    tf.data.experimental.cardinality(
        validation_dataset
    ).numpy()
)

print(
    "Test batches:",
    tf.data.experimental.cardinality(
        test_dataset
    ).numpy()
)


# ---------------------------------------------------------
# 6. Class distribution
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(
    "Training malignant:",
    (train_df["label"] == 1).sum()
)

print(
    "Training non-malignant:",
    (train_df["label"] == 0).sum()
)

print(
    "Validation malignant:",
    (validation_df["label"] == 1).sum()
)

print(
    "Validation non-malignant:",
    (validation_df["label"] == 0).sum()
)

print(
    "Test malignant:",
    (test_df["label"] == 1).sum()
)

print(
    "Test non-malignant:",
    (test_df["label"] == 0).sum()
)

print("\n" + "=" * 70)
print("DATA PIPELINE READY")
print("=" * 70)
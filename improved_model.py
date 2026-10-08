import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight


# =========================================================
# 1. SETTINGS
# =========================================================

BASE_DIR = r"C:\Cancer_ANN_Project"

TRAIN_CSV = os.path.join(BASE_DIR, "train_split.csv")
VAL_CSV = os.path.join(BASE_DIR, "validation_split.csv")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# =========================================================
# 2. LOAD DATA
# =========================================================

print("=" * 60)
print("IMPROVED CNN MODEL")
print("=" * 60)

print("\nLoading training and validation data...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print(f"Training records: {len(train_df)}")
print(f"Validation records: {len(val_df)}")

print("\nTraining class distribution:")
print(train_df["label"].value_counts())

print("\nValidation class distribution:")
print(val_df["label"].value_counts())


# =========================================================
# 3. IMAGE PATH RESOLUTION
# =========================================================

print("\nSearching for image files...")

image_index = {}

for root, dirs, files in os.walk(BASE_DIR):

    for filename in files:

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            full_path = os.path.join(
                root,
                filename
            )

            image_index[filename.lower()] = full_path


print(
    f"JPEG/PNG images found: {len(image_index)}"
)


def resolve_image_path(path):

    filename = os.path.basename(
        str(path)
    ).lower()

    if filename in image_index:
        return image_index[filename]

    return None


train_paths = train_df["image_path"].apply(
    resolve_image_path
).values

val_paths = val_df["image_path"].apply(
    resolve_image_path
).values

train_labels = train_df["label"].values.astype(
    np.float32
)

val_labels = val_df["label"].values.astype(
    np.float32
)


# =========================================================
# 4. CHECK PATHS
# =========================================================

train_missing = np.sum(
    pd.isna(train_paths)
)

val_missing = np.sum(
    pd.isna(val_paths)
)

print("\nTraining images missing:", train_missing)
print("Validation images missing:", val_missing)

if train_missing > 0 or val_missing > 0:

    raise FileNotFoundError(
        "Some training or validation images could not be found."
    )


# =========================================================
# 5. IMAGE PREPROCESSING
# =========================================================

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


# =========================================================
# 6. CREATE DATASETS
# =========================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)


# Shuffle training data
train_dataset = train_dataset.shuffle(
    buffer_size=len(train_df),
    seed=SEED
)

train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


# =========================================================
# 7. DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        mode="horizontal"
    ),

    layers.RandomRotation(
        0.05
    ),

    layers.RandomZoom(
        0.10
    ),

    layers.RandomContrast(
        0.10
    )

], name="data_augmentation")


# =========================================================
# 8. BATCH DATA
# =========================================================

train_dataset = train_dataset.batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
).prefetch(
    tf.data.AUTOTUNE
)


# =========================================================
# 9. CALCULATE CLASS WEIGHTS
# =========================================================

classes = np.array([0, 1])

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_labels.astype(int)
)

class_weights = {
    0: float(class_weights_array[0]),
    1: float(class_weights_array[1])
}

print("\nClass weights:")
print(class_weights)


# =========================================================
# 10. BUILD IMPROVED CNN
# =========================================================

print("\nBuilding improved CNN...")

model = models.Sequential([

    layers.Input(
        shape=(224, 224, 1)
    ),

    # Data augmentation
    data_augmentation,

    # Convolution block 1
    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution block 2
    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution block 3
    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution block 4
    layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Global pooling instead of Flatten
    layers.GlobalAveragePooling2D(),

    # Dense layer
    layers.Dense(
        128,
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Dropout(
        0.40
    ),

    # Binary output
    layers.Dense(
        1,
        activation="sigmoid"
    )

])


# =========================================================
# 11. COMPILE MODEL
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


# =========================================================
# 12. DISPLAY MODEL
# =========================================================

print("\nImproved CNN architecture:")

model.summary()


# =========================================================
# 13. CALLBACKS
# =========================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=0.00001,

    verbose=1
)


# =========================================================
# 14. TRAIN MODEL
# =========================================================

print("\n")
print("=" * 60)
print("STARTING IMPROVED MODEL TRAINING")
print("=" * 60)

history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# =========================================================
# 15. SAVE MODEL
# =========================================================

model_path = os.path.join(
    BASE_DIR,
    "improved_cnn.keras"
)

model.save(
    model_path
)

print("\nImproved model saved to:")
print(model_path)


# =========================================================
# 16. SAVE TRAINING HISTORY
# =========================================================

history_df = pd.DataFrame(
    history.history
)

history_path = os.path.join(
    BASE_DIR,
    "improved_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)

print("\nTraining history saved to:")
print(history_path)


# =========================================================
# 17. DISPLAY FINAL TRAINING RESULTS
# =========================================================

print("\n")
print("=" * 60)
print("FINAL TRAINING RESULTS")
print("=" * 60)

best_epoch = np.argmin(
    history.history["val_loss"]
) + 1

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation loss: "
    f"{min(history.history['val_loss']):.4f}"
)

print(
    f"Best validation accuracy: "
    f"{max(history.history['val_accuracy']):.4f}"
)

print(
    f"Best validation precision: "
    f"{max(history.history['val_precision']):.4f}"
)

print(
    f"Best validation recall: "
    f"{max(history.history['val_recall']):.4f}"
)


# =========================================================
# 18. TRAINING CURVES
# =========================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Improved CNN Training and Validation Accuracy")
plt.legend()

plt.tight_layout()

accuracy_plot = os.path.join(
    BASE_DIR,
    "improved_accuracy_curve.png"
)

plt.savefig(
    accuracy_plot,
    dpi=300
)

plt.close()


plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Improved CNN Training and Validation Loss")
plt.legend()

plt.tight_layout()

loss_plot = os.path.join(
    BASE_DIR,
    "improved_loss_curve.png"
)

plt.savefig(
    loss_plot,
    dpi=300
)

plt.close()


print("\nTraining curves saved.")

print("\n")
print("=" * 60)
print("IMPROVED MODEL TRAINING COMPLETE")
print("=" * 60)
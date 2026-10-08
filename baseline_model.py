import os
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models

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


# =========================================================
# 1. Load metadata
# =========================================================

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)


# =========================================================
# 2. Image preprocessing
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
# 3. TensorFlow dataset creation
# =========================================================

def create_dataset(dataframe, training=False):

    paths = dataframe["local_image_path"].values

    labels = dataframe["label"].values.astype(
        "float32"
    )

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
    validation_df
)

test_dataset = create_dataset(
    test_df
)


# =========================================================
# 4. Build baseline CNN
# =========================================================

model = models.Sequential([

    layers.Input(
        shape=(224, 224, 1)
    ),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(
        0.30
    ),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# =========================================================
# 5. Compile model
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
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
# 6. Display architecture
# =========================================================

print("=" * 70)
print("BASELINE CNN ARCHITECTURE")
print("=" * 70)

model.summary()


# =========================================================
# 7. Train baseline model
# =========================================================

print("\n" + "=" * 70)
print("STARTING BASELINE TRAINING")
print("=" * 70)

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=10
)


# =========================================================
# 8. Evaluate on final test set
# =========================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_results = model.evaluate(
    test_dataset,
    return_dict=True
)

for metric, value in test_results.items():

    print(
        f"{metric}: {value:.4f}"
    )


# =========================================================
# 9. Save model
# =========================================================

model_path = os.path.join(
    DATASET_PATH,
    "baseline_cnn.keras"
)

model.save(model_path)

print("\nBaseline model saved to:")
print(model_path)

print("\n" + "=" * 70)
print("BASELINE TRAINING COMPLETE")
print("=" * 70)
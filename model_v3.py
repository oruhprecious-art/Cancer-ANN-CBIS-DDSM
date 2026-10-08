import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = r"C:\Cancer_ANN_Project"

TRAIN_CSV = os.path.join(
    PROJECT_DIR,
    "train_split.csv"
)

VAL_CSV = os.path.join(
    PROJECT_DIR,
    "validation_split.csv"
)

IMAGE_ROOT = os.path.join(
    PROJECT_DIR,
    "jpeg"
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "model_v3.keras"
)

HISTORY_PATH = os.path.join(
    PROJECT_DIR,
    "model_v3_history.csv"
)

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 30

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


print("=" * 65)
print("MODEL V3 - BALANCED CNN")
print("=" * 65)


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print("\nTraining records:", len(train_df))
print("Validation records:", len(val_df))


print("\nTraining class distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nValidation class distribution:")
print(val_df["label"].value_counts().sort_index())


# ============================================================
# IMAGE INDEX
# ============================================================

print("\nSearching for images...")

image_files = {}

for root, dirs, files in os.walk(IMAGE_ROOT):

    for file in files:

        if file.lower().endswith(".jpg"):

            image_files[file] = os.path.join(
                root,
                file
            )

print(
    "Images found:",
    len(image_files)
)


# ============================================================
# RESOLVE IMAGE PATH
# ============================================================

def resolve_image_path(path):

    filename = os.path.basename(
        str(path)
    )

    return image_files.get(
        filename
    )


train_df["resolved_path"] = train_df[
    "image_path"
].apply(resolve_image_path)

val_df["resolved_path"] = val_df[
    "image_path"
].apply(resolve_image_path)


train_df = train_df[
    train_df["resolved_path"].notna()
].copy()

val_df = val_df[
    val_df["resolved_path"].notna()
].copy()


print("\nResolved training images:", len(train_df))
print("Resolved validation images:", len(val_df))


# ============================================================
# DATA GENERATOR
# ============================================================

class MammographyGenerator(
    tf.keras.utils.Sequence
):

    def __init__(
        self,
        dataframe,
        batch_size=32,
        shuffle=False
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.indices = np.arange(
            len(self.df)
        )

        self.on_epoch_end()


    def __len__(self):

        return int(
            np.ceil(
                len(self.df)
                /
                self.batch_size
            )
        )


    def __getitem__(self, index):

        batch_indices = self.indices[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        batch_df = self.df.iloc[
            batch_indices
        ]

        images = []
        labels = []

        for _, row in batch_df.iterrows():

            image = load_img(
                row["resolved_path"],
                color_mode="grayscale",
                target_size=IMAGE_SIZE
            )

            image = img_to_array(
                image
            )

            image = image / 255.0

            images.append(image)

            labels.append(
                float(row["label"])
            )

        return (
            np.asarray(
                images,
                dtype=np.float32
            ),
            np.asarray(
                labels,
                dtype=np.float32
            )
        )


    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


# ============================================================
# GENERATORS
# ============================================================

train_generator = MammographyGenerator(
    train_df,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_generator = MammographyGenerator(
    val_df,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential(
    [

        layers.RandomFlip(
            mode="horizontal"
        ),

        layers.RandomRotation(
            factor=0.02
        ),

        layers.RandomZoom(
            height_factor=0.05,
            width_factor=0.05
        )

    ],
    name="conservative_augmentation"
)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

model = models.Sequential(
    [

        layers.Input(
            shape=(224, 224, 1)
        ),

        augmentation,

        # ----------------------------------------------------
        # BLOCK 1
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # BLOCK 2
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # BLOCK 3
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # BLOCK 4
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        layers.GlobalAveragePooling2D(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Dropout(
            0.25
        ),

        layers.Dense(
            1,
            activation="sigmoid"
        )

    ],
    name="Cancer_ANN_Model_V3"
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n")
model.summary()


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=optimizers.Adam(
        learning_rate=0.0003
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


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = callbacks.EarlyStopping(

    monitor="val_loss",

    patience=7,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=1e-6,

    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 65)
print("STARTING TRAINING")
print("=" * 65)


history = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print("\nModel saved to:")
print(MODEL_PATH)


# ============================================================
# SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.insert(
    0,
    "epoch",
    range(
        1,
        len(history_df) + 1
    )
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)

print("\nTraining history saved to:")
print(HISTORY_PATH)


# ============================================================
# BEST EPOCH
# ============================================================

best_epoch = (
    history_df["val_loss"]
    .idxmin()
)

best_row = history_df.iloc[
    best_epoch
]


print("\n")
print("=" * 65)
print("BEST VALIDATION RESULT")
print("=" * 65)

print(
    "Epoch:",
    int(best_row["epoch"])
)

print(
    "Validation loss:",
    round(
        float(best_row["val_loss"]),
        4
    )
)

print(
    "Validation accuracy:",
    round(
        float(best_row["val_accuracy"]),
        4
    )
)

print(
    "Validation precision:",
    round(
        float(best_row["val_precision"]),
        4
    )
)

print(
    "Validation recall:",
    round(
        float(best_row["val_recall"]),
        4
    )
)


print("\n")
print("=" * 65)
print("MODEL V3 TRAINING COMPLETE")
print("=" * 65)
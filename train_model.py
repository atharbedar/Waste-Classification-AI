# ==========================================
#  WASTE CLASSIFICATION
# MobileNetV2 Transfer Learning + Fine Tuning
# ==========================================

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
import os

# ==========================================
# SETTINGS
# ==========================================

dataset_path = "dataset/trashnet-master/data/dataset-resized/dataset-resized"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

SEED = 42

# ==========================================
# CHECK DATASET
# ==========================================

if not os.path.exists(dataset_path):
    print(" Dataset path not found!")
    print(dataset_path)
    exit()

print("\n================================")
print(" WASTE CLASSIFICATION AI")
print("================================")

print("\nDataset Path:")
print(dataset_path)

# ==========================================
# LOAD TRAINING DATA
# ==========================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ==========================================
# LOAD VALIDATION DATA
# ==========================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================================
# CLASS NAMES
# ==========================================

class_names = train_dataset.class_names

NUM_CLASSES = len(class_names)

print("\n================================")
print("CLASSES")
print("================================")

print(class_names)

print("\nNumber of classes:", NUM_CLASSES)

# ==========================================
# PERFORMANCE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

# ==========================================
# DATA AUGMENTATION
# ==========================================

data_augmentation = models.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.10
    ),

    layers.RandomZoom(
        0.10
    ),

], name="data_augmentation")

# ==========================================
# MOBILE NET V2
# ==========================================

base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(224, 224, 3)
)

# ==========================================
# FREEZE BASE MODEL
# ==========================================

base_model.trainable = False

# ==========================================
# BUILD MODEL
# ==========================================

model = models.Sequential([

    layers.Input(
        shape=(224, 224, 3)
    ),

    # Data augmentation
    data_augmentation,

    # MobileNetV2 preprocessing
    layers.Rescaling(
        1.0 / 127.5,
        offset=-1
    ),

    # Pretrained model
    base_model,

    # Convert feature maps into vector
    layers.GlobalAveragePooling2D(),

    # Dense layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Dropout
    layers.Dropout(
        0.4
    ),

    # Output
    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

# ==========================================
# COMPILE
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

# ==========================================
# MODEL SUMMARY
# ==========================================

print("\n================================")
print("MODEL SUMMARY")
print("================================")

model.summary()

# ==========================================
# CALLBACKS
# ==========================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(

    "best_waste_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_accuracy",

    patience=4,

    restore_best_weights=True,

    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=0.00001,

    verbose=1
)

# ==========================================
# INITIAL TRAINING
# ==========================================

print("\n================================")
print("INITIAL TRAINING STARTED")
print("================================")

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=INITIAL_EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)

# ==========================================
# FINE TUNING
# ==========================================

print("\n================================")
print("🔧 FINE TUNING STARTED")
print("================================")

# Unfreeze MobileNetV2
base_model.trainable = True

# Freeze most layers
# Only last 30 layers will train

for layer in base_model.layers[:-30]:

    layer.trainable = False

# ==========================================
# RECOMPILE
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

print("\nMobileNetV2 layers:")
print(len(base_model.layers))

print("Trainable layers:")

for layer in base_model.layers:

    if layer.trainable:

        print(layer.name)

# ==========================================
# FINE TUNE
# ==========================================

history_fine = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)

# ==========================================
# LOAD BEST MODEL
# ==========================================

print("\n================================")
print("LOADING BEST MODEL")
print("================================")

model = tf.keras.models.load_model(
    "best_waste_model.keras"
)

# ==========================================
# FINAL EVALUATION
# ==========================================

print("\n================================")
print("FINAL MODEL EVALUATION")
print("================================")

loss, accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)

print("\nValidation Loss:")
print(loss)

print("\nValidation Accuracy:")
print(
    f"{accuracy * 100:.2f}%"
)

# ==========================================
# SAVE FINAL MODEL
# ==========================================

model.save(
    "waste_classifier.keras"
)

print("\n================================")
print(" MODEL SAVED")
print("================================")

print(
    "waste_classifier.keras"
)

# ==========================================
# SAVE CLASS NAMES
# ==========================================

with open(
    "class_names.txt",
    "w"
) as file:

    for name in class_names:

        file.write(
            name + "\n"
        )

print("\nClass names saved:")
print("class_names.txt")

# ==========================================
# TEST ONE BATCH
# ==========================================

print("\n================================")
print("TEST PREDICTION")
print("================================")

images, labels = next(
    iter(validation_dataset)
)

predictions = model.predict(
    images,
    verbose=0
)

for i in range(
    min(10, len(images))
):

    predicted_index = np.argmax(
        predictions[i]
    )

    actual_index = labels[i].numpy()

    print(
        f"\nActual: "
        f"{class_names[actual_index]}"
    )

    print(
        f"Predicted: "
        f"{class_names[predicted_index]}"
    )

    print(
        f"Confidence: "
        f"{predictions[i][predicted_index] * 100:.2f}%"
    )

print("\n================================")
print(" TRAINING COMPLETED")
print("================================")
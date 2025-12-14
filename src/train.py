import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

# -------------------------------
# 1️⃣ Paths
# -------------------------------
DATA_DIR = "data/"
IMG_DIR_1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
IMG_DIR_2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")
META_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

# -------------------------------
# 2️⃣ Load metadata
# -------------------------------
metadata = pd.read_csv(META_PATH)

# map lesion ID to diagnosis
metadata = metadata[["image_id", "dx"]]

# -------------------------------
# 3️⃣ Add full image path
# -------------------------------
image_paths = []

for img in metadata["image_id"]:
    file1 = os.path.join(IMG_DIR_1, img + ".jpg")
    file2 = os.path.join(IMG_DIR_2, img + ".jpg")

    if os.path.exists(file1):
        image_paths.append(file1)
    elif os.path.exists(file2):
        image_paths.append(file2)
    else:
        image_paths.append(None)

metadata["path"] = image_paths
metadata = metadata.dropna()

# -------------------------------
# 4️⃣ Encode labels
# -------------------------------
labels = sorted(metadata["dx"].unique())
label_to_index = {label: idx for idx, label in enumerate(labels)}
metadata["label"] = metadata["dx"].map(label_to_index)

# -------------------------------
# 5️⃣ Train/Validation split
# -------------------------------
train_df, val_df = train_test_split(metadata, test_size=0.2, random_state=42, stratify=metadata["label"])

# -------------------------------
# 6️⃣ Image Generator
# -------------------------------
IMG_SIZE = 224
BATCH = 32

datagen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=20,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_dataframe(
    train_df,
    x_col="path",
    y_col="dx",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH
)

val_gen = datagen.flow_from_dataframe(
    val_df,
    x_col="path",
    y_col="dx",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH
)

# -------------------------------
# 7️⃣ Build Model (MobileNetV2)
# -------------------------------
base = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False  # freeze

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
output = Dense(len(labels), activation="softmax")(x)

model = Model(inputs=base.input, outputs=output)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# 8️⃣ Train model
# -------------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# -------------------------------
# 9️⃣ Save model
# -------------------------------
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/skin_cancer_model.h5"
model.save(MODEL_PATH)

print("\n✅ Training Complete!")
print("Model saved at:", MODEL_PATH)

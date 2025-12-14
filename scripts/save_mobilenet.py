from tensorflow.keras.applications import MobileNetV2

model = MobileNetV2(weights="imagenet", include_top=True)
model.save("models/mobilenet_best.h5")

print("Saved models/mobilenet_best.h5")

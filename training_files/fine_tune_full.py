import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import os

# ======================================================
# 1️⃣ Load existing trained model
# ======================================================
model_path = "model/resnet50v2_best.keras"
assert os.path.exists(model_path), "❌ Model not found! Make sure resnet50v2_best.keras exists."
model = load_model(model_path)
print(f"✅ Loaded model: {model_path}")

# ======================================================
# 2️⃣ Updated full dataset (old + new Gemini data)
# ======================================================
dataset_dir = "dataset_original"  # make sure you've added your new Gemini images here

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.15  # 15% validation
)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

print("✅ Dataset ready — Training on full dataset (old + new Gemini images)")

# ======================================================
# 3️⃣ Fine-tune setup
# ======================================================
# Unfreeze last 50 layers for adaptation
for layer in model.layers[-100:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ======================================================
# 4️⃣ Train (fine-tuning)
# ======================================================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,               # 5–8 is enough
    verbose=1
)

# ======================================================
# 5️⃣ Save the fine-tuned model
# ======================================================
new_model_path = "model/resnet50v2_finetuned_full.keras"
model.save(new_model_path)
print(f"✅ Fine-tuned model saved as: {new_model_path}")

# ======================================================
# 6️⃣ Evaluate performance
# ======================================================
loss, acc = model.evaluate(val_gen)
print(f"\n🎯 Fine-tuned Validation Accuracy: {acc*100:.2f}%")

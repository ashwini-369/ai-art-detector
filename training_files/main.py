import os
import shutil
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# =========================
# 1️⃣ Paths
# =========================
original_dataset = "dataset_original"
dataset_dir = "dataset"
model_dir = "model"
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# =========================
# 2️⃣ Function to split dataset
# =========================
def split_dataset(class_name, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    src_folder = os.path.join(original_dataset, class_name)
    images = os.listdir(src_folder)
    random.shuffle(images)
    
    train_end = int(train_ratio * len(images))
    val_end = train_end + int(val_ratio * len(images))
    
    splits = {
        "train": images[:train_end],
        "validation": images[train_end:val_end],
        "test": images[val_end:]
    }
    
    for split, img_list in splits.items():
        split_folder = os.path.join(dataset_dir, split, class_name)
        os.makedirs(split_folder, exist_ok=True)
        for img in img_list:
            shutil.copy(os.path.join(src_folder, img), os.path.join(split_folder, img))

# Split both classes
split_dataset("ai_generated")
split_dataset("human_generated")
print("Dataset split completed!")

# =========================
# 3️⃣ Data Generators
# =========================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
).flow_from_directory(
    os.path.join(dataset_dir, "train"),
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(dataset_dir, "validation"),
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)

# =========================
# 4️⃣ Model
# =========================
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# =========================
# 5️⃣ Callbacks
# =========================
checkpoint = ModelCheckpoint(os.path.join(model_dir, "resnet50v2_best.keras"),
                             monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# =========================
# 6️⃣ Training
# =========================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=[checkpoint, early_stop]
)

# =========================
# 7️⃣ Test evaluation
# =========================
test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(dataset_dir, "test"),
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

loss, acc = model.evaluate(test_gen)
print("Test Accuracy:", acc)

# =========================
# 8️⃣ Plot accuracy/loss
# =========================
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title("Loss")
plt.show()
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 9️⃣ Confusion Matrix + Classification Report
# =========================

# Get true labels and predicted labels
y_true = test_gen.classes
y_pred = (model.predict(test_gen) > 0.5).astype(int).flatten()

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred          )
print("\nConfusion Matrix:")
print(cm)

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['AI Generated', 'Human Generated']))

# =========================
#  🔹 Plot confusion matrix
# =========================
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['AI Generated', 'Human Generated'],
            yticklabels=['AI Generated', 'Human Generated'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - AI vs Human Paintings')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

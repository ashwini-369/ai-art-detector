import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load your saved model
model = tf.keras.models.load_model("model/resnet50v2_best.keras")

# Reload test generator
test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    "dataset/test",
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Predictions
y_true = test_gen.classes
y_pred = (model.predict(test_gen) > 0.5).astype(int).flatten()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['AI Generated', 'Human Generated']))

# Plot
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

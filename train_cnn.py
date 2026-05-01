import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# -----------------------------
# 📦 DATA PREPROCESSING
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    'dataset',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'dataset',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

NUM_CLASSES = train_data.num_classes

# -----------------------------
# 🧠 CNN MODEL
# -----------------------------
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# ⏹️ EARLY STOPPING
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# -----------------------------
# 🚀 TRAIN
# -----------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# -----------------------------
# 💾 SAVE MODEL
# -----------------------------
model.save("cnn_model.keras")

# Save labels
with open("cnn_labels.json", "w") as f:
    json.dump(train_data.class_indices, f)

print("✅ CNN Model trained and saved!")

# -----------------------------
# 📊 ACCURACY GRAPH
# -----------------------------
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("CNN Accuracy")
plt.tight_layout()
plt.savefig("cnn_accuracy.png")

# -----------------------------
# 📉 LOSS GRAPH
# -----------------------------
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("CNN Loss")
plt.tight_layout()
plt.savefig("cnn_loss.png")

# -----------------------------
# 📌 CONFUSION MATRIX
# -----------------------------
val_data.reset()
predictions = model.predict(val_data)
y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(val_data.classes, y_pred)

class_labels = list(val_data.class_indices.keys())
class_labels_clean = [label.replace("_", " ") for label in class_labels]

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels_clean,
            yticklabels=class_labels_clean)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CNN Confusion Matrix")
plt.tight_layout()
plt.savefig("cnn_confusion_matrix.png")

# -----------------------------
# 📄 CLASSIFICATION REPORT
# -----------------------------
report = classification_report(
    val_data.classes,
    y_pred,
    target_names=class_labels_clean
)

print("\n📊 CNN Classification Report:\n")
print(report)

with open("cnn_classification_report.txt", "w") as f:
    f.write(report)

# -----------------------------
# 🎯 BEST ACCURACY
# -----------------------------
best_val_acc = max(history.history['val_accuracy'])
print(f"\n🔥 CNN Best Validation Accuracy: {best_val_acc*100:.2f}%")
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# Data augmentation
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

# Load MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze all layers
for layer in base_model.layers:
    layer.trainable = False

# Fine-tune last layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Custom layers
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# Save model
model.save("model.keras")

# Save labels
with open("labels.json", "w") as f:
    json.dump(train_data.class_indices, f)

print("✅ Model trained and saved!")

# -----------------------------
# 📊 PLOTS (Accuracy & Loss)
# -----------------------------
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.savefig("accuracy.png")

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss Graph")
plt.savefig("loss.png")

# -----------------------------
# 📌 CONFUSION MATRIX
# -----------------------------
val_data.reset()
predictions = model.predict(val_data)
y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(val_data.classes, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")

# -----------------------------
# 📄 CLASSIFICATION REPORT
# -----------------------------
class_labels = list(val_data.class_indices.keys())

report = classification_report(val_data.classes, y_pred, target_names=class_labels)

with open("classification_report.txt", "w") as f:
    f.write(report)

print("📊 Metrics generated successfully!") 
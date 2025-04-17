# src/train_model.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import save_model

# Paths
data_dir = "../dataset"
model_path = "../trained_model/model.h5"

# Data Preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    data_dir, target_size=(64, 64), color_mode='grayscale',
    class_mode='categorical', subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir, target_size=(64, 64), color_mode='grayscale',
    class_mode='categorical', subset='validation'
)

# Build Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=10)
save_model(model, model_path)
print(f"Model saved to {model_path}")


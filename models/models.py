import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, IMAGE_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure RGB format
    image = image / 255.0  # Normalize pixel values
    return image

data_dir = r'C:\Users\Lenovo\Desktop\pro\data'
class_names = os.listdir(data_dir)
num_classes = len(class_names)
print(num_classes)

data = []
labels = []

for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        image = preprocess_image(image_path)
        data.append(image)
        labels.append(class_idx)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42)

# Create a CNN model (using a pre-trained VGG16 base)
base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)
cnn_model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

cnn_model.compile(optimizer=Adam(lr=0.0001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

# Train the CNN model
cnn_model.fit(datagen.flow(np.array(X_train), np.array(y_train), batch_size=BATCH_SIZE),
              epochs=10, validation_data=(np.array(X_test), np.array(y_test)))

# Evaluate the CNN model
cnn_predictions = cnn_model.predict(np.array(X_test))
cnn_pred_labels = np.argmax(cnn_predictions, axis=1)
accuracy = accuracy_score(y_test, cnn_pred_labels)
print("CNN Model Accuracy:", accuracy)

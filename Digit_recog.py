import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


print("Enter the image having 1:1 aspect ratio and the number at center")
img_url=input("Enter the path of image you want to recognize")
img = cv2.imread(img_url, cv2.IMREAD_GRAYSCALE) 

if np.mean(img) > 127:
    img = 255 - img

img = cv2.resize(img, (28, 28))
img = img / 255.0
img = img.reshape(1, 28, 28, 1).astype("float32")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train_cnn, y_train, epochs=3)

pred = model.predict(img)
print("Predicted Digit:", np.argmax(pred))
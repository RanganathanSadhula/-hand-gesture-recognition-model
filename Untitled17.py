#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from sklearn.model_selection import train_test_split

# Load the MNIST dataset (contains digits 0-9)
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.uint8)

# Filter images for digits 1 to 5
mask = (mnist.target >= 1) & (mnist.target <= 5)
images = mnist.data[mask]
labels = mnist.target[mask]

# Normalize pixel values to be between 0 and 1
images = images / 255.0

# Flatten the images
images_flattened = images.reshape((images.shape[0], -1))

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images_flattened, labels, test_size=0.2, random_state=42)

# Define the simple neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.rand(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))

    def forward(self, X):
        self.layer1 = 1 / (1 + np.exp(-(np.dot(X, self.weights1) + self.bias1)))
        self.output = 1 / (1 + np.exp(-(np.dot(self.layer1, self.weights2) + self.bias2)))
        return self.output

    def backward(self, X, y, learning_rate):
        error = y - self.output
        output_delta = error * (self.output * (1 - self.output))
        error_hidden = output_delta.dot(self.weights2.T)
        hidden_delta = error_hidden * (self.layer1 * (1 - self.layer1))

        self.weights2 += self.layer1.T.dot(output_delta) * learning_rate
        self.bias2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights1 += X.T.dot(hidden_delta) * learning_rate
        self.bias1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)

# Create and train the neural network
input_size = images_flattened.shape[1]
hidden_size = 128
output_size = 5
learning_rate = 0.01
epochs = 10

model = NeuralNetwork(input_size, hidden_size, output_size)
model.train(X_train, y_train, epochs, learning_rate)

# Test the model
predictions = np.argmax(model.forward(X_val), axis=1)
accuracy = np.mean(predictions == y_val)
print(f'Test accuracy: {accuracy}')


# In[3]:


import cv2
import numpy as np
import math
from tensorflow.keras.models import load_model

# Load the pre-trained hand gesture recognition model
model = load_model("sign_mnist_cnn_50_Epochs.h5")  # Replace with the actual path to your model file

vid = cv2.VideoCapture(0)

while True:
    flag, imgFlip = vid.read()

    # Check if the frame was successfully read
    if not flag:
        print("Error reading frame from webcam.")
        break

    img = cv2.flip(imgFlip, 1)  # Flip horizontally for a mirrored view

    cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), 0)

    # Add a check to ensure imgCrop is not None
    imgCrop = img[100:300, 100:300]
    if imgCrop is None:
        print("Error cropping image.")
        break

    imgBlur = cv2.GaussianBlur(imgCrop, (3, 3), 0)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    imgHSV = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)

    lower = np.array([2, 0, 0])
    upper = np.array([20, 255, 255])
    mask = cv2.inRange(imgHSV, lower, upper)

    kernel = np.ones((5, 5))

    dilation = cv2.dilate(mask, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    filtered_img = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, imgBin = cv2.threshold(filtered_img, 127, 255, 0)

    contours, hierarchy = cv2.findContours(imgBin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(imgCrop, (x, y), (x + w, y + h), (0, 0, 255), 0)

        con_hull = cv2.convexHull(contour)

        con_hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, con_hull)
        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            if angle <= 90:
                count_defects += 1
                cv2.circle(imgCrop, far, 2, [0, 0, 255], -1)

            cv2.line(imgCrop, start, end, [0, 255, 0], 2)

        if count_defects > 0:
            # Preprocess the image for the model
            roi = cv2.resize(imgCrop, (28, 28), interpolation=cv2.INTER_AREA)
            roi = roi.reshape(1, 28, 28, 3) / 255.0  # Normalize the image

            # Make a prediction using the pre-trained model
            prediction = model.predict(roi)
            predicted_class = np.argmax(prediction)

            # Display the recognized number on the video window
            gesture_classes = ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE"]
            cv2.putText(img, f"Predicted: {gesture_classes[predicted_class]}", (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    except:
        pass

    cv2.imshow("Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()


# In[ ]:





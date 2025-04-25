# Import necessary libraries
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow import keras 

# Load the CIFAR-10 dataset from Keras (contains 60,000 32x32 images in 10 classes)
(X_train, Y_teain), (X_test, Y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values to scale them between 0 and 1 (improves training efficiency)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Convert class labels to one-hot encoded format (10 categories for CIFAR-10)
Y_teain = keras.utils.to_categorical(Y_teain, 10)
Y_test = keras.utils.to_categorical(Y_test, 10)

# Define the Convolutional Neural Network (CNN) model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model with Adam optimizer, categorical crossentropy loss, and accuracy as the evaluation metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model's architecture
model.summary()

# Train the model on the training data
# Batch size: 64, Epochs: 25, 20% of training data used for validation
history = model.fit(X_train, Y_teain, batch_size=64, epochs=25, validation_split=0.2)
test_acc , test_loss=model.evaluate(X_test , Y_test)
print(f'Test Accuracy : {test_acc} , Test Loss : {test_loss}')

y_pred=model.predict(X_test)
print(f'Final Prediction : {y_pred[10]}')

index=5 # Choose the index of the test imag
plt.imshow(X_test[index] , cmap='gray')
plt.title(f'Predicted : {np.argmax(y_pred[index])} , Actual : {np.argmax(Y_test[index])}')
plt.show()

import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow import keras 

(X_train , Y_teain) , (X_test , Y_test) = keras.datasets.cifar10.load_data()

X_train , X_test = X_train / 255.0 , X_test / 255.0 

Y_teain=keras.utils.to_categorical(Y_teain , 10)
Y_test=keras.utils.to_categorical(Y_test , 10)

model=keras.Sequential([
    keras.layers.Conv2D(32 , (3,3) , activation='relu' , input_shape=(32,32,3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64 , (3,3) , activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    
    keras.layers.Flatten(),
    
    keras.layers.Dense(128 , activation='relu'),
    keras.layers.Dense(10 , activation='softmax')
])

model.compile(optimizer='adam' , loss='categorical_crossentropy' , metrics=['accuracy'])

model.summary()

history=model.fit(X_train , Y_teain , batch_size=64 , epochs=25 , validation_split=0.2)

test_acc , test_loss=model.evaluate(X_test , Y_test)
print(f'Test Accuracy : {test_acc} , Test Loss : {test_loss}')

y_pred=model.predict(X_test)
print(f'Final Prediction : {y_pred[10]}')

index=5
plt.imshow(X_test[index] , cmap='gray')
plt.title(f'Predicted : {np.argmax(y_pred[index])} , Actual : {np.argmax(Y_test[index])}')
plt.show()

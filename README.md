MNIST digits classification dataset
data set link:https://keras.io/api/datasets/mnist/

keras.datasets.mnist.load_data(path="mnist.npz")
Loads the MNIST dataset.

This is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images

(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

(x_train, y_train), (x_test, y_test)

x_train:NumPy array of grayscale image data with shapes (60000, 28, 28), containing the training data. Pixel values range from 0 to 255.

y_train:NumPy array of digit labels (integers in range 0-9) with shape (60000,) for the training data.

x_test:NumPy array of grayscale image data with shapes (10000, 28, 28), containing the test data. Pixel values range from 0 to 255.

y_test:NumPy array of digit labels (integers in range 0-9) with shape (10000,) for the test data



OUTPUTS:
1)INPUT LAYER ,OUTPUT LAYER(No Hidden Layers)

Training Accuracy: 0.89

Training Loss: 0.42

Test Accuracy: 0.39

Test Loss: 0.90


2)INPUT LAYER ,HIDDEN LAYER,OUTPUT LAYER

Training Accuracy: 0.967

Training Loss: 0.10

Test Accuracy: 0.963

Test Loss: 0.123


1)INPUT LAYER(Flatten Layer),HIDDEN LAYER ,OUTPUT LAYER

Training Accuracy: 0.97

Training Loss: 0.08

Test Accuracy: 0.967

Test Loss: 0.11



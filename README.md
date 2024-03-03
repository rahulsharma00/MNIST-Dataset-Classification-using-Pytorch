# MNIST-Dataset-Classification

This repository contains code for classifying the MNIST dataset using deep learning techniques. The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9) and is a popular benchmark dataset in the field of machine learning. The code implements a convolutional neural network (CNN) using PyTorch to achieve high accuracy in digit recognition. It includes data preprocessing, model training, evaluation, and visualization of results.

# Steps 
Sure, here are the steps you can include in the README for your GitHub repository:

1. **Importing Libraries:** Import necessary libraries including PyTorch, TorchVision, and Matplotlib.

2. **Downloading the Dataset:** Download the MNIST dataset using TorchVision's datasets module.

3. **Creating a PyTorch Neural Network:**
    - Define a neural network class inheriting from `nn.Module`.
    - Create a sequential model consisting of convolutional and linear layers.
    - Specify the architecture including input channels, filter shapes, and activation functions.

4. **Instance of the Neural Network, Loss, and Optimizers:**
    - Instantiate the defined neural network.
    - Define the loss function and optimizer.

5. **Training Flow:**
    - Iterate through the dataset for a specified number of epochs.
    - For each batch:
        - Perform forward pass to get predictions.
        - Calculate the loss.
        - Backpropagate the gradients.
        - Update the model parameters.

6. **Testing Prediction of a Single Image:**
    - Obtain a single image from the test set.
    - Perform inference on the image using the trained model.
    - Calculate the accuracy of the prediction.

7. **Testing Prediction of Multiple Images:**
    - Iterate through a few images from the test set.
    - For each image:
        - Perform inference on the image.
        - Calculate the accuracy of the prediction.

8. **Prediction with Image Visualization:**
    - Visualize predictions on a subset of test images using Matplotlib.
    - Display the predicted labels along with the true labels.


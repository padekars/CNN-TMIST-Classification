# CNN-TMIST-Classification

This project is about developing a Convolutional Neural Network (CNN) model to classify the TMNIST dataset, which consists of grayscale images of handwritten characters. The dataset includes 94 different characters. The goal is to train a neural network on a subset of the dataset (training set) and then use the trained model to predict the character class of images in a separate subset (test set).

Here is a summary of the main steps in the project:

Data Exploration and Preprocessing:

The dataset is loaded from a CSV file.
Exploratory data analysis is performed to understand the structure of the data.
The dataset is split into features (X) and labels (y).
Visualization of sample images from the dataset is done.
Data Splitting:

The dataset is split into training and testing sets with an 80-20 split.
Data Normalization and Encoding:

The pixel values of the images are normalized to the range [0, 1].
One-hot encoding is applied to convert categorical character labels into numerical format.
CNN Model Architecture:

A Convolutional Neural Network (CNN) model is defined using TensorFlow and Keras.
The model consists of convolutional layers, max pooling layers, and fully connected layers.
The output layer uses softmax activation for multi-class classification.
Model Training:

The model is compiled with the Adam optimizer and categorical crossentropy loss function.
It is then trained on the training set for 20 epochs with a batch size of 128.
Model Evaluation:

The trained model is evaluated on the test set to measure its accuracy.
Performance Visualization:

The accuracy and loss of the model during training and testing are visualized using plots.
Conclusion:

A conclusion is drawn summarizing the entire process, and the success of the CNN in recognizing handwritten characters from the TMNIST dataset is highlighted.


# Description

This assignment focuses on building fully connected neural networks (NN) and convolutional neural networks (CNN). It consists of four parts where you practice dealing with various datasets and implement, train, and adjust neural networks.
The first part consists of performing data analysis and building a basic NN. In the second part, we learn how to optimize and improve your NN using various techniques. In the third part, we will implement a basic CNN and apply optimization and data augmentation techniques in the fourth part.
Apart from this do not miss out some extra bonus points, by completing the task described at the end of this description.

# Note for using libraries:
For this assignment, any pre-trained or pre-built neural networks or CNN architectures cannot be used (e.g. torchvision.models, keras.applications). This time you can use scikit-learn for data preprocessing.
For this assignment you can use PyTorch or Keras/Tensorflow deep learning framework (works using sklearn.neural_network.MLPClassifier won't be evaluated):
- Pytorch [60 mins blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- Keras / Tensorflow [Getting started](https://keras.io/getting_started/)

<img width="949" alt="Screenshot 2023-10-08 at 11 02 02 PM" src="https://github.com/ItsTheDeeKay/Image-Classification-Using-CNN/assets/113076076/06665255-c841-42cd-8b49-49a39ee3257c">

As you can see from above structure, this is an 8-layered neural network comprising both convolutional layers and fully connected layers.
- The first layer takes an input tensor of size (batch_size, 3, 64, 64) and applies a convolutional operation with 64 filters, a kernel size of 3, and a stride of 1, producing an output tensor of size (batch_size, 64, 62, 62).
- This is followed by a ReLU activation function, preserving the output tensor's size.
- The second layer applies a max-pooling operation with a kernel size of 3 and a stride of 2, reducing the output tensor's size to (batch_size, 64, 30, 30).
- The third layer applies a convolutional operation with 192 filters, a kernel size of 3, padding of 2, and a stride of 1, producing an output tensor of size (batch_size, 192, 30, 30).
- This is followed by another ReLU activation function.
- The fourth layer applies a max-pooling operation with a kernel size of 3 and a stride of 2, reducing the output tensor's size to (batch_size, 192, 14, 14).
- The fifth layer applies a convolutional operation with 384 filters, a kernel size of 3, padding of 1, and a stride of 1, producing an output tensor of size (batch_size, 384, 14, 14).
- This is followed by another ReLU activation function.
- The sixth layer applies a convolutional operation with 256 filters, a kernel size of 3, padding of 1, and a stride of 1, producing an output tensor of size (batch_size, 256, 14, 14).
- This is followed by another ReLU activation function.
- The seventh layer applies a convolutional operation with 256 filters, a kernel size of 3, padding of 1, and a stride of 1, producing an output tensor of size (batch_size, 256, 14, 14).
- This is followed by another ReLU activation function.
- The eighth layer applies a max-pooling operation with a kernel size of 3 and a stride of 2, reducing the output tensor's size to (batch_size, 256, 6, 6).
- After the convolutional layers, the architecture includes an adaptive average pooling layer that resizes the output tensor to (batch_size, 256, 4, 4). - - Then, three fully connected layers are used for classification, with dropout and ReLU activation layers in between.
- The final fully connected layer produces an output tensor of size (batch_size, num_classes), where “num_classes” is 3 in our case.

# Part I: Building a Basic NN
In this assignment, you will implement a neural network using the PyTorch/Keras library. You will train the network on the dataset provided, which contains of seven features and a target. Your goal is to predict a target, that has a binary representation.
# Step 1: Loading the Dataset
Load the dataset. It is provided on UBlearns > Assignments.
You can use the pandas library to load the dataset into a [DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
# Step 2: Preprocessing the Dataset
First, we need to preprocess the dataset before we use it to train the neural network. Preprocessing typically involves converting categorical variables to numerical variables, scaling numerical variables, and splitting the dataset into training and validation sets.
For this dataset, you can use the following preprocessing steps:

# Step 3: Defining the Neural Network
Now, we need to define the neural network that we will use to make predictions on the dataset. For this part, you can define a simple neural network.

Decide your NN architecture:
- How many input neurons are there?
- What activation function will you choose?
- Suggestion: try ReLU
- What is the number of hidden layers?
- Suggestion: start with a small network, e.g. 2 or 3 layers
- What is the size of each hidden layer?
- Suggestion: try 64 or 128 nodes for each layer
- What activation function is used for the hidden and output layer?

# Step 4: Training the Neural Network
Training has to be defined from scratch, e.g. code with in-built .fit() function won’t be evaluated.

# Part II: Optimizing NN 
Based on your NN model defined in Part I, tune the hyperparameters and apply different tools to increase the accuracy. Try various setups and draw conclusions.

# Part III: Implementing & Improving AlexNet
In this part implement AlexNet, check how to improve the model, and apply that to solve an image dataset containing three classes: dogs, cars and food.
The expected accuracy for this part is more than 94%.

# Part IV: Optimizing CNN + Data Argumentation 
Apply your improved version of CNN model defined in Part III to Google Street View House Numbers (SVHN).


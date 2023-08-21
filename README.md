
# Description

This assignment focuses on building fully connected neural networks (NN) and convolutional neural networks (CNN). It consists of four parts where you practice dealing with various datasets and implement, train, and adjust neural networks.
The first part consists of performing data analysis and building a basic NN. In the second part, we learn how to optimize and improve your NN using various techniques. In the third part, we will implement a basic CNN and apply optimization and data augmentation techniques in the fourth part.
Apart from this do not miss out some extra bonus points, by completing the task described at the end of this description.

# Note for using libraries:
For this assignment, any pre-trained or pre-built neural networks or CNN architectures cannot be used (e.g. torchvision.models, keras.applications). This time you can use scikit-learn for data preprocessing.
For this assignment you can use PyTorch or Keras/Tensorflow deep learning framework (works using sklearn.neural_network.MLPClassifier won't be evaluated):
- Pytorch [60 mins blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- Keras / Tensorflow [Getting started](https://keras.io/getting_started/)

Let’s consider a generic structure for defining a neural network in Pytorch. [Click here for more details](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)



This code defines a neural network with three layers:
- Flatten layer that flattens the input image tensor
- Two fully connected layers with 512 hidden units and ReLU activation functions
- Final fully connected layer with 10 output units (corresponding to the 10 possible
classes in the MNIST dataset)

The forward method specifies how input data is passed through the layers of the network.
It defines a neural network using PyTorch's nn.Module class and the nn.Sequential module, and then uses the network to make a prediction.
nn.Module is a base class in the PyTorch module that provides a framework for building and organizing complex neural network architectures.
When defining a neural network module in PyTorch, you usually create a class that inherits from nn.Module. The nn.Module class provides a set of useful methods and attributes that help in building, training and evaluating the neural network.
Some of the key methods provided by nn.Module are:
__init__(): This is the constructor method that initializes the different layers and
parameters of the neural network.
forward(self, x): This method defines the forward pass of the neural network. It
takes the input tensor (x) and returns the predicted probabilities for each class.
nn.Linear is a PyTorch module that applies a linear transformation to the input data. It is one of the most commonly used modules in deep learning for building neural networks.
The nn.Linear module takes two arguments: in_features and out_features - the number of input/output features. When an input tensor is passed through an nn.Linear module, it is first flattened into a 1D tensor and then multiplied by a weight matrix of size out_features x in_features. A bias term of size out_features is also added to the output.
The output of an nn.Linear module is given by the following equation: 𝑜𝑢𝑡𝑝𝑢𝑡 = 𝑤𝑇𝑥 + 𝑏
model = NeuralNetwork().to(device) print(model)
Here we create an instance of the NeuralNetwork class and moves it to the device specified by the device variable (which should be set to 'cuda' for GPU or 'cpu' for CPU). It is also good practice to print the summary of the architecture of the NN.

This code generates a random input image tensor of size 1x28x28 (representing a single 28x28 grayscale image) and passes it through the neural network using the model instance. The output of the network is a tensor of size 1x10, representing the predicted probabilities for each of the 10 possible classes.
nn.Softmax module is used to convert these probabilities to a valid probability distribution, and the argmax method is used to obtain the class with the highest probability.

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


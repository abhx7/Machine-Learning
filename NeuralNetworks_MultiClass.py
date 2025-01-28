import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
#how to use for numpy made model?

np.set_printoptions(precision=2)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell


def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False


###------------------------------Activation Functions an others--------------------------------------###
def softmax(z):  
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """    
    ez = np.exp(z - np.max(z, axis=0, keepdims=True))  # Subtract max value for numerical stability
    return ez / np.sum(ez, axis=0, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def linear(x):
    return x

# Define loss function: Sparse Categorical Cross-Entropy
def sparse_categorical_crossentropy(y_true, y_pred):
    m = y_true.shape[0]
    p = softmax(y_pred)  # Apply softmax to get probabilities from logits
    log_likelihood = -np.log(p[y_true, range(m)])  # For sparse labels
    loss = np.sum(log_likelihood) / m
    return loss

def compute_loss_with_regularization(y_true, y_pred, parameters, lambda_):
    # Sparse categorical cross-entropy loss
    loss = sparse_categorical_crossentropy(y_true, y_pred)

    # L2 Regularization term
    reg_loss = 0
    for key in parameters.keys():
        if "W" in key:  # Only apply to weight matrices
            reg_loss += np.sum(np.square(parameters[key]))

    reg_loss = (lambda_ / (2 * y_true.shape[0])) * reg_loss
    return loss + reg_loss


# Derivative of ReLU for backpropagation
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def one_hot_encode(y, num_classes):
    """
    Converts a vector of class labels (y) into a one-hot encoded matrix.

    Args:
        y (ndarray): Array of integer class labels with shape (m,).
        num_classes (int): Total number of classes.

    Returns:
        ndarray: One-hot encoded matrix with shape (num_classes, m).
    """
    m = y.shape[0]  # Number of examples
    one_hot = np.zeros((num_classes, m))  # Initialize the one-hot encoded matrix
    one_hot[y, np.arange(m)] = 1  # Set the appropriate element to 1
    return one_hot

###---------------------------------------MODEL------------------------------------------------###

# Initialize parameters (weights and biases)
def initialize_parameters(num_classes):
    parameters = {
        "W1": np.random.randn(25, 400)*0.01,  # Weight matrix for layer 1
        "b1": np.random.rand(25, 1),  # Random integers for bias vector
        "W2": np.random.randn(15, 25)*0.01,   # Weight matrix for layer 2
        "b2": np.random.rand(15, 1),  # Random integers for bias vector
        "W3": np.random.randn(10, 15)*0.01,   # Weight matrix for layer 3 (output layer)
        "b3": np.random.rand(num_classes, 1)  # Random integers for bias vector
    }
    return parameters
# Forward propagation
def forward_propagation(X, parameters):
    Z1 = np.matmul(parameters["W1"], X) + parameters["b1"]
    A1 = relu(Z1)

    Z2 = np.matmul(parameters["W2"], A1) + parameters["b2"]
    A2 = relu(Z2)

    Z3 = np.matmul(parameters["W3"], A2) + parameters["b3"]
    A3 = linear(Z3)  # No activation for the output layer (logits)
    
    return A1, A2, A3

def backward_propagation(X, Y, A1, A2, A3, parameters):
    m = X.shape[0]  # Number of examples

    # Compute softmax to get probabilities
    Y_hat = softmax(A3)

    # Compute dZ3 (gradient w.r.t. logits)
    dZ3 = Y_hat - Y  # Y should be one-hot encoded

    # Gradients for Layer 3
    dW3 = np.matmul(dZ3, A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m

    # Gradients for Layer 2
    dZ2 = np.matmul(parameters["W3"].T, dZ3) * relu_derivative(A2)
    dW2 = np.matmul(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    # Gradients for Layer 1
    dZ1 = np.matmul(parameters["W2"].T, dZ2) * relu_derivative(A1)
    dW1 = np.matmul(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    gradients = {
        "dW1": dW1, "db1": db1,
        "dW2": dW2, "db2": db2,
        "dW3": dW3, "db3": db3
    }

    return gradients


# Update parameters using gradient descent
def update_parameters(parameters, gradients, learning_rate):
    parameters["W1"] -= learning_rate * gradients["dW1"]
    parameters["b1"] -= learning_rate * gradients["db1"]
    parameters["W2"] -= learning_rate * gradients["dW2"]
    parameters["b2"] -= learning_rate * gradients["db2"]
    parameters["W3"] -= learning_rate * gradients["dW3"]
    parameters["b3"] -= learning_rate * gradients["db3"]

    return parameters

def adam_optimizer(parameters, gradients, v, s, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    v_corrected = {}
    s_corrected = {}

    for key in parameters.keys():
        # Update biased first moment estimate
        v[key] = beta1 * v[key] + (1 - beta1) * gradients[f"d{key}"]

        # Update biased second raw moment estimate
        s[key] = beta2 * s[key] + (1 - beta2) * (gradients[f"d{key}"] ** 2)

        # Compute bias-corrected first moment estimate
        v_corrected[key] = v[key] / (1 - beta1 ** t)

        # Compute bias-corrected second raw moment estimate
        s_corrected[key] = s[key] / (1 - beta2 ** t)

        # Update parameters
        parameters[key] -= learning_rate * v_corrected[key] / (np.sqrt(s_corrected[key]) + epsilon)

    return parameters, v, s

# Plot loss during training
def plot_loss(losses):
    plt.plot(losses)
    plt.title("Loss over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()


###------------------------------------------DATASET---------------------------------------------------###

def load_data():
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    return X, y

# Training with Adam optimizer
np.random.seed(1234)
X, y = load_data()

# # Normalize the dataset by standardisation
X_mean = np.mean(X, axis=0)
epsilon = 1e-8  # Small constant to prevent division by zero
X_std = np.std(X, axis=0) + epsilon
X_normalised = (X - X_mean) / X_std

# Normalise data by min-max
X_normalised = X/255.0

print ('The first element of X is: ', X[0])
# print ('The first element of y is: ', y[0,0])
# print ('The last element of y is: ', y[-1,0])
print ('The shape of X is: ' + str(X.shape))
print ('The shape of y is: ' + str(y.shape))

m, n = X.shape
fig, axes = plt.subplots(8,8, figsize=(5,5))
fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

#fig.tight_layout(pad=0.5)
widgvis(fig)
for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20,20)).T
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')
    
    # Display the label above the image
    ax.set_title(y[random_index,0])
    ax.set_axis_off()
    fig.suptitle("Label, image", fontsize=14)
plt.show()


###-------------------------------------TRAINING---------------------------------------------------###

# number of outputs
num_classes = 10

# Initialize parameters
parameters = initialize_parameters(num_classes)

# Initialize Adam variables
v = {key: np.zeros_like(value) for key, value in parameters.items()}  # Momentum term
s = {key: np.zeros_like(value) for key, value in parameters.items()}  # RMSProp term
t = 0  # iteration step

# Training hyperparameters
learning_rate = 0.001
epochs = 40
losses = []

# Define the number of examples for the smaller dataset
subset_size = 4000

# Randomly select indices for the subset
np.random.seed(42)  # For reproducibility
random_indices = np.random.choice(X.shape[0], subset_size, replace=False)

# Create the subset
X_subset = X_normalised[random_indices, :]  # Select subset_size rows from X
y_subset = y[random_indices, :]  # Select corresponding rows from y

print(f"Subset X shape: {X_subset.shape}")
print(f"Subset y shape: {y_subset.shape}")


# Training loop
for epoch in range(epochs):
    t += 1  # Increment iteration step

    # Forward propagation
    A1, A2, A3 = forward_propagation(np.transpose(X_subset), parameters)

    # Compute loss
    #loss = sparse_categorical_crossentropy(y_subset, A3)
    lambda_ = 0.001  # Regularization strength
    loss = compute_loss_with_regularization(y_subset, A3, parameters, lambda_)

    losses.append(loss)

    # Backward propagation
    #gradients = backward_propagation(np.transpose(X), np.transpose(y), A1, A2, A3, parameters)
    gradients = backward_propagation(np.transpose(X_subset), one_hot_encode(y_subset, num_classes), A1, A2, A3, parameters)

    # Update parameters using Adam optimizer
    parameters, v, s = adam_optimizer(parameters, gradients, v, s, t, learning_rate)
    #parameters = update_parameters(parameters, gradients, learning_rate)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

# Plot the loss curve
plot_loss(losses)


###-----------------------------------------TEST-------------------------------------------###


# Define prediction function using numpy
def predict(X, parameters):
    # Perform forward propagation for prediction
    A1, A2, A3 = forward_propagation(X, parameters)
    prediction = A3  # Logits are the raw output values
    prediction_p = softmax(prediction)  # Apply softmax to get probabilities
    return prediction_p

# Display function (assuming you have an appropriate display function for images)
def display_digit(image):
    # Reshape the image and display it
    image = image.reshape(20, 20).T
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

# # Example: Use the first image (index 1015) for prediction and display
# image_of_two = X[1015]  # Get an image to display
# display_digit(image_of_two)

# # Perform prediction using the model (with numpy)
# prediction = predict(image_of_two.reshape(400, 1), parameters)

# print(f"Predicted a Two: \n{prediction}")
# print(f"Largest Prediction index: {np.argmax(prediction)}")

# # Display the probability vector
# print(f"Prediction probabilities: \n{prediction}")
# print(f"Total of predictions: {np.sum(prediction):0.3f}")

# # Get the class prediction
# yhat = np.argmax(prediction)
# print(f"Predicted class index: {yhat}")

# Plot random images and display predictions
m, n = X.shape
fig, axes = plt.subplots(8, 8, figsize=(5, 5))
fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # Adjust layout

for i, ax in enumerate(axes.flat):
    # Select random index
    random_index = np.random.randint(m)

    # Reshape the image to (20, 20) for displaying
    X_random_reshaped = X[random_index].reshape(20, 20).T

    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')

    # Perform prediction on the selected image
    prediction = predict(X[random_index].reshape(400, 1), parameters)
    yhat = np.argmax(prediction)

    # Display label above the image (true label and predicted class)
    ax.set_title(f"{y[random_index, 0]},{yhat}", fontsize=10)
    ax.set_axis_off()

fig.suptitle("Label, yhat", fontsize=14)
plt.show()

# Function to calculate and display errors (misclassifications)
def display_errors(parameters, X, y):
    errors = 0
    for i in range(X.shape[0]):
        # Make a prediction for each image
        prediction = predict(X[i].reshape(400, 1), parameters)
        yhat = np.argmax(prediction)
        if yhat != y[i]:
            errors += 1
    return errors

# Calculate and display the number of errors (misclassifications)
errors = display_errors(parameters, X, y)
print(f"{errors} errors out of {len(X)} images")

# # Evaluate errors on the training subset
# errors = display_errors(parameters, X_subset, y_subset.flatten())
# print(f"{errors} errors out of {len(X_subset)} images")

# # Calculate and display training accuracy
# accuracy = calculate_accuracy(parameters, X_subset, y_subset.flatten())
# print(f"Training Accuracy: {accuracy * 100:.2f}%")

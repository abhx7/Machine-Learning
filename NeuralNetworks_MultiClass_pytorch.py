import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define Neural Network Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(400, 25)
        self.fc2 = nn.Linear(25, 15)
        self.fc3 = nn.Linear(15, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Logits (softmax will be applied later)
        return x

# Load Data
X = np.load("data/X.npy")
y = np.load("data/y.npy").flatten()  # Ensure labels are 1D

# Normalize Data and Convert to PyTorch Tensors
X = torch.tensor(X / 255.0, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Define Model, Loss, and Optimizer
model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train Model
def train_model(model, criterion, optimizer, X_train, y_train, epochs=40):
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    return losses

losses = train_model(model, criterion, optimizer, X, y)

# Plot Loss
plt.plot(losses)
plt.title("Loss over iterations")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Prediction Function
def predict(model, X):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        logits = model(X)  # Forward pass
        return torch.argmax(logits, dim=1)  # Get class predictions

# Display function for images
def display_digit(image):
    image = image.reshape(20, 20).T
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

# Display random images with predictions
def display_predictions(model, X, y, num_images=64):
    fig, axes = plt.subplots(8, 8, figsize=(5, 5))
    fig.tight_layout(pad=0.3)

    for i, ax in enumerate(axes.flat):
        random_index = np.random.randint(len(X))
        image = X[random_index].reshape(1, -1)  # Reshape for model input
        yhat = predict(model, torch.tensor(image, dtype=torch.float32)).item()

        ax.imshow(X[random_index].reshape(20, 20).T, cmap='gray')
        ax.set_title(f"{y[random_index]},{yhat}", fontsize=10)
        ax.set_axis_off()

    plt.show()

display_predictions(model, X.numpy(), y.numpy())

# Evaluate Errors
def evaluate_model(model, X, y):
    predictions = predict(model, X)
    errors = (predictions != y).sum().item()
    accuracy = (predictions == y).float().mean().item()
    return errors, accuracy

errors, accuracy = evaluate_model(model, X, y)
print(f"{errors} errors out of {len(X)} images")
print(f"Training Accuracy: {accuracy * 100:.2f}%")

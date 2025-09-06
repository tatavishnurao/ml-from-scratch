import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X_train = np.array([[0,0],[0,1],[1,0],[1,1]])
y_train = np.array([[0],[1],[1],[0]])

np.random.seed(42)
hidden_neurons = 4
W1 = np.random.randn(2, hidden_neurons)
b1 = np.zeros((1, hidden_neurons))
W2 = np.random.randn(hidden_neurons, 1)
b2 = np.zeros((1, 1))

lr = 0.1
epochs = 5000
losses = []

n = X_train.shape[0]

for epoch in range(epochs):
    epoch_loss = 0
    for i in range(n):
        x = X_train[i].reshape(1, -1)
        y = y_train[i].reshape(1, -1)

        z1 = np.dot(x, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        loss = np.mean((y - a2) ** 2)
        epoch_loss += loss

        d_loss_output = 2 * (a2 - y) * sigmoid_derivative(a2)

        dW2 = np.dot(a1.T, d_loss_output)
        db2 = d_loss_output

        d_hidden = np.dot(d_loss_output, W2.T) * sigmoid_derivative(a1)
        dW1 = np.dot(x.T, d_hidden)
        db1 = d_hidden

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    losses.append(epoch_loss / n)

z1 = np.dot(X_train, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
a2 = sigmoid(z2)
preds = (a2 > 0.5).astype(int)

for i in range(len(losses)):
    if i % 1000 == 0:
        print(f"Epoch {i}, Loss: {losses[i]}")
plt.plot(losses)
plt.title("SGD Training Loss (XOR, 4 hidden neurons)")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.savefig("sgd_xor_loss.png")

preds, y_train.ravel()

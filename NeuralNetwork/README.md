# XOR Neural Network (From Scratch) - Complete Project README

## 🔍 Project Objective

Build a neural network **from first principles** to solve the **XOR classification task**, using:

* Manual implementation of forward and backward propagation
* Sigmoid activations
* Stochastic Gradient Descent (SGD)
* No frameworks (no TensorFlow, PyTorch, etc.)

This README captures all the design choices, mathematical logic, statistics, and reasoning involved.

---

## 📚 Why XOR?

### Historical Context

* XOR was the dataset that **defeated the original Perceptron** (Minsky & Papert, 1969).
* It forced the machine learning community to go **beyond linear models**.

### Mathematical Reason

* XOR is **non-linearly separable**.
* A network must learn to **curve the decision boundary**.

### Minimal yet Complete

* Just 4 binary inputs: (0,0), (0,1), (1,0), (1,1)
* Forces the use of:

  * A hidden layer
  * Nonlinear activation (sigmoid)
  * Chain rule via backpropagation
  * Loss function and gradient updates

---

## 📊 Dataset

| x1 | x2 | y |
| -- | -- | - |
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

Shape:

* `X`: (4,2)
* `y`: (4,1)

---

## 🧮 Math: Network Structure

### Architecture

* Input layer: 2 neurons (x1, x2)
* Hidden layer: 4 neurons
* Output layer: 1 neuron
* Activation: Sigmoid

### Forward Propagation

```
z1 = X @ W1 + b1
A1 = sigmoid(z1)
z2 = A1 @ W2 + b2
A2 = sigmoid(z2) = ˆy
```

### Loss Function: Mean Squared Error

```
L = (1/n) * sum((y - ˆy)^2)
dL/dA2 = 2*(ˆy - y)
```

### Backward Propagation

```
dA2 = 2*(A2 - y) * sigmoid_derivative(A2)
dW2 = A1.T @ dA2
db2 = sum(dA2)

dA1 = (dA2 @ W2.T) * sigmoid_derivative(A1)
dW1 = X.T @ dA1
db1 = sum(dA1)
```

### Parameter Update (SGD)

```
W -= lr * dW
b -= lr * db
```

---

## ⚙️ Learning Mechanism

* **SGD:** Updates are made **after each sample**, not batch.
* **Learning rate (lr):** 0.1
* **Epochs:** 5000
* **Convergence:** Measured via loss curve

---

## 📉 Observations from Training

* Loss steadily decreases
* Final predictions: `[0, 1, 1, 0]` (Perfect match)
* Decision boundary: **non-linear**, curved as expected

---

## 📈 Visualization

* **Loss vs Epochs:** Shows convergence
* **Decision boundary plot:** Confirms class separation in 2D input space

---

## 🔑 Insights

* XOR forces full **gradient flow**, non-linearity, and network depth
* Proves your **math and code are working**
* Ideal for beginners and for debugging custom networks

---

## 🧠 Final Takeaway

If your neural network can solve XOR **without frameworks**, it means:

* You understand forward & backpropagation
* Your gradients are correct
* Your model is learning general patterns, not memorizing

This is the true "Hello World" of deep learning.

---

**Author:** Vishnu

**Environment:** uv(astral)

**Libraries:** Numpy and Matplotlib 

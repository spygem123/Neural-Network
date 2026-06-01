# Neural Network

A fully functional feedforward neural network implemented using **Python** and **NumPy**

Trained to solve the **XOR problem**: a classic benchmark that requires a hidden layer to solve, since the data is not linearly separable.

---

## Architecture

```
Input (2)  →  Hidden Layer (8, ReLU)  →  Output Layer (1, Sigmoid)
```

| Layer  | Inputs | Neurons | Activation |
|--------|--------|---------|------------|
| Hidden | 2      | 8       | ReLU       |
| Output | 8      | 1       | Sigmoid    |

---

## How It Works

### Forward Pass
Each layer computes:
```
output = activation(input @ weights + bias)
```

### Loss
Mean Squared Error:
```
loss = mean((y_true - y_pred)²)
```

### Backward Pass
Gradients flow back through the network using the chain rule, updating every weight and bias to minimise loss.

---

## Project Structure

```
├── main.py                 # Entry point — builds and trains the network
├── network.py              # Network class — chains layers, runs training loop
├── network_layer.py        # Layer class — weights, biases, forward & backward
└── activation_function.py  # ReLU and Sigmoid with derivatives
```

---

## Results

After 10,000 training epochs with a learning rate of `0.1`:

| Input      | Expected | Predicted |
|------------|----------|-----------|
| [0, 0]     | 0        | ~0.02     |
| [0, 1]     | 1        | ~0.97     |
| [1, 0]     | 1        | ~0.97     |
| [1, 1]     | 0        | ~0.03     |

The network successfully learns XOR.


---

## Key Concepts Implemented

- **Weight initialisation** — small random values scaled by `0.1` to avoid saturation
- **Matrix multiplication** — the core of the forward pass (`@` operator)
- **Chain rule** — how gradients are passed back through each layer
- **ReLU** — fast, sparse activation; avoids vanishing gradients in hidden layers
- **Sigmoid** — squashes output to `[0, 1]` for binary classification
- **MSE loss** — measures prediction error; its derivative seeds the backward pass


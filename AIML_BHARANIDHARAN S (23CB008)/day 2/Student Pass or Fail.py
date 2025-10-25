#Predict Student Pass/Fail using a Neural Network

import numpy as np

X = np.array([[1, 30],[2, 50],[3, 70],[5, 90]])
y = np.array([[0], [0], [1], [1]])

X = X / np.max(X)

np.random.seed(42)

input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
epochs = 10000

W1 = np.random.rand(input_size, hidden_size)
b1 = np.random.rand(1, hidden_size)
W2 = np.random.rand(hidden_size, output_size)
b2 = np.random.rand(1, output_size)

print("Initial Weights and Biases:")
print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

for epoch in range(epochs):
    # Forward Propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    error = y - final_output

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    W2 += hidden_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if (epoch+1) % 2000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

print("\nTraining Complete!")
print("Final W1:", W1)
print("Final b1:", b1)
print("Final W2:", W2)
print("Final b2:", b2)

print("\nTesting the Neural Network:")
test_data = np.array([[4, 80], [1, 50]]) / np.max(X)
predicted_output = sigmoid(sigmoid(test_data.dot(W1) + b1).dot(W2) + b2)
predicted_labels = (predicted_output > 0.5).astype(int)

for i in range(len(test_data)):
    print(f"Input: {test_data[i]*np.max(X)} -> Predicted Output: {predicted_labels[i][0]} (Pass=1, Fail=0)")

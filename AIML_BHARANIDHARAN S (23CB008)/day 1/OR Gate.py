import numpy as np

X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
y = np.array([0, 1, 1, 1])
np.random.seed(42)
weights = np.random.rand(2)
bias = np.random.rand(1)
learning_rate = 0.1
print("Initial weights:", weights)
print("Initial bias:", bias)

def step_function(x):
    return 1 if x >= 0 else 0
epochs = 10
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        predicted = step_function(linear_output)
        error = y[i] - predicted
        weights = weights + learning_rate * error * X[i]
        bias = bias + learning_rate * error
        print(f"Input: {X[i]}, Target: {y[i]}, Predicted: {predicted}")
        print(f"Updated weights: {weights}, Updated bias: {bias}")

print("\nTraining complete!")
print("Final weights:", weights)
print("Final bias:", bias)

print("\nTesting the Trained Perceptron")
for i in range(len(X)):
    linear_output = np.dot(X[i], weights) + bias
    predicted = step_function(linear_output)
    print(f"Input: {X[i]} -> Predicted Output: {predicted}")
import numpy as np
import re
import pandas as pd

class AdvancedMLP:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001, l2_lambda=0.01, dropout_rate=0.2, clip_value=1.0):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.clip_value = clip_value

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2. / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, x * alpha)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    def dropout(self, x, rate):
        mask = np.random.binomial(1, 1 - rate, size=x.shape)
        return x * mask / (1 - rate)

    def forward(self, X):
        self.z = []
        self.a = [X]

        for i in range(len(self.weights)):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.z.append(z)
            if i == len(self.weights) - 1:
                a = z  # No activation in the output layer
            else:
                a = self.leaky_relu(z)
                if i < len(self.weights) - 1:
                    a = self.dropout(a, self.dropout_rate)
            self.a.append(a)

        return self.a[-1]

    def backward(self, X, y):
        m = y.shape[0]
        dz = self.a[-1] - y
        dw = np.dot(self.a[-2].T, dz) / m + (self.l2_lambda / m) * self.weights[-1]
        db = np.sum(dz, axis=0, keepdims=True) / m

        # Gradient clipping
        dw = np.clip(dw, -self.clip_value, self.clip_value)
        db = np.clip(db, -self.clip_value, self.clip_value)

        self.weights[-1] -= self.learning_rate * dw
        self.biases[-1] -= self.learning_rate * db

        for i in range(len(self.weights) - 2, -1, -1):
            dz = np.dot(dz, self.weights[i + 1].T) * self.leaky_relu_derivative(self.z[i])
            dw = np.dot(self.a[i].T, dz) / m + (self.l2_lambda / m) * self.weights[i]
            db = np.sum(dz, axis=0, keepdims=True) / m

            # Gradient clipping
            dw = np.clip(dw, -self.clip_value, self.clip_value)
            db = np.clip(db, -self.clip_value, self.clip_value)

            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db

    def train(self, X, y, epochs, batch_size, validation_data=None, patience=10, lr_scheduler=False):
        best_val_loss = float('inf')
        no_improvement_count = 0

        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            for start in range(0, X.shape[0], batch_size):
                end = start + batch_size
                X_batch, y_batch = X[start:end], y[start:end]

                self.forward(X_batch)
                self.backward(X_batch, y_batch)

            if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
                output = self.forward(X)
                loss = np.mean((y - output) ** 2)
                print(f'Epoch {epoch+1}, Loss: {loss}')

            if validation_data:
                val_X, val_y = validation_data
                val_predictions = self.forward(val_X)
                val_loss = np.mean((val_y - val_predictions) ** 2)
                print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

            if lr_scheduler:
                self.learning_rate *= 0.95  # Reduce learning rate by 5% each epoch

def read_file(filepath):
    ds = []
    with open(filepath) as fp:
        for line in fp:
            comp = re.compile("\d+")
            dataList = comp.findall(line)
            row = list(map(int, dataList[1:]))
            ds.append(row)
    return ds

def create_df(dataset):
    dataframe = pd.DataFrame(dataset, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'label'])
    return dataframe

if __name__ == "__main__":
    dataset = read_file("dataset.txt")
    df = create_df(dataset)

    # Splitting training and holdout set
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    holdout = df[~msk]
    items = train.drop(columns="label")
    targets = train["label"].values.reshape(-1, 1)
    inp = np.array(items)
    tar = np.array(targets)

    # Normalizing the inputs
    inp = (inp - np.mean(inp, axis=0)) / np.std(inp, axis=0)

    # Create an advanced MLP
    mlp = AdvancedMLP(input_size=10, hidden_sizes=[32, 16], output_size=1, learning_rate=0.001, l2_lambda=0.001, dropout_rate=0.2, clip_value=1.0)

    # Train the network
    mlp.train(inp, tar, epochs=1000, batch_size=32, validation_data=(vinp, vtar), patience=10, lr_scheduler=True)

    # Validation set
    val = read_file("validationSet.txt")
    vf = create_df(val)
    vitems = vf.drop(columns="label")
    vtargets = vf["label"].values.reshape(-1, 1)
    vinp = np.array(vitems)
    vtar = np.array(vtargets)

    # Normalizing the validation inputs
    vinp = (vinp - np.mean(vinp, axis=0)) / np.std(vinp, axis=0)

    # Evaluate on validation set
    predictions = mlp.forward(vinp)
    val_error = np.mean((vtar - predictions) ** 2)
    print(f"Validation MSE: {val_error}")


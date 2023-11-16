import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_gaussian_quantiles

def initialize_gaussian_params(num_features):
    mean = np.zeros(num_features)
    covariance_matrix = np.eye(num_features)
    return mean, covariance_matrix

def quantile_loss(y_true, y_pred, quantile):
    e = y_true - y_pred
    return np.mean(np.maximum(quantile * e, (quantile - 1) * e))

def gradient_quantile_loss(y_true, y_pred, quantile):
    e = y_true - y_pred
    return -np.mean(np.where(e >= 0, (1 - quantile), quantile) * e)

def train_quantile_regression(X, y, quantile, learning_rate=0.01, n_iterations=1000):
    num_features = X.shape[1]
    mean, covariance_matrix = initialize_gaussian_params(num_features)
    weights = np.zeros(num_features)
    bias = 0

    for _ in range(n_iterations):
        y_pred = X.dot(weights) + bias
        gradient = gradient_quantile_loss(y, y_pred, quantile)
        weights -= learning_rate * gradient * X.mean(axis=0)
        bias -= learning_rate * gradient

    return weights, bias

# Generar datos gaussianos
N = 1000 # muestras
gaussian_quantiles = make_gaussian_quantiles(mean=None,
                        cov=0.1,
                        n_samples=N,
                        n_features=2,
                        n_classes=2,
                        shuffle=True,
                        random_state=None)

X, Y = gaussian_quantiles
Y = Y.ravel()

# Dividir datos en conjuntos de entrenamiento y prueba
split_ratio = 0.8
split_index = int(split_ratio * N)
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

# Cuantil que deseas estimar
quantile = 0.5

# Entrenar modelo de regresión cuantílica
weights, bias = train_quantile_regression(X_train, Y_train, quantile)

# Evaluar el rendimiento del modelo en el conjunto de prueba
predictions = X_test.dot(weights) + bias
mse = np.mean((Y_test - predictions)**2)
r2 = 1 - mse / np.var(Y_test)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# Visualizar datos generados
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.title('Datos Generados')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
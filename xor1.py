import random
import math
import csv

# Función de activación sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Inicialización de pesos con Xavier/Glorot y sesgo de manera aleatoria
def initialize_weights(input_size):
    weights = [random.uniform(-math.sqrt(6) / math.sqrt(input_size + 1), math.sqrt(6) / math.sqrt(input_size + 1)) for _ in range(input_size)]
    bias = random.uniform(-0.5, 0.5)
    return weights, bias

# Normalización de datos
def normalize_data(data):
    max_values = [max(column) for column in zip(*data)]
    min_values = [min(column) for column in zip(*data)]

    normalized_data = [[(x - min_val) / (max_val - min_val) for x, max_val, min_val in zip(row, max_values, min_values)] for row in data]

    return normalized_data

# Entrenamiento del modelo usando el conjunto de datos XOR_trn.csv con descenso de gradiente estocástico (SGD)
def train(X, y, learning_rate, epochs, regularization_factor=0.0001, print_interval=100):
    input_size = len(X[0])
    weights, bias = initialize_weights(input_size)

    for epoch in range(epochs):
        for i in range(len(X)):
            # Propagación hacia adelante
            weighted_sum = sum([X[i][j] * weights[j] for j in range(input_size)]) + bias
            prediction = sigmoid(weighted_sum)  # Cambia la función de activación a sigmoid

            # Cálculo del error
            error = y[i] - prediction

            # Regularización L2
            for j in range(input_size):
                weights[j] = weights[j] + learning_rate * (error * X[i][j] - regularization_factor * weights[j])
            bias = bias + learning_rate * error
           

    return weights, bias

# Evaluación del modelo en el conjunto de datos XOR_tst.csv
def test(X, weights, bias):
    predictions = []
    for i in range(len(X)):
        weighted_sum = sum([X[i][j] * weights[j] for j in range(len(X[0]))]) + bias
        prediction = round(sigmoid(weighted_sum))
        predictions.append(prediction)
    return predictions

# Generación de partículas aleatorias para generalización
def generate_particles(num_particles, input_size):
    particles = []
    for _ in range(num_particles):
        particle = [random.choice([0, 1]) for _ in range(input_size)]
        particles.append(particle)
    return particles

# Guardar conjunto de datos en un archivo CSV
def save_to_csv(file_path, data):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

# Cargar conjunto de datos desde un solo archivo CSV
def load_data(file_path):
    data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append([float(x) for x in row])
    return data

# Conjunto de datos XOR de entrenamiento
X_train = load_data('XOR_trn.csv')
y_train = [row[-1] for row in X_train]

# Normalización de datos de entrenamiento
X_train_normalized = normalize_data(X_train)

# Generación de conjunto de prueba (XOR_tst.csv)
num_particles = 10
X_test = generate_particles(num_particles, len(X_train[0]) - 1)

# Hiperparámetros ajustados
learning_rate = 0.00001  # Disminuye la tasa de aprendizaje
epochs = 20000  # Aumenta el número de épocas
regularization_factor = 0.00001  # Disminuye el factor de regularización

# Entrenamiento con descenso de gradiente estocástico
weights, bias = train(X_train_normalized, y_train, learning_rate, epochs, regularization_factor, print_interval=100)

# Generalización
predictions = test(X_test, weights, bias)

# Guardar las partículas generadas en XOR_tst.csv
particles_and_predictions = [particle + [prediction] for particle, prediction in zip(X_test, predictions)]
save_to_csv('XOR_tst.csv', particles_and_predictions)

# Resultados
print("Pesos finales:", weights)
print("Sesgo final:", bias)
print("Predicciones en conjunto de prueba:", predictions)
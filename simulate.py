import time
import random


# Symulacja zużycia energii (w Watach)
def estimate_energy_usage(elapsed_time, batch_size):
    return elapsed_time * batch_size * 0.5  # Założenie: 0.5W na jednostkę czasu i rozmiar batcha

# Klasa symulująca warstwę Dense
class DenseLayer:
    def __init__(self, units):
        self.units = units
        # Dla uproszczenia: inicjalizujemy wagi jako listę jedynek
        self.weights = [1.0] * units

    def get_weights(self):
        return self.weights

    def set_weights(self, new_weights):
        self.weights = new_weights


# Klasa symulująca model sieci neuronowej
class SimpleModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        # Symulujemy trzy warstwy analogiczne do modelu TensorFlow:
        # - Pierwsza warstwa: 64 jednostki
        # - Druga warstwa: 32 jednostki
        # - Trzecia warstwa: 10 jednostek (np. dla 10 klas)
        self.layers = [DenseLayer(64), DenseLayer(32), DenseLayer(10)]
        self.current_accuracy = 0.5  # Początkowa dokładność

    def compile(self, optimizer, loss, metrics):
        # W tej symulacji metoda compile jedynie zapisuje przekazane parametry
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def fit(self, x, y, batch_size, epochs, verbose):
        start_time = time.time()
        # Symulujemy czas treningu – opóźnienie by odwzorować działanie treningu
        time.sleep(0.01)
        # Symulacja przyrostu dokładności
        self.current_accuracy += 0.05
        if self.current_accuracy > 1.0:
            self.current_accuracy = 1.0
        elapsed_time = time.time() - start_time
        # Zwracamy symulowany "history" treningu
        history = {'accuracy': [self.current_accuracy]}
        return History(history)


# Klasa do przechowywania historii treningu
class History:
    def __init__(self, history):
        self.history = history


# Implementacja przycinania modelu (pruning)
def apply_pruning(model):
    print("Applying pruning to the model.")
    # Przykładowe przycięcie: redukujemy wagi w każdej warstwie Dense o 10%
    for layer in model.layers:
        if isinstance(layer, DenseLayer):
            new_weights = [w * 0.9 for w in layer.get_weights()]
            layer.set_weights(new_weights)
    return model


# Funkcja do optymalizacji procesu uczenia
def green_ml_optimizer(model, data, epochs, energy_limit):
    energy_consumption = []
    batch_size = 32  # Początkowy rozmiar batcha
    results = {'epochs': [], 'energy': [], 'accuracy': []}

    for epoch in range(epochs):
        start_time = time.time()

        # Trening modelu (symulacja)
        history = model.fit(data['train'], data['train_labels'], batch_size=batch_size, epochs=1, verbose=0)

        elapsed_time = time.time() - start_time
        energy = estimate_energy_usage(elapsed_time, batch_size)
        energy_consumption.append(energy)

        accuracy = history.history['accuracy'][0]
        results['epochs'].append(epoch + 1)
        results['energy'].append(sum(energy_consumption))
        results['accuracy'].append(accuracy)

        # Adaptacja hiperparametrów, jeśli zużycie energii jest zbyt wysokie
        if sum(energy_consumption) > energy_limit:
            print(f"Epoch {epoch + 1}: Energy limit exceeded. Adapting model...")
            batch_size = min(batch_size + 16, len(data['train']))  # Zwiększamy batch_size
            model = apply_pruning(model)  # Stosujemy przycinanie modelu

        print(f"Epoch {epoch + 1}: Energy = {energy:.2f}W, Accuracy = {accuracy:.4f}")

        if sum(energy_consumption) > energy_limit:
            print("Energy limit exceeded. Stopping training.")
            break

    return model, results


# Funkcja tworząca prosty model symulacyjny
def create_model(input_shape):
    model = SimpleModel(input_shape)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Przygotowanie fikcyjnych danych testowych
def generate_fake_data():
    num_samples = 1000
    num_features = 20
    # Generujemy listę próbek, gdzie każda próbka to lista 20 losowych wartości
    x_train = [[random.random() for _ in range(num_features)] for _ in range(num_samples)]
    # Generujemy etykiety – losowe liczby całkowite z przedziału 0-9 (10 klas)
    y_train = [random.randint(0, 9) for _ in range(num_samples)]
    return {'train': x_train, 'train_labels': y_train}


# Testowanie algorytmu
if __name__ == "__main__":
    data = generate_fake_data()
    model = create_model(input_shape=(20,))  # Zakładając 20 cech w danych wejściowych
    energy_limit = 10  # Limit energii w Watach

    model, results = green_ml_optimizer(model, data, epochs=20, energy_limit=energy_limit)

    print("\nTraining results:")
    print(f"Total energy consumed: {results['energy'][-1]:.2f}W")
    print(f"Final accuracy: {results['accuracy'][-1]:.4f}")

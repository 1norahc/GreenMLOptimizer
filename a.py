import tensorflow as tf
import numpy as np
import time


# Funkcja symulująca szacowanie zużycia energii (w Watach)
def estimate_energy_usage(elapsed_time, batch_size):
    return elapsed_time * batch_size * 0.5  # Założenie: 0.5W na jednostkę czasu i rozmiar batcha


# Funkcja implementująca przycinanie modelu (pruning)
def apply_pruning(model):
    print("Applying pruning to the model.")
    # Przykładowe przycięcie: redukujemy wagi w każdej warstwie Dense o 10%
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            # Pobieramy aktualne wagi i przesuwamy je o 10%
            new_weights = [w * 0.9 for w in layer.get_weights()]
            layer.set_weights(new_weights)
    return model


# Główna funkcja optymalizująca proces uczenia
def green_ml_optimizer(model, data, epochs, energy_limit, initial_batch_size=32):
    energy_consumption = []
    batch_size = initial_batch_size
    results = {'epochs': [], 'energy': [], 'accuracy': []}

    for epoch in range(epochs):
        start_time = time.time()

        # Trening modelu – uruchamiamy jedną epokę
        history = model.fit(data['train'], data['train_labels'], batch_size=batch_size, epochs=1, verbose=0)

        elapsed_time = time.time() - start_time
        energy = estimate_energy_usage(elapsed_time, batch_size)
        energy_consumption.append(energy)

        accuracy = history.history['accuracy'][0]
        results['epochs'].append(epoch + 1)
        results['energy'].append(sum(energy_consumption))
        results['accuracy'].append(accuracy)

        # Adaptacja hiperparametrów, jeśli zużycie energii przekracza limit
        if sum(energy_consumption) > energy_limit:
            print(f"Epoch {epoch + 1}: Energy limit exceeded. Adapting model...")
            batch_size = min(batch_size + 16, len(data['train']))
            model = apply_pruning(model)

        print(f"Epoch {epoch + 1}: Energy = {energy:.2f}W, Accuracy = {accuracy:.4f}")

        if sum(energy_consumption) > energy_limit:
            print("Energy limit exceeded. Stopping training.")
            break

    return model, results


# --- Sekcja tworzenia modeli ---

def create_dense_model(input_shape, num_classes):
    """Tworzy prosty model w pełni połączony (dense)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def create_cnn_model(input_shape, num_classes):
    """Tworzy model CNN – przykładowo dla danych obrazowych."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def model_factory(model_type, input_shape, num_classes):
    """
    Funkcja fabrykująca modele.
    Aby dodać nowy model, wystarczy:
      1. Napisać funkcję tworzącą nowy model,
      2. Dodać odpowiedni warunek w tej funkcji.
    """
    if model_type == 'dense':
        return create_dense_model(input_shape, num_classes)
    elif model_type == 'cnn':
        return create_cnn_model(input_shape, num_classes)
    else:
        raise ValueError("Unsupported model type: " + model_type)


# --- Sekcja generowania danych testowych ---

def generate_fake_data(model_type='dense'):
    """Generuje fikcyjne dane treningowe w zależności od typu modelu."""
    if model_type == 'dense':
        x_train = np.random.rand(1000, 20)  # 1000 próbek, 20 cech
        y_train = np.random.randint(0, 10, 1000)  # 10 klas
        return {'train': x_train, 'train_labels': y_train}
    elif model_type == 'cnn':
        # Przykład dla danych obrazowych – np. obrazy 28x28 z 1 kanałem (np. MNIST)
        x_train = np.random.rand(1000, 28, 28, 1)
        y_train = np.random.randint(0, 10, 1000)
        return {'train': x_train, 'train_labels': y_train}
    else:
        raise ValueError("Unsupported model type for data generation.")


# --- Główna część programu ---

if __name__ == "__main__":
    # Wybieramy typ modelu ('dense' lub 'cnn')
    model_type = 'dense'  # Możesz zmienić na 'cnn'
    data = generate_fake_data(model_type=model_type)

    # Ustalanie kształtu wejściowego i liczby klas na podstawie typu modelu
    if model_type == 'dense':
        input_shape = (20,)  # 20 cech
    elif model_type == 'cnn':
        input_shape = (28, 28, 1)  # obrazy 28x28 z 1 kanałem
    num_classes = 10

    # Tworzymy model wykorzystując funkcję fabrykującą
    model = model_factory(model_type, input_shape, num_classes)

    # Ustalanie limitu energii (w Watach) – przykładowa wartość
    energy_limit = 10

    # Uruchamiamy optymalizację treningu
    model, results = green_ml_optimizer(model, data, epochs=20, energy_limit=energy_limit)

    print("\nTraining results:")
    print(f"Total energy consumed: {results['energy'][-1]:.2f}W")
    print(f"Final accuracy: {results['accuracy'][-1]:.4f}")

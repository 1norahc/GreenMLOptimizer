import tensorflow as tf
import numpy as np
import time

# Symulacja zużycia energii (w Watach)
def estimate_energy_usage(elapsed_time, batch_size):
    return elapsed_time * batch_size * 0.5  # Założenie: 0.5W na jednostkę czasu i rozmiar batcha

# Implementacja przycinania modelu
def apply_pruning(model):
    print("Applying pruning to the model.")
    # Przykładowe przycięcie: usuwamy 10% wag w warstwie Dense
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.set_weights([w * 0.9 for w in layer.get_weights()])
    return model

# Funkcja do optymalizacji procesu uczenia
def green_ml_optimizer(model, data, epochs, energy_limit):
    energy_consumption = []
    batch_size = 32  # Początkowy rozmiar batcha
    results = {'epochs': [], 'energy': [], 'accuracy': []}

    for epoch in range(epochs):
        start_time = time.time()
        
        # Trening modelu
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

# Przykład prostego modelu
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # Zakładając, że mamy 10 klas
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Przygotowanie danych testowych (fikcyjne dane)
def generate_fake_data():
    x_train = np.random.rand(1000, 20)  # 1000 próbek, 20 cech
    y_train = np.random.randint(0, 10, 1000)  # 10 klas
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

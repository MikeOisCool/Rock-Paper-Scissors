import random
import numpy as np 
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models, mixed_precision

# GPU-Konfiguration
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print("Mixed Precision aktiviert!", "Verfügbare GPUs:", physical_devices)

# Spielkonstanten
actions = ["R", "P", "S"]
action_to_int = {"R": 0, "P": 1, "S": 2}
int_to_action = {0: "R", 1: "P", 2: "S"}

# Initialisierung von Variablen
state_size = 10
learning_rate = 0.001
epsilon, epsilon_min, epsilon_decay = 1.0, 0.01, 0.995
total_games_played = 0
opponent_history, player_history, X_train, y_train = [], [], [], []
results = {"p1": 0, "p2": 0, "tie": 0}

# Win-Rate Metrik
def win_rate(y_true, y_pred):
    correct_moves = K.cast(K.equal(K.argmax(y_pred, axis=-1), K.cast(y_true, dtype='int64')), dtype='float32')
    return K.mean(correct_moves)

# Zustandsermittlung
def get_enhanced_state(opponent_history, player_history, state_size):
    # Fülle die Historie mit Standardwerten (z. B. 0), falls sie zu kurz ist
    opponent_state = [action_to_int.get(a, 0) for a in opponent_history[-state_size:]]
    player_state = [action_to_int.get(a, 0) for a in player_history[-state_size:]]
    
    # Fülle die Historie auf, falls sie kürzer als state_size ist
    if len(opponent_state) < state_size:
        opponent_state = [0] * (state_size - len(opponent_state)) + opponent_state
    if len(player_state) < state_size:
        player_state = [0] * (state_size - len(player_state)) + player_state
    
    opponent_last_move = opponent_state[-1] if opponent_state else 0
    player_last_move = player_state[-1] if player_state else 0
    move_difference = (player_last_move - opponent_last_move) % 3
    
    # Kombiniere die Zustände zu einem einzigen Vektor
    state = opponent_state + player_state + [opponent_last_move, player_last_move, move_difference]
    return np.array(state, dtype=np.float32).reshape(1, -1)

# Mustererkennung
def detect_pattern(history, max_pattern_length=10):
    for length in range(1, max_pattern_length + 1):
        pattern = history[-length:]
        if len(history) >= length * 2 and history[-length * 2: -length] == pattern:
            return pattern
    return None

# Modellarchitektur
model = models.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(state_size * 2 + 3, 1)),
    layers.LSTM(64),
    layers.Dense(64, activation="relu"),
    layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9, staircase=True)),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", win_rate]
)

# Trainingsfunktion
def train_model():
    if len(X_train) > 200:
        try:
            X_train_arr = np.array(X_train, dtype=np.float32).reshape(-1, state_size * 2 + 3, 1)
            y_train_arr = np.array(y_train, dtype=np.int32)
            if X_train_arr.shape[0] == y_train_arr.shape[0]:
                model.fit(X_train_arr, y_train_arr, epochs=15, verbose=0)
                X_train.clear()
                y_train.clear()
        except Exception as e:
            print(f"Fehler beim Trainieren: {e}")

# Spielzähler
def count_games():
    global total_games_played, results
    total_games_played += 1
    if total_games_played >= 1000:
        total_games = sum(results.values())
        if total_games > 0:
            win_rate = results["p1"] / total_games * 100
            print(f"Win Rate nach {total_games_played} Spielen: {win_rate:.2f}%")
        results = {"p1": 0, "p2": 0, "tie": 0}
        total_games_played = 0

# Spieler-Funktion
def player(prev_play):
    global epsilon, opponent_history, player_history
    prev_play = prev_play if prev_play else random.choice(actions)
    opponent_history.append(prev_play)
    state = get_enhanced_state(opponent_history, player_history, state_size)

    # Mustererkennung
    pattern = detect_pattern(opponent_history)
    if pattern:
        guess = {"R": "P", "P": "S", "S": "R"}[pattern[0]]
    else:
        if np.random.rand() < epsilon:
            guess = random.choice(actions)
        else:
            prediction_move = np.argmax(model.predict(state.reshape(1, state_size * 2 + 3, 1), verbose=0))
            guess = int_to_action[prediction_move]
    
    player_history.append(guess)
    if len(player_history) > 1:
        prev_player_move = player_history[-2]
        if (prev_player_move == "R" and prev_play == "S") or \
           (prev_player_move == "P" and prev_play == "R") or \
           (prev_player_move == "S" and prev_play == "P"):
            results["p1"] += 1
        elif prev_player_move == prev_play:
            results["tie"] += 1
        else:
            results["p2"] += 1

    # Trainingsdaten sammeln
    if len(opponent_history) > 1:
        X_train.append(state.flatten().tolist())
        y_train.append(action_to_int[prev_play])
        if len(X_train) % 10 == 0:
            train_model()
    
    count_games()
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    return guess

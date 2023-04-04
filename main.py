import numpy as np

# Inicializar el ambiente y los parámetros del algoritmo
num_states = 10
num_actions = 2
Q_table = np.zeros((num_states, num_actions))
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
num_episodes = 1000

# Definir una función para elegir una acción con base en la política epsilon-greedy
def choose_action(state):
    if np.random.uniform() < epsilon:
        action = np.random.choice(num_actions)
    else:
        action = np.argmax(Q_table[state])
    return action

# Ciclo principal de aprendizaje
for episode in range(num_episodes):
    state = 0
    done = False
    while not done:
        # Elegir una acción con base en la política actual
        action = choose_action(state)
        # Ejecutar la acción y recibir una recompensa
        if action == 0:
            next_state = state - 1
            reward = -1 if next_state == 0 else 0
        else:
            next_state = state + 1
            reward = 1 if next_state == num_states - 1 else 0
        # Actualizar la tabla Q con base en la ecuación de Q-Learning
        Q_table[state, action] += learning_rate * (reward + discount_factor * np.max(Q_table[next_state]) - Q_table[state, action])
        # Actualizar el estado actual
        state = next_state
        # Verificar si se ha llegado al estado final
        done = (state == 0) or (state == num_states - 1)

# Imprimir la tabla Q aprendida
print(Q_table)

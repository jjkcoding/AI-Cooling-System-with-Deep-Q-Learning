

import numpy as np
from keras.models import load_model
import env


# Setting the paramaters
num_actions = 5
dir_boundary = (num_actions - 1) / 2
temp_step = 1.5

# Building environment
env = env.Env(optimal_temp = (18.0, 24.0), init_month = 0, init_num_users = 20, init_rate_data = 30)

model = load_model("model.h5")

# Choosing the training mode
train = False


# Running a 1 Year Simulation
env.train = train
curr_state, _, _ = env.observe()
for timestep in range(0, 6*30*24*60):
    q_vals = model.predict(curr_state)
    action = np.argmax(q_vals[0])
    if (action - dir_boundary) < 0:
        direction = -1
    else:
        direction = 1
    energy_ai = abs(action - dir_boundary) * temp_step
    next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep / (30*24*60)))
    curr_state = next_state



# Printing training results for each epoch
print("\n")
print("Total Energy Spent with AI: {:.0f}".format(env.total_energy_ai))
print("Total Energy Spent without AI: {:.0f}".format(env.total_energy_noai))
print("Energy Saved Percentage: {:.0f}".format((env.total_energy_noai - env.total_energy_ai) / env.total_energy_noai * 100))


                
                
                
                
                
                
                
                
                
                
                
                
                
                
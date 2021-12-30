

import numpy as np
import env
import brain
import dqn


# Setting the paramaters
epsilon = 0.3
num_actions = 5
dir_boundary = (num_actions - 1) / 2
num_epochs = 100
max_memory = 3000
batch_size = 512
temp_step = 1.5
early_stop = True
delta_loss_thresh = 0.0005
prev_loss = 1e5


# Building environment
env = env.Env(optimal_temp = (18.0, 24.0), init_month = 0, init_num_users = 20, init_rate_data = 30)

# Building brain
brain = brain.Brain(learning_rate = 0.00001, num_actions = num_actions)

# Building DQN Model
dqn = dqn.DQN(max_memory = max_memory, discount = 0.9)

# Choosing the training mode
train = True


# Training AI
env.train = train
model = brain.model
if env.train:
    # Starting loop over all epochs (1 epoch = 5 months)
    for epoch in range(1, num_epochs):
        # Initializing variables
        total_reward = 0
        loss = 0.0
        new_month = np.random.randint(0, 12)
        env.reset(new_month = new_month)
        game_over = False
        curr_state, _, _ = env.observe()
        timestep = 0 
        
        # Starting loop over timesteps (1 timestep = 1 min)
        while ((not game_over) and timestep <= 5*30*24*60):
            # Random action by exploration
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, num_actions)
                if action - dir_boundary < 0:
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - dir_boundary) * temp_step
            # Playing action by prediction from brain
            else:
                q_vals = model.predict(curr_state)
                action = np.argmax(q_vals[0])
                if (action - dir_boundary) < 0:
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - dir_boundary) * temp_step
            
            # Updating environment and next state
            next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep / (30*24*60)))
            total_reward += reward
            
            # Storing transition into memory
            dqn.remember([curr_state, action, reward, next_state], game_over)
            
            # Gathering in two separate batches the inputs and targets
            inputs, targets = dqn.get_batch(model, batch_size = batch_size)
            
            # Computing the loss over batches to train model
            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            curr_state = next_state

                
        # Printing training results for each epoch
        print("\n")
        print("Epoch: {:03d}/{:03d}".format(epoch, num_epochs))
        print("Total Energy Spent with AI: {:.0f}".format(env.total_energy_ai))
        print("Total Energy Spent without AI: {:.0f}".format(env.total_energy_noai))
        print("Current Loss: {:.6f}".format(loss))
        print("Loss Difference: {:.6f}".format(abs(prev_loss - loss)))
        
        # Saving model
        model.save("model.h5")
        
        if early_stop and abs(prev_loss - loss) < delta_loss_thresh:
            print("Broke Loss Threshold and Stopping Program")
            break
        prev_loss = loss
                
                
                
                
                
                
                
                
                
                
                
                
                

import numpy as np

class Env(object):
    
    # Initializing initial variables
    
    def __init__(self, optimal_temp = (18.0, 24.0), init_month = 0, init_num_users = 10, init_rate_data = 60):
        self.monthly_atmospheric_temp = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0, 23.0, 24.0, 22.0, 10.0, 5.0, 1.0]
        self.init_month = init_month
        self.atmospheric_temp = self.monthly_atmospheric_temp[init_month]
        self.optimal_temp = optimal_temp
        self.min_temp = -20
        self.max_temp = 80
        self.min_num_users = 10
        self.max_num_users = 100
        self.max_update_users = 5
        self.min_rate_data = 20
        self.max_rate_data = 300
        self.max_update_data = 10
        self.init_num_users = init_num_users
        self.curr_num_users = init_num_users
        self.init_rate_data = init_rate_data
        self.curr_rate_data = init_rate_data
        self.intrinsic_temp = self.atmospheric_temp + 1.25 * self.curr_num_users + 1.25 * self.curr_rate_data
        self.temp_ai = self.intrinsic_temp
        self.temp_noai = np.mean(optimal_temp)
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1
    
    
    # Purpose: Update environment after each action by the AI
    def update_env(self, direction, energy_ai, month):
        energy_noai = 0
        # Energy spent by the system is the absolute value of the temperature change caused by the server cooling system
        # If the temperture goes out of bounds, cooling system spends energy to bring temperature back to bounds
        
        if self.temp_noai < self.optimal_temp[0]:
            energy_noai = self.optimal_temp[0] - self.temp_noai
            self.temp_noai = self.optimal_temp[0]
        elif self.temp_noai > self.optimal_temp[1]:
            energy_noai = self.temp_noai - self.optimal_temp[0]
            self.temp_noai = self.optimal_temp[1]
       
        # Computing and scaling reward
        self.reward = energy_noai - energy_ai
        self.reward = 0.001 * self.reward
        
        # Update atmospheric temperature
        self.atmospheric_temp = self.monthly_atmospheric_temp[month]
        
        # Update number of users
        self.curr_num_users += np.random.randint(-self.max_update_users, self.max_update_users)
        if self.curr_num_users > self.max_num_users:
            self.curr_num_users = self.max_num_users
        elif self.curr_num_users < self.min_num_users:
            self.curr_num_users = self.min_num_users
        
        # Updating rate of data
        self.curr_rate_data += np.random.randint(-self.max_update_data, self.max_update_data)
        if self.curr_rate_data > self.max_rate_data:
            self.curr_rate_data = self.max_rate_data
        elif self.curr_rate_data < self.min_rate_data:
            self.curr_rate_data = self.min_rate_data
        
        # Computing Delta of Intrinsic Temperature
        past_intrinsic_temp = self.intrinsic_temp
        self.intrinsic_temp = self.atmospheric_temp + 1.25 * self.curr_num_users + 1.25 * self.curr_rate_data
        delta_intrinsic_temp = self.intrinsic_temp - past_intrinsic_temp
        
        # Computing Delta of Temperature caused by AI
        if direction == -1:
            delta_temp_ai = -energy_ai
        elif direction == 1:
            delta_temp_ai = energy_ai
        
        # Updating new server's temperature when there is an AI
        self.temp_ai += delta_intrinsic_temp + delta_temp_ai
        
        # Updating new server's temperature when there is no ai
        self.temp_noai += delta_intrinsic_temp
        
        # Checks if game over
        if self.temp_ai < self.min_temp:
            if self.train == 1:
                self.game_over = 1
            else:
                self.total_energy_ai += self.optimal_temp[0] - self.temp_ai
                self.temp_ai = self.optimal_temp[0]
        elif self.temp_ai > self.max_temp:
            if self.train == 1:
                self.game_over = 1
            else:
                self.total_energy_ai += self.temp_ai - self.optimal_temp[1]
                self.temp_ai = self.optimal_temp[1]
                
        # Updating total energy spent by AI
        self.total_energy_ai += energy_ai
        
        # Updating the total energy spent by server's cooling system without AI
        self.total_energy_noai += energy_noai
        
        # Scaling the next state
        scaled_temp_ai = (self.temp_ai - self.min_temp) / (self.max_temp - self.min_temp)
        scaled_num_users = (self.curr_num_users - self.min_num_users) / (self.max_num_users - self.min_num_users)
        scaled_rate_data = (self.curr_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)
        next_state = np.matrix([scaled_temp_ai, scaled_num_users, scaled_rate_data])
        
        return next_state, self.reward, self.game_over
        
    # Purpose: Resetting variables for environment
    def reset(self, new_month):
        self.atmospheric_temp = self.monthly_atmospheric_temp[new_month]
        self.init_month = new_month
        self.curr_num_users = self.init_num_users
        self.curr_rate_data = self.init_rate_data
        self.intrinsic_temp = self.atmospheric_temp + 1.25 * self.curr_num_users + 1.25 * self.curr_rate_data
        self.temp_ai = self.intrinsic_temp
        self.temp_noai = np.mean(self.optimal_temp)
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1
    
    
    # Purpose: Gives time of the current state, reward, and checks if game is over
    def observe(self):
        scaled_temp_ai = (self.temp_ai - self.min_temp) / (self.max_temp - self.min_temp)
        scaled_num_users = (self.curr_num_users - self.min_num_users) / (self.max_num_users - self.min_num_users)
        scaled_rate_data = (self.curr_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)
        curr_state = np.matrix([scaled_temp_ai, scaled_num_users, scaled_rate_data])
        return curr_state, self.reward, self.game_over
        
        
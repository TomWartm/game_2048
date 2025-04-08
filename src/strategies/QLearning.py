import numpy as np
import os
import random
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim

ACTION_MAP = {"A": 0, "D": 1, "S": 2, "W": 3}
REVERSE_ACTION_MAP = {v: k for k, v in ACTION_MAP.items()}

class DQNetwork(nn.Module):
    def __init__(self):
        super(DQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=2, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=2, padding=1)

        self.fc1 = nn.Linear(784, 1024)  
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 4)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x)) 
        x = torch.relu(self.conv3(x)) 
        x = x.view(x.size(0), -1) # flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class QLearning:
    def __init__(self, model_path="data/qmodel.pt"):
        self.counter = 0
        self.training_steps = 0
        self.start_time = datetime.now()
        self.model_path = model_path
        
        self.model = DQNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # reload model if it exists
        self.load_model()

        # Replay memory
        self.memory = []
        self.memory_size = 2048 # Size of memory: note that this is not too good when we have more moves then that
        self.batch_size = 512
        self.train_moves = 100

        # Discount factor
        self.gamma = 0.9

    def __str__(self):
        return "Deep Q-Learning"

    def save_model(self):
        store_path = self.model_path.split("/")
        store_path = "/".join(store_path[:-1]) + "/" + str(self.training_steps) + "_" + store_path[-1]
        # Ensure the directory exists
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        
        # Save the model's state dictionary
        torch.save(self.model.state_dict(), store_path)
        print(f"Model saved to {store_path}")

    def load_model(self, from_training_step = 48988):
        self.training_steps = from_training_step
        load_path = self.model_path.split("/")
        load_path = "/".join(load_path[:-1]) + "/" + str(from_training_step) + "_" + load_path[-1]
        if os.path.exists(load_path):
            self.model.load_state_dict(torch.load(load_path))
            self.model.eval()  # Set the model to evaluation mode
            print(f"Model loaded from {load_path}")
        else:
            print(f"No model found at {load_path}")

    def move(self, game, board):

        state = torch.FloatTensor(board).unsqueeze(0).unsqueeze(0)  

        # Epsilon-greedy policy
        epsilon = 0.2
        possible_actions = [direction for direction in ["A", "D", "S", "W"] if game.checkmove(board, direction)]
        if len(possible_actions) == 0:
            raise ValueError("No possible actions")
        if np.random.rand() < epsilon:
            action = np.random.choice(possible_actions)
            
        else:
            with torch.no_grad():
                q_values = self.model(state)
                action = REVERSE_ACTION_MAP[torch.argmax(q_values).item()]
                if action not in possible_actions:
                    action = np.random.choice(possible_actions)
                    

        action_index = ACTION_MAP[action]
        
        # Perform the action and get the next state
        next_board, _ = game.move(board, action)
        next_state = torch.FloatTensor(next_board).unsqueeze(0).unsqueeze(0) 

        # Get reward
        reward = 0
        reward += game.future_av_value(board, action)
        reward = reward * (len(game.empty_spaces(next_board)) - len(game.empty_spaces(board)) + 1)
        if game.highest_tile(next_board) > game.highest_tile(board):
            reward *= 1.5
        if game.is_over(next_board):
            reward *= 0.5  

        # Check if the game is over
        done = game.is_over(next_board)

        # Store transition in memory
        self.memory.append((state, action_index, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

        # Train the model
        if len(self.memory) >= self.batch_size and self.counter % self.train_moves == 0:
            self.train_model()
            
        # store model every 5minutes
        if datetime.now() - self.start_time > timedelta(minutes=60):
            self.save_model()
            self.start_time = datetime.now()

        self.counter += 1
        return next_board

    def train_model(self):
        if len(self.memory) < self.batch_size:
            return  # Not enough samples to train

        # Sample a batch of experiences from memory
        batch = random.sample(self.memory, self.batch_size)

        # Prepare batch data
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.cat(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones, dtype=torch.bool)

        # Compute current Q-values
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (~dones)

        # Compute loss
        loss = self.criterion(q_values, target_q_values)

        # Perform a gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.training_steps += 1
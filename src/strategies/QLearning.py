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
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 128)        # Second hidden layer
        self.fc3 = nn.Linear(128, output_size)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class QLearning:
    def __init__(self, input_size = 16, output_size=4, model_path="data/qmodel.pt"):
        self.counter = 0
        self.start_time = datetime.now()
        self.model_path = model_path
        # Initialize the DQNetwork
        self.model = DQNetwork(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # reload model if it exists
        self.load_model()

        # Replay memory
        self.memory = []
        self.memory_size = 10000
        self.batch_size = 64

        # Discount factor
        self.gamma = 0.9

    def __str__(self):
        return "Deep Q-Learning"

    def save_model(self):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save the model's state dictionary
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()  # Set the model to evaluation mode
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"No model found at {self.model_path}")

    def move(self, game, board):
        # Convert board to a flattened tensor
        state = torch.FloatTensor(board.flatten()).unsqueeze(0)

        # Epsilon-greedy policy
        epsilon = 0.3
        possible_actions = [direction for direction in ["A", "D", "S", "W"] if game.checkmove(board, direction)]
        if len(possible_actions) == 0:
            raise ValueError("No possible actions")
        if np.random.rand() < epsilon:
            action = np.random.choice(possible_actions)
        else:
            with torch.no_grad():
                q_values = self.model(state)
                action = possible_actions[torch.argmax(q_values[0, :len(possible_actions)]).item()]

        # Convert action to integer
        action_index = ACTION_MAP[action]

        # Perform the action and get the next state
        next_board, _ = game.move(board, action)
        next_state = torch.FloatTensor(next_board.flatten()).unsqueeze(0)

        # Get reward
        reward = 0
        reward += game.future_av_value(board, action)
        reward += len(game.empty_spaces(next_board)) - len(game.empty_spaces(board)) + 1
        if game.highest_tile(next_board) > game.highest_tile(board):
            reward += 1

        # Check if the game is over
        done = game.is_over(next_board)

        # Store transition in memory
        self.memory.append((state, action_index, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

        # Train the model
        if len(self.memory) >= self.batch_size:
            self.train_model()
            
        # store model every 5minutes
        if datetime.now() - self.start_time > timedelta(minutes=5):
            self.save_model()
            self.start_time = datetime.now()

        self.counter += 1
        return next_board

    def train_model(self):
        if len(self.memory) < self.batch_size:
            return  # Not enough samples to train

        # Sample a batch of experiences from memory
        batch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in batch:
            state = state.squeeze(0)  # Ensure state has shape [1, input_size]
            next_state = next_state.squeeze(0)  # Ensure next_state has shape [1, input_size]

            # Get the current Q-values for the state
            q_values = self.model(state)
            target = q_values.clone().detach()

            if done:
                target[action] = reward  # Set the target for the action to the reward
            else:
                # Get the maximum Q-value for the next state
                next_q_values = self.model(next_state)
                target[action] = reward + self.gamma * torch.max(next_q_values).item()

            # Perform a gradient descent step
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
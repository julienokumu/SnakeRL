# Import necessary libraries
import pygame  # For creating the game graphics and handling user input
import random  # For generating random values, used in game mechanics like food placement
import numpy as np  # For handling arrays and mathematical operations
import torch  # For deep learning with PyTorch
import torch.nn as nn  # For building neural network layers
import torch.optim as optim  # For optimization algorithms like Adam
import collections  # For using the deque data structure in experience replay memory
from itertools import cycle  # For cycling through a sequence repeatedly

# Initialize pygame (sets up the underlying pygame system)
pygame.init()

# Game Constants (defining constants for game settings)
WIDTH, HEIGHT = 640, 480  # Width and height of the game window
GRID_SIZE = 20  # The size of each grid block in pixels
WHITE = (255, 255, 255)  # RGB color for white (used for the background)
GREEN = (0, 255, 0)  # RGB color for green (used for the snake)
RED = (255, 0, 0)  # RGB color for red (used for food)
BLACK = (0, 0, 0)  # RGB color for black (used for background or boundaries)
ACTION_SPACE = [(GRID_SIZE, 0), (-GRID_SIZE, 0), (0, GRID_SIZE), (0, -GRID_SIZE)]  # Possible actions: move right, left, down, or up

# Define Deep Q-Network (DQN) class, a neural network for approximating the Q-function
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()  # Initialize the parent class (nn.Module)
        # Define a simple feedforward neural network with ReLU activations
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),  # First hidden layer: input_dim -> 128 neurons
            nn.ReLU(),  # ReLU activation function
            nn.Linear(128, 128),  # Second hidden layer: 128 -> 128 neurons
            nn.ReLU(),  # ReLU activation function
            nn.Linear(128, output_dim)  # Output layer: 128 -> output_dim (number of actions)
        )
    
    # Define the forward pass through the network
    def forward(self, x):
        return self.fc(x)  # Pass the input through the defined layers

# Experience Replay Memory class, to store and sample past experiences for training
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = collections.deque(maxlen=capacity)  # Initialize deque with a fixed size (capacity)
    
    # Store a new experience (state, action, reward, next state, done flag) in memory
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # Append the experience to the deque
    
    # Sample a batch of experiences from memory
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)  # Randomly sample 'batch_size' experiences
        return map(np.array, zip(*batch))  # Return as a tuple of numpy arrays
    
    # Get the current size of the memory (number of experiences stored)
    def __len__(self):
        return len(self.memory)

# Training Parameters (hyperparameters for training the DQN)
gamma = 0.99  # Discount factor for future rewards (closer to 1 means more emphasis on future rewards)
lr = 0.001  # Learning rate for the optimizer
epsilon = 1.0  # Exploration rate (probability of choosing a random action)
epsilon_min = 0.01  # Minimum value for epsilon (to ensure some exploration even after training)
epsilon_decay = 0.995  # Decay rate for epsilon after each episode (decays towards epsilon_min)
batch_size = 64  # Number of experiences to sample from memory for each training step
memory_capacity = 100000  # Maximum size of the experience replay memory
target_update_freq = 1000  # Frequency of updating the target network
train_start = 1000  # Number of experiences to collect before starting training

# Initialize model (input_dim represents the state space size, output_dim represents the action space size)
input_dim = 9  # The state space is represented by 9 features (as defined in get_state function)
output_dim = len(ACTION_SPACE)  # The action space has 4 possible actions (left, right, up, down)
policy_net = DQN(input_dim, output_dim)  # Initialize the policy network (Q-function approximator)
target_net = DQN(input_dim, output_dim)  # Initialize the target network
target_net.load_state_dict(policy_net.state_dict())  # Copy the policy network's weights to the target network
target_net.eval()  # Set the target network to evaluation mode (disables dropout/BatchNorm)
optimizer = optim.Adam(policy_net.parameters(), lr=lr)  # Initialize Adam optimizer for policy network
memory = ReplayMemory(memory_capacity)  # Initialize replay memory with a maximum capacity

# Initialize pygame screen for displaying the game
screen = pygame.display.set_mode((WIDTH, HEIGHT))  # Create the game window with the specified width and height
pygame.display.set_caption("Reinforcement Learning: Snake by Julien Okumu")  # Set the window caption (title)

# Function to get the current state of the game (for feeding into the DQN model)
def get_state(snake, food, direction):
    head_x, head_y = snake[0]  # Get the x and y coordinates of the snake's head
    food_x, food_y = food  # Get the x and y coordinates of the food
    # Check if there is any immediate danger in the direction the snake is moving
    danger_straight = (head_x + direction[0], head_y + direction[1]) in snake or head_x + direction[0] < 0 or head_x + direction[0] >= WIDTH or head_y + direction[1] < 0 or head_y + direction[1] >= HEIGHT
    direction_left = (direction[1], -direction[0])  # The left turn relative to the current direction
    danger_left = (head_x + direction_left[0], head_y + direction_left[1]) in snake  # Check if the left turn is dangerous
    direction_right = (-direction[1], direction[0])  # The right turn relative to the current direction
    danger_right = (head_x + direction_right[0], head_y + direction_right[1]) in snake  # Check if the right turn is dangerous
    # Return the state as an array of binary features
    return np.array([danger_straight, danger_left, danger_right, direction[0], direction[1], food_x < head_x, food_x > head_x, food_y < head_y, food_y > head_y])  # The state features

# Function to select an action based on the current state
def select_action(state):
    # Check if a random action should be selected based on the exploration rate (epsilon)
    global epsilon
    if random.random() < epsilon:  # epsilon-greedy approach: random action if epsilon condition is met
        return random.randint(0, output_dim - 1)  # Select a random action from the available actions
    else:
        # Otherwise, select the action with the highest Q-value from the policy network (exploitation)
        with torch.no_grad():  # Disable gradient calculation for this operation to save memory
            # Pass the state to the policy network and return the action with the maximum Q-value
            return policy_net(torch.tensor(state, dtype=torch.float32)).argmax().item()

# Function to perform model optimization (training step)
def optimize_model():
    # Check if there are enough samples in the memory to start training
    if len(memory) < train_start:
        return  # Exit if there are not enough samples in memory to train
    
    # Sample a batch of experiences (states, actions, rewards, next states, and done flags) from memory
    states, actions, rewards, next_states, dones = memory.sample(batch_size)
    
    # Convert the sampled states, actions, rewards, next states, and done flags to PyTorch tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)  # Ensure actions are in the right shape
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    
    # Get the predicted Q-values for the selected actions from the policy network
    q_values = policy_net(states).gather(1, actions)  # gather() is used to get the Q-values of the chosen actions
    # Get the maximum Q-value for each next state from the target network (used in the Bellman equation)
    next_q_values = target_net(next_states).max(1)[0].detach()  # Detach to avoid tracking gradients for target network
    # Compute the target Q-values using the Bellman equation
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))  # Incorporates reward, discount, and terminal flag
    
    # Compute the loss between the predicted Q-values and the target Q-values
    loss = nn.MSELoss()(q_values.squeeze(), target_q_values)  # Mean Squared Error Loss
    
    # Perform a gradient descent step
    optimizer.zero_grad()  # Zero the gradients before backpropagation
    loss.backward()  # Backpropagate the error
    optimizer.step()  # Update the model's weights using the optimizer

# Main loop that simulates the Snake game and trains the agent
step_counter = 0  # Initialize the step counter
while True:  # Run the game loop indefinitely
    snake = [(WIDTH // 2, HEIGHT // 2)]  # Initialize the snake at the center of the screen
    direction = ACTION_SPACE[0]  # Set the initial direction of the snake
    food = (random.randrange(0, WIDTH, GRID_SIZE), random.randrange(0, HEIGHT, GRID_SIZE))  # Place the food randomly
    clock = pygame.time.Clock()  # Initialize the game clock to control frame rate
    running = True  # Flag to track if the game is still running
    
    while running:  # Run the game until it is over
        screen.fill(BLACK)  # Clear the screen by filling it with the background color (black)
        for event in pygame.event.get():  # Process events (like user input)
            if event.type == pygame.QUIT:  # If the quit event is triggered (e.g., user closes the window)
                pygame.quit()  # Quit pygame
                exit()  # Exit the program
        
        # Get the current state of the game (snake, food, direction)
        state = get_state(snake, food, direction)
        # Select an action based on the current state using the select_action function
        action_index = select_action(state)
        # Update the direction based on the selected action
        direction = ACTION_SPACE[action_index]
        
        # Calculate the new position of the snake's head based on the current direction
        new_head = (snake[0][0] + direction[0], snake[0][1] + direction[1])
        
        # Determine the reward based on the new position of the snake
        # -1000 for invalid positions (snake collision or out of bounds), otherwise 10 points
        reward = -1000 if new_head in snake or new_head[0] < 0 or new_head[0] >= WIDTH or new_head[1] < 0 or new_head[1] >= HEIGHT else 10
        
        # If the snake's head reaches the food, give a large reward and spawn new food
        if new_head == food:
            reward = 10000  # Large reward for eating food
            food = (random.randrange(0, WIDTH, GRID_SIZE), random.randrange(0, HEIGHT, GRID_SIZE))  # New food location
        else:
            # Remove the last segment of the snake's body if it doesn't eat food
            snake.pop()
        
        # Add the new head to the front of the snake
        snake.insert(0, new_head)
        # Check if the game is over (snake collided with itself or went out of bounds)
        done = new_head in snake[1:] or new_head[0] < 0 or new_head[0] >= WIDTH or new_head[1] < 0 or new_head[1] >= HEIGHT
        
        # Get the next state after performing the action
        next_state = get_state(snake, food, direction)
        # Store the current transition (state, action, reward, next_state, done) in memory
        memory.push(state, action_index, reward, next_state, done)
        
        # Perform a model optimization step (train the agent) after each action
        optimize_model()
        
        # Draw the snake's segments on the screen
        for segment in snake:
            pygame.draw.rect(screen, GREEN, (segment[0], segment[1], GRID_SIZE, GRID_SIZE))
        # Draw the food on the screen
        pygame.draw.rect(screen, RED, (food[0], food[1], GRID_SIZE, GRID_SIZE))
        pygame.display.flip()  # Update the screen
        clock.tick(10)  # Control the game loop speed (frame rate)
        
        step_counter += 1  # Increment the step counter
        # Every few steps, update the target network to match the policy network
        if step_counter % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Decay epsilon to reduce exploration over time
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Break the inner game loop if the game is over
        if done:
            break


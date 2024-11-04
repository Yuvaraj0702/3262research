import numpy as np
import pandas as pd
import gym
import yfinance as yf
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the DRQN class
class DRQN:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(None, self.state_size)))
        model.add(LSTM(64, return_sequences=True))
        model.add(LSTM(64))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def predict(self, state):
        state = np.reshape(state, [1, state.shape[0], self.state_size])
        return self.model.predict(state)

    def train(self, state, target):
        state = np.reshape(state, [1, state.shape[0], self.state_size])
        self.model.fit(state, target, epochs=1, verbose=0)

# Define the Trading Environment
class TradingEnvironment(gym.Env):
    def __init__(self, data):
        super(TradingEnvironment, self).__init__()
        self.data = data
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)  # [Hold, Buy, Sell]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(data.shape[1],), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.data[self.current_step]

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        next_state = self.data[self.current_step]

        # Calculate reward
        reward = 0
        if action == 1:  # Buy
            reward = (self.data[self.current_step][0] - self.data[self.current_step - 1][0])
        elif action == 2:  # Sell
            reward = (self.data[self.current_step - 1][0] - self.data[self.current_step][0])
        elif action == 0:  # Hold
            reward = -0.01  # Small penalty for holding

        return next_state, reward, done, {}

# Test the trained model
def test_agent(env, drqn, episodes=10):
    total_profit = 0
    for episode in range(episodes):
        state = env.reset()
        episode_profit = 0
        done = False
        logging.info(f"Starting Test Episode {episode + 1}")
        while not done:
            action = np.argmax(drqn.predict(state))  # Always exploit during testing
            next_state, reward, done, _ = env.step(action)
            episode_profit += reward
            state = next_state
            
            # Log the action taken
            action_name = ['Hold', 'Buy', 'Sell'][action]
            logging.info(f"Action taken: {action_name}, Reward: {reward}, Total Episode Profit: {episode_profit}")

        total_profit += episode_profit
        logging.info(f"Test Episode {episode + 1} finished, Total Profit: {episode_profit}")

    average_profit = total_profit / episodes
    logging.info(f"Average Profit over {episodes} episodes: {average_profit}")
    logging.info("Profit Explanation: The profit indicates the cumulative returns from the trading actions taken during the episode.")

# Example Usage
if __name__ == "__main__":
    # Load historical stock data from Yahoo Finance
    stock_symbol = 'AAPL'  # You can change this to any stock symbol
    data = yf.download(stock_symbol, start='2023-01-01', end='2023-05-05', progress=False)
    
    # Prepare the features: Use only the 'Close' prices for this example
    close_prices = data['Close'].values
    data_length = len(close_prices)

    # Normalize the data (between 0 and 1)
    normalized_data = (close_prices - np.min(close_prices)) / (np.max(close_prices) - np.min(close_prices))
    reshaped_data = normalized_data.reshape(-1, 1)  # Reshape for the environment

    # Split data into training and testing datasets
    split_ratio = 0.8
    train_size = int(data_length * split_ratio)
    train_data = reshaped_data[:train_size]
    test_data = reshaped_data[train_size:]

    # Initialize environments
    train_env = TradingEnvironment(train_data)
    test_env = TradingEnvironment(test_data)

    state_size = reshaped_data.shape[1]  # Number of features (1 for 'Close' price)
    action_size = train_env.action_space.n
    drqn = DRQN(state_size, action_size)

    episodes = 100  # Reduced number of episodes for faster training
    epsilon = 1.0  # Start with full exploration
    epsilon_min = 0.01
    epsilon_decay = 0.995

    # Experience replay buffer
    replay_buffer = deque(maxlen=2000)
    batch_size = 16  # Reduced batch size for less memory usage

    # Training Phase
    for episode in range(episodes):
        state = train_env.reset()
        total_reward = 0
        logging.info(f"Starting Training Episode {episode + 1}")

        while True:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = train_env.action_space.sample()  # Explore
            else:
                action = np.argmax(drqn.predict(state))  # Exploit

            next_state, reward, done, _ = train_env.step(action)  # Take action
            total_reward += reward

            # Store the experience in the replay buffer
            replay_buffer.append((state, action, reward, next_state, done))

            # Update the model every few steps
            if len(replay_buffer) >= batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                for m_state, m_action, m_reward, m_next_state, m_done in minibatch:
                    target = drqn.predict(m_state)
                    if m_done:
                        target[0][m_action] = m_reward  # No future reward
                    else:
                        target[0][m_action] = m_reward + 0.95 * np.max(drqn.predict(m_next_state))  # Q-learning update
                    drqn.train(m_state, target)

            state = next_state
            if done:
                break

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        logging.info(f"Episode {episode + 1}/{episodes} finished, Total Reward: {total_reward}")

    # Testing Phase
    test_agent(test_env, drqn, episodes=10)

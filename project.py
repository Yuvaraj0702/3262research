import numpy as np
import pandas as pd
import gymnasium as gym
import yfinance as yf
import logging
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer
from tensorflow.keras.layers import ELU
from tensorflow.keras.initializers import HeNormal, Identity, RandomNormal, Zeros
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from collections import deque
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Define the DRQN class
class DRQN:
    def __init__(self, state_size, action_size, learning_rate=0.00025, tau = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.tau = tau

        # Main Q-network
        self.model = self.build_model()
        self.model.optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        # Target Q-network
        self.target_model = self.build_model()
        # Initialize target network weights
        self.update_target_network()
        #print("Trainable variables in the model:")
        #for var in self.model.trainable_variables:
        #    print(" -", var.name, "shape:", var.shape)
        
        #print("Model Summary:")
        self.model.summary()

    def build_model(self):
        model = Sequential()

        # First Dense layer with He Normal initialization for weights
        model.add(Dense(256, input_shape=(None, self.state_size), kernel_initializer=HeNormal()))
        model.add(ELU())
        # Second Dense layer with He Normal initialization for weights
        model.add(Dense(256, kernel_initializer=HeNormal()))
        model.add(ELU())
        # LSTM layer with identity initialization for recurrent kernel, zero for bias, and He normal for kernel
        lstm_initializer = Identity()
        model.add(LSTM(256, 
                       kernel_initializer=HeNormal(),
                       recurrent_initializer=lstm_initializer, 
                       bias_initializer=Zeros(),  
                       return_sequences=True))
        
        # Adjust the forget gate bias to 1
        lstm_layer = model.layers[-1]
        lstm_layer.set_weights([lstm_layer.get_weights()[0], lstm_layer.get_weights()[1], np.ones(1024)])

        model.add(Dense(self.action_size, activation='linear', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001)))

        return model

    def update_target_network(self):
        for target_param, local_param in zip(self.target_model.trainable_variables, self.model.trainable_variables):
            target_param.assign((1 - self.tau) * target_param + self.tau * local_param)

    def predict(self, state, use_target=False):
        state = np.reshape(state, (1, 1, self.state_size))
        model = self.target_model if use_target else self.model
        q_values = model.predict(state)
        return q_values

    def train_on_batch(self, minibatch, discount_factor):
        #print("\n--- Training Batch with Action Augmentation ---")
        #print("Minibatch size:", len(minibatch))
        loss = 0

        with tf.GradientTape() as tape:
            for sequence_data in minibatch:
                # Unpack the sequence data, including hypothetical rewards for all actions
                states, actions, rewards, next_states, dones, all_hypothetical_rewards = sequence_data
                
                # Convert states and next_states to tensors and reshape for LSTM input
                state_sequence = np.reshape(np.array(states), (1, len(states), self.state_size))
                next_state_sequence = np.reshape(np.array(next_states), (1, len(next_states), self.state_size))
                #print("state_sequence sample values:", state_sequence[0, :5, :])
                state_sequence = np.nan_to_num(state_sequence, nan=0.0, posinf=0.0, neginf=0.0)
                # Predict Q-values for the current state sequence
                q_values_sequence = self.model(state_sequence, training=True)
                # Copy the Q-values for manipulation
                target_q_values_sequence = q_values_sequence.numpy().copy()

                for t in range(len(states)):
                    hypothetical_rewards = all_hypothetical_rewards[t]
                    
                    for a in range(self.action_size):
                        # For each action, calculate the target Q-value based on hypothetical rewards
                        if dones[t]:
                            target_q_value = hypothetical_rewards[a]
                        else:
                            # Expand the next state for model input and get next Q-values
                            next_actions = self.model.predict(next_state_sequence)  # Main network
                            best_action = tf.argmax(next_actions[0, t])
                            single_next_state = tf.expand_dims(tf.expand_dims(next_state_sequence[0, t], axis=0), axis=0)
                            next_q_values = self.target_model(single_next_state)
                            print("next_q_values shape:", next_q_values.shape)
                            print("best_action:", best_action)
                            max_next_q_value = next_q_values[0, 0, best_action]
                            #max_next_q_value = tf.reduce_max(next_q_values)
                            #max_next_q_value = tf.clip_by_value(max_next_q_value, -1.0, 1.0)
                            # Check types and values
                            target_q_value = hypothetical_rewards[a] + discount_factor * max_next_q_value
                        print("target_q_value", target_q_value)
                        # Update target Q-values for each action at time t
                        target_q_values_sequence[0, t, a] = target_q_value

                # Calculate the action augmentation loss for this sequence
                print("target_q_values_sequence shape:", target_q_values_sequence.shape)
                print("q_values_sequence shape:", q_values_sequence.shape)
                #print("target_q_values_sequence sample values:", target_q_values_sequence[0, :5, :])
                #print("q_values_sequence sample values:", q_values_sequence[0, :5, :])
                sample_loss = tf.reduce_mean(tf.square(target_q_values_sequence - q_values_sequence))
                print(sample_loss)
                loss += sample_loss

            # Average the loss over the minibatch
            loss /= len(minibatch)
            print("Average Loss for Minibatch:", loss)

        # Apply gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss 
          
    #def train(self, state, target):
        #state = np.reshape(state, [1, state.shape[0], self.state_size])
        #self.model.fit(state, target, epochs=1, verbose=0)

# Define the Trading Environment with additional indicators
class TradingEnvironment(gym.Env):
    def __init__(self, raw_data, data, dates, initial_cash=10000000, trade_size=100000, commission_rate=0.0000278, spread=0.0015):
        super(TradingEnvironment, self).__init__()
        self.raw_data = raw_data
        self.data = data
        self.dates = dates  # Pass dates separately
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(data.shape[1] + 4,), dtype=np.float32)

        # Trading parameters
        self.initial_cash = initial_cash
        self.trade_size = trade_size
        self.cash_balance = initial_cash
        self.unrealized_pnl = 0
        #self.position_size = 0
        self.portfolio_value = initial_cash
        self.commission_rate = commission_rate
        self.spread = spread

        # Initial portfolio state
        self.position = [0, 1, 0]
        self.current_position = 0

    def reset(self):
        self.current_step = 0
        self.position = [0, 1, 0]
        self.cash_balance = self.initial_cash
        self.unrealized_pnl = 0
        self.portfolio_value = self.initial_cash
        self.current_position = 0
        return self._get_observation()

    def _get_observation(self):
        # All features are already contained in self.data[self.current_step - 1]
        ohlcv_features = self.data[self.current_step]
        
        # Encode day of the week as a sinusoidal feature
        current_date = pd.Timestamp(self.dates[self.current_step])
        #print("Current date in get_observation:", current_date)
        day_of_week = current_date.weekday()
        day_encoding = np.sin(2 * np.pi * day_of_week / 7)
        # Return full observation including position encoding
        return np.concatenate([ohlcv_features, self.position])

    def step(self, action):

        if self.current_step >= len(self.data) - 1:
            done = True
            return self._get_observation(), 0, done, {}
        
        # Take action at the start of the day based on the open price
        open_price = self.raw_data[self.current_step][0]
        #print("Open price in step: ", open_price)
        

        # Calculate transaction cost if position changes
        transaction_cost = 0
        if action != 1:
            transaction_cost = self.trade_size * self.commission_rate * self.spread

        # Get reward for the chosen action (without applying changes)
        reward = self.calculate_reward(action, open_price, transaction_cost)

        # Apply the actual update based on the chosen action
        self.update_position(action, open_price, transaction_cost)
        
        done = self.current_step >= len(self.data) - 1
        self.current_step += 1
        # Prepare the next state
        next_state = self._get_observation()
        
        return next_state, reward, done, {}

    def calculate_reward(self, action, open_price, transaction_cost=0):
            #print(self.current_step)
            # Get the closing price for the current step
            close_price = self.raw_data[self.current_step][3]  # Assuming the close price is at index 3

            # Initialize hypothetical cash balance and unrealized PnL
            hypothetical_unrealized_pnl = self.unrealized_pnl
            hypothetical_cash_balance = self.cash_balance
            starting_portfolio_value = hypothetical_cash_balance + hypothetical_unrealized_pnl
            if action == 0:  # Sell action
                # Check if we have enough position to sell
                if hypothetical_unrealized_pnl >= self.trade_size:
                    hypothetical_unrealized_pnl = hypothetical_unrealized_pnl - self.trade_size  # Reduce position size by trade size
                    hypothetical_cash_balance = hypothetical_cash_balance + self.trade_size - transaction_cost
                else:
                    # Sell remaining position
                    hypothetical_cash_balance = hypothetical_cash_balance + hypothetical_unrealized_pnl - transaction_cost
                    hypothetical_unrealized_pnl = 0

            elif action == 1:  # Neutral action
                # Unrealized PnL changes with market movements
                pass

            elif action == 2:  # Buy action
                # Increase position size by trade size
                if hypothetical_cash_balance < self.trade_size + transaction_cost or hypothetical_cash_balance < 0:
                    pass
                else:
                    hypothetical_unrealized_pnl = hypothetical_unrealized_pnl + self.trade_size
                    hypothetical_cash_balance = hypothetical_cash_balance - self.trade_size - transaction_cost

            hypothetical_unrealized_pnl = hypothetical_unrealized_pnl * (close_price / self.raw_data[self.current_step][0])
            # Calculate hypothetical portfolio value for reward
            hypothetical_portfolio_value = hypothetical_cash_balance + hypothetical_unrealized_pnl

            # Reward is the log return based on the change in portfolio value
            reward = np.log(hypothetical_portfolio_value / starting_portfolio_value)
            alternative_reward = hypothetical_portfolio_value/starting_portfolio_value
            # Print debugging information
            #print("\n--- Hypothetical Reward Calculation ---")
            #print("Action:", action)
            #print("Date:", pd.Timestamp(self.dates[self.current_step]))
            #print("Open Price:", self.raw_data[self.current_step][0])
            #print("Close Price:", close_price)
            #print("Hypothetical Unrealized PnL:", hypothetical_unrealized_pnl)
            #print("Hypothetical Cash Balance:", hypothetical_cash_balance)
            #print("Transaction Cost:", transaction_cost)
            #print("Hypothetical Portfolio Value:", hypothetical_portfolio_value)
            #print("Previous Portfolio Value:", self.portfolio_value)
            #print("Reward:", reward)

            return reward

    def update_position(self, action, open_price, transaction_cost):
        #print(self.current_step)
        #"""Update the actual position based on the action taken."""
        #print("\n--- Before taking action ---")
        #print("Current date: ", pd.Timestamp(self.dates[self.current_step]))
        #print("Open Price:", open_price)
        #print("Current Position Size (in dollars):", self.unrealized_pnl)
        #print("Current cash in hand:", self.cash_balance)
        #print("Position Indicator:", self.position)
        #print("Portfolio Value:", self.portfolio_value)
        close_price = self.raw_data[self.current_step][3]  # Close price for the current day
        if self.unrealized_pnl > 0:
            # Update position size based on action
            if action == 2:  # Buy action
                if self.cash_balance < self.trade_size + transaction_cost or self.cash_balance < 0:
                    pass
                else:
                    self.unrealized_pnl = self.unrealized_pnl + self.trade_size  # Increase position size by $100,000
                    self.cash_balance = self.cash_balance - self.trade_size - transaction_cost

            elif action == 1:  # Neutral action
                pass  # Hold position, no change in position size

            elif action == 0:  # Sell action
                if self.unrealized_pnl >= self.trade_size:
                    self.unrealized_pnl = self.unrealized_pnl - self.trade_size  # Decrease position size by $100,000
                    self.cash_balance = self.cash_balance + self.trade_size - transaction_cost
                else:
                    self.cash_balance = self.cash_balance + self.unrealized_pnl - transaction_cost
                    self.unrealized_pnl = 0  # Set to zero if position size is below $100,000
        else:
            if action == 2:  # Buy action
                if self.cash_balance < self.trade_size + transaction_cost or self.cash_balance < 0:
                    pass
                else:
                    self.unrealized_pnl = self.unrealized_pnl + self.trade_size  # Increase position size by $100,000
                    self.cash_balance = self.cash_balance - self.trade_size - transaction_cost

            elif action == 1 or 0:  # Neutral action
                pass  # Hold position, no change in position size
        self.unrealized_pnl = (self.unrealized_pnl) * (close_price / open_price)  # Account for fluctuations
        # Update portfolio value
        self.portfolio_value = self.unrealized_pnl + self.cash_balance

        # Update position indicator for tracking decisions
        if action == 2:
            self.position = [0, 0, 1]  # Buy
        elif action == 1:
            self.position = [0, 1, 0]  # Neutral
        elif action == 0:
            self.position = [1, 0, 0]  # Sell

        #print("\n--- Actual Action Taken and Position Update ---")
        #print("Action Taken:", action)
        #print("Current Position Size (in dollars):", self.unrealized_pnl)
        #print("Position Indicator:", self.position)
        #print("Updated Portfolio Value:", self.portfolio_value)

def get_financial_data(stock_symbol, start_date, end_date):
    # Load historical stock data
    data = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)
    # Calculate RSI, 20-day moving average, and 50-day moving average
    data['RSI'] = calculate_rsi(data['Close'])
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()

    data = data.dropna()

    features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MA_20', 'MA_50']]
    normalized_data = z_score_normalization(features, period=96, clip_value=10)
    #print(data)
    return features, normalized_data, normalized_data.index

def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Testing function for the trained agent
def test_agent(env, drqn, episodes=10):
    total_profit = 0
    max_profit = 0
    max_profit_episode = 0
    for episode in range(episodes):
        state = env.reset()
        episode_profit = 0
        cumulative_returns = []
        done = False
        print(f"Starting Test Episode {episode + 1}")
        
        while not done:
            action = np.argmax(drqn.predict(state))
            next_state, reward, done, _ = env.step(action)
            episode_profit += reward
            action_name = ['Sell', 'Neutral', 'Buy'][action]
            cumulative_returns.append((env.current_step, episode_profit))  # Store step and cumulative profit as (x, y)
            state = next_state

        # Plot cumulative returns for the current episode
        x_vals, y_vals = zip(*cumulative_returns)  # Unpack cumulative returns into x and y lists
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label=f'Episode {episode + 1}')
        plt.xlabel('Trading Days')
        plt.ylabel('Cumulative Returns')
        plt.title(f'Cumulative Returns Over Time - Episode {episode + 1}')
        plt.legend()
        plt.show()

        print(f"Episode {episode + 1} finished with Total Profit: {episode_profit:.4f}")

        total_profit += episode_profit
        if episode_profit > max_profit:
            max_profit = episode_profit
            max_profit_episode = episode
        print(f"Test Episode {episode + 1} finished, Total Profit: {episode_profit:.4f}, Max Profit Obtained: {max_profit:.4f}, Episode for Max Profit: {max_profit_episode}")

def z_score_normalization(data, period=96, clip_value=10):
    # Calculate the rolling mean and rolling standard deviation with a window of 96 for each feature
    rolling_mean = data.rolling(window=period, min_periods=1).mean()
    rolling_std = data.rolling(window=period, min_periods=1).std()

    # Apply Z-score normalization
    z_normalized_data = (data - rolling_mean) / rolling_std

    # Clip the values to be within the range [-10, 10]
    z_normalized_data = z_normalized_data.clip(-clip_value, clip_value)

    return z_normalized_data

# Main training loop
if __name__ == "__main__":
    stock_symbol = 'SPY'
    raw_data, data, dates = get_financial_data(stock_symbol, start_date='2020-01-01', end_date='2023-12-31')
    train_data = data[dates <= '2023-01-01']
    test_data = data[dates > '2023-01-01']
    raw_train_data = raw_data[dates <= '2023-01-01']
    raw_test_data = raw_data[dates > '2023-01-01']
    train_env = TradingEnvironment(raw_train_data.values, train_data.values, train_data.index)
    test_env = TradingEnvironment(raw_test_data.values, test_data.values, test_data.index)

    state_size = train_data.shape[1] + 3
    action_size = train_env.action_space.n
    drqn = DRQN(state_size, action_size=action_size, learning_rate=0.00025, tau=0.001)

    replay_buffer = deque(maxlen=480)
    sequence_length = 96
    batch_size = 4
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    discount_factor = 0.99
    num_steps_per_episode = len(train_data)
    for episode in range(20):
        print(f"\nStarting Episode {episode + 1}")
        state = train_env.reset()
        total_reward = 0
        steps = 0
        episode_experiences = []
        start_index = random.randint(0, len(train_data) - sequence_length - 1)
        train_env.current_step = start_index
        print(f"Starting from index {start_index} in the data.")
        for step in range(sequence_length):
            if np.random.rand() < epsilon:
                action = train_env.action_space.sample()
            else:
                action = np.argmax(drqn.predict(state))
            hypothetical_rewards = [
                train_env.calculate_reward(
                action,
                open_price=train_env.data[train_env.current_step, 0],  # Corrected for 2D indexing
                transaction_cost= 500000 * 0.0000278 * 0.001
                ) for action in [0, 1, 2]
            ]

            # Print the selected action and hypothetical rewards
            print("Chosen Action:", action)
            print("Hypothetical Rewards:", hypothetical_rewards)

            next_state, reward, done, _ = train_env.step(action)
            total_reward += reward
            episode_experiences.append((state, action, reward, next_state, done, hypothetical_rewards))
            state = next_state
            #print("Total Rewards:", total_reward)
            if len(episode_experiences) >= sequence_length:
                for start in range(len(episode_experiences) - sequence_length + 1):
                    sequence = episode_experiences[start:start + sequence_length]
                    states, actions, rewards, next_states, dones, all_hypothetical_rewards = zip(*sequence)
                    replay_buffer.append((list(states), list(actions), list(rewards), list(next_states), list(dones), list(all_hypothetical_rewards)))
                    print(f"Replay Buffer Size: {len(replay_buffer)}")
        # Break if done is reached within the episode
        if done:
            print(f"Breaking due to 'done' at step {step + 1} in episode {episode + 1}")
            break

        if len(replay_buffer) > batch_size:
            minibatch = random.sample(replay_buffer, batch_size)
            loss = drqn.train_on_batch(minibatch, discount_factor)
            print(f"Episode {episode + 1}, Step {steps}, Loss: {loss.numpy() if not np.isnan(loss.numpy()) else 'NaN'}")

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        print(f"Episode {episode + 1} finished | Total Reward: {total_reward}, Epsilon: {epsilon:.5f}, Steps: {steps}")
        update_frequency = 1 
        if episode % update_frequency == 0:
            drqn.update_target_network()

    test_agent(test_env, drqn, episodes=1)




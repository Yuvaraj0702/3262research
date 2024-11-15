
# DRQN-based Trading System

This project implements a **Deep Recurrent Q-Network (DRQN)** for training a trading agent in a custom-built environment using historical stock market data. The DRQN model leverages LSTMs to handle sequential dependencies, making it suitable for time series tasks like trading.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Features](#features)
- [Usage](#usage)
- [Initialization](#initialization)
- [Training](#training)
- [Environment](#environment)
- [Model Architecture](#model-architecture)
- [Customization](#customization)
- [Ensemble replication](#ensemble-replication)
- [License](#license)

## Introduction
This code defines:
1. A **DRQN class** for training a trading agent with LSTM layers, designed to capture temporal dependencies.
2. A custom **TradingEnvironment** class that simulates market trading, with reward based on portfolio value changes and various transaction costs.

## Installation

To set up the environment and run the DRQN model, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Yuvaraj0702/3263research.git
   cd 3263research
   ```

## Dependencies
To use this project, you will need the following Python packages:
```python
numpy
pandas
tensorflow
gymnasium
yfinance
matplotlib
logging
```

To install dependencies:
```bash
pip install numpy pandas tensorflow gymnasium yfinance matplotlib
```

## Features
- **DRQN model** with LSTM-based architecture, using He Normal initialization and customized weight settings.
- **Trading Environment** in OpenAI Gym format, with features including:
  - Dynamic position updates.
  - Sinusoidal encoding of the day of the week for time awareness.
  - Customizable transaction costs and reward structure.


## Usage
To run the code: 
  ```bash
   python DRQN_replication.py
   ```

## Initialization
How the code set up objects and models before commencing training:
1. **Prepare Market Data:** Fetch historical data using `yfinance` or load a dataset with OHLCV information.
2. **Initialize Environment:** Set up `TradingEnvironment` with the required parameters like initial cash, trade size, and data.
3. **Initialize DRQN Agent:** Use the `DRQN` class by providing state size, action size, learning rate, etc.


## Training
How the agent is trained:
1. Gather a set of episodes where the agent interacts with the `TradingEnvironment`.
2. Save experience tuples and use them in batches for training with the `train_on_batch` method in `DRQN`.

## Environment
### `TradingEnvironment`
A custom trading environment that:
- Observes an agent's action to **Buy**, **Sell**, or **Hold**.
- Computes transaction costs, rewards based on portfolio returns.
- Includes sinusoidal encoding for day-of-the-week to account for temporal cycles.

### Observation Space
The state includes market data features and additional position indicators:
- `[OHLCV, Position, Day Encoding]`

### Action Space
Actions:
- `0` - Sell
- `1` - Hold
- `2` - Buy

## Model Architecture
The DRQN model consists of:
- Two dense layers with ELU activations.
- An LSTM layer with identity initialization for recurrent kernel and He Normal for kernel.
- Output layer for Q-values with a linear activation.

```plaintext
Input -> Dense(256, ELU) -> Dense(256, ELU) -> LSTM(256) -> Dense(action_size, Linear)
```

## Customization
### Hyperparameters
- Adjust `learning_rate`, `tau`, and layer dimensions in the DRQN model.
- Modify transaction costs in `TradingEnvironment`.

### Environment
The environment is built to be flexible and can be adjusted for different financial instruments, trade sizes, and portfolio configurations.

## Ensemble replication
Run the file Ensemble_replication.ipynb on jupyterhub to see the results of the ensembling model.

## License
This project is licensed under the MIT License.

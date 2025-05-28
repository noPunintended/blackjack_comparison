import numpy as np
import random
# from blackjack_env import ACTION_HIT, ACTION_STAND # Import if in separate file

class RLPlayerAgent:
    def __init__(self, action_space_size, observation_space_shape, learning_rate=0.01, discount_factor=0.99, epsilon=1.0, epsilon_decay_rate=0.0001, min_epsilon=0.01):
        self.action_space_size = action_space_size
        self.observation_space_shape = observation_space_shape # e.g., (32, 11, 2)

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon

        # Initialize Q-table with zeros
        # The size of Q_table will be (player_sum_range, dealer_up_card_range, usable_ace_range, action_space_size)
        # Note: Player sum goes up to 21 (or 31 if multiple 10s and an Ace), so a range of 32 covers it.
        # Dealer up card values are 1-10 (Ace is 1, 10/J/Q/K are 10), so range of 11 (index 0 unused or map Ace to 0).
        # Let's map Ace to 1, and 2-10 as their values for dealer upcard.
        # So, player_sum_range: up to 31 (index 0-31), dealer_up_card_range: 1-10 (index 0-10, can map Ace to 10 as per obs)
        # However, our env's _get_obs already maps Ace to 1 for dealer_up_card_value, so 1-10 works fine for obs.
        self.q_table = np.zeros(self.observation_space_shape + (self.action_space_size,))

    def choose_action(self, state):
        """
        Epsilon-greedy policy to choose an action.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            return random.randint(0, self.action_space_size - 1)
        else:
            # Exploit: choose the action with the highest Q-value for the current state
            # state is a tuple (player_sum, dealer_up_card_value, usable_ace_bool)
            # Need to convert boolean to int for indexing
            player_sum, dealer_up_card_value, usable_ace = state
            
            # Ensure valid indices for Q-table
            # player_sum could go above 21 before busting (e.g., 20+10=30), cap it at max observed index if needed.
            # Assuming observation_space_shape[0] covers max player_sum observed in state
            player_sum = min(player_sum, self.observation_space_shape[0] - 1) 
            dealer_up_card_value = min(dealer_up_card_value, self.observation_space_shape[1] - 1)


            return np.argmax(self.q_table[player_sum, dealer_up_card_value, usable_ace])

    def learn(self, state, action, reward, next_state, terminated):
        """
        Update the Q-value for the given state-action pair using the Q-learning formula.
        """
        player_sum, dealer_up_card_value, usable_ace = state
        next_player_sum, next_dealer_up_card_value, next_usable_ace = next_state

        # Ensure valid indices for Q-table
        player_sum = min(player_sum, self.observation_space_shape[0] - 1)
        dealer_up_card_value = min(dealer_up_card_value, self.observation_space_shape[1] - 1)
        next_player_sum = min(next_player_sum, self.observation_space_shape[0] - 1)
        next_dealer_up_card_value = min(next_dealer_up_card_value, self.observation_space_shape[1] - 1)

        # Q(S,A)
        current_q = self.q_table[player_sum, dealer_up_card_value, usable_ace, action]

        if terminated:
            # If terminal state, no future reward
            target_q = reward
        else:
            # Q(S', A') = max Q-value for the next state
            max_future_q = np.max(self.q_table[next_player_sum, next_dealer_up_card_value, next_usable_ace])
            target_q = reward + self.gamma * max_future_q
        
        # Q-learning update rule
        self.q_table[player_sum, dealer_up_card_value, usable_ace, action] = current_q + self.lr * (target_q - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_rate)

    def get_policy(self):
        """
        Returns the optimal policy derived from the Q-table (for visualization).
        """
        policy = np.zeros(self.observation_space_shape, dtype=int)
        for ps in range(self.observation_space_shape[0]):
            for du in range(self.observation_space_shape[1]):
                for ua in range(self.observation_space_shape[2]):
                    policy[ps, du, ua] = np.argmax(self.q_table[ps, du, ua])
        return policy
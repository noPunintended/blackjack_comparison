import numpy as np
from env.blackjack import BlackjackEnv
from agents.det_dealer import _dealer_play
from agents.player import RLPlayerAgent

env = BlackjackEnv(num_decks=6, dealer_hits_on_soft_17=True)

# Initialize RL agent
agent = RLPlayerAgent(
    action_space_size=env.action_space.n,
    observation_space_shape=env.observation_space.shape,
    learning_rate=0.01,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_decay_rate=0.000001,  # Smaller decay for more episodes
    min_epsilon=0.1
)

num_episodes = 500000  # Number of games to train for
rewards_per_episode = []

print(f"\n--- Starting RL Agent Training for {num_episodes} episodes ---")

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    episode_reward = 0

    # Handle immediate natural blackjacks from reset for reward calculation
    if 'outcome' in info and (info['outcome'] == 'player_blackjack' or
                            (info['outcome'] == 'push' and len(env.player_hand) == 2 and len(env.dealer_hand) == 2)):

        # The reward for natural blackjack is already set in env.reset
        # No action taken by agent, so no Q-update, just record reward
        episode_reward = env.natural_blackjack_payout if info['outcome'] == 'player_blackjack' else 0
        done = True

    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        # Learn from the experience
        agent.learn(state, action, reward, next_state, terminated)

        state = next_state
        episode_reward += reward
        done = terminated or truncated

    agent.decay_epsilon()
    rewards_per_episode.append(episode_reward)

    if (episode + 1) % 10000 == 0:
        avg_reward = np.mean(rewards_per_episode[-10000:])
        print(f"Episode {episode + 1}/{num_episodes}, Avg Reward (last 10k): {avg_reward:.4f}, Epsilon: {agent.epsilon:.4f}")
        # Optional: Evaluate the agent's win rate periodically
        # You would run a separate evaluation loop without epsilon-greedy policy

        print("\n--- Training Complete ---")

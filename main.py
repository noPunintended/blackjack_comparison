from env.blackjack import BlackjackEnv
from agents.det_dealer import _dealer_play

if __name__ == "__main__":
    env = BlackjackEnv(num_decks=6, dealer_hits_on_soft_17=True) 
    player_input = input("Do you want to play as the user player? (y/n): ").lower()
    if player_input != 'y':
        player = False
    else:
        player = True
    num_games = 0
    total_reward = 0

    # Define actions locally for the __main__ block if they aren't imported or globally accessible.
    # Alternatively, you could reference them as env.ACTION_HIT if they were class attributes,
    # but as currently defined, they are global constants.
    # The safest bet is to redefine them or import them if the RLPlayerAgent was in a separate file.
    # For now, let's redefine them for clarity in this block:
    LOCAL_ACTION_STAND = 0
    LOCAL_ACTION_HIT = 1

    if player:
        while True:
            num_games += 1
            print(f"\n--- Starting Game {num_games} (User Player) ---")
            observation, info = env.reset()
            
            env.last_reward = 0
            if info.get('outcome') == 'player_blackjack':
                env.last_reward = env.natural_blackjack_payout
            elif info.get('outcome') == 'push' and len(env.player_hand) == 2 and len(env.dealer_hand) == 2:
                env.last_reward = 0

            env.render()

            done = False
            if 'outcome' in info: 
                done = True
                print(f"--- Game Over! Outcome: {info['outcome']} (Reward: {env.last_reward}) ---")
                total_reward += env.last_reward
                print(f"Total Reward across games: {total_reward}")

            while not done:
                player_sum, dealer_upcard, usable_ace = observation
                print(f"Your Hand: {player_sum} (Usable Ace: {'Yes' if usable_ace else 'No'}) | Dealer Up: {dealer_upcard}")
                
                action_choice = input("Enter 'h' for Hit or 's' for Stand: ").lower()
                
                if action_choice == 'h':
                    action = LOCAL_ACTION_HIT  # Use the locally defined action constant
                elif action_choice == 's':
                    action = LOCAL_ACTION_STAND # Use the locally defined action constant
                else:
                    print("Invalid input. Please enter 'h' or 's'.")
                    continue

                observation, reward, terminated, truncated, info = env.step(action)
                env.last_reward = reward
                env.render()
                done = terminated or truncated
            
            if not (hasattr(env, 'outcome') and env.outcome in ['player_blackjack', 'push'] and len(env.player_hand) == 2 and len(env.dealer_hand) == 2):
                total_reward += reward
            print(f"Total Reward across games: {total_reward}")

            play_again = input("Play another game? (y/n): ").lower()
            if play_again != 'y':
                break

    else:
        
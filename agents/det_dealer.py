def _dealer_play(self):
        """
        Dealer's fixed strategy: hit on 16 or less, stand on 17 or more.
        Handles Soft 17 rule based on `self.dealer_hits_on_soft_17`.
        """
        while True:
            dealer_sum, usable_ace = self._get_hand_value(self.dealer_hand)

            # 1. Dealer busts: Game over for dealer
            if dealer_sum > 21:
                break
            
            # 2. Dealer stands on hard 17 or more
            if dealer_sum > 17:
                break
            
            # 3. Handle Soft 17 rule
            if dealer_sum == 17:
                if usable_ace and self.dealer_hits_on_soft_17:
                    # If dealer_hits_on_soft_17 is True (e.g., Las Vegas Strip rules), dealer hits on Soft 17
                    self.dealer_hand.append(self._deal_card())
                else:
                    # If dealer_hits_on_soft_17 is False (e.g., Reno rules), dealer stands on Soft 17
                    # This also covers hard 17
                    break
            
            # 4. Dealer hits on 16 or less (or on Soft 17 if applicable from point 3)
            elif dealer_sum < 17:
                self.dealer_hand.append(self._deal_card())
            else: 
                # This 'else' should theoretically not be hit if the above conditions are exhaustive.
                # It's a defensive break.
                break
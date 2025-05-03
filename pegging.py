class Pegging:
    """ The history of cards played during the pegging phase of a hand. """

    def __init__(self):
        """ Creates an empty pegging history. """
        self._prev_round = None
        self._prev_play = None
        self._card = None
        self._player = None
        self._passed = [False, False]
        self._total = 0
        self._hash = 0
        

    def play(self, game, card, player):
        """ Returns the pegging history and score resulting from the given player
            playing the given card after this history.

            game -- a Cribbage game
            card -- a legal card to play for the given player
            player -- 0 or 1 for the dealer or other player, respectively
        """
        result = Pegging()

        if card is None:
            result._hash = self._hash * 3
            if self._passed[1 - player]:
                # both players have passed
                result._prev_round = self
                result._player = player
            else:
                result._passed = self._passed[:]
                result._passed[player] = True
                result._player = player
                result._total = self._total
                result._prev_round = self._prev_round
                result._prev_play = self
        else:
            result._hash = (self._hash * 7 + card.rank()) * 19 + (ord(card.suit()) - ord("A"))
            result._card = card
            result._player = player
            result._total = self._total + game.rank_value(card.rank())
            result._prev_round = self._prev_round
            result._prev_play = self
            result._passed = self._passed[:]

        return result, self.score(game, card, player)


    def is_start_round(self):
        """ Determines if this pegging history represents the start
            of a round of pegging.
        """
        return self._total == 0
    

    def total_points(self):
        """ Returns the current total points in the current round of
            pegging in this history.
        """
        return self._total


    def has_passed(self, player):
        """ Determines if this given player has passed in the current
            round of pegging in this history.

            player -- 0 or 1 for the dealer and non-dealer respectively
        """
        return self._passed[player]


    def is_legal(self, game, card, player):
        """ Determines if it is legal for the player to play the given
            card.  This assumes the player has the card available to play.

            game -- a cribbage game
            card -- a card
            player -- 0 for the dealer or 1 for the non-dealer
        """
        return self._total + game.rank_value(card.rank()) <= game.pegging_limit()


    def has_legal_play(self, game, hand, player):
        """ Determines if the given hand contains a legal play for the given
            player.

            game -- a cribbage game
            hand -- an interable over cards
            player -- 0 for the dealer or 1 for the non-dealer
        """
        if self._passed[player]:
            return False
        else:
            for card in hand:
                if self.is_legal(game, card, player):
                    return True
            return False
            
    
    def score(self, game, card, player):
        """ Returns the score earned by the given player when playing the given card
            (or None for "go").
            The score is negative to indicate that the other player scores points (as for a "go")
            and None if the play is illegal.

            game -- a Cribbage game
            card -- a (rank, suit) pair, or None
            player -- 0 or 1 for the dealer and non-dealer respectively
        """
        if card is None:
            if self._passed[player]:
                # pass after "go"
                return 0
            elif not self._passed[1 - player]:
                if self._total == game.pegging_limit():
                    # pass because limit reached
                    return 0
                else:
                    # first pass this round; "go" for other player
                    return -1
            else:
                # current player finally passes after "go"
                return 0
        else:
            if self._passed[player]:
                # player already passed in this round but is playing a card
                return None

        if self._total + game.rank_value(card.rank()) > game.pegging_limit():
            # played card makes total too high
            return None

        count = 1
        curr_matches = 1
        max_matches = 1
        max_straight = 1
        min_rank = card.rank()
        max_rank = card.rank()
        curr = self
        doubles = False
        ranks_seen = set()
        ranks_seen.add(card.rank())
        while curr is not None and (count == 0 or curr_matches == max_matches or not doubles):
            if curr._card is not None:
                count += 1
                
                if curr._card.rank() == card.rank():
                    # ranks match
                    if curr_matches != -1:
                        curr_matches += 1
                    max_matches = max(max_matches, curr_matches)
                else:
                    # ranks don't match
                    curr_matches = -1

                if curr._card.rank() in ranks_seen:
                    doubles = True
                else:
                    ranks_seen.add(curr._card.rank())

                min_rank = min(curr._card.rank(), min_rank)
                max_rank = max(curr._card.rank(), max_rank)
                if max_rank - min_rank + 1 == count and not doubles:
                    max_straight = count


            curr = curr._prev_play

        pair_score = game.peg_pair_value(max_matches)
        straight_score = game.peg_straight_value(max_straight)
        fifteen_score = game.peg_sum_value(self._total + game.rank_value(card.rank()))
        if self._total + game.rank_value(card.rank()) == game.pegging_limit():
            thirtyone_score = game.pegging_exact_value(self._passed[1 - player])
        else:
            thirtyone_score = 0
        
        return pair_score + straight_score + fifteen_score + thirtyone_score


    def plays(self):
        history = []
        curr_round = []
        curr = self
        while curr is not None:
            if curr._player is not None:
                curr_round.append((curr._player, curr._card))
            if curr._prev_play is None:
                if len(curr_round) > 0:
                    history.append(list(reversed(curr_round)))
                curr = curr._prev_round
                curr_round = []
            else:
                curr = curr._prev_play
        if len(curr_round) > 0:
            history.append(list(reversed(curr_round)))

        return list(reversed(history))


    def __hash__(self):
        """ Returns the hash code for this history.
        """
        return self._hash


    def __eq__(self, other):
        if self._card is None and other._card is not None:
            return False
        elif self._card is not None and other._card is None:
            return False
        else:
            if self._card is not None and not (self._card == other._card):
                return False

            # cards are the same (including possibly both None)
            
            if self._prev_play is not None and other._prev_play is None:
                return False
            elif self._prev_play is None and other._prev_play is not None:
                return False
            elif self._prev_play is not None:
                return self._prev_play == other._prev_play
            else:
                if self._prev_round is None and other._prev_round is not None:
                    return False
                elif self._prev_round is not None and other._prev_round is None:
                    return False
                elif self._prev_round is not None:
                    return self._prev_round == other._prev_round
                else:
                    return True
                
            
    def __repr__(self):
        if self._card is None and self._prev_round is None and self._prev_play is None:
            return ""
        else:
            if self._prev_play is not None:
                start = str(self._prev_play)
                end = ""
            else:
                start = str(self._prev_round)
                end = " | "
            sep = "," if len(start) > 0 and start[-1] != " " else ""
            return "".join([start, sep, str(self._player), ":", str(self._card), end])
        

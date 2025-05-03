from game import Game, State
import cribbage
from pegging import Pegging


class PeggingGame(Game):
    def __init__(self, cards):
        """ Creates a cribbage pegging game where each player is dealt the
            given number of cards.

            cards -- a positive integer
        """
        self._cards = cards
        self._game = cribbage.Game()


    def initial_state(self):
        deck = self._game.deck();
        deck.shuffle()
        p0_hand = deck.deal(self._cards)
        p1_hand = deck.deal(self._cards)
        return PeggingGame.State(self._game, p0_hand, p1_hand, Pegging())


    class State(State):
        """ A state in a cribbage pegging game.
        """
        def __init__(self, game, p0, p1, hist):
            self._game = game
            self._cards = [tuple(p0), tuple(p1)]
            self._history = hist
            self._turn = 1
            self._score = [0, 0]


        def __hash__(self):
            return self._history.__hash__() ^ self._cards[0].__hash__() ^ self._cards[1].__hash__()


        def __eq__(self, other):
            return (self._history == other._history
                    and self._cards[0] == other._cards[0]
                    and self._cards[1] == other._cards[1])


        def __repr__(self):
            return str(self._history) + " // " + str(self._cards) + str(self._score)
        
        
        def is_terminal(self):
            return sum(len(cards) for cards in self._cards) == 0


        def payoff(self):
            return self._score[0] - self._score[1]


        def score(self):
            return self._score[:]


        def actor(self):
            return self._turn


        def get_actions(self):
            cards = [c for c in self._cards[self._turn]
                     if self._history.is_legal(self._game, c, self._turn)]
            if len(cards) == 0:
                return [None]
            else:
                return cards

            
        def is_legal(self, card):
            if card is None:
                return not self._history.has_legal_play(self._game, self._cards[self._turn], self._turn)
            else:
                return self._history.is_legal(self._game, card, self._turn)


        def successor(self, action):
            # update cards available
            remaining = self._cards[:]
            if action is not None:
                remaining[self._turn] = tuple(c for c in self._cards[self._turn] if not (c == action))
            new_history, pts = self._history.play(self._game, action, self._turn)
            # make new state
            succ = PeggingGame.State(self._game, *remaining, new_history)

            # update score in new state
            succ._score = self._score[:]
            if pts > 0:
                succ._score[self._turn] += pts
            elif pts < 0:
                # negative score means points to other player
                succ._score[1 - self._turn] += -pts

            # update turn in new state
            succ._turn = 1 - self._turn

            return succ



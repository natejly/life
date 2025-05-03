from abc import ABC, abstractmethod

class Game(ABC):
    """ A two-player zero-sum perfect information game.
    """
    
    @abstractmethod
    def initial_state(self):
        """ Returns the initial state of this game.
        """
        pass
    
    
class State(ABC):
    """ A state in a game.
    """
    
    @abstractmethod
    def is_terminal(self):
        """ Determines if this state is terminal.  Return value is true is so and false otherwise.

            self -- a state
        """
        pass


    def payoff(self):
        """ Returns the payoff for player 0 at this terminal state.

            self -- a terminal state
        """
        return 0.0      # default is a draw


    @abstractmethod
    def actor(self):
        """ Determines which player is the actor in this nonterminal state.

            self -- a nonterminal state
        """
        pass

    
    @abstractmethod
    def get_actions(self):
        """ Returns a list of possible actions in this nonterminal state.
            The representation of each state is left to the implementation.

            self -- a nonterminal state
        """
        pass


    def is_legal(self, action):
        """ Determines if the given action is legal in this state.

            self -- a state
            action -- an action
        """
        return False
    
    
    @abstractmethod
    def successor(self, action):
        """ Returns the state that results from the given action in this nonterminal state.

            self -- a nonterminal state
            action -- one of the actions in the list returned by get_actions for this state
        """
        pass

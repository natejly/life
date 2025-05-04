# Revised code from test_mcts.py from pset 4
import mcts
import time
import math
C = math.sqrt(2)
import random
from peg_game import PeggingGame
from mcts import mcts_policy
import minimax as minimax
class TESTAGENT():
    def __init__(self, time_limit=None, constant=None, rollouts=None, max_depth=None, weight=None, decay=None):
        self.time_limit = time_limit
        self.constant = constant
        self.rollouts = rollouts
        self.max_depth = max_depth
        self.weight = weight
        self.decay = decay

        self.policy_generator = mcts_policy(
            limit=self.time_limit,
            constant=self.constant, 
            rollouts=self.rollouts, 
            max_depth=self.max_depth, 
            weight=self.weight, 
            decay=self.decay
        )
    
    def policy(self):
        return self.policy_generator


def compare_policies(game, p1, p2, games):
    p1_wins = 0
    p2_wins = 0
    p1_score = 0

    for i in range(games):
        p1_policy = p1()
        p2_policy = p2()
        position = game.initial_state()
        
        while not position.is_terminal():
            if position.actor() == i % 2:
                move = p1_policy(position)
            else:
                move = p2_policy(position)
            position = position.successor(move)

        p1_score += position.payoff() * (1 if i % 2 == 0 else -1)
        if position.payoff() == 0:
            p1_wins += 0.5
            p2_wins += 0.5
        elif (position.payoff() > 0 and i % 2 == 0) or (position.payoff() < 0 and i % 2 == 1):
            p1_wins += 1
        else:
            p2_wins += 1
            
    return p1_score / games, p1_wins / games


def test_game(game, count, p1_policy_fxn, p2_policy_fxn):

    margin, wins = compare_policies(game, p1_policy_fxn, p2_policy_fxn, count)

    # print("NET: ", margin, "; WINS: ", wins, sep="")
    return margin, wins

if __name__ == '__main__':
    time_limit = 0.005
    game = PeggingGame(4)
    num_games = 10000
    h = (lambda pos: pos.score()[0] - pos.score()[1])
    time_limit = 0.005
    default = TESTAGENT(time_limit=0.005, constant=C, rollouts=50, max_depth=4, weight=1, decay=1)
    population = TESTAGENT(time_limit=0.005, constant=1.044465166456834, rollouts=125.23529411764706, max_depth=2.6176470588235294, weight=0.7355062877045861, decay=1.0833588282288886)

    margin, wins = compare_policies(game, population.policy, default.policy, num_games)
    print("Population vs Default")
    print("NET: ", margin, "; WINS: ", wins, sep="")
    margin, wins = compare_policies(game, default.policy, lambda: minimax.minimax_policy(14, minimax.Heuristic(h)), num_games)
    print("Default vs Minimax")
    print("NET: ", margin, "; WINS: ", wins, sep="")
    margin, wins = compare_policies(game, population.policy, lambda: minimax.minimax_policy(14, minimax.Heuristic(h)), num_games)
    print("Default vs Minimax")
    print("NET: ", margin, "; WINS: ", wins, sep="")
    


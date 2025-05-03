# Revised code from test_mcts.py from pset 4

import mcts
import time
import math
C = math.sqrt(2)
import random
from peg_game import PeggingGame


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

    print("NET: ", margin, "; WINS: ", wins, sep="")
    return margin, wins
    

if __name__ == '__main__':
    start = time.time()
    game = PeggingGame(4)
    num_games = 1000
    mctspolicy1 = mcts.mcts_policy
    mctspolicy2 = mcts.mcts_policy
    test_game(game, num_games, mctspolicy1, mctspolicy2)
    print("Time taken: ", time.time() - start)
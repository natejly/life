import itertools as it
import random

def greedy_throw(game, deal, crib):
    """ Returns a greedy choice of which cards to throw.  The greedy choice
        is determined by the score of the four cards kept and the two cards
        thrown in isolation, without considering what the turned card
        might be or what the opponent might throw to the crib.  If multiple
        choices result in the same net score, then one is chosen randomly.

        game -- a Cribbage game
        deal -- a list of the cards dealt
        crib -- 1 for owning the crib, -1 for opponent owning the crib
    """
    def score_split(indices):
        keep = []
        throw = []
        for i in range(len(deal)):
            if i in indices:
                throw.append(deal[i])
            else:
                keep.append(deal[i])
        return keep, throw, score(game, keep, None, False)[0] + crib * score(game, throw, None, True)[0]

    throw_indices = game.throw_indices()
    
    # to randomize the order in which throws are considered to have the effect
    # of breaking ties randomly
    random.shuffle(throw_indices)

    # pick the (keep, throw, score) triple with the highest score
    return max(map(lambda i: score_split(i), throw_indices), key=lambda t: t[2])
        

def score(game, hand, turn, crib):
    """ Returns the score for the given hand and turn card.  The score
        is returned as a six-element list with the total score in the
        first element and the pairs, 15s, runs, flushes, and nobs subscores
        in the remaining elements in that order.

        game -- a cribbage game 
        hand -- a list of cards
        turn -- a card, or None
        crib -- true to score by crib scoring rules
    """
    rank_count = {rank: 0 for rank in game.all_ranks()}
    suit_count = {suit: 0 for suit in game.all_suits()}
    value_count = {value: 0 for value in game.all_values()}
    
    all_cards = list(hand) + ([] if turn is None else [turn])
    for card in all_cards:
        rank_count[card.rank()] += 1
        suit_count[card.suit()] += 1
        value_count[game.rank_value(card.rank())] += 1

    fifteens = 0
    for size in range(2, 6):
        for cards in it.combinations(all_cards, size):
            if sum(game.rank_value(c.rank()) for c in cards) == 15:
                fifteens += 1
    fifteens *= game.fifteen_value()
        
    pairs = 0
    for r in game.all_ranks():
        pairs += len(list(it.combinations(range(rank_count[r]), 2)))
    pairs *= game.pair_value()
    
    straights = 0
    curr_run = 0
    combinations = 1
    for r in game.all_ranks():
        if rank_count[r] == 0:
            straights += game.straight_value(curr_run, combinations)
            curr_run = 0
            combinations = 1
        else:
            curr_run += 1
            combinations *= rank_count[r]
    straights += game.straight_value(curr_run, combinations)

    flushes = 0
    max_flush = max(suit_count[s] for s in game.all_suits())
    if max_flush == len(hand) and not crib and (turn is None or turn.suit() != hand[0].suit()):
        # flush in hand does not match turn card
        flushes += game.hand_flush_value(len(hand))
    elif max_flush == len(hand) + 1:
        # flush using turn card
        flushes += game.turn_flush_value(max_flush)

    nobs = sum(game.nob_value(card, turn) for card in hand)
        
    return pairs + fifteens + straights + flushes + nobs, pairs, fifteens, straights, flushes, nobs

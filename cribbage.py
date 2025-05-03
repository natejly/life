import itertools as it
import sys
import copy

import scoring
from deck import Deck,Card
from pegging import Pegging

class Game:
    def __init__(self):
        # make this a tuple so it is immutable and so doesn't need to be copied
        # in the accessor?
        self._throw_indices = list(it.combinations(range(self.keep_cards() + self.throw_cards()), self.throw_cards()))

        
    def all_ranks(self):
        return range(1, 14)

    
    def all_suits(self):
        return ['S', 'H', 'D', 'C']


    def rank_value(self, rank):
        """ Returns the pegging value of the given rank in this game.

            rank -- an integer from 1 (A) to 13 (K)
        """
        return min(rank, 10)

    
    def all_values(self):
        return range(1, 11)

    
    def fifteen_value(self):
        return 2

    
    def pair_value(self):
        return 2


    def turn_card_value(self, card):
        return 2 if card.rank() == 11 else 0
    
    
    def straight_value(self, length, count):
        if length >= 3:
            return length * count
        else:
            return 0


    def nob_value(self, card, turn):
        if turn is not None and card.rank() == 11 and card.suit() == turn.suit():
            return 1
        else:
            return 0
        
    
    def hand_flush_value(self, size):
        if size >= 4:
            return size
        else:
            return 0


    def turn_flush_value(self, size):
        if size >= 5:
            return size
        else:
            return 0


    def keep_cards(self):
        return 4


    def throw_cards(self):
        return 2


    def throw_indices(self):
        return self._throw_indices[:]
    
    
    def pegging_limit(self):
        return 31


    def pegging_exact_value(self, go):
        return 1 if go else 2

        
    def peg_pair_value(self, count):
        if count < 2:
            return 0
        elif count == 2:
            return 2
        elif count == 3:
            return 6
        elif count == 4:
            return 12


    def peg_straight_value(self, length):
        return 0 if length < 3 else length

        
    def peg_sum_value(self, total):
        return 2 if total == 15 else 0
    

    def winning_score(self):
        return 121


    def game_value(self, p0_score, p1_score):
        """ Returns the point value of the game ending with the
            given score.  Positive indicates points won by P0;
            negative indicates points won by P1.  If neither player
            has reached the threshold required to win, then the
            return value is 0.

            p0_score -- an integer
            p1_score -- an integer
        """
        if max(p0_score, p1_score) < self.winning_score():
            # no winner yet
            return 0
        else:
            loser = min(p0_score, p1_score)
            if loser <= 60:
                # double skunk
                points = 3
            elif loser <= 90:
                # skunk
                points = 2
            else:
                points = 1

            return points * (1 if p0_score > p1_score else -1)

        
    def deck(self):
        return Deck(self.all_ranks(), self.all_suits(), 1)
    
        
    def deal(self, count):
        deck = Deck(self.all_ranks(), self.all_suits(), 1)
        deck.shuffle()
        return deck.deal(count)


    def is_legal_split(self, hand, split):
        """ Determines if split is a partition of hand.

            hand -- an iterable over cards
            split -- an iterable over iterables of cards
        """
        # count cards in hand
        card_count = dict()
        for c in hand:
            if c not in card_count:
                card_count[c] = 1
            else:
                card_count[c] += 1

        # match cards in partition with cards in hand
        part_size = 0
        for part in split:
            part_size += len(part)
            for c in part:
                if c not in card_count or card_count[c] == 0:
                    # card in partition not in hand, or too many in partition
                    return False
                card_count[c] -= 1

        # if no problems yet, everything must have matched
        return part_size == len(hand)
    

    def play(self, p0_policy, p1_policy, log):
        scores = [0, 0]
        policies = [p0_policy, p1_policy]
        dealer = 0
        handsPlayed = 0

        # loop until one player has points to win
        while max(scores) < self.winning_score():
            log("Score: " + str(scores))
            # deal cards
            per_player = self.keep_cards() + self.throw_cards()
            in_play = self.deal(2 * per_player + 1)
            hands = [in_play[per_player * p : per_player * (p + 1)] for p in [0, 1]]
            handsPlayed += 1
            turn = in_play[-1]
            keeps = [policies[p].keep(copy.deepcopy(hands[p]),
                                      scores[:] if p == 0
                                               else list(reversed(scores)),
                                      dealer == p)
                     for p in [0, 1]]

            # check that policy's action is a partition of the hand
            for p in [0, 1]:
                if not self.is_legal_split(hands[p], keeps[p]):
                    raise Exception("split does not partition hand")
                if len(keeps[p][1]) != self.throw_cards():
                    raise Exception("discarded {n} cards".format(n=len(keeps[p][1])))
                
            scores[dealer] += self.turn_card_value(turn)
            log("Turn: " + str(turn) + " " + str(scores))

            # playing/pegging
            passes = [False, False]
            peg_cards = [k[0] for k in keeps]
            peg_turn = 1 - dealer
            history = Pegging()
            played = 0
            last_played = 0
            while max(scores) < self.winning_score() and (sum(len(cards) for cards in peg_cards) > 0 or not history.is_start_round()):
                if not passes[peg_turn]:
                    play = policies[peg_turn].peg(copy.deepcopy(peg_cards[peg_turn]), history, scores[:] if peg_turn == 0 else list(reversed(scores)), dealer == peg_turn)

                    # check legality of play
                    if play is None and history.has_legal_play(self, peg_cards[peg_turn], 0 if peg_turn == dealer else 1):
                        raise Exception("passing when has legal play")
                    elif play is not None and not history.is_legal(self, play, 0 if peg_turn == dealer else 1):
                        raise Exception("invalid card")
                    
                    if play is None:
                        passes[peg_turn] = True
                    else:
                        played += 1
                else:
                    play = None

                log(play)
                history, score = history.play(self, play, 0 if peg_turn == dealer else 1)
                if score > 0:
                    scores[peg_turn] += score
                    log(scores)
                elif score < 0:
                    scores[1 - peg_turn] += -score
                    log(scores)

                # remove played card from player's possible cards
                if play is not None:
                    new_cards = [card for card in peg_cards[peg_turn] if card != play]
                    if len(new_cards) == len(peg_cards[peg_turn]):
                        raise Exception("played card not in hand")
                    
                    peg_cards[peg_turn] = new_cards
                    
                # next player's turn
                peg_turn = 1 - peg_turn
                
                if sum(1 if passed else 0 for passed in passes) == 2:
                    # both players passed; reset for next round of pegging
                    passes = [False, False]
                    last_played = played
                    played = 0

            # score non-dealer's hand
            if max(scores) < self.winning_score():
                hand_score = scoring.score(self, keeps[1 - dealer][0], turn, False)
                log("NON-DEALER: " + str(keeps[1 - dealer][0]) + " " + str(hand_score))
                scores[1 - dealer] += hand_score[0]
                log(scores)

            # score dealer's hand
            if max(scores) < self.winning_score():
                hand_score = scoring.score(self, keeps[dealer][0], turn, False)
                log("DEALER: " + str(keeps[dealer][0]) + " " + str(hand_score))
                scores[dealer] += hand_score[0]
                log(scores)

            # score crib
            if max(scores) < self.winning_score():
                crib = keeps[dealer][1] + keeps[1 - dealer][1]
                hand_score = scoring.score(self, crib, turn, True)
                log("CRIB: " + str(crib) + str(hand_score))
                scores[dealer] += hand_score[0]
                log(scores)
            
            dealer = 1 - dealer
            
        return self.game_value(*scores), handsPlayed


def peg(game, list):
    history = Pegging()
    player = 1
    for play in list:
        history, score = history.play(game, play, player)
        print(play, score)
        player = 1 - player


def evaluate_policies(game, p0_policy, p1_policy, count):
    scores = dict()
    p0_total = 0
    p1_total = 0
    total_hands = 0
    for g in range(count):
        if g % 2 == 0:
            results = game.play(p0_policy, p1_policy, lambda mess: None)
            p0_pts = results[0]
        else:
            results = game.play(p1_policy, p0_policy, lambda mess: None)
            p0_pts = -results[0]
        if p0_pts not in scores:
            scores[p0_pts] = 0
        scores[p0_pts] += 1
        if p0_pts > 0:
            p0_total += p0_pts
        else:
            p1_total += -p0_pts
            
        total_hands += results[1]

    return (p0_total - p1_total) / count, p0_total / count, p1_total / count, scores, total_hands / count

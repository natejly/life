import itertools as it
import random
class Card:
    rank_str = [None, "A"] + [str(n) for n in range(2, 10)] + ["T", "J", "Q", "K"]
    def __init__(self, rank, suit):
        """ Creates a card of the given rank and suit.

            rank -- an integer
            suit -- a character
        """
        self._rank = rank
        self._suit = suit
        self._hash = str(self).__hash__()
            

    def rank(self):
        return self._rank


    def suit(self):
        return self._suit


    def same_suit(self, other):
        return self._suit == other._suit


    def __repr__(self):
        return "[" + Card.rank_str[self._rank] + str(self._suit) + "]"


    def __eq__(self, other):
        return self._rank == other._rank and self._suit == other._suit


    def __hash__(self):
        return self._hash


class Deck:
    def __init__(self, ranks, suits, copies):
        """ Creates a deck of cards including the given number of copies
            of each possible combination of the given ranks and the
            given suits.

            ranks -- an iterable of integers
            suits -- an iterable
            copies -- a nonnegative integer
        """
        self._cards = []
        for copy in range(copies):
            self._cards.extend(map(lambda c: Card(*c), it.product(ranks, suits)))
            
    def shuffle(self):
        """ Shuffles this deck. """
        random.shuffle(self._cards)


    def size(self):
        """ Returns the number of cards remaining in this deck. """
        return len(self._cards)
    

    def deal(self, n):
        """ Removes and returns the next n cards from this deck.

            n -- an integer between 0 and the size of this deck (inclusive)
        """
        dealt = self._cards[-n:]
        dealt.reverse()
        del self._cards[-n:]
        return dealt


    def peek(self, n):
        """ Returns the next n cards from this deck without removing them.

            n -- an integer between 0 and the size of this deck (inclusive)
        """
        dealt = self._cards[-n:]
        dealt.reverse()
        return dealt

            
    def remove(self, cards):
        """ Removes the given cards from this deck.  If there is a card
            to remove that isn't present in this deck, then the effect is
            the same as if that card had not been included in the list to
            remove.  If there are multiple occurrences of a given card
            in the list to remove, then the corresponding number of occurrences
            of that card in this deck are removed.

            cards -- an iterable over Cards
        """
        counts = dict()
        for card in cards:
            if card not in counts:
                counts[card] = 0
            counts[card] += 1

        remaining = []
        for card in self._cards:
            if card in counts and counts[card] > 0:
                counts[card] -= 1
            else:
                remaining.append(card)
        self._cards = remaining

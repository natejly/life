# modified from pset 4
class Heuristic:
    ''' A wrapper for a heuristic function that counts how many times the
        heuristic is called.
    '''
    def __init__(self, h):
        ''' Creates a wrapper for the given function.

            h -- a heuristic function that takes a game position and returns its heiristic value,
                 or its actual value if the position is terminal.
        '''
        self.calls = 0
        self.heuristic = h
        self.inf = float("inf") 

        
    def evaluate(self, pos):
        ''' Returns the underlying heuristic applied to the given position.

            pos -- a game position
        '''
        # calls on terminal positions don't count
        if not pos.is_terminal():
            self.calls += 1
        return self.heuristic(pos)

    
    def count_calls(self):
        ''' Returns the number of times this heiristic has been called.
        '''
        return self.calls


def seeds_stored_heuristic(pos):
    ''' A simple heuristic for Kalah.  Returns the difference in the number of seeds
        in P1's store vs. P2's store (P1 - P2) unless the position is terminal,
        in which case it returns +/- total seeds in the game (positive for P1 win,
        negative for P2 win).

        pos -- a Kalah position
    '''
    if pos.is_terminal():
        value = pos.payoff() * (pos._seeds_stored(0) + pos._seeds_stored(1))
    else:
        value =  pos._seeds_stored(0) - pos._seeds_stored(1)
    return value


def seeds_stored_heuristic_soft_winner(pos):
    ''' A heuristic function for Kalah.  Returns the difference in seeds stored for
        each player (P1 - P2), unadjusted for terminal positions.

        pos -- a Kalah position
    '''
    return pos._seeds_stored(0) - pos._seeds_stored(1)


def minimax_policy(depth, h):
    def fxn(pos):
        value, move = minimax(pos, depth, h)
        return move
    return fxn


def minimax(pos, depth, h):
    ''' Returns the minimax value of the given position, with the given heuristic function
        applied at the given depth.

        pos -- a game position
        depth -- a nonnegative integer
        h -- a heuristic function that can be applied to pos and all its successors
    '''
    if pos.is_terminal() or depth == 0:
        return (h.evaluate(pos), None)
    else:
        if pos.actor() == 0:
            # max player
            best_value = -h.inf
            best_move = None
            moves = pos.get_actions()
            for move in moves:
                child = pos.successor(move)
                mm, _ = minimax(child, depth - 1, h)
                if mm > best_value:
                    best_value = mm
                    best_move = move
            return (best_value, best_move)
        else:
            # min player
            best_value = h.inf
            best_move = None
            moves = pos.get_actions()
            for move in moves:
                child = pos.successor(move)
                mm, _ = minimax(child, depth - 1, h)
                if mm < best_value:
                    best_value = mm
                    best_move = move
            return (best_value, best_move)
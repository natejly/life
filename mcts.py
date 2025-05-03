import math
import random
import time
C = math.sqrt(2)
class Node():
    def __init__(self, position, parent=None, move=None):
        self.pos = position
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.move = move
        if position.is_terminal():
            self.possible_moves = []
        else:
            self.possible_moves = position.get_actions()

def is_min(node):
    return node.pos.actor() == 0

def is_leaf(node):
    return node.pos.is_terminal()

def ucb(node, constant = None):
    ave_val = node.value / node.visits
    explore = constant * math.sqrt(math.log(node.parent.visits) / node.visits)
    if is_min(node.parent):
        return ave_val + explore
    else:
        return explore - ave_val

def traverse(node, constant=None):
    current = node
    while True:
        if is_leaf(current):
            return current
        if current.possible_moves:
            move = current.possible_moves.pop()
            child_pos = current.pos.successor(move)
            child = Node(child_pos, current, move)
            current.children.append(child)
            return child
        current = max(current.children, key=lambda x: ucb(x, constant))
    
def rollout(node, max_depth = None):
    current = node.pos
    history = []
    depth = 0
    while True:
        if current.is_terminal():
            payoff = current.payoff()
            return payoff
        if max_depth is not None and depth >= max_depth:
            return current.payoff()
        actions = current.get_actions()
        if not actions:
            return current.payoff()
        action = random.choice(actions)
        history.append((current.actor(), action))
        current = current.successor(action)
        depth += 1

    
def bp(node, payoff, weight=None, decay=None):
    current = node
    current_weight = weight if weight is not None else 1.0
    
    while current:
        current.visits += 1
        current.value += payoff * current_weight
        current = current.parent
        if decay is not None:
            current_weight *= decay
    
def policy(position, limit, constant=None, rollouts=None, max_depth=None, weight=None, decay=None):
    root = Node(position)
    root.visits = 1  
    start = time.time()
    end = start + limit
    rollout_count = 0
    
    while True:
        if time.time() >= end:
            break
            
        if rollouts is not None and rollout_count >= rollouts:
            break
            
        node = traverse(root, constant)
        node.visits = 1  # Initialize visits for new node
        payoff = rollout(node, max_depth)
        bp(node, payoff, weight, decay)
        rollout_count += 1
        # print(f"rollout count: {rollout_count}")
    if not root.children:
        return random.choice(position.get_actions()) if position.get_actions() else None
    
    best_child = max(root.children, key=lambda x: x.visits)
    return best_child.move

def mcts_policy(limit=None, constant=None, rollouts=None, max_depth=None, weight=None, decay=None):
    def policy_wrapper(position):
        return policy(position, limit=limit, constant=constant, rollouts=rollouts, 
                     max_depth=max_depth, weight=weight, decay=decay)
    return policy_wrapper
import heapq, random

class PriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, node, priority):
        entry = (priority + (len(self.heap),random.random()), node)
        heapq.heappush(self.heap, entry)

    def pop(self):
        _, node = heapq.heappop(self.heap)
        return node

    def is_empty(self):
        return len(self.heap) == 0
    
class Node:
    def __init__(self, parent_node, parent_clause, goal, priority, kb):
        """
        parent_node: the parent node
        parent_clause: the clause used to backchain to produce current node's goal
        goal: the current goal
        priority: the current node's priority (confidence, num_steps)
        kb: the set of clauses we can still use to continue the proof
        """
        self.parent_node = parent_node
        self.parent_clause = parent_clause

        self.goal = goal

        self.priority = priority

        self.kb = kb
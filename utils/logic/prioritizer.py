from abc import ABC, abstractmethod 
from utils.logic.nli import NLI
from utils.logic.fol import *



class Prioritizer(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, recipe_premise, query_hypothesis):
        pass

    def _grammar2str(self, grammar):
        flatenned_list = flatten_and(grammar)
        # extract the predicate names from the grammar
        processed_predicates = ["not " + str(x.arg.name) if isinstance(x, Not) else str(x.name) for x in flatenned_list]
        # replace underscores with spaces
        processed_predicates = [x.replace("_", " ") for x in processed_predicates]
        return processed_predicates
            
    def _str2grammar(self, list_str):
        # s: ~spicy
        # check if s starts with ~
        #conjuncts = [Not(Predicate(s[1:], Variable('x'))) if s.startswith('~') else Predicate(s, Variable('x')) for s in list_str]
        # no longer appending not, but using the LLM for negations
        conjuncts = [Predicate(s, Variable('x')) for s in list_str]
        ands = and_list(conjuncts)
        return ands

class Pure_prioritizer(Prioritizer):
    def __init__(self):
        pass
    
    def __call__(self, premise, goal):
        """
        calculate the score of a clause based on its match or entailment with the goal
        """
        if not isinstance(premise, str):
            premise_str = self._grammar2str(premise)[0]
            goal_str = self._grammar2str(goal)[0]
        else:
            premise_str = premise
            goal_str = goal
        nli = NLI()
        score = nli(premise_str, goal_str)['scores'][0]
        return score



class GD_prioritizer(Prioritizer):
    def __init__(self):
        pass
    
    def __call__(self, kb, goal):
        
        clause_scores = []
        for clause in kb:
            
            clause_scores.append(self._score_clause(clause, goal))
        return clause_scores
    
    def _score_clause(self, clause, goal):
        """
        calculate the score of a clause based on its match or entailment with the goal
        """
        nli = NLI()
        if isinstance(goal, Not):
            unsigned_goal = goal.arg
        else:
            unsigned_goal = goal
        goal_str = self._grammar2str(goal)[0]

        if isinstance(clause, Predicate):
            if str(clause.name) == str(unsigned_goal.name):
                if isinstance(goal, Not):
                    return (clause, -2)
                elif isinstance(goal, Predicate):
                    return (clause, 2)
            else:
                score = nli(str(clause.name), goal_str)['scores'][0]
                return (clause, score)
                
        elif isinstance(clause, ForAll):
            if isinstance(clause.body.rhs, Predicate):
                if str(clause.body.rhs.name) == str(unsigned_goal.name):
                    if isinstance(goal, Not):
                        return (clause, -1)
                    elif isinstance(goal, Predicate):
                        return (clause, 1)
                else:
                    rhs_str = self._grammar2str(clause.body.rhs)[0]
                    score = nli(rhs_str, goal_str)['scores'][0]
                    return (clause, score)


            elif isinstance(clause.body.rhs, Not):
                if str(clause.body.rhs.arg.name) == str(unsigned_goal.name):
                    if isinstance(goal, Not):
                        return (clause, 1)
                    elif isinstance(goal, Predicate):
                        return (clause, -1)
                else:
                    negated_rhs = self._grammar2str(self.negate_preds(clause.body.rhs.arg))[0]
                    score = nli(negated_rhs, goal_str)['scores'][0]
                    return (clause, score)

            
    
    def negate_preds(self, literal):
        """
        return the negation of a literal
        """
        if isinstance(literal, Predicate):
            return Not(literal)
        elif isinstance(literal, Not):
            return literal.arg
        else:
            raise ValueError("literal is neither a Predicate nor a Not")
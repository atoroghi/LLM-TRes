# main file for our agent
import os, sys, json, re, random, copy
random.seed(98)
from abc import ABC, abstractmethod 
import numpy as np
from agent.llm.llm_actions import *
from utils.logic.fol import *
from utils.logic.nli import NLI
from utils.logger import logger
from utils.retriever import ST_Retriever, GPT3_Retriever
from utils.logic.queue import PriorityQueue, Node
from utils.logic.prioritizer import GD_prioritizer, Pure_prioritizer
from utils.llm.clarifier import Clarifier

class Agent(ABC):
    def __init__(self):
        pass 
    @abstractmethod
    def __call__(self):
        # convert query to FOL 
        # convert recipe option to FOL KB 
        # resolution algo that calls resolve() 
        pass 

    # the children should only overwrite the resolve function

    def query2fol(self, query, query2predicates):
        # convert query to FOL 
        if query in query2predicates:
            query_conjuncts_str = query2predicates[query]
        else:
            query_conjuncts_str = ExtractQueryPreds('agent/llm/llm_prompts')(query=query)
            if query2predicates:
                query2predicates[query] = query_conjuncts_str
            else:

                query2predicates = {query: query_conjuncts_str}
        query_fol = self.str2grammar(query_conjuncts_str)
        return query_fol, query2predicates
    
    def query2negation(self, query, predicate2negations):
        query_str_list = self.grammar2str(query)
        negated_query_dict = {}
        for predicate in query_str_list:
            if predicate in predicate2negations:
                negated_query_dict[predicate] = predicate2negations[predicate]

            else:
                negated_predicate = NegatePreds('agent/llm/llm_prompts')(Property=predicate)
                if predicate2negations:
                    predicate2negations[predicate] = negated_predicate
                else:
                    predicate2negations = {predicate: negated_predicate}
                negated_query_dict[predicate] = negated_predicate
        return negated_query_dict, predicate2negations

    
    def option2fol(self, options, options2predicates):
        # convert recipe option to FOL KB 
        fol_options = {}
        for option in options:
            if options[option] in options2predicates:
                option_conjuncts_str = options2predicates[options[option]]
            else:
                option_conjuncts_str = [x.replace("_", " ") for x in ExtractOptionPreds('agent/llm/llm_prompts')(recipe=options[option])]
                
                if options2predicates:
                    options2predicates[options[option]] = option_conjuncts_str
                else:
                    options2predicates = {options[option]: option_conjuncts_str}
            fol_options[option] = self.str2grammar(option_conjuncts_str)
        return fol_options, options2predicates
    
    
        

    @abstractmethod
    def prove(self):
        pass 

class Monolithic_llm(Agent):
    def __init__(self):
        super().__init__()

    def __call__(self, query, kb, aspects, log, max_steps):
        answer = self.prove(query, kb, aspects, log=log, max_steps=max_steps)
        return answer
    
    def prove(self, query, kb, aspects, log, max_steps=100):
        if not aspects:
            answer, proof = GetMonolithicProof('agent/llm/llm_prompts')(Query=query, KB=kb)
        else:
            if isinstance(query, str):
                answer, proof = GetGPTRecipe('agent/llm/llm_prompts')(query=query, options=kb)
            else:
                answer, proof = GetMonolithicProofCOPA('agent/llm/llm_prompts')(Event=aspects, KB=kb, Consequence1=query[0], Consequence2=query[1])
        
        log("final answer: ", answer); log("response: ", proof)
        return answer



class GD_Resolution(Agent):
    def __init__(self):
        super().__init__()
        self.retriever = GPT3_Retriever(os.path.join(os.getcwd(), 'models'))
        self.prioiritizer = GD_prioritizer()
        
    def __call__(self, query, kb, aspects, log, max_steps):
        answer = self.prove(query, kb, aspects, log=log, max_steps=max_steps)
        return str(answer)
    

    def prove(self, query, kb, aspects, log=None, max_steps=100):
        if not aspects:
            # Pronto
            for rule in kb:
                if isinstance(rule, Predicate):
                    premise = rule
                    break
            pos_query = query
            neg_query = self.negate_preds(query)
        
        else:
            # COPA
            premise = aspects
            pos_query = query[0]
            neg_query = query[1]
        
        log("negated_query", neg_query)
    
        pos_proof, pos_score = self._gd_resolution(pos_query, kb, premise, max_steps)
        neg_proof, neg_score = self._gd_resolution(neg_query, kb, premise, max_steps)
        
        log("pos_proof", pos_proof); log("pos_score", pos_score); log("neg_proof", neg_proof); log("neg_score", neg_score)

        if not pos_proof and not neg_proof:
            return 'unsolvable'
        else:
            if aspects:
                if abs(pos_score) > abs(neg_score):
                    return pos_query
                else:
                    return neg_query
            else:
                return abs(pos_score) > abs(neg_score)
    

    

    def _gd_resolution(self, query, kb, premise, max_steps=1000):
        q = PriorityQueue()
        root_priority = (-1, 0)
        root = Node(None, None, query, root_priority, kb)
        q.push(root, root_priority)

        while not q.is_empty() and max_steps > 0:
            max_steps -= 1
            cur_node = q.pop()

            # empty goal => found complete proof
            if cur_node.goal is None:
                return self._construct_proof(cur_node), cur_node.priority[0]
            
            new_nodes = self._expand_node(cur_node, premise)

            for node in new_nodes:
                q.push(node, node.priority)

        return None, 0
    def _expand_node(self, node, premise):
        new_nodes = []
        top_k = self._rank_clauses(premise, node.goal, node.kb)

        for (clause, score) in top_k:
            # new_goal is clause's LHS if clause is an implication or None if clause is an assumption
            if isinstance(clause, Predicate):
                new_goal = None 
            elif isinstance(clause, ForAll):
                new_goal = clause.body.lhs

            new_node_priority = (node.priority[0]*score, node.priority[1]+1)
            new_node_kb = copy.deepcopy(node.kb)
            new_node_kb = self._remove_clause(clause, new_node_kb)
            new_node = Node(node, clause, new_goal, new_node_priority, new_node_kb)
            new_nodes.append(new_node)

        return new_nodes
    
    def _remove_clause(self, clause, kb):
        matching_clauses = [c for c in kb if str(c) == str(clause)]
        if matching_clauses:
            kb.remove(matching_clauses[0])
        return kb
    
    def _rank_clauses(self, premise, goal, kb, k=15):
        """
        rank the clauses in kb to backchain from node.goal
        return a list where each item is (clause, score)
        """
        # 1. semantic search of kb clause RHSs with the goal to get rid of red-herrings
        #kb = self.retriever(goal, premise, kb, k)
        kb.append(premise)
        # 2. top k clauses (exact matches, entailment scores)
        
        clause_scores = self.prioiritizer(kb, goal)

        return clause_scores

    def _construct_proof(self, node):
        """
        repeatedly traverse up parent_node to reconstruct proof
        """
        used_clauses = []
        while node.parent_node is not None:
            used_clauses.append(node.parent_clause)
            node = node.parent_node

        return used_clauses



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




class Pure_entailment(Agent):
    def __init__(self):
        super().__init__()
        self.prioiritizer = Pure_prioritizer()

    def __call__(self, query, kb, aspects, log, max_steps):
        answer = self.prove(query, kb, aspects, log=log, max_steps=max_steps)
        return str(answer)
    
    def prove(self, query, kb, aspects, log=None, max_steps=100):
        if not aspects:
            # Pronto
            for rule in kb:
                if isinstance(rule, Predicate):
                    premise = rule
                    break
            pos_query = query
            neg_query = self.negate_preds(query, aspects)
        else:
            # COPA
            premise = aspects
            pos_query = query[0]
            neg_query = query[1]
        log("positive_query", pos_query)
        log("negated_query", neg_query)
        pos_score = self._pure_entailment(pos_query, kb, premise, max_steps)
        neg_score = self._pure_entailment(neg_query, kb, premise, max_steps)
        log("pos_score", pos_score); log("neg_score", neg_score)
        if not aspects:
            return abs(pos_score) > abs(neg_score)
        else:
            if pos_score > neg_score:
                return pos_query
            else:
                return neg_query

    def _pure_entailment(self, query, kb, premise, max_steps):
        score = self.prioiritizer(premise, query)
        return score

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














class Backchaining_pronto(Agent):
    def __init__(self):
        super().__init__()
        self.retriever = GPT3_Retriever(os.path.join(os.getcwd(), 'models'))
        self.clarifier = Clarifier()
    def __call__(self, query, rules, kb, log, masked_rules_no, misleading_rules_no, max_steps):
        
        rules_fol = self.perturb_rules(rules, kb, masked_rules_no, misleading_rules_no)
        log("Perturbed Rules: ", rules_fol)

        # rules_fol_str = [
        #     "FOR_ALL x, real(x) => number(x)",
        #     "FOR_ALL x, complex(x) => imaginary(x)",
        #     "FOR_ALL x, integer(x) => real(x)",
        #     "FOR_ALL x, natural(x) => integer(x)",
        #     "FOR_ALL x, natural(x) => ~negative(x)",
        #     "FOR_ALL x, prime(x) => natural(x)",
        #     "FOR_ALL x, mersenne_prime(x) => prime(x)",
        #     "mersenne_prime(x)"
        # ]
        # rules_fol = [parse_fol(rule)[1] for rule in rules_fol_str]
        # _, query = parse_fol('~imaginary(x)')

        for rule in rules_fol:
            if isinstance(rule, Predicate):
                premise = rule
                break
        log("premise: ", premise)
        premise = self.clarifier(premise, rules_fol)
        log("Clarified premise: ", premise)
        answer = self.prove(query, rules_fol, premise, first_step=True, log=log, max_steps=max_steps)
        return str(answer)
    
    def perturb_rules(self, rules_fol, kb, masked_rules_no, misleading_rules_no):
        # mask some rules except the premise

        universal_rules = rules_fol[:-1]
        masked_rules = random.sample(universal_rules, masked_rules_no)
        unmasked_rules = [rule for rule in rules_fol if rule not in masked_rules] + [rules_fol[-1]]

        #masked_rules = rules_fol[-(masked_rules_no+1):-1]
        #unmasked_rules = rules_fol[:-(masked_rules_no+1)] + [rules_fol[-1]]

        random.shuffle(kb)
        misleading_rules = []
        for rule in kb:
            if len(misleading_rules) == misleading_rules_no:
                break
            if str(rule) not in [str(x) for x in masked_rules]:
                misleading_rules.append(rule)
            
                

        perturbed_rules = [] + unmasked_rules
        for rule in misleading_rules:
            if str(rule) not in [str(x) for x in perturbed_rules]:
                perturbed_rules.append(rule)
        return perturbed_rules

    
    def prove(self, query, all_rules, premise, first_step=False, log=None, used_rules=[], max_steps=100):
        if len(all_rules) == 1:
            return 'unsolvable'
        result = 'idk'
        log('query', query)

        # first identify the premise which is the only predicate
        
        # if first_step:
        #     rules = self.retriever(query, premise, rules, 15)
        #     log('retrieved rules', rules)
        
        
        
        # backchaining algorithm
        while result == 'idk' and max_steps > 0:
            max_steps -= 1
            rules = self.retriever(query, premise, all_rules, 15)
            if used_rules:
                rules = [rule for rule in rules if rule not in used_rules]
            # specify the next rule to be used
            selected_rules, selected_rules_scores, polarities = self._find_next_rule(query, rules, first_step)
            if selected_rules is None:
                result = 'unsolvable'
                break
            # query cannot be resolved with any RHS, so need to backtrack
            if abs(max(selected_rules_scores.values())) < 0.9:
                log("backtracking from proving", query)
                break
            #if more than one rule was selected, choose the one that is most likely to allow continuing the proof
            
            if len(selected_rules) > 1:
                log('selected rules all', selected_rules)
                next_rule_max_scores = {}
                other_rules = [rule for rule in rules if rule not in selected_rules]
                for rule in selected_rules:
                    _, next_hop_scores, _ = self._find_next_rule(rule.body.lhs, other_rules, False)
                    if not next_hop_scores:
                        result = 'unsolvable'
                        break
                    next_rule_max_scores[rule] = max(next_hop_scores.values())
                selected_rule = max(next_rule_max_scores, key=next_rule_max_scores.get)

            else:
                selected_rule = selected_rules[0]

            log('selected rule', selected_rule)
            
            # polarity determines if the rule changes the sign of the predicate
            polarity = polarities[selected_rule]
            log('polarity', polarity)

            used_rules = used_rules + [selected_rule]
            if polarity:
                if self.ask(selected_rule, premise):
                    result = True
                    log('result', result)
                    break
                else:
                    rules.remove(selected_rule)
                    inner_result = self.prove(selected_rule.body.lhs, all_rules, premise, False, log, used_rules)
                    if inner_result == 'unsolvable':
                        result = 'unsolvable'
                        log('result', result)
                        break
                    else:
                        if inner_result != 'idk':
                            result = inner_result
                            log('result', result)
                            break
            elif not polarity:
                if self.ask(selected_rule, premise):
                    result = False
                    log('result', result)
                    break
                else:
                    rules.remove(selected_rule)
                    inner_result = self.prove(selected_rule.body.lhs, all_rules, premise, False, log, used_rules)
                    if inner_result == 'unsolvable':
                        result = 'unsolvable'
                        log('result', result)
                        break
                    else:
                        if inner_result != 'idk':
                            result = not(inner_result)
                            log('result', result)
                            break
             



        return result


    
    def _find_next_rule(self, query, rules, first_step):
        candidate_rule_scores = {}
        nli = NLI()
        query_str = self.grammar2str(query)[0]
        negated_query = self.negate_query(query)
        negated_query_str = self.grammar2str(negated_query)[0]
        if isinstance(query, Not):
            unsigned_query = query.arg
        else:
            unsigned_query = query

        # for first step, we need to explore both positive and negative possible rhss
        if first_step:
            for rule in rules:
                if isinstance(rule, Predicate):
                    if str(rule.name) == str(unsigned_query.name):
                        if isinstance(query, Not):
                            candidate_rule_scores[rule] = -2
                        else:
                            candidate_rule_scores[rule] = 2
                elif isinstance(rule, ForAll):
                    if isinstance(rule.body.rhs, Predicate):
                        if str(rule.body.rhs.name) == str(unsigned_query.name):
                            if isinstance(query, Not):
                                candidate_rule_scores[rule] = -1
                            else:
                                candidate_rule_scores[rule] = 1
                        else:
                            rule_rhs = self.neutralize_predicate(rule.body.rhs)
                            pos_score = nli(rule_rhs, query_str)['scores'][0]
                            neg_score = nli(rule_rhs, negated_query_str)['scores'][0]
                            if pos_score > neg_score:
                                candidate_rule_scores[rule] = pos_score
                            else:
                                candidate_rule_scores[rule] = -1 * neg_score
                    elif isinstance(rule.body.rhs, Not):
                        if str(rule.body.rhs.arg.name) == str(unsigned_query.name):
                            #return rule, False
                            if isinstance(query, Not):
                                candidate_rule_scores[rule] = 1
                            else:
                                candidate_rule_scores[rule] = -1
                        
                        # commented out because nli scores for negated rhss are not reliable
                        # else:
                        #     rule_rhs = 'not '+ self.neutralize_predicate(rule.body.rhs)
                        #     pos_score = nli(rule_rhs, query_str)['scores'][0]
                        #     neg_score = nli(rule_rhs, negated_query_str)['scores'][0]
                        #     if pos_score > neg_score:
                        #         candidate_rule_scores[rule] = pos_score
                        #     else:
                        #         candidate_rule_scores[rule] = -1 * neg_score
                else:
                    raise ValueError("rule is neither a Predicate nor a ForAll")
        # for subsequent steps, we only need to explore the rhss with similar polarity to the query
        else:
            for rule in rules:
                if isinstance(rule, Predicate):
                    if str(rule.name) == str(unsigned_query.name):
                        if isinstance(query, Not):
                            candidate_rule_scores[rule] = 2
                        else:
                            candidate_rule_scores[rule] = -2
                elif isinstance(rule, ForAll):
                    if isinstance(query, Predicate):
                        # both query and rhs most be positive
                        if isinstance(rule.body.rhs, Predicate):
                            if str(rule.body.rhs.name) == str(query.name):
                                candidate_rule_scores[rule] = 1
                            else:
                                rule_rhs = self.neutralize_predicate(rule.body.rhs)
                                candidate_rule_scores[rule] = nli(rule_rhs, query_str)['scores'][0]
                    elif isinstance(query, Not):
                        if isinstance(rule.body.rhs, Not):
                            if str(rule.body.rhs.arg.name) == str(unsigned_query.name):
                                candidate_rule_scores[rule] = 1
                            
                            # commented out because nli scores for negated rhss are not reliable
                            # else:
                            #     rule_rhs = self.neutralize_predicate(rule.body.rhs)
                            #     candidate_rule_scores[rule] = nli('not ' + rule_rhs, query_str)['scores'][0]
                else:
                    raise ValueError("rule is neither a Predicate nor a ForAll")

        if candidate_rule_scores == {}:
            return None, None, None
        selected_rules, max_scoring_rule_scores, polarities = self.pick_highest_score(candidate_rule_scores)
        return selected_rules, max_scoring_rule_scores, polarities
    
    def negate_query(self, query):
        if isinstance(query, Predicate):
            return Not(query)
        elif isinstance(query, Not):
            return query.arg
        else:
            raise ValueError("query is neither a Predicate nor a Not")

    def neutralize_predicate(self, predicate):
        if isinstance(predicate, Predicate):
            return str(predicate.name)
        elif isinstance(predicate, Not):
            return str(predicate.arg.name)
        else:
            raise ValueError("rhs is neither a Predicate nor a Not")

    def ask(self, rule, premise):
        nli = NLI()
        if isinstance(rule.body.lhs, Predicate):
            if str(rule.body.lhs.name) == str(premise.name) or nli(premise.name, rule.body.lhs.name)['scores'][0]>0.9:
                return True
        return False
    
    def pick_highest_score(self, rhs_entailment_scores):
        max_abs_score = max(abs(val) for val in rhs_entailment_scores.values())
        max_scoring_rules, polarities = [], {}
        max_scoring_rule_scores = {}
        for rule, score in rhs_entailment_scores.items():
            if abs(score) == max_abs_score:
                max_scoring_rules.append(rule)
                polarities[rule] = (score > 0)
                max_scoring_rule_scores[rule] = score
        return max_scoring_rules, max_scoring_rule_scores, polarities
    





class Symbolic_Backchaining(Agent):
    def __init__(self):
        super().__init__()
    
    def __call__(self, clauses, goal):
        log = logger()
        # assuming clauses and goal are "fol" objects (parsed to the fol grammar from strings)
        #log('clauses', clauses)
        #log('goal', goal)
        
        result = self.prove(clauses, goal)
        #log('result', result)
        return result
    

         
    def prove(self, clauses, goal):
        # backchaining algorithm
        #log = logger()
        #log('clauses', clauses)
        #log('goal', goal)
        for clause in clauses:
            if isinstance(clause, And):
                broken_clause = flatten_and(clause)
                clauses.remove(clause)
                clauses.extend(broken_clause)

        if self.ask(clauses, goal):
            return True
        for clause in clauses:
            if hasattr(clause, 'body'):
                unifiability, substitution = self.unify(clause.body.rhs, goal)
                #if str(clause.body.rhs) == str(goal):
                if unifiability:
                    new_clause = self.subst(clause, substitution)
                    clauses.append(new_clause)
                    premise_disjuncts = flatten_or(clause.body.lhs)
                    for disjunct in premise_disjuncts:
                        conjuncts = flatten_and(disjunct)
                        active_conjuncts = set(conjuncts)
                        for conjunct in conjuncts:
                            if self.prove(clauses, conjunct):
                                active_conjuncts.remove(conjunct)
                            if len(active_conjuncts) == 0:
                                return True

                            # what if not proven?
                    
                            

    def unify(self, literal1, literal2):
        unifiability = False
        substitution = {}
        if isinstance(literal1, Predicate) and isinstance(literal2, Predicate):
            # unifiablility and the substitution
            if literal1.name == literal2.name:

                for i in range(len(literal1.args)):
                    if isinstance(literal1.args[i], Variable) and isinstance(literal2.args[i], Variable):
                        unifiability = True
                        substitution[str(literal1.args[i].name)] = literal2.args[i]
                        substitution[str(literal2.args[i])] = literal2.args[i]
                    elif isinstance(literal1.args[i], Variable) and isinstance(literal2.args[i], Constant):
                        unifiability = True
                        substitution[str(literal1.args[i])] = literal2.args[i]
                        substitution[str(literal2.args[i])] = literal2.args[i]
                    elif isinstance(literal1.args[i], Constant) and isinstance(literal2.args[i], Variable):
                        unifiability = True
                        substitution[literal1.args[i]] = literal1.args[i]
                        substitution[literal2.args[i]] = literal1.args[i]
                    elif isinstance(literal1.args[i], Constant) and isinstance(literal2.args[i], Constant):
                        continue
        
        return unifiability, substitution
            

    def subst(self, clause, substitution):
        if isinstance(clause, Predicate):
            return Predicate(clause.name, Variable(substitution[clause.args.name]))
        elif isinstance(clause, Not):
            return Not(self.subst(clause.arg, substitution))
        elif isinstance(clause, And):
            clauses = flatten_and(clause)
            new_clauses = []
            for clause in clauses:
                new_clauses.append(self.subst(clause, substitution))
            return and_list(new_clauses)
        elif isinstance(clause, Or):
            clauses = flatten_or(clause)
            new_clauses = []
            for clause in clauses:
                new_clauses.append(self.subst(clause, substitution))
            return or_list(new_clauses)
        elif isinstance(clause, Implies):
            return Implies(self.subst(clause.lhs, substitution), self.subst(clause.rhs, substitution))
        elif isinstance(clause, ForAll):
            clause.var = substitution[clause.var.name]
            return ForAll(clause.var, self.subst(clause.body, substitution))



    def ask(self, clauses, goal):
        for clause in clauses:
            if str(clause) == str(goal):
                return True
            unifiability, substitution = self.unify(clause, goal)
            if unifiability:
                for clause in clauses:
                    clause = self.subst(clause, substitution)
                    clauses.append(clause)
                return True
                

        return False








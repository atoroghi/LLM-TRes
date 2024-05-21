from typing import Any
from utils.logic.fol import *
from utils.logic.nli import NLI
from agent.llm.llm_actions import *

class Clarifier:
    def __init__(self):
        pass

    def __call__(self, premise, rules):
        """
        If the premise doesn't have a strong entailment with any of the rules, consults the LLM
        to replace the premise with a one-sentence explanation of it
        """
        # get the reolvent with premise
        resolvable_rules = []
        for rule in rules:
            if isinstance(rule, ForAll):
                resolvable_rules.append(self.ask(rule, premise))
        if any(resolvable_rules):
            return premise
        else:
            if isinstance(premise, Predicate):
                explanation = ConsultGPT('agent/llm/llm_prompts')(concept=premise.name)
                new_premise = Predicate(explanation, Variable('x'))
            elif isinstance(premise, Not):
                explanation = ConsultGPT('agent/llm/llm_prompts')(concept=premise.arg.name)
                new_premise = Not(Predicate(explanation, Variable('x')))
            return new_premise


    def ask(self, rule, premise):
        nli = NLI()
        if isinstance(rule.body.lhs, Predicate):
            if str(rule.body.lhs.name) == str(premise.name) or nli(premise.name, rule.body.lhs.name)['scores'][0]>0.9:
                return True
        return False

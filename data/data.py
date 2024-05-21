# generator functions to iterate over the datasets 

import json, os, sys, re, pickle, random
from utils.llm.clarifier import Clarifier
from utils.logic.fol import *
random.seed(98)

class Dataset:
    def __init__(self, dataset_name, path, misleading_rules_no, masked_rules_no, use_logic, log):
        self.dataset_name = dataset_name
        self.use_logic = use_logic
        self.data = self._load_dataset(path)
        self.len = len(self.data)
        self.misleading_rules_no = misleading_rules_no
        self.masked_rules_no = masked_rules_no
        self.dataset_kb = self._load_kb(path, dataset_name)
        self.log = log

    
    def __call__(self, row_number):
        row = self.data[row_number]
        if self.dataset_name == 'Recipe-MPR':
            query, options, answer, aspects = row['query'], row['options'], row['answer'], list(row['correctness_explanation'].keys())
        elif self.dataset_name == 'ProntoQA':
            query, rules, answer, aspects = row['query'], row['context'], row['answer'], None
            self.log("Query: ", query); self.log("Answer: ", answer); self.log("Rules: ", rules)

            self.clarifier = Clarifier()
            perturbed_rules = self._perturb_rules(rules, self.dataset_kb, self.masked_rules_no, self.misleading_rules_no)
            
            # for rule in perturbed_rules:
            #     # if the premise isn't known to the NLI model, we should consult the LLM
            #     if isinstance(rule, Predicate):
            #         clarified_rule = self.clarifier(rule, perturbed_rules)
            #         perturbed_rules.remove(rule)
            #         perturbed_rules.append(clarified_rule)
            
            options = perturbed_rules
            self.log("Perturbed Rules: ", options)
        
        elif self.dataset_name == 'COPA':
            query, options, answer, aspects = row['options'], row['rules'], row['answer'], row['query']

            

        return query, options, answer, aspects
    


    def _perturb_rules(self, rules_fol, dataset_kb, masked_rules_no, misleading_rules_no):
        """
        Perturbs the rules by masking some rules and adding some misleading rules
        """
        # mask some rules except the premise

        universal_rules = rules_fol[:-1]
        masked_rules = random.sample(universal_rules, masked_rules_no)
        unmasked_rules = [rule for rule in rules_fol if rule not in masked_rules]
        random.shuffle(dataset_kb)
        misleading_rules = []
        for rule in dataset_kb:
            if len(misleading_rules) == misleading_rules_no:
                break
            if str(rule) not in [str(x) for x in masked_rules]:
                misleading_rules.append(rule)
            
        perturbed_rules = [] + unmasked_rules
        for rule in misleading_rules:
            if str(rule) not in [str(x) for x in perturbed_rules]:
                perturbed_rules.append(rule)
        return perturbed_rules
    

    def _load_dataset(self, file_path):
        """
        Loads the dataset from the file_path
        for ProntoQA, rules are FOL objects
        """

        if self.dataset_name == 'Recipe-MPR':
            with open(os.path.join(file_path, self.dataset_name + '.json') , "r") as f:
                data = json.load(f)
        elif self.dataset_name == 'ProntoQA':
            if self.use_logic == False:
                with open(os.path.join(file_path, self.dataset_name + '_list_context.json') , "r") as f:
                    data = json.load(f)
            else:
                with open(os.path.join(file_path, self.dataset_name + '.pickle') , "rb") as f:
                    data = pickle.load(f)
        elif self.dataset_name == 'COPA':
            if self.use_logic == False:
                with open(os.path.join(file_path, self.dataset_name + '.json') , "r") as f:
                    data = json.load(f)
            else:
                with open(os.path.join(file_path, self.dataset_name + '_fol.pickle') , "rb") as f:
                    data = pickle.load(f)
        return data
    
    def _load_kb(self, file_path, dataset_name):
        """
        Loads the knowledge base including all axioms for the dataset
        """
        if dataset_name == 'Recipe-MPR':
            kb = None
        elif dataset_name == 'ProntoQA':
            if self.use_logic == False:
                with open(os.path.join(file_path, 'ProntoQA_KB_NL.json') , "r") as f:
                    kb = json.load(f)
            else:
                with open(os.path.join(file_path, 'ProntoQA_KB.pickle') , "rb") as f:
                    kb = pickle.load(f)

        elif dataset_name == 'COPA':
            if self.use_logic == False:
                with open(os.path.join(file_path, 'COPA_KB_NL.json') , "r") as f:
                    kb = json.load(f)
            else:
                with open(os.path.join(file_path, 'COPA_KB.pickle') , "rb") as f:
                    kb = pickle.load(f)
        return kb
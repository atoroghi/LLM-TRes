from abc import ABC, abstractmethod 
from sentence_transformers import SentenceTransformer, util
from utils.logic.fol import *
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import openai
from utils.embeddings_utils import get_embedding

load_dotenv()
client = OpenAI()


class Retriever(ABC):

    def __init__(self):
        pass 
    @abstractmethod
    def __call__(self): 
        pass 

    def query2nl(self, query):
        if isinstance(query, Predicate):
            return f"x is a {query.name}"
        elif isinstance(query, Not):
            return f"x is not a {query.arg.name}"
    
    def rules2nl(self, rules):
        nl2rules = {}
        rules_nl = []
        for rule in rules:
            if isinstance(rule, ForAll):
                if isinstance(rule.body.rhs, Predicate):
                    nl = f"Every {rule.body.lhs.name} is a {rule.body.rhs.name}"
                    nl2rules[nl] = rule
                    rules_nl.append(nl)
                elif isinstance(rule.body.rhs, Not):
                    nl = f"Every {rule.body.lhs.name} is not a {rule.body.rhs.arg.name}"
                    nl2rules[nl] = rule
                    rules_nl.append(nl)
        return rules_nl, nl2rules
    def encode(self, s):
        pass

class GPT3_Retriever(Retriever):
    def __init__(self, model_dir):
        pass
    

    def __call__(self, query, assumption, rules, k_facts=10):
        query_nl = self.query2nl(query)
        assumption_nl = self.query2nl(assumption)
        query_nl = f"{query_nl} and {assumption_nl}"
        for rule in rules:
            if isinstance(rule, Predicate):
                assumption = rule
                break
        if assumption in rules:
            rules.remove(assumption)
        rules_nl, nl_rules = self.rules2nl(rules)
        query_embedding = self.encode(query_nl)
        rules_embeddings = self.encode(rules_nl)
        results = self.select_most_similar(query_embedding, rules_embeddings, k_facts)
        selected_rules = [rules[result] for result in results] + [assumption]
        rules.append(assumption)
        return selected_rules
    
    def encode(self, s):
        return client.embeddings.create(input=s, model= "text-embedding-3-small").data

    def select_most_similar(self, query_embedding, rules_embeddings, k_facts):
        rule_scores = {}
        query_vector = query_embedding[0].embedding
        for i, rule_embedding in enumerate(rules_embeddings):
            rule_vector = rule_embedding.embedding
            rule_scores[i] = self.cosine_similarity(query_vector, rule_vector)
        top_k = sorted(rule_scores, key=rule_scores.get, reverse=True)[:k_facts]
        return top_k
    
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class ST_Retriever(Retriever):
    def __init__(self, model_dir):
        self.mpnet = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            cache_folder=f"{model_dir}/sentence_transformers",
        )

    def __call__(self, query, rules, k_facts=10):
        query_nl = self.query2nl(query)
        for rule in rules:
            if isinstance(rule, Predicate):
                assumption = rule
                break
        rules.remove(assumption)
        rules_nl, nl_rules = self.rules2nl(rules)

        query_embedding = self.encode([query_nl])
        rules_embeddings = self.encode(rules_nl)

        results = util.semantic_search(
            query_embedding, rules_embeddings, score_function=util.dot_score, top_k=k_facts
        )[0]

        selected_rules = [nl_rules[rules_nl[result["corpus_id"]]] for result in results] + [assumption]
        return selected_rules


    def encode(self, s):
        return self.mpnet.encode(s, normalize_embeddings=True, convert_to_tensor=True)
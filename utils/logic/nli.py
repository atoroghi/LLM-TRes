import os, sys
from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

class NLI:
    def __init__(self):
        pass

    def __call__(self, recipe_premise, query_hypothesis):
        return classifier(recipe_premise, query_hypothesis)

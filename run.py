# main file to run experiments 
from data import data
import os, sys, copy
import tqdm, json
import numpy as np
import pickle
import argparse
from agent import agent
from utils.logger import logger
from utils.logic.fol import parse_fol

def get_args():
    parser = argparse.ArgumentParser(description="Run experiments for the dataset")
    parser.add_argument('path', help='Path to directory containing dataset')
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--scoring_method", type=str, default="GD resolution", help="Method to use for scoring the options")
    parser.add_argument("--experiment_name", type=str, required=True, default="res", help="Name for saving the results of the experiment")
    parser.add_argument("--masked_rules", type=int, default=0, help="number of masked rules")
    parser.add_argument("--misleading_rules", type=int, default=0, help="number of misleading rules added")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum steps allowed for reasoining")
    parser.add_argument("--query_predicate_extraction", type=str, default="false", help="Whether to extract query predicates using the LLM or use the dataset provided predicates")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    #



    dataset_name = args.dataset_name
    path = args.path
    scoring_method = args.scoring_method
    masked_rules_no = args.masked_rules
    misleading_rules_no = args.misleading_rules
    max_steps = args.max_steps
    use_logic = False if scoring_method == "monolithic llm" else True
 



    if scoring_method == "GD resolution":
        Reasoner = agent.GD_Resolution()
    elif scoring_method == "Backchaining":
        Reasoner = agent.Backchaining_pronto()
    elif scoring_method == "pure entailment":
        Reasoner = agent.Pure_entailment()
    elif scoring_method == "monolithic llm":
        Reasoner = agent.Monolithic_llm()
    else:
        raise ValueError("Scoring method not supported")



    log = logger(args.experiment_name)
    # load the dataset
    dataset = data.Dataset(dataset_name, path, misleading_rules_no, masked_rules_no, use_logic, log)
    outcomes = []

    

    #for i in tqdm.tqdm(range(dataset.len)):
    for i in tqdm.tqdm(range(30, 500)):

        log("Example no: ", i)

        # for COPA, aspects is the premise and query the list of options
        query, options, answer, aspects = dataset(i)

        print(query); print(options)

        selected_option = Reasoner(query , options, aspects, log, max_steps)
        
        log("Selected option:", selected_option)
        correctness = str(answer).lower() in str(selected_option).lower()
        outcomes.append(correctness)
        if not correctness:
            log("Incorrect Answer", answer)

        print("Accuracy so far: ", np.mean(outcomes))

    
    print(f"Accuracy: {np.mean(outcomes)}"); print(f"Correct: {np.sum(outcomes)}"); print(f"Incorrect: {np.sum(np.logical_not(outcomes))}")
    log("Accuracy: ", np.mean(outcomes)); log("Correct: ", np.sum(outcomes)); log("Incorrect: ", np.sum(np.logical_not(outcomes)))
    


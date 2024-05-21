# main file to run experiments 
from data import data
import os, sys
import tqdm
import numpy as np

import argparse
from agent import agent
from utils.logger import logger
from utils.logic.fol import parse_fol

def get_args():
    parser = argparse.ArgumentParser(description="Run experiments for the dataset")
    parser.add_argument('path', help='Path to directory containing dataset')
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    #

    dataset_name = args.dataset_name
    path = args.path
    # load the dataset
    dataset = data.Dataset(dataset_name, path)
    # load the agent
    #Reasoner = agent.NLI_Backchaining()
    Reasoner = agent.Symbolic_Backchaining()
    log = logger()
    outcomes = []


    P = "FOR_ALL x, meat_pasta(x) && watching_weight(x) && ~spicy(x) => likes(U, x)"
    Q = "FOR_ALL x, low_cal(x) => watching_weight(x)"
    R = "low_cal(R)"
    cluses_str = [Q, P, R]
    fol_clauses = []
    for clause in cluses_str:
        tree, fol = parse_fol(clause)
        fol_clauses.append(fol)
    

    _, goal = parse_fol("likes(U, R)")
    proof = Reasoner.prove(fol_clauses, goal)


    
import re, tqdm
import json, pickle
from agent.llm.llm_actions import *
from utils.logic.fol import *
from agent.agent import *


# def _form_story(entry):
#     story = f'''event 1: {query}\n possible continuation 1: {options[0]}
#  possible continuation 2: {options[1]}\n answer: {answer}\nrules: {str(context)}'''
#     return story

def _str2fol_str(entry):
    query = entry['query']
    options = entry['options']
    context = entry['rules']
    answer = entry['answer']

    fol_query = ConvertFOLCOPAPremise('agent/llm/llm_prompts')(sentence=query)
    fol_options = []
    for option in options:
        fol_option = ConvertFOLCOPAPremise('agent/llm/llm_prompts')(sentence=option)
        fol_options.append(fol_option)
        if option == answer:
            fol_answer = fol_option
        
    fol_rules = []
    for rule in context:
        triple = f'{rule[0]}, {rule[1]}, {rule[2]}'
        fol_rule = ConvertFOLCOPAForall('agent/llm/llm_prompts')(triple=triple)
        fol_rules.append(fol_rule)
    # story = _form_story(entry)
    new_entry = {'query': fol_query, 'options': fol_options, 'answer': fol_answer, 'rules': fol_rules}
    return new_entry

# def grammar2str(grammar):
#     flatenned_list = flatten_and(grammar)
#     # extract the predicate names from the grammar
#     processed_predicates = ["not " + str(x.arg.name) if isinstance(x, Not) else str(x.name) for x in flatenned_list]
#     # replace underscores with spaces
#     processed_predicates = [x.replace("_", " ") for x in processed_predicates]
#     return processed_predicates


def premise2fol(strfol):
    # convert query to FOL 
    #story_fol = _str2fol_str(query_raw)[0]
    tree, fol = parse_fol(strfol)
    return fol

def rules2fol(rules_strforall):
    # convert rules to FOL
    rules_fol = []
    for rule in rules_strforall:
        tree, rule_fol = parse_fol(rule)
        rules_fol.append(rule_fol)
    return rules_fol


mode = 'Lambada_preparation'

# extracting query, options, answer, and explanations from the log file
if mode == 'text extraction':

    # Define the file paths
    input_file_path = 'data/test-explained.jsonl'
    output_file_path = 'data/copa.json'

    data = []

    # Open the log file and read its contents
    with open(input_file_path, 'r') as input_file:
        input = input_file.read()

        for line in tqdm.tqdm(input.split('\n')[:-1]):
            json_line = json.loads(line)

            if json_line['asks-for'] == 'effect':
                query = json_line['p']
                options = [json_line['a1'], json_line['a2']]
                if json_line['most-plausible-alternative'] == '1':
                    answer = json_line['a1']
                else:
                    answer = json_line['a2']
                
                explanations = json_line['human-explanations']
                expl_ratings = {}
                for i, explanation in enumerate(explanations):
                    expl_ratings[i] = explanation['filtered-avg-rating']
                triples_no = max(expl_ratings, key=lambda k: expl_ratings[k])
                rules = explanations[triples_no]['triples']
                if len(rules)>1:
                    data.append({'query': query, 'options': options, 'answer': answer, 'rules': rules})

           

    
    # Save the query entries as JSON to the output file
    with open(output_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print('Queries extracted and saved to queries.json.')


elif mode == 'Str FOL extraction':
    with open('data/copa.json', 'r') as f:
        data = json.load(f)

    fol_data = []
    for i in tqdm.tqdm(range(len(data))):
        entry = data[i]        


        new_entry = _str2fol_str(entry)
        fol_data.append(new_entry)

    # with open('data/ProntoQA.pickle', 'wb') as f:
    #     pickle.dump(fol_data, f)
    with open('data/copa_folstr.json', 'w') as f:
        json.dump(fol_data, f, indent=4)
    

elif mode == 'FOL extraction':
    with open('data/copa_folstr.json', 'r') as f:
        data = json.load(f)

    fol_data = []
    for i in tqdm.tqdm(range(len(data))):
        entry = data[i]        
        query = premise2fol(entry['query'])
        options = [premise2fol(option) for option in entry['options']]
        answer = premise2fol(entry['answer'])
        rules = rules2fol(entry['rules'])

        new_entry = {'query': query, 'options': options, 'answer': answer, 'rules': rules}
        fol_data.append(new_entry)
    with open('data/copa_fol.pickle', 'wb') as f:
        pickle.dump(fol_data, f)


elif mode == 'KB Forming':
    with open('data/copa_fol.pickle', 'rb') as f:
        data = pickle.load(f)

    kb = set()
    kb_str = []
    for i in tqdm.tqdm(range(len(data))):
        query = data[i]['query']
        options = data[i]['options']
        rules = data[i]['rules']
        answer = data[i]['answer']
        for rule in rules:
            if isinstance(rule, ForAll):
                if str(rule) not in kb_str:
                    kb_str.append(str(rule))
                    kb.add(rule)

    kb = list(kb)
    with open('data/COPA_KB.pickle', 'wb') as f:
        pickle.dump(kb, f)




elif mode == 'KB Forming NL':
    with open('data/copa.json', 'r') as f:
        data = json.load(f)

    kb_nl = []
    for i in tqdm.tqdm(range(len(data))):
        query = data[i]['query']
        options = data[i]['options']
        rules = data[i]['rules']
        answer = data[i]['answer']
        for rule in rules:
            kb_nl.append(rule)

    with open('data/COPA_KB_NL.json', 'w') as f:
        json.dump(kb_nl, f, indent=4)


elif mode == 'Lambada_preparation':
    with open('data/copa.json', 'r') as f:
        data = json.load(f)
    
    lambada_data_correct = []
    lambada_data_incorrect = []
    for i in tqdm.tqdm(range(143)):
        query = data[i]['query']
        options = data[i]['options']
        answer = data[i]['answer']
        rules = data[i]['rules']
        for option in options:
            if option == answer:
                correct_option = option
            else:
                incorrect_option = option
        body_text = " ".join([str(rule) for rule in rules])
        lambada_data_correct.append({'id': i, 'body_text': body_text, 'world_model': [], 'goal': correct_option, 'label': True, 'program': []})
        lambada_data_incorrect.append({'id': i, 'body_text': body_text, 'world_model': [], 'goal': incorrect_option, 'label': False, 'program': []})

    with open('data/copa_lambada_correct.json', 'w') as f:
        json.dump(lambada_data_correct, f, indent=4)

    with open('data/copa_lambada_incorrect.json', 'w') as f:
        json.dump(lambada_data_incorrect, f, indent=4)




        
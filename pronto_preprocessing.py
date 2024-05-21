import re, tqdm
import json, pickle
from agent.llm.llm_actions import *
from utils.logic.fol import *
from agent.agent import *


def _str2fol_str(text):
    sentences = text.lower().split('. ')
    sentences = [sentence + "." if not sentence.endswith('.') else sentence for sentence in sentences]
    fol_str_list = []
    for sentence in sentences:
        fol_str = ConvertFOLPronto('agent/llm/llm_prompts')(sentence=sentence)
        fol_str = re.sub(r'\((?!(x)\)).*?\)', r'(x)', fol_str)
        fol_str_list.append(fol_str)
    return fol_str_list

def goal_to_sentence(input_str):
    # Split the input string into parts
    parts = input_str.split('(')
    
    # Extract the variable (x), condition (something), and negate flag (if present)
    variable = parts[1].split(',')[0].strip()
    condition = parts[1].split(',')[1].split(')')[0].strip()
    negate_flag = 'not' in condition
    
    # Remove 'not' if it exists
    if negate_flag:
        condition = condition.replace('not ', '')
    
    # Construct the sentence based on the negate flag
    if negate_flag:
        sentence = f"{variable} is not {condition}."
    else:
        sentence = f"{variable} is {condition}."

    return sentence



def _perturb_rules(rules_fol, dataset_kb, masked_rules_no, misleading_rules_no):
        """
        Perturbs the rules by masking some rules and adding some misleading rules
        """
        # mask some rules except the premise

        universal_rules = rules_fol[:-1]
        premise = rules_fol[-1]
        masked_rules = random.sample(universal_rules, masked_rules_no)
        unmasked_rules = [rule for rule in universal_rules if rule not in masked_rules]
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
        perturbed_rules.append(premise)
        return perturbed_rules

def grammar2str(grammar):
    flatenned_list = flatten_and(grammar)
    # extract the predicate names from the grammar
    processed_predicates = ["not " + str(x.arg.name) if isinstance(x, Not) else str(x.name) for x in flatenned_list]
    # replace underscores with spaces
    processed_predicates = [x.replace("_", " ") for x in processed_predicates]
    return processed_predicates


def query2fol(query_raw):
    # convert query to FOL 
    query = _str2fol_str(query_raw)[0]
    tree, query_fol = parse_fol(query)
    return query_fol

def rules2fol(rules_raw):
    # convert rules to FOL
    rules = _str2fol_str(rules_raw) 
    rules_fol = []
    for rule in rules:
        tree, rule_fol = parse_fol(rule)
        rules_fol.append(rule_fol)
    return rules_fol


mode = 'perturbation'

# extracting query, context, and answer from the log file
if mode == 'text extraction':

    # Define the file paths
    input_file_path = 'data/ProntoQA.log'
    output_file_path = 'data/queries.json'

    # Regular expression pattern to match multi-line queries
    pattern = r'Q: (.+?)True or false:(.+?)(?=\nQ:|\Z)'

    # Initialize a list to store query entries
    query_entries = []

    # Open the log file and read its contents
    with open(input_file_path, 'r') as log_file:
        log_content = log_file.read()

        # Find all matches of the pattern in the log content
        matches = re.findall(pattern, log_content, re.DOTALL)

        # Iterate through the matches and extract query and context
        for match in matches:
            context = match[0].strip()
            query_and_answer = match[1].strip()
            query_full = query_and_answer.split('A:')[0].strip()
            answer_full = query_and_answer.split('A:')[1].strip()

            if answer_full.startswith('Predicted'):
                true_answer_full = answer_full.split("Expected answer:")[1].strip()
                answer = true_answer_full.split("\n")[0].split()[-1].strip("'")
            else:
                answer = answer_full.split()[-1].strip("'")

            # Extract only the part of the query before the newline character
            query = query_full.split('\n')[0].strip()


            # Create a dictionary for the query entry
            query_entry = {'context': context, 'query': query, 'answer': answer}

            # Append the query entry to the list
            query_entries.append(query_entry)

    # Save the query entries as JSON to the output file
    with open(output_file_path, 'w') as json_file:
        json.dump(query_entries, json_file, indent=4)

    print('Queries extracted and saved to queries.json.')


elif mode == 'FOL extraction':
    with open('data/ProntoQA.json', 'r') as f:
        data = json.load(f)

    fol_data = []
    for i in tqdm.tqdm(range(20)):
        query = data[i]['query']
        context = data[i]['context']
        answer = data[i]['answer']

        query_fol = query2fol(query)
        rules_fol = rules2fol(context)
        fol_data.append({'query': query_fol, 'context': rules_fol, 'answer': answer})

    with open('data/ProntoQA.pickle', 'wb') as f:
        pickle.dump(fol_data, f)
    


elif mode == 'KB Forming':
    with open('data/ProntoQA.pickle', 'rb') as f:
        data = pickle.load(f)

    kb = set()
    kb_str = []
    for i in tqdm.tqdm(range(len(data))):
        query = data[i]['query']
        context = data[i]['context']
        answer = data[i]['answer']
        for rule in context:
            if isinstance(rule, ForAll):
                if str(rule) not in kb_str:
                    kb_str.append(str(rule))
                    kb.add(rule)

    kb = list(kb)
    with open('data/ProntoQA_KB.pickle', 'wb') as f:
        pickle.dump(kb, f)




elif mode == 'KB Forming NL':
    with open('data/ProntoQA_KB.pickle', 'rb') as f:
        fol_kb = pickle.load(f)

    kb_nl = []
    for rule in fol_kb:
        lhs = grammar2str(rule.body.lhs)[0]
        rhs = grammar2str(rule.body.rhs)[0]
        kb_nl.append("all " + lhs + "s" " are " + rhs + ".")

    with open('data/ProntoQA_KB_NL.json', 'w') as f:
        json.dump(kb_nl, f, indent=4)


elif mode == 'make list context':
    with open('data/ProntoQA.json', 'r') as f:
        data = json.load(f)

    new_data = []
    for i in tqdm.tqdm(range(len(data))):
        query = data[i]['query']
        # replace the object in query with x
        query_words = query.split()
        query_words[0] = "x"
        query = " ".join(query_words)


        context_text = data[i]['context']
        context_list = re.split(r'\.\s*', context_text.strip())
        context_list = context_list[:-1]
        premise = context_list[-1]
        words = premise.split()
        words[0] = "x"
        replaced_premise = " ".join(words)
        context_list[-1] = replaced_premise
        answer = data[i]['answer']
        new_data.append({'query': query, 'context': context_list, 'answer': answer})

    with open('data/ProntoQA_list_context.json', 'w') as f:
        json.dump(new_data, f, indent=4)



elif mode == 'Lambada_preparation':
    with open('data/ProntoQA_list_context.json', 'r') as f:
        data = json.load(f)
    
    new_data = []
    for i in tqdm.tqdm(range(500)):
        query = data[i]['query']
        context = data[i]['context']
        answer = data[i]['answer']

        query_parts = query.split()
        prop = ' '.join(query_parts[2:])
        if 'not' in prop:
            goal = f"not is(x, {prop[4:-1]})"
        else:
            goal = f"is(x, {prop[:-1]})"
        goal = goal.replace('-', '_')
        if answer.lower() == 'false':
            label = False
        elif answer.lower() == 'true':
            label = True

        body_text = ''
        for rule in context:
            rule = rule.replace('-', '_')
            body_text += rule + '. '
        
        body_text = body_text[:-1]
        

        new_data.append({'id': i, 'body_text': body_text, 'world_model': [], 'goal': goal, 'label': label, 'program': []})
    
    with open('data/ProntoQA_Lambada.json', 'w') as f:
        json.dump(new_data, f, indent=4)


elif mode == 'perturbation':
    masked_rules_no = 2
    misleading_rules_no = 75
    
    with open('data/ProntoQA_KB_NL.json', 'r') as f:
        dataset_kb = json.load(f)

    with open('data/ProntoQA_Lambada.json', 'r') as f:
        data = json.load(f)

    perturbed_data = []
    perturbed_data_lambada = []
    for i in tqdm.tqdm(range(len(data))):
        id = data[i]['id']
        body_text = data[i]['body_text']
        world_model = data[i]['world_model']
        goal = data[i]['goal']
        label = data[i]['label']
        program = data[i]['program']

        rules = body_text.split('. ')
        perturbed_rules = _perturb_rules(rules, dataset_kb, masked_rules_no, misleading_rules_no)
        perturbed_body_text = '. '.join(perturbed_rules)
        perturbed_data_lambada.append({'id': id, 'body_text': perturbed_body_text, 'world_model': world_model, 'goal': goal, 'label': label, 'program': program})

        query = goal_to_sentence(goal)
        perturbed_data.append({'query': query, 'context': perturbed_rules, 'answer': str(label)})



    with open(f'data/ProntoQA_Lambada_misleading{misleading_rules_no}_masked{masked_rules_no}.json', 'w') as f:
        json.dump(perturbed_data_lambada, f, indent=4)

    with open(f'data/ProntoQA_list_context_misleading{misleading_rules_no}_masked{masked_rules_no}.json', 'w') as f:
        json.dump(perturbed_data, f, indent=4)
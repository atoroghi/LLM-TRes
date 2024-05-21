import os
import pickle
import json
import numpy as np
from utils.logic.fol import *


with open('data/ProntoQA_old.pickle', 'rb') as f:
    data = pickle.load(f)

_, new = parse_fol('real(x)')
data[26]['query'] = new

_, new = parse_fol('mersenne_prime(x)')
data[122]['context'][-1] = new

_, new = parse_fol('warm_blooded(x)')
data[294]['query'] = new

_, new = parse_fol('FOR_ALL x, lepidopteran(x) => insect(x)')
data[357]['context'][-4] = new

_, new = parse_fol('mersenne_prime(x)')
data[364]['context'][-1] = new

_, new = parse_fol('FOR_ALL x, lepidopteran(x) => insect(x)')
data[392]['context'][-5] = new

_, new = parse_fol('FOR_ALL x, lepidopteran(x) => insect(x)')
data[403]['context'][-4] = new

_, new = parse_fol('FOR_ALL x, lepidopteran(x) => insect(x)')
data[439]['context'][-4] = new

_, new = parse_fol('FOR_ALL x, lepidopteran(x) => insect(x)')
data[446]['context'][-4] = new

_, new = parse_fol('FOR_ALL x, lepidopteran(x) => insect(x)')
data[459]['context'][-4] = new

_, new = parse_fol('FOR_ALL x, lepidopteran(x) => insect(x)')
data[471]['context'][-4] = new

_, new = parse_fol('~small(x)')
data[492]['query'] = new

_, new = parse_fol('FOR_ALL x, lepidopteran(x) => insect(x)')
data[574]['context'][-5] = new

_, new = parse_fol('FOR_ALL x, lepidopteran(x) => insect(x)')
data[674]['context'][-4] = new

_, new = parse_fol('FOR_ALL x, lepidopteran(x) => insect(x)')
data[697]['context'][-5] = new

_, new = parse_fol('FOR_ALL x, lepidopteran(x) => insect(x)')
data[707]['context'][-4] = new

_, new = parse_fol('FOR_ALL x, lepidopteran(x) => insect(x)')
data[741]['context'][-5] = new

_, new = parse_fol('FOR_ALL x, lepidopteran(x) => insect(x)')
data[813]['context'][-4] = new

_, new = parse_fol('FOR_ALL x, lepidopteran(x) => insect(x)')
data[836]['context'][-5] = new

_, new = parse_fol('FOR_ALL x, lepidopteran(x) => insect(x)')
data[864]['context'][-5] = new

_, new = parse_fol('FOR_ALL x, lepidopteran(x) => insect(x)')
data[964]['context'][-5] = new

_, new = parse_fol('FOR_ALL x, lepidopteran(x) => insect(x)')
data[972]['context'][-4] = new

with open('data/ProntoQA.pickle', 'wb') as f:
    pickle.dump(data, f)




# directory_path = "data/firsts"

# all_queries_fol = []

# folders = os.listdir(directory_path)
# folders.sort()


# for folder in folders:
#     folder_path = os.path.join(directory_path, folder)
#     pickle_file_path = os.path.join(folder_path, "prontoQA.pickle")

#     if os.path.exists(pickle_file_path):
#         with open(pickle_file_path, 'rb') as f:
#             data = pickle.load(f)
#             print(f"Loaded {folder_path}")
#             all_queries_fol.extend(data)

# print(len(all_queries_fol))


# with open('data/ProntoQA.pickle', 'wb') as f:
#     pickle.dump(all_queries_fol, f)
# print("Saved data to ProntoQA.pickle")





# with open('data/ProntoQA.json', 'r') as f:
#     text_data = json.load(f)

# rules_lens = []
# new_data = []
# for i in range(len(all_queries_fol)):
#     number = i
#     query = text_data[i]['query']
#     query_fol = all_queries_fol[i]['query']
#     context = text_data[i]['context']
#     rules_fol = all_queries_fol[i]['context']
#     answer = text_data[i]['answer']

#     rules_lens.append(len(rules_fol))

#     new_data.append({'number': number ,'query': str(query), 'query_fol': str(query_fol),
#                       'context': context, 'rules_fol': str(rules_fol), 'answer': answer})

# print("average Rules lengths: ", np.mean(rules_lens))

# with open('data/ProntoQA_withfol.json', 'w') as f:
#     json.dump(new_data, f, indent=4)
#     print("Saved data to ProntoQA_withfol.json")

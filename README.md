
Thank you for visiting this repository!
This repository contains the implementation of our EMNLP-24 paper **"Verifiable, Debuggable, and Repairable Commonsense Logical Reasoning via LLM-based Theory Resolution"**.

In order to use the code, please follow these steps:

## 1- Install requirements
~~~
pip install -r requirements.txt
~~~

## 2- Running Experiments
You can run our model on prontoQA and COPA-SSE using commands like the following:
~~~
python -m run ----dataset_name ProntoQA --scoring_method GD\ resolution --masked_rules 0 --misleading_rules 0 --experiment_name test
~~~

Here, "masked_rules" and "misleading_rules" arguemnts represent the number of rules you would like to ablate from the KB or add to the KB using axioms from other queries.

You can also try GPT baseline purely using the BART entailment model by switching the "scoring_method" argument to "monolithic llm" or "pure_entailment" respectively.

For running other LLMs, please refer to the "Other_LLMs.ipynb" notebook.

For running our model on Recipe-MPR, please run the notebook "recipe-mpr.ipynb".


## Citation
If you find our work useful, please consider giving a 🌟 to our repo and citing our paper.

```text
@inproceedings{toroghi2024verifiable,
  title={Verifiable, debuggable, and repairable commonsense logical reasoning via llm-based theory resolution},
  author={Toroghi, Armin and Guo, Willis and Pesaranghader, Ali and Sanner, Scott},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages={6634--6652},
  year={2024}
}
```

Thank you for your attention!


## Directory Structure


```text
/
├── agent/
│   ├── agent.py
│   └── llm/
│       ├── llm_actions.py
│       └── llm_prompts/
│           ├── ConsultGPT.yaml
│           ├── ConvertFOLCOPAForall.yaml
│           ├── ConvertFOLCOPAPremise.yaml
│           ├── ConvertFOLPronto.yaml
│           ├── ExtractOptionPreds.yaml
│           ├── ExtractOptionPreds_rev2.yaml
│           ├── ExtractOptionPreds_rev3.yaml
│           ├── ExtractQueryPreds.yaml
│           ├── ExtractQueryPreds_rev2.yaml
│           ├── ExtractQueryPreds_rev3.yaml
│           ├── ExtractQueryPreds_rev4.yaml
│           ├── GetGPTRecipe.yaml
│           ├── GetMonolithicProof.yaml
│           ├── GetMonolithicProofCOPA.yaml
│           └── NegatePreds.yaml
├── copa_preprocessing.py
├── data/
│   ├── copa.json
│   ├── copa_fol.pickle
│   ├── copa_folstr.json
│   ├── COPA_KB.pickle
│   ├── COPA_KB_NL.json
│   ├── data.py
│   ├── ProntoQA.json
│   ├── ProntoQA.log
│   ├── ProntoQA.pickle
│   ├── ProntoQA_KB.pickle
│   ├── ProntoQA_KB_NL.json
│   ├── ProntoQA_Lambada.json
│   ├── ProntoQA_Lambada_misleading0_masked1.json
│   ├── ProntoQA_Lambada_misleading0_masked2.json
│   ├── ProntoQA_Lambada_misleading0_masked3.json
│   ├── ProntoQA_Lambada_misleading0_masked4.json
│   ├── ProntoQA_Lambada_misleading0_masked5.json
│   ├── ProntoQA_Lambada_misleading0_masked6.json
│   ├── ProntoQA_Lambada_misleading15_masked2.json
│   ├── ProntoQA_Lambada_misleading30_masked2.json
│   ├── ProntoQA_Lambada_misleading45_masked2.json
│   ├── ProntoQA_Lambada_misleading60_masked2.json
│   ├── ProntoQA_Lambada_misleading75_masked2.json
│   ├── ProntoQA_list_context.json
│   ├── ProntoQA_list_context_misleading0_masked1.json
│   ├── ProntoQA_list_context_misleading0_masked2.json
│   ├── ProntoQA_list_context_misleading0_masked3.json
│   ├── ProntoQA_list_context_misleading0_masked4.json
│   ├── ProntoQA_list_context_misleading0_masked5.json
│   ├── ProntoQA_list_context_misleading0_masked6.json
│   ├── ProntoQA_list_context_misleading15_masked2.json
│   ├── ProntoQA_list_context_misleading30_masked2.json
│   ├── ProntoQA_list_context_misleading45_masked2.json
│   ├── ProntoQA_list_context_misleading60_masked2.json
│   ├── ProntoQA_list_context_misleading75_masked2.json
│   ├── ProntoQA_withfol.json
│   └── Recipe-MPR.json
├── OtherLLMs.ipynb
├── pronto_preprocessing.py
├── prontodata.py
├── Query_Predicates.json
├── README.md
├── recipe-mpr.ipynb
├── recipe_mpr_final/
│   ├── caches/
│   │   ├── aspects_cache.json
│   │   ├── entailment_cache_2.pkl
│   │   └── negations_cache.json
│   ├── logs/
│   │   ├── hybrid.txt
│   │   ├── monolithic-llm.json
│   │   └── monolithic-nli.json
│   ├── prompts/
│   │   ├── aspects.yaml
│   │   ├── monolithic_llm.yaml
│   │   └── negations.yaml
│   └── recipe-mpr.ipynb
├── requirements.txt
├── run.py
├── test.py
├── treegen.py
└── utils/
    ├── embeddings_utils.py
    ├── llm/
    │   ├── chain.py
    │   ├── clarifier.py
    │   ├── llm.py
    │   └── prompt.py
    ├── logger.py
    ├── logic/
    │   ├── fol.lark
    │   ├── fol.py
    │   ├── nli.py
    │   ├── prioritizer.py
    │   ├── ProntoQA_withfol.json
    │   └── queue.py
    └── retriever.py
```
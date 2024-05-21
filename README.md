
Thank you for visiting this repository!
This repository contains the implementation of our NeurIPS-24 paper **Verifiable, Debuggable, and Repairable Commonsense Logical Reasoning via LLM-based Theory Resolution**.

In order to use the code, please follow these steps:

## 1- Install requirements
~~~
pip install -r requirements.txt
~~~

## 2- Running Experiments for ProntoQA and COPA-SSE
You can run our model on these datasets using commands like the following:
~~~
python -m run ----dataset_name ProntoQA --scoring_method GD resolution --masked_rules 0 --misleading_rules 0 --experiment_name test
~~~

Here, "masked_rules" and "misleading_rules" arguemnts represent the number of rules you would like to ablate from the KB or add to the KB using axioms from other queries.

You can also try GPT baseline by switching the "scoring_method" argument to "monolithic llm".

For running other LLMs, please refer to the "Other_LLMs.ipynb" notebook.

For running our model on Recipe-MPR, please run the notebook "recipe-mpr.ipynb".


Thank you for your attention!

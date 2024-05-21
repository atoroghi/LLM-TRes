# custom logger to save LLM responses and resolution processes 
class logger:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

    def __call__(self, variable_name, variable_value):
        with open(f'{self.experiment_name}_records.txt', 'a') as file:
            file.write(f"{variable_name}: {variable_value}\n")

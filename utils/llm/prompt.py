from ruamel.yaml import YAML
from jinja2 import Template


class Prompt:
    def __init__(self, prompt_file):
        yaml = YAML()

        with open(prompt_file, 'r') as f:
            self.prompt_info = yaml.load(f)

        self.input_template = Template(
            self.prompt_info['input_template'], trim_blocks=True, lstrip_blocks=True)

    def __call__(self, **input):
        messages = []

        messages.append(self._create_system_message())

        try:
            messages += self._create_few_shot_messages()
        except:
            pass

        input_message = {'role': 'user', 'content': self._format_input(input)}
        messages.append(input_message)

        return messages

    def _create_system_message(self):
        return {'role': 'system', 'content': self.prompt_info['system']}

    def _create_few_shot_messages(self):
        messages = []

        for example in self.prompt_info['few_shot']:
            input = self._format_input(example['input'])
            output = example['output']

            user_msg = {'role': 'user', 'content': input}
            assistant_msg = {'role': 'assistant', 'content': output}

            messages += [user_msg, assistant_msg]

        return messages

    def _format_input(self, input):
        return self.input_template.render(**input)

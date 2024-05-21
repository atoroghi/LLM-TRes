from abc import ABC, abstractmethod

from utils.llm.llm import LLM
from utils.llm.prompt import Prompt


class Chain(ABC):
    def __init__(self, prompt_f):
        self.prompt = Prompt(prompt_f)
        self.llm = LLM()

    def __call__(self, **prompt_input):
        prompt_input = {k.upper(): v for k, v in prompt_input.items()}

        messages = self.prompt(**prompt_input) 
        response = self.llm(messages)
        result = self.parse_response(response)

        #return messages, response, result
        return result
    
    @abstractmethod
    def parse_response(self, response):
        """Parse the LLM's response.
        
        Args:
            response (str): the LLM's response.
        
        Returns:
            The parsed response.
        """

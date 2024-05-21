from utils.llm.chain import Chain

class GetGPTRecipe(Chain):
    def __init__(self, prompts_dir):
        prompt_f = f"{prompts_dir}/GetGPTRecipe.yaml"
        super().__init__(prompt_f)
    
    def parse_response(self, response):
        try:
            parts = response.strip().split("Therefore,")
            final_answer = parts[1]
        except:
            final_answer = response
        return final_answer, response

class ConvertFOLCOPAPremise(Chain):
    def __init__(self, prompts_dir):
        prompt_f = f"{prompts_dir}/ConvertFOLCOPAPremise.yaml"
        super().__init__(prompt_f)
    def parse_response(self, response):
        return response

class ConvertFOLCOPAForall(Chain):
    def __init__(self, prompts_dir):
        prompt_f = f"{prompts_dir}/ConvertFOLCOPAForall.yaml"
        super().__init__(prompt_f)
    def parse_response(self, response):
        return response

class ConvertFOLPronto(Chain):
    def __init__(self, prompts_dir):
        prompt_f = f"{prompts_dir}/ConvertFOLPronto.yaml"
        super().__init__(prompt_f)

    def parse_response(self, response):
        return response

class GetMonolithicProofCOPA(Chain):
    def __init__(self, prompts_dir):
        prompt_f = f"{prompts_dir}/GetMonolithicProofCOPA.yaml"
        super().__init__(prompt_f)

    def parse_response(self, response):
        parts = response.strip().split("\n")
        final_answer = parts[0]
        return final_answer, response

class GetMonolithicProof(Chain):
    def __init__(self, prompts_dir):
        prompt_f = f"{prompts_dir}/GetMonolithicProof.yaml"
        super().__init__(prompt_f)

    def parse_response(self, response):
        if 'false' in response.lower() and 'true' not in response.lower():
            final_answer = 'false'
        elif 'true' in response.lower() and 'false' not in response.lower():
            final_answer = 'true'
        else:
            final_answer = 'misformatted answer'
        return final_answer, response


class ConsultGPT(Chain):
    def __init__(self, prompts_dir):
        prompt_f = f"{prompts_dir}/ConsultGPT.yaml"
        super().__init__(prompt_f)

    def parse_response(self, response):
        return response

class ExtractQueryPreds(Chain):
    def __init__(self, prompts_dir):
        prompt_f = f"{prompts_dir}/ExtractQueryPreds_rev4.yaml"
        super().__init__(prompt_f)

    def parse_response(self, response):
        #make it compatible with the AIMA code
        predicates_list = response.split(", ")
        processed_predicates = [x.replace(" ", "_").lower().rstrip('.') for x in predicates_list]
        return processed_predicates
    
class ExtractOptionPreds(Chain):
    def __init__(self, prompts_dir):
        prompt_f = f"{prompts_dir}/ExtractOptionPreds_rev3.yaml"
        super().__init__(prompt_f)


    def parse_response(self, response):
        #make it compatible with the AIMA code
        predicates_list = response.split(" & ")
        processed_predicates = [x.replace(" ", "_").lower().rstrip('.') for x in predicates_list]
        return processed_predicates
    
class NegatePreds(Chain):
    def __init__(self, prompts_dir):
        prompt_f = f"{prompts_dir}/NegatePreds.yaml"
        super().__init__(prompt_f)

    def parse_response(self, response):
        return response
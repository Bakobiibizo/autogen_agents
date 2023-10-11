import os
import json
import autogen
import datasets
from functools import partial
from dotenv import load_dotenv
from typing import Coroutine, List, Dict, Optional

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


config_list: List[Dict[str, str]] = [
    {
        "model": "gpt-4",
        "api_key": openai_api_key
    },
    {
        "model": "gpt-3.5-turbo",
        "api_key": openai_api_key
    }
]


class TuningAgent:
    def __init__(self):
        self.config_list = config_list
        autogen.ChatCompletion.start_logging()
        
    def get_tuning_agent(self):
        return self
    
    def get_logged_history(self):
        return json.dump(autogen.ChatCompletion.logged_history, open("conversations.json", "w"), indent = 4)
        
    def load_dataset(self, seed = 41, n_tune_data = 20, n_test_data = 100):
        data = datasets.load_dataset("openai_humaneval")["test"].shuffle(seed=seed)
        tune_data = [
            {
                "definition": data[x]["prompt"],
                "test": data[x]["test"],
                "entry_point": data[x]["entry_point"],
            }
            for x in range(n_tune_data)
        ]
        test_data = [
            {
                "definition": data[x]["prompt"],
                "test": data[x]["test"],
                "entry_point": data[x]["entry_point"],
            }
            for x in range(n_tune_data, n_tune_data + n_test_data)
        ]
        return tune_data, test_data
    
    def evalutation_with_assertions(
        self, 
        config_list: Dict[str, str], 
        use_docker=False
        ):
        return partial(
            autogen.code_utils.eval_function_completions,
            assertions=partial(
                autogen.code_utils.generate_assertions,
                config_list=config_list
                ),
            use_docker=use_docker,
            )

    def get_tuning_config(self, tune_data, eval_func):
        return {
            "data": f"{tune_data}",
            "metric": "success",
            "mode": "max",
            "eval_func": f"{eval_func}",
            "inference_budget": 0.05,
            "optimization_budget": 3,
            "num_samples": -1,
        }
        
    def tune_openai_model(self, tunning_config: Dict[str, str])-> Coroutine:
        config, analysis = autogen.ChatCompletion.tune_openai_model(tunning_config)
        
        response= autogen.Completion.create(context=analysis, **config)
        return response.gi_yieldfrom
        
def main(ntest: Optional[int], ntune: Optional[int], noise_seed: Optional[int]):
    if not noise_seed:
        seed=noise_seed
    input_seed=seed
    tunner = TuningAgent()
    agent = tunner.get_tuning_agent()
    agent.get_tuning_config(tune_data=agent.load_dataset(seed=input_seed, n_tune_data=ntune)[0], eval_func=agent.evalutation_with_assertions(config_list, use_docker=False))
    agent.load_dataset(n_test_data=ntest, n_tune_data=ntune, seed=input_seed)
    tunner.get_logged_history()
    
    
if __name__ == "__main__":
    main()
    
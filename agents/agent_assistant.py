import os
import json
import autogen
from autogen import AssistantAgent
from dotenv import load_dotenv
from typing import List, Dict, Optional


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


class AgentAssistant(AssistantAgent):
    def __init__(self, name = "assistant"):
        self.name = name
        self.config_list: List[Dict[str, str]] = config_list
        self.assistant_template = AssistantAgent(name=name, llm_config={"seed":41, "config_lilst": config_list})
        self.assistant = self.assistant_agent(name="assistant", llm_config={"seed":41, "config_lilst": config_list}),
        autogen.ChatCompletion.start_logging()
        
    def assistant_agent(self, name: Optional[str]="assistant", llm_config={"seed":41, "config_lilst": config_list})-> AssistantAgent:
        return self.assistant_template(name=name, llm_config=llm_config)
    
    def get_logged_history(self):
        json.dump(autogen.ChatCompletion.logged_history, open("conversations.json", "w"), indent = 4)
 
def main():
    agent = AgentGenerator()
    agent.get_logged_history()
    
if __name__ == "__main__":
    main()
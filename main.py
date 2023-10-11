import os
import autogen
from dotenv import load_dotenv
from typing import List, Dict
from agents.agent_assistant import AssistantAgent
from agents.agent_tuning import TuningAgent
from agents.agent_ipython import IPythonUserProxyAgent
from agents.agent_user_proxy import UserProxyAgent
__all__ = [
    "AssistantAgent",
    "TuningAgent",
    "IPythonUserProxyAgent",
    "UserProxyAgent",
    ]

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

def main(**kwargs):
    agent = AssistantAgent(name="assistant", llm_config={"seed":41, "config_lilst": config_list})
    agent.get_logged_history()
    usr_proxy = UserProxyAgent(name="ipython_proxy", human_input_mode="ALWAYS", is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"), code_execution_config={"max_consecutive_auto_replys": 3})
    usr_proxy.get_logged_history()
    ipython_proxy = IPythonUserProxyAgent(name="ipython_proxy", human_input_mode="ALWAYS", is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"), code_execution_config={"max_consecutive_auto_replys": 3})
    ipython_proxy.get_logged_history()
    ipython_proxy.generate_init_message(**kwargs)
    usr_proxy.max_consecutive_auto_replys= 3

if __name__ =="__main__":
    main()
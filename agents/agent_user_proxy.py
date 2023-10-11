import os
import json
import autogen
from autogen import UserProxyAgent
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
init_message= """Please write a PyQt5 chat application. These are the requirements:
    1. a chat history that displays both the user and agent messages. This should be scrollable
    2. A text input for the user that has send and clear buttons. In addition it should have a toggle to save the current prompt into a list of prompts.
    3. A list of prompts should be displayed in some kind of container on the right hand side of the chat window. The list should be scrollable. You should be able to move the items in the list and drop them into your message box to chain the messages. This could be broken down further into:
    - a list of messages in a container
    - drag and drop functionality
    - option to add, remove and edit items in the list.
    4. An api request made to openai's gpt-4 model. This should be a post request using the openai library. It should send messages in a list of dictionaries like this {"role": "user | assistant | system", "content": "some content"}
    """

class IpythonProxyAgent(UserProxyAgent):
    def __init__(self):
        super().__init__(
            name="ipython_proxy",
            code_execution_config=config_list[1], 
            max_consecutive_auto_reply=5,
            system_message=init_message
            )
        self.config_list = config_list
        self.ipython_agent = self.ipython_proxy_agent(name="ipython_proxy", human_input_mode="ALWAYS", is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"))
        autogen.ChatCompletion.start_logging()
    
    def ipython_proxy_agent(self, name: Optional[str]="ipython_proxy", human_input_mode="ALWAYS", is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"))-> UserProxyAgent:
        return UserProxyAgent(name=name, human_input_mode=human_input_mode, is_termination_msg=is_termination_msg)
    
    def get_logged_history(self):
        return json.dump(autogen.ChatCompletion.logged_history, open("conversations.json", "w"), indent = 4)
        
        
def main():
    agent = IpythonProxyAgent()
    print(agent.get_logged_history())
    
if __name__ == "__main__":
    main()
from typing import Dict, Union
from IPython import get_ipython
import autogen
import json

init_message= """Please write a PyQt5 chat application. These are the requirements:
    1. a chat history that displays both the user and agent messages. This should be scrollable
    2. A text input for the user that has send and clear buttons. In addition it should have a toggle to save the current prompt into a list of prompts.
    3. A list of prompts should be displayed in some kind of container on the right hand side of the chat window. The list should be scrollable. You should be able to move the items in the list and drop them into your message box to chain the messages. This could be broken down further into:
    - a list of messages in a container
    - drag and drop functionality
    - option to add, remove and edit items in the list.
    4. An api request made to openai's gpt-4 model. This should be a post request using the openai library. It should send messages in a list of dictionaries like this {"role": "user | assistant | system", "content": "some content"}
    """


class IPythonUserProxyAgent(autogen.UserProxyAgent):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._ipython = get_ipython()

    def generate_init_message(self, *args, **kwargs) -> Union[str, Dict]:
        return super().generate_init_message(*args, **kwargs) + init_message

    def run_code(self, code, **kwargs):
        result = self._ipython.run_cell("%%capture --no-display cap\n" + code)
        log = self._ipython.ev("cap.stdout")
        log += self._ipython.ev("cap.stderr")
        if result.result is not None:
            log += str(result.result)
        exitcode = 0 if result.success else 1
        if result.error_before_exec is not None:
            log += f"\n{result.error_before_exec}"
            exitcode = 1
        if result.error_in_exec is not None:
            log += f"\n{result.error_in_exec}"
            exitcode = 1
        return exitcode, log, None
    
    def get_logged_history(self):
        return json.dump(autogen.ChatCompletion.logged_history, open("conversations.json", "w"), indent = 4)
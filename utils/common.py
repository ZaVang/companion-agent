import re
import os 
import shutil
import json
from utils.path import PERSONA_DB_DIR

MAX_IMPORTANCE = 5
TIME_DECAY_RATE = 0.995
MAX_OBSERVATIONS = 10
EMBEDDING_CACHE_SIZE = 30
MAX_TURNS = 5
PASSWORD = 'xxxxxxx'
DEFAULT_AREA = 'Asia/Shanghai'

mbti_list = [
    "INTP", "INTJ", "INFP", "INFJ",
    "ISTP", "ISTJ", "ISFP", "ISFJ",
    "ENTP", "ENTJ", "ENFP", "ENFJ",
    "ESTP", "ESTJ", "ESFP", "ESFJ"
]

relation_types = [
    "Friend", "Spouse", "Colleague", "Sibling", "Parent", "Child", "Relative",
    "Partner", "Mentor", "Mentee", "Neighbor", "Roommate", "Classmate",
    "Teammate", "Other"
]

daily_schedule_example = """
[
    {
        "start_time": "09:00:00",
        "end_time": null,
        "event": "在自己的床上。",
        "duration": null
    },
    {
        "start_time": "10:00:00",
        "end_time": null,
        "event": "起床，坐到电脑前。",
            "duration": null
    },
    {
        "start_time": "12:00:00",
        "end_time": null,
        "event": "在自己房间里，站在沙发旁。",
        "duration": null
    }
]
"""

schedule_example = """
[
    {
        "start_time": "2023-11-14 09:00:00",
        "event": "在自己的床上。",
        "end_time": null,
        "duration": null
    },
    {
        "start_time": "2023-11-14 10:00:00",
        "event": "起床，坐到电脑前。",
        "end_time": null,
        "duration": null
    },
    {
        "start_time": "2023-11-14 12:00:00",
        "event": "在自己房间里，站在沙发旁。",
        "end_time": null,
        "duration": null
    }
]
"""

def extract_response_components(llm_output: str, parse_action: bool = True):
    """
    Extracts the chat response, thought, and optionally the action from the LLM output.

    :param llm_output: The output string from the LLM.
    :param parse_action: A boolean to decide whether to parse the action or not.
    :return: A tuple containing the chat response, the thought, and optionally the action.
    """
    # Regex patterns to match the chat response and thought
    chat_pattern = r"Response: (.*?)\nThought:"
    thought_pattern = r"Thought: (.*)"

    # Extract the chat and thought using regex
    chat_match = re.search(chat_pattern, llm_output, re.DOTALL)
    thought_match = re.search(thought_pattern, llm_output, re.DOTALL)

    # Extracted chat and thought
    chat = chat_match.group(1).strip() if chat_match else ""
    thought = thought_match.group(1).strip() if thought_match else ""

    # If parse_action is True, also extract the action
    action = ""
    if parse_action:
        action_pattern = r"Action: (.*)"
        action_match = re.search(action_pattern, llm_output, re.DOTALL)
        action = action_match.group(1).strip() if action_match else ""

    return chat, thought, action

def format_retrieved_data(retrieved_data):
    json_strings = []

    for data_dict in retrieved_data:
        # Convert each dictionary into a JSON string
        json_string = json.dumps(data_dict, indent=2, ensure_ascii=False)
        json_strings.append(json_string)

    return json_strings

def get_next_id(directory_path=PERSONA_DB_DIR):
    # 获取目录中的所有文件
    files = os.listdir(directory_path)
    
    # 过滤出所有的 JSON 文件，并提取文件名中的数字部分
    json_files = [file for file in files if file.endswith('.json')]
    ids = [int(file.split('.')[0]) for file in json_files]

    # 如果目录为空，则下一个 ID 为 001
    if not ids:
        return "001"

    # 获取最大的 ID 并加 1
    next_id = max(ids) + 1

    # 返回下一个 ID，格式化为三位数字字符串
    return f"{next_id:03d}"

def get_file_names(folder_path=PERSONA_DB_DIR):
    return [file[:-5] for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    
def clear_folder_contents(folder_path):
    """
    删除指定文件夹及其所有子文件夹中的内容，但保留文件夹结构。

    Args:
        folder_path (str): 要清空的文件夹的路径。
    """
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            os.makedirs(item_path)


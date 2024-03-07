DEFAULT_BEHAVIOR_PROMPT = """
现在时间是{time}，正在执行的动作是{action}
"""

def default_behavior(time: str, s: str):
    print(DEFAULT_BEHAVIOR_PROMPT.format(time=time, action=s))
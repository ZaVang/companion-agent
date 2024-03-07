import os
import json
import argparse

from agent.agent import AgentConfig, Agent
from agent.persona import Persona
from observation.schedule_module import Schedule
from observation.time_module import TimeModule
from observation.time_trigger import TimeTrigger
from memory.memory import Memory
from retrieve.retrieve import retrieve_memory
from utils.path import *
from utils.template import *
from utils.model import model_mapping, get_embedding_model
from utils.common import *


## 启动系统
def launch(args):
    ### 加载agent，包括memory、schedule、persona
    agent = Agent.from_id(args.agent_id)

    ## 加载agent的trigger
    agent_trigger = TimeTrigger(timezone=args.area)
    if args.init_trigger:
        agent_trigger.add_schedule_job(dailyschedule=agent.schedule.daily_schedule)
        agent_trigger.add_schedule_job(dailyschedule=agent.schedule.special_schedule)

    ### 识别用户uid,加载用户的memory和schedule
    memory = Memory.from_id(args.user_id)

    ### 加载embedding model
    emb_model = get_embedding_model(args.emb_model_name)

    memory.embedding_memory(emb_model)

    return agent, agent_trigger, memory, emb_model


def interact(agent, agent_trigger, memory, emb_model, user_id, query, **kwargs):
    ### 获取用户输入和uid，如果uid和当前uid不同，重新加载用户的memory和schedule
    if user_id != memory.info.persona.id:
        memory = Memory.from_id(user_id)
        memory.embedding_memory(emb_model)

    ### 设置适当的prompt

    # 根据query从memory中retrieve信息
    retrieved_data = retrieve_memory(emb_model, query, memory)

    # 获取时间和日期
    request_time = agent_trigger.time_module.get_current_time()
    date = request_time.split(" ")[0]

    # 获取agent的persona和schedule
    PERSONA_PROMPT = agent.persona.to_dict(keep_schedule=False)
    daily_schedule = agent.schedule.daily_schedule.to_dict().get(date, None)

    ### agent回复
    user_persona = memory.info.persona.to_dict(keep_schedule=False)
    chat_history = [i.model_dump() for i in memory.chat_history.chathistory]
    input_dicts = {
        "persona": user_persona,
        "chat_history": chat_history,
        "retrieved_data": retrieved_data,
        "current_time": request_time,
        "daily_schedule": daily_schedule,
    }
    res_prompt = PERSONA_PROMPT + MEMORY_PROMPT.format(**input_dicts) + ANSWER_PROMPT
    response = agent.response(query, res_prompt, **kwargs)

    print(response)

    ### 更新用户的memory
    response_time = agent_trigger.time_module.get_current_time()
    memory.update_history("user", query, request_time)
    memory.update_history("assistant", response, response_time)

    ## 更新observation
    input_dicts["chat_history"] = [
        i.model_dump() for i in memory.chat_history.chathistory
    ]
    obs_prompt = MEMORY_PROMPT.format(**input_dicts) + OBSERVATION_PROMPT
    new_obs = agent.response(None, obs_prompt, **kwargs)
    memory.update_info(new_obs, response_time)
    memory.embedding_memory(emb_model)

    ## 定期更新observations
    if len(memory.info.observation) > MAX_OBSERVATIONS:
        info = {
            "persona": PERSONA_PROMPT,
            "observation": [i.model_dump() for i in memory.info.observation],
            "max_observations": memory.info.max_observations,
        }
        reflection_prompt = REFLECTION_PROMPT.format(**info)
        new_info = agent.response(None, reflection_prompt, **kwargs)
        memory.reflection(new_info)
        memory.embedding_memory(emb_model)

    ## 更新special_schedule
    special_schedule = agent.schedule.special_schedule.to_dict().get(date, None)
    plan_prompt = PLAN_PROMPT.format(
        special_schedule=special_schedule,
        chat_history=input_dicts["chat_history"],
        user_name=memory.info.persona.name,
    )
    new_event_str = agent.response(None, plan_prompt, **kwargs)
    new_event = extract_new_event_from_string(new_event_str)
    if new_event:
        for datetiem, event in new_event.items():
            date, time = datetiem.split(" ")
            time = time[:-3]
            agent.schedule.update_daily_events(date, time, event, special=True)

    ### 如有必要，加载新的job到trigger
    ### TODO: 从agent的schedule中获取新的job


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-uid",
        "--user_id",
        default="000001",
        help="determine which user's memory to use",
    )
    parser.add_argument(
        "-aid", "--agent_id", default="001", help="determine which agent to use"
    )
    parser.add_argument(
        "--init_trigger",
        type=bool,
        default=False,
        help="whether initialize trigger with job in agent schedule",
    )
    parser.add_argument(
        "-a", "--area", default="Asia/Shanghai", help="determine timezone and region"
    )
    parser.add_argument(
        "-m",
        "--model",
        default="GPT3.5",
        help="determine which generative model to use",
    )
    parser.add_argument("-q", "--query", required=True, help="query to answer")
    parser.add_argument(
        "-emb",
        "--emb_model_name",
        default="text2vec",
        help="choose embedding models to cal similarity",
    )
    parser.add_argument(
        "-tk", "--top_k", type=int, default=5, help="retrieve top k observation"
    )
    parser.add_argument("-vb", "--verbose", action="store_true", help="print message")
    args = parser.parse_args()

    agent, agent_trigger, memory, emb_model = launch(args)
    resopnse = interact(agent, agent_trigger, memory, emb_model, **args.__dict__)

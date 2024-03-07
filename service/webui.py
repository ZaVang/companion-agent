import sys
from pathlib import Path
current_directory = Path(__file__).parent.resolve()
project_top_level_directory = current_directory.parent
sys.path.append(str(project_top_level_directory))

import argparse
import os
import json
import uuid
from typing import Literal, Optional
import asyncio
import gradio as gr
from pydantic import UUID1

from agent.persona import (
    Persona, 
    Height, 
    Relationship, 
    Weight, 
    MBTI, 
    LikesAndDislikes, 
    Relationship
)
from agent.agent import Agent
from agent.brain import MasterBrain
from utils.path import PERSONA_DB_DIR
from utils.common import (
    mbti_list, 
    relation_types, 
    daily_schedule_example,
    schedule_example,
)
from observation.schedule_module import (
    DailySchedule,
    Schedule,
    DailyEvent,
    EventInstance
)
from memory.event import ExperienceEvent
from memory.engram import EngramManager

master_brain = MasterBrain.initialize()

async def async_setup(ui: gr.Blocks, share: bool) -> None:
    ui.launch(share=share)


def create_refresh_button(refresh_component, 
                          refresh_method, 
                          refresh_value, 
                          refreshed_args,
                          icon: Optional[str] = None, 
                          variant: Literal["primary", "secondary", "stop"] = "primary",
                          size: Literal["sm", "lg"] = "lg") -> gr.Button:
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args
        for k, v in args.items():
            setattr(refresh_component, k, v)
        return gr.update(**(args or {}))

    refresh_button = gr.Button(value=refresh_value, icon = icon, variant=variant, size=size)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )
    return refresh_button
    

def submit_persona(name, age, gender, 
                    height_value, height_unit, 
                    weight_value, weight_unit,
                    introduction, personality, 
                    mbti, mbti_trait, 
                    likes, dislikes,
                    rel_name1, rel_type1, rel_details1,
                    rel_name2, rel_type2, rel_details2,
                    rel_name3, rel_type3, rel_details3,
                    story1, story2, story3,
                    daily_schedule, schedule
                    ):
    global master_brain
    ## weight and height
    height = Height(height=height_value, unit=height_unit) if height_unit and height_value else None
    weight = Weight(weight=weight_value, unit=weight_unit) if weight_unit and weight_value else None
    
    ## relationship
    def create_relationship(name, rel_type, details):
        if name and rel_type:
            return Relationship(name=name, relation=rel_type, details=details)

    relationship_data = [
        (rel_name1, rel_type1, rel_details1),
        (rel_name2, rel_type2, rel_details2),
        (rel_name3, rel_type3, rel_details3),
    ]
    relationship = [create_relationship(name, rel_type, details) for name, rel_type, details in relationship_data if name and rel_type]
    
    ## schedule
    regular_schedule_data = json.loads(daily_schedule)
    daily_events = [DailyEvent(**event_data) for event_data in regular_schedule_data]
    regular_schedule = DailySchedule(events=daily_events)
    
    schedule_data = json.loads(schedule)
    events = [EventInstance(**event_data) for event_data in schedule_data]
    special_schedule = Schedule(events=events)
    
    ## story
    story_data = [story1, story2, story3]
    story = [ExperienceEvent(actor=name, content=s) for s in story_data if s] # type: ignore
    
    persona_dict = {
        "name": name,
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "introduction": introduction,
        "personality": personality,
        "mbti": MBTI(mbti_type=mbti, trait=mbti_trait) if mbti else None,
        "likes_and_dislikes": LikesAndDislikes(likes=likes.split(';'), dislikes=dislikes.split(';')),
        "relationship": relationship,
        "regular_schedule": regular_schedule,
        "story": story
    }
    persona = Persona(**persona_dict)
    agent = Agent(persona=persona, schedule=special_schedule)
    agent.save()
    
    master_brain.agent_registry.id2name[agent.id] = agent.persona.name
    master_brain.agent_registry.name2id[agent.persona.name] = agent.id
    return f"Persona with name {name} created! You can now use it in the Chat Interface."


def collect_chat(like_data, evt: gr.LikeData):
    # 文件路径
    file_path = 'db/collect/like.json' if evt.liked else 'db/collect/dislike.json'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []
    except json.JSONDecodeError:
        data = []

    data.append(like_data)

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    return "记录成功"

async def chat(agent_name, user_name, query, chat_history, model_name):
    global master_brain
    agent_id = master_brain.agent_registry.name2id[agent_name]
    master_brain.add_agent([agent_id])
    response, thought = await master_brain.handle_interaction(
        agent_id=agent_id, 
        audience=[user_name],
        user=user_name,
        user_input=query,
        model_name=model_name
    )
    
    response = master_brain.agent_brains[agent_id].agent.persona.name + ': ' + response

    # 将用户消息和系统回复添加到聊天历史中
    if not chat_history:
        chat_history = []
    chat_history.append((f'{user_name}: {query}', response))

    return "", chat_history, thought, ""

async def reflection(agent_name, user, model_name):
    global master_brain
    agent_id = master_brain.agent_registry.name2id[agent_name]
    master_brain.add_agent([agent_id])
    return await master_brain.perform_reflection(agent_id, user, model_name, verbose=False)

def save_chat():
    global master_brain
    try:
        master_brain.save_memory()
        return "Save finished"
    except Exception as e:
        return str(e)

def create_chat_tab():
    global master_brain
    with gr.Tab('Chat'):
        with gr.Row():
            with gr.Column():
                agent_name = gr.Dropdown(label="选择Agent name", 
                                        choices=master_brain.get_agent_names(), # type: ignore
                                        interactive=True) 
                agent_refresh_button = create_refresh_button(
                    refresh_component=agent_name, 
                    refresh_method=master_brain.get_agent_names, 
                    refresh_value="Refresh agent list",
                    refreshed_args=lambda: {"choices": master_brain.get_agent_names()},
                    variant='secondary',
                    size='sm'
                )
            user_name = gr.Textbox(label="User name", placeholder="Small Jin")
            model_name = gr.Dropdown(label="选择模型",
                                    choices=['GPT3.5', "GPT4"],
                                    value='GPT3.5',
                                    interactive=True)
        
        with gr.Row():
            with gr.Column(scale=4):
                chat_history = gr.Chatbot(show_copy_button=True, likeable=True)
            with gr.Column(scale=1):
                thought_box = gr.Textbox(label="内心OS:", interactive=False, lines=10)
                message_box = gr.Textbox(label="Message:", interactive=False)
                
        with gr.Column(scale=3):
            query = gr.Textbox(info="query", placeholder="你是谁？")
            with gr.Row():
                clear_btn = gr.ClearButton([query, chat_history, thought_box, message_box], value='清除对话框', variant='secondary')
                show_history_btn = gr.Button(value='显示历史对话', variant='primary')
                chat_btn = gr.Button(value='发送', variant='primary')
            with gr.Row():
                reflection_btn = gr.Button(value='Reflection', variant='primary')
                save_btn = gr.Button(value='保存记忆', variant='primary')
        
        chat_history.like(collect_chat, inputs=chat_history, outputs=message_box)
        
        chat_btn.click(
            chat, 
            inputs=[agent_name, user_name, query, chat_history, model_name],
            outputs=[query, chat_history, thought_box, message_box]
        )
        
        reflection_btn.click(
            reflection,
            inputs = [agent_name, user_name, model_name],
            outputs = [message_box]
        )
        
        save_btn.click(
            save_chat,
            inputs = [],
            outputs = [message_box]
        )

# def visualize_and_display(agent_name: str, memory_type, engram_id):
#     global master_brain
#     agent_id = master_brain.agent_registry.name2id[agent_name]
#     master_brain.add_agent([agent_id])
#     message = "成功输出图像"

#     brain = master_brain.agent_brains[agent_id]
#     try:
#         if 'ShortTermMemory' in memory_type:
#             image, legend = brain.visualize_memory(brain.short_term_memory)

#         if 'EpisodicMemory' in memory_type:
#             image, legend = brain.visualize_memory(brain.episodic_memory)

#         if 'Engram' in memory_type:
#             engram_dict = {}
#             engram_uuids = [uuid.UUID(s) for s in engram_id.split(';')]
#             for engram_uuid in engram_uuids:
#                 engram = brain.episodic_memory.get_engram_by_id(engram_uuid)
#                 if engram:
#                     engram_dict[engram_uuid] = engram
#                 else:
#                     raise ValueError(f"UUID {engram_uuid} 不存在.")
#             engram_manager = EngramManager(engram_dict=engram_dict)
#             image, legend = brain.visualize_memory(engram_manager)

#     except ValueError as ve:
#         message = str(ve)
#     except Exception as e:  # 捕获其他异常
#         message = f"发生错误: {str(e)}"

#     if image:
#         legend_str = ''
#         for k,v in legend.items():
#             legend_str += f'{k}: {v}' + '\n'
            
#         return image, legend_str, message
#     else:
#         return None, None, message
    
# def create_visualization_tab():
#     with gr.Tab("Image Viewer"):
#         with gr.Row():
#             with gr.Column():
#                 agent_name = gr.Dropdown(label="选择Agent name", 
#                                         choices=master_brain.get_agent_names(), 
#                                         interactive=True) 
#                 memory_type = gr.Radio(["ShortTermMemory", "EpisodicMemory", "Engram"], 
#                                         label="选择需要可视化的memory类型",
#                                         info="如果选择可视化某一具体的engram，需要将uuid填在下方的uuid栏中。"
#                                         )
#                 engram_id = gr.Textbox(label="填入需要可视化的engram的uuid，多个id间用';'分割，如果选择可视化另外两项则可忽略",
#                                     interactive=True)
#             with gr.Column():
#                 visualize_btn = gr.Button("Visualize", variant='primary')
#                 visualization_message = gr.Textbox(label='Message', interactive=False)

#         with gr.Row():
#             image_output = gr.Image(label="Visualization")
#         with gr.Row():
#             legend_output = gr.Textbox(label="Legend", interactive=False)

#         visualize_btn.click(
#             visualize_and_display,
#             inputs=[agent_name, memory_type, engram_id],
#             outputs=[image_output, legend_output, visualization_message]
#         )

def create_persona_tab():
    with gr.Tab('Create New Agents'):
        with gr.Row():
            name = gr.Textbox(label="Name", info="The name of the agent.")
            age = gr.Number(label="Age", precision=0, info="The age of the agent.")
            gender = gr.Dropdown(label="Gender", 
                                    choices=['Male', 'Female', 'Other', 'Prefer not to say'],
                                    info="The gender of the agent.")
        
        with gr.Row():
            height_value = gr.Number(label="Height value", info="The height of the agent.")
            height_unit = gr.Dropdown(label="Height unit", 
                                        choices=['cm', 'inch', 'foot'],
                                        info="The unit of the height, cm is recommended.")
            weight_value = gr.Number(label="Weight value", info="The weight of the agent.")
            weight_unit = gr.Dropdown(label="Weight unit", 
                                        choices=['kg', 'pound', 'jin'],
                                        info="The unit of the weight, kg is recommended.")
        
        with gr.Row():
            introduction = gr.Textbox(label="Introduction", info="A brief introduction of the agent.")                
            personality = gr.Textbox(label="Personality", info="Describe the personality of the agent.")
            mbti = gr.Dropdown(label="MBTI", choices=mbti_list)  # type: ignore
            mbti_trait = gr.Radio(["A", "T", "None"], label="MBTI trait", 
                                    info="Trait for MBTI, for example INFJ-A.")
        
        with gr.Row():
            likes = gr.Textbox(label="Likes", info="Likes of the agent, must be splited by ';'.")
            dislikes = gr.Textbox(label="Dislikes", info="Dislikes of the agent, must be splited by ';'.")

        with gr.Row():
            with gr.Row():
                rel_name1 = gr.Textbox(label="Relationship 1 - Name", placeholder="Chandler")
                rel_type1 = gr.Dropdown(label="Type", choices=relation_types) # type: ignore
                rel_details1 = gr.Textbox(label="Details", info="Describe the details of the relationship.",
                                            placeholder="Chandler is Monica's spouse.")
            with gr.Row():
                rel_name2 = gr.Textbox(label="Relationship 2 - Name")
                rel_type2 = gr.Dropdown(label="Type", choices=relation_types) # type: ignore
                rel_details2 = gr.Textbox(label="Details", info="Describe the details of the relationship.")
            with gr.Row():
                rel_name3 = gr.Textbox(label="Relationship 3 - Name")
                rel_type3 = gr.Dropdown(label="Type", choices=relation_types) # type: ignore
                rel_details3 = gr.Textbox(label="Details", info="Describe the details of the relationship.")

        with gr.Row():
            with gr.Row():
                story1 = gr.Textbox(label="Story 1", info="Describe the experience of the agent.",
                                    placeholder="Monica used to be very fat, but she finally became very slim.")
            with gr.Row():    
                story2 = gr.Textbox(label="Story 2", info="Describe the experience of the agent.")
            with gr.Row():
                story3 = gr.Textbox(label="Story 3", info="Describe the experience of the agent.")
                
        with gr.Row():
            daily_schedule = gr.TextArea(label="Daily Schedule of the agent", info="Agent's daily schedule that in json format.",
                                            placeholder=daily_schedule_example)
            schedule = gr.TextArea(label="Schedule of the agent", info="Agent's schedule that in json format.",
                                            placeholder=schedule_example)

        create_button = gr.Button("Create")
        output = gr.Textbox(label="Output")

        create_button.click(
            submit_persona, 
            inputs=[name, age, gender, 
                    height_value, height_unit, 
                    weight_value, weight_unit,
                    introduction, personality, 
                    mbti, mbti_trait, 
                    likes, dislikes,
                    rel_name1, rel_type1, rel_details1,
                    rel_name2, rel_type2, rel_details2,
                    rel_name3, rel_type3, rel_details3,
                    story1, story2, story3,
                    daily_schedule, schedule,
                    ],
            outputs=output
        )

def webui():
    with gr.Blocks(theme=gr.themes.Default()) as ui:
        gr.Markdown(
        """
        <h1 style="text-align: center; font-size: 40px;">一个简易的AI-Companion</h1>
        """
        )
        create_chat_tab()
        # create_visualization_tab()
        create_persona_tab()

    return ui


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action='store_true', help="make link public (used in colab)")
    
    args = parser.parse_args()
    ui = webui()
    asyncio.run(async_setup(ui, args.share))


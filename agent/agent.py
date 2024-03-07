from typing import Optional, List, Dict, Literal
from pydantic import BaseModel, Field, UUID1
from pathlib import Path
import uuid

from agent.persona import Persona
from observation.schedule_module import Schedule
from utils.template import *
from utils.path import AGENT_DB_DIR


class AgentRegistry(BaseModel):
    id2name: Dict[UUID1, str]
    name2id: Dict[str, UUID1]
    
    @classmethod
    def load_registry(cls,
                    file_name: str = 'registry.json',
                    directory: Path = AGENT_DB_DIR) -> 'AgentRegistry':
        file_path = directory / file_name
        if not file_path.exists():
            return cls()
        else:
            with open(file_path, 'r', encoding="utf-8") as f:
                registry_data = f.read()
            return cls.model_validate_json(registry_data)
    
    def save_registry(self,
                    filename: str = 'registry.json',
                    registry_directory: Path = AGENT_DB_DIR) -> None:
        registry_directory.mkdir(exist_ok=True, parents=True)
        
        with open(registry_directory / filename, 'w', encoding="utf-8") as file:
            file.write(self.model_dump_json(exclude={"engrams"}, indent=4))
                    

class Agent(BaseModel):
    id: UUID1 = Field(default_factory=uuid.uuid1)
    # id: str
    persona: Persona
    schedule: Schedule

    @classmethod
    def from_id(cls, agent_id: UUID1) -> "Agent":
        persona = Persona.from_json(f"{str(agent_id)}.json")
        schedule = Schedule.from_json(f"{str(agent_id)}.json")
        item = {
            "id": agent_id,
            "persona": persona,
            "schedule": schedule
        }
        return cls(**item)

    def save(self) -> None:
        self.persona.to_json(f"{str(self.id)}.json")
        self.schedule.to_json(f"{str(self.id)}.json")
        
    def chat_prompt(self, user_input: str,
                    actor: str, 
                    audience: List[str], 
                    chat_history_list: Optional[List[str]]=None,
                    retrieved_data: Optional[List[str]]=None,
                    prompt: str = CHAT_PROMPT) -> str:
        
        if len(audience) > 1:
            audience_str = ', '.join(audience[:-1]) + '和' + audience[-1]
        else:
            audience_str = audience[0]
        
        chat_history = '\n'.join(chat_history_list) if chat_history_list else ''
        format_dict = {
            'audience': audience_str,
            'query': user_input,
            'user': actor,
            'chat_history': chat_history
        }
        if retrieved_data:
            retrieved_memory = '\n'.join(retrieved_data)
            format_dict['retrieved_memory'] = retrieved_memory
            
        chat_prompt = prompt.format(**format_dict)
        
        return chat_prompt
    
    def system_prompt(self, 
                      fields: Optional[List[Literal[
                                          'name',
                                          'age',
                                          'gender',
                                          'height',
                                          'weight',
                                          'introduction', 
                                          'personality', 
                                          'mbti', 
                                          'likes_and_dislikes',    
                                          'relationship', 
                                          'regular_schedule',
                                          'story']]] = None, 
                        include=False) -> str:
        
        if fields is None:
            fields = []

        name_included = 'name' in fields
        if include and name_included:
            # 如果是包含模式且fields中有name，从fields中移除name
            fields.remove('name')
            # 单独提取name
        elif not include and not name_included:
            # 如果是排除模式且fields中没有name，添加name到fields
            fields.append('name')
            
        # 获取剩余的persona信息
        character_info = self.persona.prompt(fields=fields, include=include)

        # 使用name和character_info格式化system prompt
        system_prompt = SYSTEM_PROMPT.format(name=self.persona.name, character_info=character_info)
        
        return system_prompt
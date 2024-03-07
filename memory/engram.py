from typing import List, Dict, Set, Literal, Union, Optional
from pydantic import BaseModel, Field, UUID1, validator
import uuid
from zoneinfo import ZoneInfo
from datetime import datetime
from pathlib import Path

from memory.neuron import NeuronCell
from memory.event import Events
from utils.schema import DateTime
from utils.path import ENGRAM_DB_DIR
from utils.common import DEFAULT_AREA

## TODO: 可以添加根据actor或者audience来筛选神经元的方法
class Engram(BaseModel):
    engram: Dict[str, List[NeuronCell]] = Field(default_factory=lambda: {
        'chat': [],
        'thought': [],
        'reflection': [],
        'perception': [],
        'experience': [],
    })
    scope: Literal["full", "partial"] = Field(default="partial")
    represent: UUID1 = Field(default_factory=uuid.uuid1)
    strength: float = 1
    time: DateTime
    summary: str = Field(default="")
    uuid: UUID1 = Field(default_factory=uuid.uuid1)
    actor: List[str]
    audience: Optional[List[str]] = None
    
    @validator('time', pre=True, always=True)
    def parse_create_time(cls, v):
        if isinstance(v, str):
            naive_datetime = datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
            return naive_datetime.replace(tzinfo=ZoneInfo(DEFAULT_AREA))
        return v 
    
    def add_event(self, events: Union[Events, List[Events]]) -> None:
        """ Adds one or multiple events, represented as NeuronCells, to the appropriate sequence. """
        events = events if isinstance(events, list) else [events]
        for event in events:
            neuron_cell = NeuronCell(event_id=event.event_id, 
                                     event_type=event.event_type, 
                                     create_time=event.create_time,
                                     actor=event.actor,
                                     audience=event.audience)
            
            if self.engram['chat']:
                last_chat_neuron = self.engram['chat'][-1]
                last_chat_neuron.connect_to(neuron_cell)

            self.engram[event.event_type].append(neuron_cell)  
    
    def add_neurons(self, neurons: Union[NeuronCell, List[NeuronCell]]):
        """
        Add a single neuroncell or a list of neuroncells to the engram.
        """
        if isinstance(neurons, NeuronCell):
            self.engram[neurons.event_type].append(neurons)
        elif isinstance(neurons, list):
            for neuron in neurons:
                self.engram[neuron.event_type].append(neuron)
        else:
            raise TypeError(f"Invalid type: {type(neurons)}. Expected NeuronCell or list of NeuronCell.")

    def remove_neuron(self, neuron: Union[UUID1, NeuronCell, List[Union[UUID1, NeuronCell]]]):
        """
        Remove a neuroncell by its ID or by the NeuronCell instance.
        Supports removal of single item or list of items.
        """
        if isinstance(neuron, (UUID1, NeuronCell)):
            neuron_ids = [neuron.event_id if isinstance(neuron, NeuronCell) else neuron]
        elif isinstance(neuron, list):
            neuron_ids = [
                item.event_id if isinstance(item, NeuronCell) else item for item in neuron
            ]
        else:
            raise TypeError(f"Invalid type: {type(neuron)}. Expected UUID1, NeuronCell or list of either.")

        # Iterate over each type of event and remove neurons
        for event_type, neuron_list in self.engram.items():
            self.engram[event_type] = [neuron for neuron in neuron_list if neuron.event_id not in neuron_ids]

        # Additionally, remove connections to the removed neurons in remaining NeuronCells
        for neuron_list in self.engram.values():
            for neuron in neuron_list:
                neuron.outgoing_connections = {conn for conn in neuron.outgoing_connections if conn.target_id not in neuron_ids}
                neuron.incoming_connections = {conn for conn in neuron.incoming_connections if conn.target_id not in neuron_ids}
    
    def retain_relevant_neurons(self, max_turns: int):
        if not self.is_empty():
            # Update the 'chat' neuron list to the last max_turns items
            self.engram['chat'] = self.engram['chat'][-max_turns:]

            # Find connected neuron IDs from the last max_turns chat neurons
            connected_neuron_ids = set()
            for chat_neuron in self.engram['chat']:
                for connection in chat_neuron.outgoing_connections:
                    connected_neuron_ids.add(connection.target_id)

            # Retain neurons in other sequences that are connected to the last max_turns chat neurons
            for neuron_type in ['thought', 'reflection', 'perception', 'experience']:
                self.engram[neuron_type] = [neuron for neuron in self.engram[neuron_type] if neuron.event_id in connected_neuron_ids]
        
    def get_neuron_by_id(self, neuron_id: UUID1) -> Optional[NeuronCell]:
        """
        Retrieves a neuron by its ID from the engram.

        Args:
            neuron_id (UUID1): The ID of the neuron to retrieve.

        Returns:
            NeuronCell: The neuron with the specified ID, if found.
        """
        for sequence in self.engram.values():
            for neuron in sequence:
                if neuron.event_id == neuron_id:
                    return neuron
        return None
    
    def get_all_neurons(self):
        """
        Retrieves a set of all unique NeuronCells in the engram.

        This method iterates through all event types in the engram and collects
        each NeuronCell into a set, ensuring all NeuronCells are unique.

        Returns:
            Set[NeuronCell]: A set of unique NeuronCells.
        """
        unique_neurons = set()
        for event_type, neurons in self.engram.items():
            unique_neurons.update(neurons)

        return unique_neurons

    @classmethod
    def from_json(cls, file_name: str, 
                  directory: Path = ENGRAM_DB_DIR, 
                  scope: Literal['full', 'partial'] = 'partial') -> 'Engram':
        
        with open(directory / scope / file_name, 'r') as f:
            engram = cls.model_validate_json(f.read())
        return engram
    
    def to_json(self, file_name: str, 
                directory: Path = ENGRAM_DB_DIR, 
                scope: Literal['full', 'partial'] = 'partial') -> None:

        directory.mkdir(exist_ok=True, parents=True)
        
        with open(directory / scope/ file_name, 'w', encoding="utf-8") as file:
            file.write(self.model_dump_json(indent=4))
    
    def is_empty(self) -> bool:
        return all(len(cells) == 0 for cells in self.engram.values())
            
    def recall(self, timestamp: datetime, create_copy: bool = False) -> 'Engram':
        if create_copy:
            engram_copy = self.model_copy(deep=True)
            engram_copy._remove_cells_after(timestamp)
            return engram_copy
        else:
            self._remove_cells_after(timestamp)

    def _remove_cells_after(self, timestamp: datetime) -> None:
        for category in self.engram:
            self.engram[category] = [
                cell for cell in self.engram[category]
                if cell.create_time <= timestamp
            ]

            for cell in self.engram[category]:
                cell.outgoing_connections = [
                    conn for conn in cell.outgoing_connections
                    if conn.create_time <= timestamp
                ]
                cell.incoming_connections = [
                    conn for conn in cell.incoming_connections
                    if conn.create_time <= timestamp
                ]
        return self
      
            
class RegistryMetadata(BaseModel):
    time: DateTime
    summary: str
    strength: float
    actor: List[str]
    audience: Optional[List[str]] = None
    scope: Literal['full', 'partial'] = 'partial'
    
    @validator('time', pre=True, always=True)
    def parse_create_time(cls, v):
        if isinstance(v, str):
            naive_datetime = datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
            return naive_datetime.replace(tzinfo=ZoneInfo(DEFAULT_AREA))
        return v 
    
    class Config:
        arbitrary_types_allowed=True
            

class EngramManager(BaseModel):
    engram_dict: Dict[UUID1, Engram] = Field(default_factory=dict)

    def add_engram(self, engram: Engram) -> None:
        """将一个新的 Engram 对象添加到管理器中。"""
        self.engram_dict[engram.uuid] = engram

    def remove_engram(self, engram_id: UUID1) -> None:
        """从管理器中移除一个 Engram 对象。"""
        self.engram_dict.pop(engram_id, None)

    def get_engram_by_id(self, engram_id: UUID1) -> Optional[Engram]:
        """根据 ID 检索特定的 Engram 对象。"""
        return self.engram_dict.get(engram_id)

    def is_empty(self) -> bool:
        return all(e.is_empty() for e in self.engram_dict.values())

    @classmethod
    def from_engram_list(cls, engram_list: List[Engram]) -> 'EngramManager':
        """
        从 Engram 对象列表创建 EngramManager 实例。

        Args:
            engram_list (List[Engram]): Engram 对象的列表。

        Returns:
            EngramManager: 新创建的 EngramManager 实例。
        """
        engram_dict = {engram.uuid: engram for engram in engram_list}
        return cls(engram_dict=engram_dict)

    def get_neuron_by_id(self, neuron_id: UUID1) -> Optional[NeuronCell]:
        """
        从所有 Engram 中根据神经元 ID 检索神经元。
        """
        for engram in self.engram_dict.values():
            neuron = engram.get_neuron_by_id(neuron_id)
            if neuron:
                return neuron
        return None
    
    def get_all_neurons(self) -> Set[NeuronCell]:
        """
        返回所有 Engram 中的所有 NeuronCell 的集合，避免重复。

        Returns:
            Set[NeuronCell]: 包含所有独特 NeuronCell 对象的集合。
        """
        all_neurons = set()
        for engram in self.engram_dict.values():
            all_neurons.update(engram.get_all_neurons())
        return all_neurons
    
    def recall(self, timestamp: datetime, create_copy: bool = False):
        if create_copy:
            new_engram_manager = EngramManager()
            for uuid, engram in self.engram_dict.items():
                recalled_engram = engram.recall(timestamp, create_copy=True)
                if not recalled_engram.is_empty():
                    new_engram_manager.engram_dict[uuid] = recalled_engram
            return new_engram_manager
        else:
            for uuid, engram in list(self.engram_dict.items()):
                engram.recall(timestamp, create_copy=False)
                if engram.is_empty():
                    del self.engram_dict[uuid]
            return self
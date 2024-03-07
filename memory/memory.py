from tkinter.tix import MAX
from typing import List, Dict, Optional, Union
from networkx import is_empty
from pydantic import BaseModel, Field, UUID1
from pathlib import Path
from datetime import datetime

from memory.event import Events
from memory.neuron import NeuronCell
from memory.engram import Engram, RegistryMetadata, EngramManager
from utils.path import REGISTRY_DB_DIR, SHORT_TERM_MEMORY_DB_DIR
from utils.common import MAX_TURNS

## TODO: 可以添加根据audience来筛选记忆的方法
class ShortTermMemory(BaseModel):
    sequences: Dict[str, Engram] = Field(default_factory=dict)
    
    def is_empty(self) -> bool:
        return all(len(engram.engram) == 0 for engram in self.sequences.values())
      
    def get_neuron_by_id(self, neuron_id: UUID1) -> Optional[NeuronCell]:
        for engram in self.sequences.values():
            neuron = engram.get_neuron_by_id(neuron_id)
            if neuron:
                return neuron
        return None
    
    def add_event(self, events: Union[Events, List[Events]]) -> None:
        events = events if isinstance(events, list) else [events]
        for event in events:
            audience = event.audience if isinstance(event.audience, list) else [event.audience]
            for individual_audience in audience:
                # Check if an Engram for the individual audience member already exists
                if individual_audience not in self.sequences:
                    self.sequences[individual_audience] = Engram(actor=[event.actor], audience=[individual_audience])                    
                # Add the event to the specific Engram of the audience member
                self.sequences[individual_audience].add_event(event)
    
    def recall(self, timestamp: datetime, create_copy: bool = False) -> Optional['ShortTermMemory']:
        """
        Initiates a recall process on all stored engrams, either modifying them in place or returning a new STM instance with the recalled engrams.
        """
        if create_copy:
            stm_copy = self.model_copy(deep=True)
            for audience, engram in stm_copy.sequences.items():
                stm_copy.sequences[audience] = engram.recall(timestamp, create_copy=True)
            return stm_copy
        else:
            for engram in self.sequences.values():
                engram.recall(timestamp)
            return None
                
    def keep_partial_stm(self, max_turns: int = MAX_TURNS) -> None:
        """
        Keep the last MAX_TURNS chat neurons and other types of neurons that are connected to the last MAX_TURNS chat neurons.
        """
        if not self.is_empty():
            # Iterate over all engrams and instruct them to keep relevant neurons
            for audience, engram in self.sequences.items():
                # Assuming a method in Engram that handles the logic of retaining relevant neurons
                engram.retain_relevant_neurons(max_turns)

    def save_memory(self, id: str, file_name: str = 'short_term_memory.json', 
                directory: Path = SHORT_TERM_MEMORY_DB_DIR) -> None:
        file_path = directory / id
        file_path.mkdir(exist_ok=True, parents=True)
        self.keep_partial_stm()
                
        with open(file_path / file_name, 'w', encoding="utf-8") as file:
            file.write(self.model_dump_json(indent=4))
    
    @classmethod
    def load_memory(cls, id: str, file_name: str = 'short_term_memory.json',
                  directory: Path = SHORT_TERM_MEMORY_DB_DIR) -> 'ShortTermMemory':
        file_path = directory / id / file_name
        if not file_path.exists():
            return cls()
        else:
            with open(file_path, 'r', encoding="utf-8") as f:
                stm_data = f.read()
            return cls.model_validate_json(stm_data)
        

class EpisodicMemory(BaseModel):
    """
    The EpisodicMemory class is responsible for storing and managing long-term episodic memories.
    It maintains a registry of engrams along with metadata about them and organizes engrams by audience.
    """
    registry: Dict[str, Dict[UUID1, RegistryMetadata]] = Field(default_factory=dict)
    engram_managers: Dict[str, EngramManager] = Field(default_factory=dict)
    
    def is_empty(self) -> bool:
        """Checks if there are any engrams across all audiences."""
        return all(manager.is_empty() for manager in self.engram_managers.values())

    def consolidate(self, audience: str, engram: Engram) -> None:
        """Incorporates a given engram into episodic memory for a specific audience."""
        if audience not in self.registry:
            self.registry[audience] = {}
        if engram.uuid not in self.registry[audience]:
            self.add_to_registry(audience, engram)
        # Now adding engram under a specific audience
        self.engram_managers[audience].add_engram(engram.uuid, engram)
        
    def add_to_registry(self, audience:str, engram: Engram) -> None:
        """
        Registers an engram with its metadata if it is not already in the registry.
        This metadata includes the timestamp and a summary of the engram.
        """
        if engram.uuid not in self.registry[audience]:
            metadata = RegistryMetadata(
                time=engram.time,
                summary=engram.summary,
                strength=engram.strength,
                actor = engram.actor,
                audience= engram.audience
            )
            self.registry[audience][engram.uuid] = metadata 
            
    def get_engram_by_id(self, engram_id: UUID1) -> Optional[Engram]:
        """Get an engram by its unique identifier."""
        for audience in self.engram_managers.keys():
            manager = self.engram_managers[audience]
            registry = self.registry[audience]
            if engram_id not in manager.engram_dict and engram_id in registry:
                self.load_partial_engram(audience, engram_id)
            engram = manager.get_engram_by_id(engram_id)
            if engram is not None:
                return engram
        return None

    def forget_engram_by_id(self, engram_id: UUID1) -> None:
        """Removes an engram from episodic memory based on its unique identifier."""
        for audience in self.engram_managers.keys():
            manager = self.engram_managers[audience]
            registry = self.registry[audience]
            manager.remove_engram(engram_id)
            registry.pop(engram_id, None)     
        
    def save_memory(self, id: str, 
                    filename: str = 'registry.json',
                    registry_directory: Path = REGISTRY_DB_DIR) -> None:
        file_path = registry_directory / id
        file_path.mkdir(exist_ok=True, parents=True)
        
        with open(file_path / filename, 'w', encoding="utf-8") as file:
            file.write(self.model_dump_json(exclude={"engrams"}, indent=4))
            
        for audience, manager in self.engram_managers.items():
            for uuid, engram in manager.engram_dict.items():
                metadata = self.registry[audience][uuid]
                engram.to_json(f"{str(uuid)}.json", scope=metadata.scope)
    
    def recall(self, timestamp: datetime, create_copy: bool = False):
        if create_copy:
            new_memory = EpisodicMemory()
            for audience, manager in self.engram_managers.items():
                new_memory.engram_managers[audience] = manager.recall(timestamp, create_copy=True)
                new_memory.registry[audience] = {
                    uuid: meta for uuid, meta in self.registry[audience].items() 
                    if meta.time <= timestamp and not self.get_engram_by_id(uuid).is_empty()
                }
            return new_memory
        else:
            for audience, manager in self.engram_managers.items():
                manager.recall(timestamp, create_copy=False)
                keys_to_delete = [
                    uuid for uuid, meta in self.registry[audience].items() 
                    if meta.time > timestamp or self.get_engram_by_id(uuid).is_empty()
                ]
                for uuid in keys_to_delete:
                    del self.registry[audience][uuid]
            return self

    @classmethod
    def load_registry(cls, id: str, 
                    file_name: str = 'registry.json',
                    directory: Path = REGISTRY_DB_DIR) -> 'EpisodicMemory':
        file_path = directory / id / file_name
        if not file_path.exists():
            return cls()
        else:
            with open(file_path, 'r', encoding="utf-8") as f:
                registry_data = f.read()
            return cls.model_validate_json(registry_data)
    
    def initialize_engram(self, partial: bool = False) -> None:
        for audience, registry in self.registry.items():
            for uuid, metadata in registry:
                if partial or metadata.scope == 'full':  # 当partial为True时，加载所有engram，否则只加载scope为'full'的engram
                    engram = Engram.from_json(str(uuid), scope=metadata.scope)
                    self.engram_managers[audience].add_engram(engram)
    
    def load_partial_engram(self, audience:str, uuid: UUID1):
        metadata = self.registry[audience].get(uuid)
        if metadata and metadata.scope == 'partial':
            engram = Engram.from_json(str(uuid)+'.json', scope=metadata.scope)
            self.engram_managers[audience].add_engram(engram)
        elif metadata and metadata.scope == 'full':
            raise ValueError("Engram is already fully loaded.")
        else:
            raise ValueError("Engram with the given UUID does not exist.")

    class Config:
        extra = "allow"
from functools import partial
from typing import List, Dict, Tuple, Union, Type, Literal, Optional
from pydantic import BaseModel, Field, UUID1
import numpy as np 
from PIL import Image
import uuid
from datetime import datetime

from agent.agent import Agent, AgentRegistry
from memory.event import (
    Events,
    EventStream,
    ChatEvent,
    PerceptionEvent,
    ThoughtEvent,
    ReflectionEvent,
    ExperienceEvent,
)
from memory.engram import Engram, EngramManager
from memory.embedding import EmbeddingManager
from memory.memory import ShortTermMemory, EpisodicMemory
from memory.visualize import visualize_engram_manager_memory, visualize_engram_memory
from utils.template import *
from utils.model import get_response_from_openai
from utils.common import (
    MAX_TURNS, PASSWORD, 
    format_retrieved_data, 
    clear_folder_contents,
    extract_response_components,
)
from utils.path import EMBEDDING_DB_DIR, ENGRAM_DB_DIR


class BaseBrain(BaseModel):
    short_term_memory: ShortTermMemory
    episodic_memory: EpisodicMemory
    agent: Agent
    event_stream: EventStream
    embedding_manager: EmbeddingManager
    
    class Config:
        arbitrary_types_allowed=True
    
    async def chat(self,
                   query: str, 
                   system_prompt: str=DEFAULT_PROMPT, 
                   model_name: Literal['GPT3.5', 'GPT4']='GPT3.5',
                   **kwargs) -> str:
        response = get_response_from_openai(query=query, 
                                            system_prompt=system_prompt, 
                                            model_name=model_name, 
                                            **kwargs)
        return response

    def transfer_message_to_event(self, 
                                  msg: str, 
                                  event_class: Type[Union[ChatEvent, PerceptionEvent, ThoughtEvent, ReflectionEvent, ExperienceEvent]],
                                  audience: List[str],
                                  **kwargs
                                  ) -> Events:
        """
        receive a message and transfer it to a specific event.
        """
        if event_class == ChatEvent and kwargs.get('user_input'):
            msg = kwargs.get('user') + ': ' + kwargs.get('user_input') + '\n' + self.agent.persona.name + ': ' + msg
            
        event = event_class(content=msg, actor=self.agent.persona.name, audience=audience)
        self.add_event_to_stream(event)
        
        embedding = self.embedding_manager.embed(msg)
        self.embedding_manager.add_embeddings({event.event_id: embedding})

        return event

    def add_event_to_stream(self, event: Events) -> None:
        """
        Add event to the stream.
        """
        self.event_stream.add_event(event)
        
    def get_chat_history(self, audience: str, max_turns: int=MAX_TURNS) -> List[str]:
        """
        Get the chat history for the given turn.
        The chat history is in the format of 'role: content'.
        """
        if self.short_term_memory.is_empty() or audience not in self.short_term_memory.sequences:
            return []
        # Get the IDs for the latest chat events
        chat_history_id = [e.event_id for e in self.short_term_memory.sequences[audience].engram['chat'][-2 * max_turns:]]

        # Retrieve each event by its ID and format the output
        chat_history = [f"{self.event_stream.get_event(eid).actor}: {self.event_stream.get_event(eid).content}" for eid in chat_history_id]

        return chat_history
                
    def retrieve_from_stm(self, 
                          query: str, 
                          audience: str,
                          topk: int = 3, 
                          threshold: float = 0.5, 
                          increase_factor: float = 0.1
                          ) -> List[str]:
        """
        Retrieve the top-k most similar events from the short-term memory based on a given query,
        adjusting the strength of the neurons using logarithmic increase and exponential decay.

        Args:
            query (str): The query string to be used for finding similar events.
            topk (int): The number of top similar events to retrieve.
            threshold (float): The threshold for combined similarity and strength score.
            increase_factor (float): The factor to logarithmically increase the strength of top-k neurons.

        Returns:
            List[str]: A list of the contents of the top-k most similar events above the threshold.

        Raises:
            ValueError: If the embedding model is not set in the Brain.
        """

        # Encode the query using the embedding model
        query_embedding = self.embedding_manager.embed(query)
        
                
        ## TODO: support milvus
        # 新检索流程
        # from db.vector_db import milvus_db
        
        # top_contents = []
        # db_vector = milvus_db("short_term_memory")
        # top_neuron_ids = db_vector.search(query_embedding)
        # for neuron_id in top_neuron_ids:
        #     neuron = self.short_term_memory.get_neuron(neuron_id)
        #     event = self.event_stream.get_event(neuron.event_id)
        #     top_contents.append(event.content)
        # return top_contents
        

        # Collect embeddings, contents, and strengths of all events, excluding recent chat events
        if audience not in self.short_term_memory.sequences:
            return []
        all_embeddings = []
        all_contents = []
        all_strengths = []
        all_neuron_id= []

        for event_type in ['reflection', 'thought', 'experience', 'perception']:
            sequence = self.short_term_memory.sequences[audience].engram[event_type]
            for neuron in sequence:
                event = self.event_stream.get_event(neuron.event_id)
                all_contents.append(event.content)
                all_strengths.append(neuron.strength)
                all_neuron_id.append(neuron.event_id)

        if not all_embeddings:
            return []
        
        # Calculate cosine similarities
        similarities = self.embedding_manager.calculate_similarities(query_embedding, all_neuron_id)
        combined_scores = [sim * strength for sim, strength in zip(similarities, all_strengths)]

        # Filter based on threshold and sort events based on combined scores
        filtered_indices_scores = [(i, score) for i, score in enumerate(combined_scores) if score >= threshold]
        filtered_indices_scores.sort(key=lambda x: x[1], reverse=True)  # Sort in descending order of score

        # Retrieve top-k contents above the threshold
        top_indices = [i for i, score in filtered_indices_scores[:topk]]
        top_contents = [all_contents[i] for i in top_indices]

        # Adjust the strength of neurons
        for i in top_indices:
            neuron = self.short_term_memory.get_neuron_by_id(all_neuron_id[i])
            neuron.strength = np.log(np.exp(neuron.strength) + increase_factor)
                                
        return top_contents

    def retrieve_from_ltm(self, 
                          query: str, 
                          audience: str,
                          topk: int = 3, 
                          threshold: float = 0.5,
                          increase_factor: float = 0.1,
                          decrease_factor: float = 0.999,
                          ) -> List[Engram]:
        """
        Retrieves the most relevant Engrams from long-term memory based on a given query.

        Args:
            query (str): The query string to be used for finding similar engrams.
            topk (int): The number of top similar engrams to retrieve.
            threshold (float): The threshold for combined similarity and strength score.
            increase_factor (float): The factor to logarithmically increase the strength of top-k engrams.
            decrease_factor (float): The factor to decrease the strength of top-k engrams.

        Returns:
            List[str]: A list of the summary of the top-k most similar engrams above the threshold.

        Raises:
            ValueError: If the embedding model is not set in the Brain.
        """
        if audience not in self.episodic_memory.registry:
            return []
        # Get the embedding for the query
        query_embedding = self.embedding_manager.embed(query)

        # Prepare embeddings, UUIDs, and strengths
        embeddings = []
        uuids = []
        strengths = []
        for uuid, metadata in self.episodic_memory.registry[audience].items():
            uuids.append(uuid)
            strengths.append(metadata.strength)

        # Return an empty list if there are no embeddings
        if not embeddings:
            return []

        # Calculate similarities and combined scores
        similarities = self.embedding_manager.calculate_similarities(query_embedding, uuids)
        combined_scores = [sim * strength for sim, strength in zip(similarities, strengths)]

        # Filter based on threshold and sort events based on combined scores
        filtered_indices_scores = [(i, score) for i, score in enumerate(combined_scores) if score >= threshold]
        filtered_indices_scores.sort(key=lambda x: x[1], reverse=True)  # Sort in descending order of score

        # Retrieve top-k Engrams based on combined scores
        top_indices = [i for i, score in filtered_indices_scores[:topk]]
        top_engrams = [self.episodic_memory.get_engram_by_id(uuids[i]) for i in top_indices]
        
        # Adjust the strength of engrams
        for i, uuid in enumerate(uuids):
            engram = self.episodic_memory.get_engram_by_id(uuid)
            if i in top_indices:
                engram.strength = np.log(np.exp(engram.strength) + increase_factor)
            else:
                engram.strength *= decrease_factor
                
        return top_engrams

    def retrieve_from_engrams(self, 
                              engrams: List[Engram], 
                              query: str,  
                              audience: str,
                              topk: int = 3,
                              threshold: float = 0.5, 
                              increase_factor: float = 0.1,
                              decrease_factor: float = 0.999,
                              ) -> List[Dict[str, str]]:
        """
        Retrieve the top-k most similar chat contents from a list of Engrams based on a given query,
        along with the context of connected events.

        Args:
            engrams (List[Engram]): The list of Engram objects to search through.
            query (str): The query string to be used for finding similar chat contents.
            topk (int): The number of top similar chat contents to retrieve.
            threshold (float): The threshold for combined similarity and strength score.
            increase_factor (float): The factor to logarithmically increase the strength of top-k neurons.
            decrease_factor (float): The factor to decrease the strength of top-k neurons.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing the contents of the top-k most similar chat NeuronCells and their connected context.

        Raises:
            ValueError: If the embedding model is not set in the Brain.
        """
        # Get the embedding for the query
        query_embedding = self.embedding_manager.embed(query)
        
        temp_engrams = EngramManager.from_engram_list(engrams)

        # Collect embeddings, contents, and strengths of chat NeuronCells
        chat_neuron_ids = []
        chat_strengths = []
        for neuron in temp_engrams.get_all_neurons():
            if neuron.event_type == 'chat':
                event = self.event_stream.get_event(neuron.event_id)
                chat_neuron_ids.append(neuron.event_id)
                chat_strengths.append(neuron.strength)

        # Calculate similarities and combined scores
        similarities = self.embedding_manager.calculate_similarities(query_embedding, chat_neuron_ids)
        combined_scores = [sim * strength for sim, strength in zip(similarities, chat_strengths)]

        # Filter based on threshold and sort events based on combined scores
        filtered_indices_scores = [(i, score) for i, score in enumerate(combined_scores) if score >= threshold]
        filtered_indices_scores.sort(key=lambda x: x[1], reverse=True)  # Sort in descending order of score

        # Retrieve top-k contents above the threshold
        top_indices = [i for i, _ in filtered_indices_scores[:topk]]

        # Create dictionaries of contents for each top chat NeuronCell and their connected context
        top_content_dicts = []

        for i in top_indices:
            neuron_id = chat_neuron_ids[i]
            chat_neuron = temp_engrams.get_neuron_by_id(neuron_id)
            content_dict = {'chat': []}
            
            ## add pre event
            for conn in chat_neuron.incoming_connections:
                uuid = conn.target_id
                event = self.event_stream.get_event(uuid)
                if event.event_type in content_dict.keys():
                    content_dict[event.event_type].append(event.content)
                else:
                    content_dict[event.event_type] = [event.content]
            
            ## add itself
            content_dict['chat'].append(self.event_stream.get_event(neuron_id).content)
            
            ## add post event
            for conn in chat_neuron.outgoing_connections:
                uuid = conn.target_id
                event = self.event_stream.get_event(uuid)
                if event.event_type in content_dict.keys():
                    content_dict[event.event_type].append(event.content)
                else:
                    content_dict[event.event_type] = [event.content]
            
            for key, value in content_dict.items():
                new_value = '\n'.join(value)
                content_dict[key] = new_value
            
            top_content_dicts.append(content_dict)
                
        # Adjust the strength of neurons
        adjusted_neurons = set()
        for i in top_indices:
            # Get the neuron ID from the top index
            neuron_id = chat_neuron_ids[i]
            # Retrieve the neuron and adjust its strength
            chat_neuron = temp_engrams.get_neuron_by_id(neuron_id)
            chat_neuron.strength = np.log(np.exp(chat_neuron.strength) + increase_factor)
            adjusted_neurons.add(neuron_id)

            # Adjust the strength of connected neurons
            for conn in chat_neuron.incoming_connections | chat_neuron.outgoing_connections:
                connected_id = conn.target_id
                connected_neuron = temp_engrams.get_neuron_by_id(connected_id)
                if connected_neuron:
                    connected_neuron.strength = np.log(np.exp(connected_neuron.strength) + increase_factor)
                    adjusted_neurons.add(connected_id)

        # Decrease the strength of non-top neurons and their connections
        for neuron in temp_engrams.get_all_neurons():
            if neuron.event_id not in adjusted_neurons:
                neuron.strength *= decrease_factor
                
        return top_content_dicts
    
    async def reflect(self, 
                      audience: str,
                      model_name: Literal['GPT3.5', "GPT4"]='GPT3.5',
                      verbose: bool = False,
                      **kwargs) -> str:
        """
        Perform reflection based on the short-term memories and other information.
        """
        exclude_items = kwargs.get('exclude_items', None)
        system_prompt = self.agent.system_prompt(exclude_items)
        memory = '\n'.join(format_retrieved_data(self.summarize_events(audience)))
        
        reflection_prompt = REFLECTION_PROMPT.format(memory=memory)
        
        if verbose:
            print(reflection_prompt)
            
        # Perform the reflection chat
        reflection_response = await self.chat(reflection_prompt, system_prompt, model_name=model_name)

        # Create a ReflectionEvent from the response
        reflection_event = self.transfer_message_to_event(
            msg=reflection_response,
            event_class=ReflectionEvent,
            user=audience,
            audience=[audience]
        )

        # Add the ReflectionEvent to the event stream and short-term memory
        self.event_stream.add_event(reflection_event)
        self.short_term_memory.add_event(reflection_event)
        
        return reflection_response
    
    def summarize_events(self, audience: str) -> List[Dict[str, List[str]]]:
        """
        Summarizes the events starting from the last chat event linked to a reflection.
        """
        if audience not in self.short_term_memory.sequences:
            return []
        chat_sequence = self.short_term_memory.sequences[audience].engram.get('chat', [])
        start_index = len(chat_sequence)
        last_linked_reflection_id = None 

        # Step 1: Find the last chat linked to a reflection
        for i in range(len(chat_sequence) - 1, -1, -1):
            chat_neuron = chat_sequence[i]
            for reflection_neuron in self.short_term_memory.sequences[audience].engram['reflection']:
                if any(reflection_neuron.event_id == conn.target_id for conn in chat_neuron.outgoing_connections):
                    start_index = i
                    last_linked_reflection_id = reflection_neuron.event_id
                    break
            if last_linked_reflection_id:
                break

        # Step 2: Extract and combine the linked chat and all subsequent events
        summarized_events = []
        if last_linked_reflection_id:
            reflection_neuron = self.event_stream.get_event(last_linked_reflection_id)
            summarized_events.append({'reflection': reflection_neuron.content})
        else:
            start_index = -1
            
        for i in range(start_index+1, len(chat_sequence)):
            chat_neuron = chat_sequence[i]
            content_dict = {'chat': self.event_stream.get_event(chat_neuron.event_id).content}
            
            for conn in chat_neuron.outgoing_connections:
                uuid = conn.target_id
                event = self.event_stream.get_event(uuid)
                if event.event_type in ['thought', 'perception', 'experience'] and self.short_term_memory.get_neuron_by_id(event.event_id):
                    content_dict[event.event_type] = event.content
            
            summarized_events.append(content_dict)

        return summarized_events
    
    def transfer_stm_to_ltm(self, audience: str, summary: str = '') -> None:
        """
        tansfer short term memory to long term memory
        """
        if self.short_term_memory.is_empty():
            return None

        represent_id = None
        strength = 1

        # Process all sequences, but only set represent_id and final_summary once
        for event_type in ['reflection', 'chat', 'thought', 'experience', 'perception']:
            sequence = self.short_term_memory.sequences[audience].engram[event_type]

            # If represent_id is not set and the sequence is not empty, set it now
            if represent_id is None and sequence:
                represent_id = sequence[-1].event_id
                strength = sequence[-1].strength
                if not summary:
                    summary = self.event_stream.get_event(represent_id).content
                break

        # Construct the Engram with data from all sequences
        params = {
            'engram': self.short_term_memory.sequences[audience].engram,
            'scope': 'partial',
            'represent': represent_id,
            'summary': summary,
            'strength': strength
        }
        engram = Engram(**params)
        embedding = self.embedding_manager.embed(summary)
        self.embedding_manager.add_embeddings({engram.uuid: embedding})
        self.episodic_memory.consolidate(audience, engram)

    def save_memory(self) -> None:
        self.short_term_memory.save_memory(id=str(self.agent.id))
        self.episodic_memory.save_memory(id=str(self.agent.id))
        self.agent.save()

    def forget_memory(self) -> None:
        """
        Forget all the memory.
        """
        new_stm = ShortTermMemory()
        new_stm.save_memory(id=str(self.agent.id))
        new_ltm = EpisodicMemory()
        new_ltm.save_memory(id=str(self.agent.id))
               
        clear_folder_contents(ENGRAM_DB_DIR)
        
    def visualize_memory(self, memory: Union[Engram, EngramManager]) -> (Image, Dict[str, str]):
        if memory.is_empty():
            return None, None
        
        if isinstance(memory, EngramManager):
            image, legend = visualize_engram_manager_memory(memory)
        else:
            image, legend = visualize_engram_memory(memory)
        
        for k,v in legend.items():
            legend[k] = self.event_stream.get_event(uuid.UUID(v)).content
            
        return (image, legend)
    
    def recall_memory(self, timestamp: datetime, create_copy: bool = False) -> 'BaseBrain':
        actor_condition = partial(EventStream.actor_condition(actor=self.agent.persona.name))
        time_condition = partial(EventStream.timestamp_condition(time=timestamp))
        
        if create_copy:
            new_stm = self.short_term_memory.recall(timestamp, create_copy=True)
            new_episodic = self.episodic_memory.recall(timestamp, create_copy=True)
            new_dict = {
                'short_term_memory': new_stm,
                'episodic_memory': new_episodic,
                'agent': self.agent,
                'event_stream': self.event_stream.model_copy(),
                'embedding_manager': self.embedding_manager.model_copy()
            }
            new_brain = BaseBrain(**new_dict)
            filter_es = new_brain.event_stream.filter_by_actor_audience_timestamp(actor_condition, time_condition)
            new_brain.event_stream.delete_events_from_stream(filter_es)
            new_brain.embedding_manager.delete_embeddings(events_after_timestamp)
            return new_brain
        
        else:
            self.short_term_memory.recall(timestamp, create_copy=False)
            self.episodic_memory.recall(timestamp, create_copy=False)
            events_after_timestamp = self.event_stream.filter_by_actor_audience_timestamp(actor_condition, time_condition)
            self.event_stream.delete_events_from_stream(filter_es)
            self.embedding_manager.delete_embeddings(events_after_timestamp)
            return self


class MasterBrain(BaseModel):
    agent_registry: AgentRegistry
    agent_brains: Dict[UUID1, BaseBrain] = Field(default_factory=dict)
    event_stream: EventStream
    embedding_manager: EmbeddingManager
    
    @classmethod
    def initialize(self) -> 'MasterBrain':
        modules = {
            'agent_registry': AgentRegistry.load_registry(),
            'event_stream': EventStream.from_json(),
            'embedding_manager': EmbeddingManager.load_registry()
        }
        return MasterBrain(**modules)
    
    def show_agent_info(self) -> str:
        return self.agent_registry.model_dump_json(indent=4)
    
    def get_agent_names(self) -> List[str]:
        return list(self.agent_registry.id2name.values())
    
    def get_agent_ids(self) -> List[str]:
        return list(self.agent_registry.id2name.keys())
    
    ##TODO: 需要加一个类似于gradio里的add agent的方法，但是因为要填的东西太多，没有ui可能有些麻烦，暂且搁置
    def add_agent(self, agent_ids: Union[List[Union[UUID1, str]], Union[UUID1, str]]) -> None:
        if not isinstance(agent_ids, list):
            agent_ids = [agent_ids]

        for agent_id in agent_ids:
            if isinstance(agent_id, str):
                agent_id = self.agent_registry.name2id[agent_id]
                if not agent_id:
                    raise ValueError(f"Agent with name {agent_id} does not exist.")

            if agent_id not in self.get_agent_ids():
                raise ValueError(f"Agent ID {agent_id} is not valid or does not exist.")

            if agent_id not in self.agent_brains:
                agent = Agent.from_id(agent_id)
                self.create_agent_brain(agent)
    
    def create_agent_brain(self, agent: Agent) -> None:
        modules = {
            "short_term_memory": ShortTermMemory.load_memory(str(agent.id)),
            "episodic_memory": EpisodicMemory.load_registry(str(agent.id)),
            "agent": agent,
            "event_stream": self.event_stream,
            "embedding_manager": self.embedding_manager
        }
        self.agent_brains[agent.id] = BaseBrain(**modules)
    
    def save_memory(self) -> None:
        for agent_brain in self.agent_brains.values():
            agent_brain.save_memory()
            
        self.event_stream.to_json()
        self.embedding_manager.save_registry()
        self.agent_registry.save_registry()
        
    def forget_memory(self) -> None:
        """
        Forget all the memory.
        """
        for agent_brain in self.agent_brains.values():
            agent_brain.forget_memory()
            
        new_event_stream = EventStream()
        new_event_stream.to_json()
        
        clear_folder_contents(ENGRAM_DB_DIR)
        clear_folder_contents(EMBEDDING_DB_DIR)
        
    def recall(self, agent_id: UUID1, timestamp: datetime, create_copy: bool = False) -> BaseBrain:
        return self.agent_brains[agent_id].recall_memory(timestamp=timestamp, create_copy=create_copy)
    
    def reset_agent_memory(self, agent_id: UUID1, input_password: str) -> str:
        if input_password == PASSWORD:
            self.agent_brains[agent_id].forget_memory()
            return "系统已成功重置。"
        else:
            return "密码错误，系统重置失败。"
        
    def reset_all(self, input_password: str) -> str:
        if input_password == PASSWORD:
            self.forget_memory()
            return "系统已成功重置。"
        else:
            return "密码错误，系统重置失败。"
        
    async def handle_interaction(self, 
                                agent_id: UUID1, 
                                audience: List[str],
                                user: str,
                                user_input: str,
                                max_turns: int = MAX_TURNS,
                                model_name: Literal['GPT3.5', "GPT4"]='GPT3.5',
                                verbose: bool = False,
                                **kwargs) -> Tuple[str, str]:
        """
        Asynchronously handles a user interaction by processing user input and generating a response.

        This method orchestrates the process of responding to a user's input in a conversation. It involves
        retrieving chat history, generating responses based on the current context, and logging the interaction
        in the agent's memory systems. It also handles the connection of retrieved engrams to the current conversation flow.

        Args:
            agent_id (UUID1): The unique identifier of the agent handling the interaction.
            audience (List[str]): List of unique identifiers representing the speakers in the conversation.
            user_input (str): The current input string from the user.
            max_turns (int, optional): The maximum number of turns to consider in the chat history. Defaults to MAX_TURNS.
            model_name (Literal['GPT3.5', 'GPT4'], optional): The name of the language model to be used. Defaults to 'GPT3.5'.
            verbose (bool, optional): If set to True, prints the chat and thought outputs. Defaults to False.
            **kwargs: Additional keyword arguments for further customization and processing.

        Returns:
            Tuple[str, str]: A tuple containing the chat response and the agent's thought process.

        TODO: Add handling for experience events and schedule processing.
        """
        self.add_agent(agent_id)
        brain = self.agent_brains[agent_id]
        # Retrieve chat history and prepare initial input
        chat_history_list = brain.get_chat_history(user, max_turns)
        combined_input = '\n'.join(chat_history_list[-1:]) + '\n' + user_input
        
        # Retrieve and generate response
        chat, thought, engrams_retrieved = await self.generate_response(brain, 
                                                                        user_input,
                                                                        user,
                                                                        audience, 
                                                                        combined_input, 
                                                                        chat_history_list, 
                                                                        model_name, 
                                                                        verbose,
                                                                        **kwargs)

        if verbose:
            print(chat + '\n' + thought)
        
        # Record processed information in event stream
        chat_event = brain.transfer_message_to_event(chat, ChatEvent, 
                                                        user_input=user_input, 
                                                        user=user,
                                                        audience=audience)
        thought_event = brain.transfer_message_to_event(thought, ThoughtEvent,
                                                        user=user,
                                                        audience=audience)
        
        # Add chat and thought events to EventStream and ShortTermMemory
        events_to_add = [chat_event, thought_event]
        brain.event_stream.add_event(events_to_add)
        brain.short_term_memory.add_event(events_to_add)
        
        # Connect retrieved engrams to the current conversation flow
        neuron = brain.short_term_memory.sequences[user].engram['chat'][-1]
        for engram in engrams_retrieved:
            represent = engram.get_neuron_by_id(engram.represent)
            if represent:
                neuron.connect_to(represent)

        return chat, thought

    async def generate_response(self, 
                                brain: BaseBrain,
                                user_input: str,
                                user: str,
                                audience: List[str],
                                combined_input: str, 
                                chat_history_list: List[str] = [],
                                model_name: Literal['GPT3.5', "GPT4"]='GPT3.5',
                                verbose: bool = False,
                                **kwargs) -> Tuple[str, str, List[Engram]]:
        """
        Asynchronously generates a response based on user input, chat history, and retrieved information.

        This method handles the interaction process by orchestrating the sequence of operations needed to 
        generate a response from the agent. It involves retrieving relevant data from short-term and long-term 
        memory, preparing prompts, and invoking the agent's chat method to get the response. Depending on the 
        agent's decision, it may involve deeper retrieval from long-term memory or engrams.

        Args:
            agent (Agent): The agent instance handling the interaction.
            user_input (str): The current input string from the user.
            users_name (List[str]): A list of unique identifiers for the speakers in the conversation.
            combined_input (str): A combined string of user input and chat history for context.
            chat_history_list (List[str], optional): A list of previous chat messages. Defaults to an empty list.
            model_name (Literal['GPT3.5', 'GPT4'], optional): The name of the language model to be used. Defaults to 'GPT3.5'.
            verbose (bool): Whether or not to print the agent's response.
            **kwargs: Additional keyword arguments that can be used for further customization.

        Returns:
            Tuple[str, str, List[Engram]]: A tuple containing the chat response, thought process of the agent, 
                                        and a list of retrieved engrams (if any).

        Raises:
            ValueError: If certain required conditions or parameters are not met during processing.
        """
        retrieve_data = brain.retrieve_from_stm(combined_input, user)
        prompt = CHAT_WITH_ACTION_AND_RETRIEVE_PROMPT if retrieve_data else CHAT_WITH_ACTION_PROMPT
        chat_prompt, system_prompt = self.prepare_prompts(brain.agent, 
                                                          user_input,
                                                          user,
                                                          audience,
                                                          chat_history_list, 
                                                          retrieve_data, 
                                                          prompt,
                                                          verbose,
                                                          **kwargs)
        
        response = await brain.chat(chat_prompt, system_prompt, model_name)
        
        chat, thought, action = extract_response_components(response)
        
        if verbose:
            print(response)
        
        ## 如果stm不够则检索ltm
        ltm_retrieved = []
        ## TODO1: 可以加一个逻辑，比如在第二次进行retrieve的时候回复给用户固定的话，比如“稍等一下，我想一下”之类的话来拖延时间
        if action == 'Retrieve':
            ltm_retrieved = brain.retrieve_from_ltm(combined_input, user)
            summary_retrieved = [e.summary for e in ltm_retrieved]
            if summary_retrieved:
                prompt = CHAT_WITH_ACTION_AND_RETRIEVE_PROMPT 
            elif retrieve_data:
                prompt = CHAT_WITH_RETRIEVE_PROMPT
            else:
                prompt = CHAT_PROMPT
            chat_prompt, system_prompt = self.prepare_prompts(brain.agent, 
                                                              user_input,
                                                              user,
                                                              audience,
                                                              chat_history_list, 
                                                              retrieve_data + summary_retrieved, 
                                                              prompt,
                                                              verbose,
                                                              **kwargs)
            
            response = await brain.chat(chat_prompt, system_prompt, model_name)
            chat, thought, action = extract_response_components(response)
            
            if verbose:
                print(response)
            
            if action == 'Retrieve':
                engrams_retrieved = brain.retrieve_from_engrams(ltm_retrieved, combined_input, user)
                if engrams_retrieved:
                    engram_summary_retrieved = format_retrieved_data(engrams_retrieved)
                    prompt = CHAT_WITH_ENGRAM_PROMPT
                else:
                    engram_summary_retrieved = summary_retrieved
                    prompt = CHAT_WITH_RETRIEVE_PROMPT
                    
                chat_prompt, system_prompt = self.prepare_prompts(brain.agent, 
                                                                  user_input,
                                                                  user,
                                                                  audience,
                                                                  chat_history_list, 
                                                                  retrieve_data + engram_summary_retrieved, 
                                                                  prompt,
                                                                  verbose,
                                                                  **kwargs)
                
                response = await brain.chat(chat_prompt, system_prompt, model_name)
                chat, thought, _ = extract_response_components(response, parse_action=False)
                
                if verbose:
                    print(response)

        return chat, thought, ltm_retrieved

    def prepare_prompts(self, 
                        agent: Agent, 
                        user_input: str,
                        user: str,
                        audience: List[str],
                        chat_history_list: List[str] = [], 
                        retrieve_data: Optional[List[str]] = None,
                        prompt: str = CHAT_PROMPT,
                        verbose: bool = False,
                        **kwargs) -> Tuple[str, str]:
        """
        Prepares the chat and system prompts for the language model based on the given context.

        This method combines user input, chat history, retrieved data, and any additional information 
        to generate the chat and system prompts that will be used by the language model. The method 
        facilitates dynamic prompt generation, adapting to different interaction contexts.

        Args:
            agent (Agent): The agent instance responsible for the interaction.
            user_input (str): The latest input received from the user.
            users_name (List[str]): A list of unique identifiers representing the speakers.
            chat_history_list (List[str], optional): A list of previous chat messages for context. Defaults to an empty list.
            retrieve_data (Optional[List[str]], optional): Additional data retrieved from memory to be included in the prompt. Defaults to None.
            prompt (str, optional): The base prompt template to be used for generating the chat prompt. Defaults to CHAT_PROMPT.
            verbose (bool): Whether or not to print the prompt.
            **kwargs: Additional keyword arguments that might influence prompt generation.

        Returns:
            Tuple[str, str]: A tuple of chat prompt and system promot.
        TODO: 后续可能的修改：将chat history移除，直接加入到调用api的函数作为history的参数。然后把其他的prompt一并移入到system prompt中。
        """
        chat_prompt = agent.chat_prompt(user_input, user, audience, chat_history_list, retrieve_data, prompt)
        exclude_items = kwargs.get('exclude_items', None)
        system_prompt = agent.system_prompt(exclude_items)
        
        if verbose:
            print("chat_prompt:\n", chat_prompt)
            print("system_prompt:\n", system_prompt)
            
        return chat_prompt, system_prompt

    async def perform_reflection(self, 
                                 agent_id: UUID1, 
                                 audience: str,
                                 model_name: Literal['GPT3.5', "GPT4"]='GPT3.5',
                                 verbose: bool = False,
                                 **kwargs) -> str:
        """
        Perform reflection based on the short-term memories and other information.
        """
        self.add_agent(agent_id)
        brain = self.agent_brains[agent_id]
        exclude_items = kwargs.get('exclude_items', None)
        system_prompt = brain.agent.system_prompt(exclude_items)
        
        reflection_response = await brain.reflect(audience = audience,
                                                       system_prompt = system_prompt,
                                                       model_name = model_name,
                                                       verbose = verbose)
        return reflection_response
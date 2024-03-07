from pathlib import Path
from typing import Callable, Dict, List, Union, Literal, Optional
from typing_extensions import Annotated
from pydantic import BaseModel, UUID1, Field, root_validator, validator
from functools import partial
import uuid
from datetime import timedelta, datetime
from zoneinfo import ZoneInfo

from memory.metadata import Metadata
from utils.schema import DateTime
from utils.path import EVENT_STREAM_DB_DIR
from utils.common import DEFAULT_AREA


class BaseEvent(BaseModel):
    event_type: Literal["base"] = "base"
    actor: str
    audience: Optional[List[str]] = None
    content: str
    location: Optional[str] = None
    # metadata: Metadata | dict = Field(default_factory=Metadata)
    event_id: UUID1 = Field(default_factory=uuid.uuid1)
    create_time: DateTime

    @property
    def text(self) -> str:
        return self.content
    
    @validator('create_time', pre=True, always=True)
    def parse_create_time(cls, v):
        if isinstance(v, str):
            naive_datetime = datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
            return naive_datetime.replace(tzinfo=ZoneInfo(DEFAULT_AREA))
        return v

    class Config:
        arbitrary_types_allowed = True


class ChatEvent(BaseEvent):
    event_type: Literal["chat"] = "chat"


class PerceptionEvent(BaseEvent):
    event_type: Literal["perception"] = "perception"


class ThoughtEvent(BaseEvent):
    event_type: Literal["thought"] = "thought"


class ReflectionEvent(BaseEvent):
    event_type: Literal["reflection"] = "reflection"
        

class ExperienceEvent(BaseEvent):
    event_type: Literal["experience"] = "experience"
    duration: Optional[int] = None
    end_time: Optional[DateTime] = None
    
    @root_validator(pre=False, skip_on_failure=True)
    def check_duration_and_end_time(cls, values):
        create_time = values.get('create_time')
        duration = values.get('duration')
        end_time = values.get('end_time')
        
        if duration is not None and end_time is not None:
            calculated_duration = (end_time - create_time).total_seconds() // 60
            if calculated_duration != duration:
                raise ValueError('end_time does not match the duration from create_time')
        elif duration is not None:
            values['end_time'] = create_time + timedelta(minutes=duration)
        elif end_time is not None:
            duration_seconds = (end_time - create_time).total_seconds()
            if duration_seconds < 0:
                raise ValueError('end_time is before create_time')
            values['duration'] = duration_seconds//60
        
        return values


Events = Annotated[
    Union[ChatEvent, PerceptionEvent, ThoughtEvent, ReflectionEvent, ExperienceEvent],
    Field(discriminator="event_type"),
]


class EventStream(BaseModel):
    """ EventStream now uses a dictionary with UUID keys. """

    log: Dict[UUID1, Events] = {}

    def add_event(self, events: Union[Events, List[Events]]):
        """ Adds one or multiple events to the log. """
        if isinstance(events, list):
            for event in events:
                self.log[event.event_id] = event
        else:
            self.log[events.event_id] = events

    def clear(self) -> bool:
        """ Clears the event log. """
        self.log.clear()
        return True

    def __add__(self, other: "EventStream") -> "EventStream":
        """ Merges two EventStreams. Conflicting UUIDs in 'other' will overwrite those in 'self'. """
        combined_log = {**self.log, **other.log}
        return EventStream(log=combined_log)

    def __len__(self) -> int:
        """ Returns the number of events in the log. """
        return len(self.log)
    
    def get_event(self, event_id: UUID1) -> Events:
        """ Retrieves an event by its UUID. """
        return self.log[event_id]

    def filter(self, *conditions: Callable[[BaseEvent], bool]) -> 'EventStream':
        """
        Returns a new EventStream with events that match all the given conditions.
        
        Args:
            conditions (Callable[[BaseEvent], bool]): A variable number of functions, 
                each taking a BaseEvent instance and returning a boolean.

        Returns:
            EventStream: A new EventStream instance containing events that match all conditions.
        """
        new_log = {event_id: event for event_id, event in self.log.items() if all(condition(event) for condition in conditions)}
        return EventStream(log=new_log)

    @staticmethod
    def actor_condition(event: BaseEvent, actor: str) -> bool:
        return event.actor == actor

    @staticmethod
    def timestamp_condition(event: BaseEvent, time: datetime, ge: bool = True) -> bool:
        return event.create_time >= time if ge else event.create_time < time

    @staticmethod
    def audience_condition(event: BaseEvent, audience: List[str]) -> bool:
        return any(audience_id in event.audience for audience_id in audience)
    
    def filter_by_actor_audience_timestamp(self, actor: str, audience: List[str], time: datetime, ge: bool = True) -> 'EventStream':
        """
        An example of filter:
        Filters the given EventStream by actor, audience, and timestamp conditions.

        Args:
            event_stream (EventStream): The EventStream to filter.
            actor (str): The actor ID to filter by.
            audience (List[str]): List of audience IDs to filter by.
            time (datetime): The timestamp to filter by.
            ge (bool, optional): If True, filters for events after the given time, 
                                else filters for events before the given time. 
                                Defaults to True.

        Returns:
            EventStream: A new EventStream instance containing events that match all conditions.
        """
        ac = partial(self.actor_condition(actor=actor))
        tc = partial(self.actor_condition(time=time))
        auc = partial(self.actor_condition(audience=audience))
        return self.filter(ac, tc, auc)
    
    def delete_events_from_stream(self, other_stream: 'EventStream') -> None:
        """
        Delete events that are present in the provided EventStream.

        Args:
            other_stream (EventStream): The EventStream containing events to be deleted.
        """
        event_ids_to_delete = set(other_stream.log.keys())
        self.log = {event_id: event for event_id, event in self.log.items() if event_id not in event_ids_to_delete}

    def keep_events_from_stream(self, other_stream: 'EventStream') -> None:
        """
        Keep only the events that are present in the provided EventStream.

        Args:
            other_stream (EventStream): The EventStream containing events to be kept.
        """
        event_ids_to_keep = set(other_stream.log.keys())
        self.log = {event_id: event for event_id, event in self.log.items() if event_id in event_ids_to_keep}
   
    @classmethod
    def from_json(cls, file: str = 'all_events.json', directory: Path = EVENT_STREAM_DB_DIR) -> "EventStream":
        with open(directory / file, "r") as f:
            return cls.model_validate_json(f.read())
    
    def to_json(self, file: str = 'all_events.json', directory: Path = EVENT_STREAM_DB_DIR) -> None:
        with open(directory / file, "w") as f:
            f.write(self.model_dump_json(indent=4))

    class Config:
        arbitrary_types_allowed = True
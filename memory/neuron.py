import uuid
import numpy as np
from typing import Literal, Optional, Set, List
from pydantic import BaseModel, Field, UUID1, validator
from datetime import datetime
from zoneinfo import ZoneInfo

from utils.common import DEFAULT_AREA
from utils.schema import DateTime
from memory.event import EventStream


class Connection(BaseModel):
    target_id: UUID1
    create_time: DateTime
    
    @validator('create_time', pre=True, always=True)
    def parse_create_time(cls, v):
        if isinstance(v, str):
            naive_datetime = datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
            return naive_datetime.replace(tzinfo=ZoneInfo(DEFAULT_AREA))
        return v
    
    def __hash__(self):
        return hash((self.target_id, self.create_time))

    def __eq__(self, other):
        if not isinstance(other, Connection):
            return NotImplemented
        return (self.target_id, self.create_time) == (other.target_id, other.create_time)

class NeuronCell(BaseModel):
    event_id: UUID1 = Field(default_factory=uuid.uuid1)
    event_type: Literal['chat', 'perception', 'thought', 'reflection', 'experience']
    create_time: DateTime
    strength: float = 1.0
    actor: str
    audience: Optional[List[str]] = None
    outgoing_connections: Set[Connection] = Field(default_factory=set)
    incoming_connections: Set[Connection] = Field(default_factory=set)
    
    @validator('create_time', pre=True, always=True)
    def parse_create_time(cls, v):
        if isinstance(v, str):
            naive_datetime = datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
            return naive_datetime.replace(tzinfo=ZoneInfo(DEFAULT_AREA))
        return v
    
    def __hash__(self):
        return hash(self.event_id)

    def __eq__(self, other):
        return isinstance(other, NeuronCell) and self.event_id == other.event_id

    def connect_to(self, other: 'NeuronCell'):
        """Create a directed connection from this neuron to another neuron."""
        connection = Connection(target_id=other.event_id,)
        self.outgoing_connections.add(connection)
        other_connection = Connection(target_id=self.event_id)
        other.incoming_connections.add(other_connection)

    def disconnect_from(self, other: 'NeuronCell'):
        """Remove the directed connection from this neuron to another neuron, if connected."""
        self.outgoing_connections = {conn for conn in self.outgoing_connections if conn.target_id != other.event_id}
        other.incoming_connections = {conn for conn in other.incoming_connections if conn.target_id != self.event_id}

    def disconnect_incoming(self, other: 'NeuronCell'):
        """Remove the directed connection from another neuron to this neuron, if connected."""
        self.incoming_connections = {conn for conn in self.incoming_connections if conn.target_id != other.event_id}
        other.outgoing_connections = {conn for conn in other.outgoing_connections if conn.target_id != self.event_id}


def calculate_connection_strength(neuron1: NeuronCell, 
                                  neuron2: NeuronCell, 
                                  eventstream: EventStream,
                                  similarity: float, 
                                  average_method: str = 'harmonic',
                                  apply_time_decay: bool = False,
                                  decay_rate: float = 0.995) -> float:
    """
    Calculate the connection strength between two neurons.

    Args:
        neuron1, neuron2: NeuronCell objects to calculate the connection strength between.
        eventstream: EventStream object to retrieve event details.
        average_method: Method for averaging strengths ('harmonic', 'arithmetic', 'geometric').
        apply_time_decay: Whether to apply time decay based on event creation time difference.
        decay_rate: Decay rate per day.

    Returns:
        Connection strength as a float.

    Note:
        - Harmonic mean emphasizes smaller values.
        - Arithmetic mean is the standard average.
        - Geometric mean is less sensitive to extremely high values.
    """
    # Retrieve events
    event1 = eventstream.get_event(neuron1.event_id)
    event2 = eventstream.get_event(neuron2.event_id)

    # Calculate average strength
    if average_method == 'harmonic':
        average_strength = 2 / (1/neuron1.strength + 1/neuron2.strength)
    elif average_method == 'arithmetic':
        average_strength = (neuron1.strength + neuron2.strength) / 2
    elif average_method == 'geometric':
        average_strength = np.sqrt(neuron1.strength * neuron2.strength)
    else:
        raise ValueError("Invalid average method. Choose 'harmonic', 'arithmetic', or 'geometric'.")

    # Calculate connection strength
    connection_strength = similarity * average_strength

    # Apply time decay if enabled
    if apply_time_decay and event1.create_time and event2.create_time:
        time_diff = abs(datetime.fromisoformat(event1.create_time) - datetime.fromisoformat(event2.create_time)).days
        decay_factor = decay_rate ** time_diff
        connection_strength *= decay_factor

    return connection_strength
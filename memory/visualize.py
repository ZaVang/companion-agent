from typing import Dict, Tuple, Union
from pydantic import UUID1
import networkx as nx
import matplotlib.pyplot as plt
import io
from PIL import Image

from memory.neuron import NeuronCell
from memory.engram import Engram, EngramManager
from memory.memory import ShortTermMemory


def visualize_engram_memory(memory: Engram,
                               prefix: str = 'E0',
                               x_offset: int = 0,
                               y_offset: int = 0,
                               return_image: bool = True) -> Union[
                                Tuple[nx.DiGraph, Dict[UUID1, Tuple[int, int]], Dict[UUID1, str], Dict[str, str]],
                                Tuple[Image.Image, Dict[str, str]]
                               ]:
    """
    Visualizes the connections of neurons in the memory.

    This method creates a directed graph representing the neurons and their connections.
    Each neuron is labeled with an abbreviation of its event type and a sequence number.
    The method displays the graph and prints a legend mapping the labels to neuron IDs.
    """
    if isinstance(memory, Engram):
        sequences = memory.engram
    elif isinstance(memory, ShortTermMemory):
        sequences = memory.sequences

    G = nx.DiGraph()
    label_mapping = {}
    label_legend = {}

    # Iterate over neuron types and their sequences
    for etype, seq in sequences.items():
        for j, neuron in enumerate(seq):
            short_label = f"{prefix}_{etype[0].upper()}{j}" 
            label_mapping[neuron.event_id] = short_label
            label_legend[short_label] = str(neuron.event_id)

            G.add_node(neuron.event_id, event_type=neuron.event_type, strength=neuron.strength)
            for conn in neuron.outgoing_connections:
                connected_id = conn.target_id
                connected_neuron = memory.get_neuron_by_id(connected_id)
                if not connected_neuron:
                    continue
                G.add_edge(neuron.event_id, connected_id)
    
    # Define positions for each neuron type
    pos = {}

    for i, chat in enumerate(sequences['chat']):
        pos[chat.event_id] = (4 * i, 0)
        for conn in chat.outgoing_connections:
            connected_id = conn.target_id
            connected_neuron = memory.get_neuron_by_id(connected_id)
            if not connected_neuron:
                continue
            # Assign positions based on event type
            pos[connected_neuron.event_id] = get_position_based_on_type((4 * i, 0), connected_neuron, i)

    # add offsets
    for k, v in pos.items():
        pos[k] = (v[0] + x_offset, v[1] + y_offset)
        
    if not return_image:
        return G, pos, label_mapping, label_legend
    
    # Compute color mapping based on strength
    max_strength = max([G.nodes[node]['strength'] for node in G.nodes()])
    node_colors = [(1 - G.nodes[node]['strength'] / max_strength, 1, 1) for node in G.nodes()]

    # # Visualize the graph
    # nx.draw(G, pos, labels=label_mapping, with_labels=True, node_color=node_colors, edge_color="gray")
    # plt.show()

    # # Print the legend
    # print("Legend:")
    # for label, uuid in label_legend.items():
    #     print(f"{label}: {uuid}")
    
    # Create a buffer to store image data
    buffer = io.BytesIO()

    # Plot the graph
    fig, ax = plt.subplots()
    nx.draw(G, pos, node_size=300, font_size=8, labels=label_mapping, with_labels=True, node_color=node_colors, edge_color="gray", ax=ax)

    # Save the plot to the buffer
    plt.savefig(buffer, format='png')
    plt.close(fig)  # Close the figure to prevent display

    # Convert buffer to PIL Image
    buffer.seek(0)
    image = Image.open(buffer)

    return image, label_legend

def get_position_based_on_type(pos: Tuple[int, int],neuron: NeuronCell, index: int) -> Tuple[int, int]:
    """
    Determines the position of a neuron based on its event type.

    Args:
        neuron (NeuronCell): The neuron whose position is being determined.
        index (int): The index of the neuron in its event sequence.

    Returns:
        tuple: The (x, y) position of the neuron.
    """
    x, y = pos
    offset_map = {
        'thought': (2, 2),
        'reflection': (2, -2),
        'perception': (0, 2),
        'experience': (0, -2),
    }

    offset = offset_map.get(neuron.event_type, (0, 0))
    return (x + offset[0], y + offset[1])

def visualize_engram_manager_memory(engram_manager: EngramManager) -> (Image, Dict[str, str]):
    engram_graphs = []
    max_x = 0
    all_pos = {}
    label_legend = {}
    label_mapping = {}

    # Loop through each engram in engram_manager
    for i, engram in enumerate(engram_manager.engram_dict.values()):
        # Visualize each engram with an updated prefix and x_offset
        G, pos, mapping, legend = visualize_engram_memory(engram, prefix=f"E{i}", x_offset=max_x, return_image=False)

        # Update the maximum x coordinate for positioning next engram
        max_x = max([x[0] for _, x in pos.items()]) + 4 

        # Update position, label mapping, and legend dictionaries
        all_pos.update({k: v for k, v in pos.items() if k not in all_pos})
        label_mapping.update({k: v for k, v in mapping.items() if k not in label_mapping})
        label_legend.update({k: v for k, v in legend.items() if k not in label_legend})

        engram_graphs.append(G)

    # Create a combined graph
    combined_G = nx.DiGraph()
    for G in engram_graphs:
        combined_G.add_nodes_from(G.nodes(data=True))

    # Add edges from all neurons in engram_manager
    all_neurons = engram_manager.get_all_neurons()
    for neuron in all_neurons:
        for conn in neuron.outgoing_connections:
            connected_id = conn.target_id
            if engram_manager.get_neuron_by_id(connected_id):
                combined_G.add_edge(neuron.event_id, connected_id)


    # Compute color mapping based on strength
    max_strength = max([combined_G.nodes[node]['strength'] for node in combined_G.nodes()])
    node_colors = [(1 - combined_G.nodes[node]['strength'] / max_strength, 1, 1) for node in combined_G.nodes()]

    # # Visualize the graph
    # nx.draw(G, pos, labels=label_mapping, with_labels=True, node_color=node_colors, edge_color="gray")
    # plt.show()

    # # Print the legend
    # print("Legend:")
    # for label, uuid in label_legend.items():
    #     print(f"{label}: {uuid}")
    
    # Create a buffer to store image data
    buffer = io.BytesIO()

    # Plot the graph
    fig, ax = plt.subplots()
    nx.draw(combined_G, all_pos, node_size=300, font_size=8, labels=label_mapping, with_labels=True, node_color=node_colors, edge_color="gray", ax=ax)

    # Save the plot to the buffer
    plt.savefig(buffer, format='png')
    plt.close(fig)  # Close the figure to prevent display

    # Convert buffer to PIL Image
    buffer.seek(0)
    image = Image.open(buffer)

    return image, label_legend
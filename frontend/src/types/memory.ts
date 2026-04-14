// Event Types
export type EventType = 'chat' | 'perception' | 'thought' | 'reflection' | 'experience';

// Memory Scope
export type MemoryScope = 'short_term' | 'long_term' | 'full';

// Neuron Node in the memory network
export interface NeuronNode {
  id: string;
  label: string;
  type: EventType;
  strength: number;
  elo: number;
  emotional_valence?: number; // -1 to 1
  emotional_arousal?: number; // 0 to 1
  is_active?: boolean;
  is_consolidated?: boolean;
  activation_threshold?: number;
  created_at?: string;
  content?: string;
}

// Connection between neurons
export interface NeuronConnection {
  source: string;
  target: string;
  weight: number;
}

// Memory network data
export interface MemoryNetwork {
  nodes: NeuronNode[];
  links: NeuronConnection[];
}

// Memory item for timeline
export interface MemoryItem {
  id: string;
  content: string;
  type: EventType;
  timestamp: string;
  strength: number;
  elo: number;
  actor: string;
}

// Add memory request
export interface AddMemoryRequest {
  content: string;
  event_type: EventType;
  actor: string;
  audience?: string[];
  impact_score?: number;
  metadata?: Record<string, unknown>;
}

// Retrieve memory request
export interface RetrieveRequest {
  query: string;
  top_k?: number;
  threshold?: number;
  memory_scope?: MemoryScope;
}

// Retrieval result
export interface RetrievalResult {
  id: string;
  content: string;
  score: number;
  activated_neurons: string[];
  retrieval_path: string[];
}

// DMN (Default Mode Network) result
export interface DMNResult {
  success: boolean;
  consolidations: number;
  prunings: number;
  new_associations: number;
  messages: string[];
  activated_neurons: string[];
}

// Network statistics
export interface NetworkStats {
  total_neurons: number;
  total_connections: number;
  neurons_by_type: Record<EventType, number>;
  avg_connections_per_neuron: number;
  avg_strength: number;
  avg_elo: number;
  active_neurons: number;
  consolidated_neurons: number;
}

// Elo ranking item
export interface EloRankingItem {
  id: string;
  label: string;
  type: EventType;
  elo: number;
  strength: number;
  rank_change: number; // positive = went up, negative = went down
}

// Decay history item
export interface DecayHistoryItem {
  neuron_id: string;
  label: string;
  previous_strength: number;
  current_strength: number;
  decay_rate: number;
  timestamp: string;
}

// System configuration
export interface SystemConfig {
  decay_rate: number;
  activation_threshold: number;
  elo_k_factor: number;
  consolidation_threshold: number;
  auto_decay: boolean;
  auto_consolidation: boolean;
}

// API Response wrapper
export interface ApiResponse<T> {
  code: number;
  msg: string;
  data: T;
}

// Graph node/link types for D3
export interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  type: EventType;
  strength: number;
  elo: number;
  emotional_valence?: number;
  is_active?: boolean;
  is_consolidated?: boolean;
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
  source: string | GraphNode;
  target: string | GraphNode;
  weight: number;
}

// Color mapping for event types
export const EVENT_TYPE_COLORS: Record<EventType, string> = {
  chat: '#60A5FA',       // blue
  thought: '#A78BFA',    // purple
  reflection: '#34D399', // green
  perception: '#FBBF24', // yellow
  experience: '#F87171', // red
};

// Event type labels
export const EVENT_TYPE_LABELS: Record<EventType, string> = {
  chat: 'Chat',
  thought: 'Thought',
  reflection: 'Reflection',
  perception: 'Perception',
  experience: 'Experience',
};

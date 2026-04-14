import { useState, useCallback, useEffect } from 'react';
import type {
  MemoryNetwork,
  NetworkStats,
  AddMemoryRequest,
  RetrieveRequest,
  RetrievalResult,
  DMNResult,
  EloRankingItem,
  DecayHistoryItem,
  SystemConfig,
  NeuronNode,
} from '../types/memory';

const API_BASE = '/ai-companion/api';

interface UseMemorySystemReturn {
  // State
  network: MemoryNetwork;
  stats: NetworkStats | null;
  eloRanking: EloRankingItem[];
  decayHistory: DecayHistoryItem[];
  config: SystemConfig;
  isLoading: boolean;
  error: string | null;
  selectedNeuron: NeuronNode | null;
  retrievalResult: RetrievalResult | null;
  dmnResult: DMNResult | null;
  
  // Actions
  refreshNetwork: () => Promise<void>;
  refreshStats: () => Promise<void>;
  addMemory: (req: AddMemoryRequest) => Promise<NeuronNode | null>;
  retrieveMemory: (req: RetrieveRequest) => Promise<RetrievalResult | null>;
  triggerDMN: () => Promise<DMNResult | null>;
  triggerEloCompetition: () => Promise<void>;
  updateConfig: (config: Partial<SystemConfig>) => Promise<void>;
  selectNeuron: (neuron: NeuronNode | null) => void;
  clearRetrievalResult: () => void;
}

export function useMemorySystem(): UseMemorySystemReturn {
  // State
  const [network, setNetwork] = useState<MemoryNetwork>({ nodes: [], links: [] });
  const [stats, setStats] = useState<NetworkStats | null>(null);
  const [eloRanking, setEloRanking] = useState<EloRankingItem[]>([]);
  const [decayHistory] = useState<DecayHistoryItem[]>([]);
  const [config, setConfig] = useState<SystemConfig>({
    decay_rate: 0.995,
    activation_threshold: 0.3,
    elo_k_factor: 32,
    consolidation_threshold: 0.7,
    auto_decay: true,
    auto_consolidation: true,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNeuron, setSelectedNeuron] = useState<NeuronNode | null>(null);
  const [retrievalResult, setRetrievalResult] = useState<RetrievalResult | null>(null);
  const [dmnResult, setDmnResult] = useState<DMNResult | null>(null);

  // Helper to make API calls
  const apiCall = useCallback(async <T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<T | null> => {
    try {
      const response = await fetch(`${API_BASE}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
        },
        ...options,
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const data = await response.json();
      return data.data || data;
    } catch (err) {
      console.error(`API call failed for ${endpoint}:`, err);
      return null;
    }
  }, []);

  // Refresh network data
  const refreshNetwork = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const data = await apiCall<MemoryNetwork>('/memory/network');
      if (data) {
        setNetwork(data);
      } else {
        // Use mock data if API not available
        setNetwork(generateMockNetwork());
      }
    } catch (err) {
      setError('Failed to fetch network');
      setNetwork(generateMockNetwork());
    } finally {
      setIsLoading(false);
    }
  }, [apiCall]);

  // Refresh stats
  const refreshStats = useCallback(async () => {
    try {
      const data = await apiCall<NetworkStats>('/memory/stats');
      if (data) {
        setStats(data);
      } else {
        setStats(generateMockStats(network));
      }
    } catch {
      setStats(generateMockStats(network));
    }
  }, [apiCall, network]);

  // Add new memory
  const addMemory = useCallback(async (req: AddMemoryRequest): Promise<NeuronNode | null> => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await apiCall<NeuronNode>('/memory/memories', {
        method: 'POST',
        body: JSON.stringify(req),
      });
      
      if (result) {
        await refreshNetwork();
        return result;
      }
      
      // Mock: create a new neuron locally
      const newNeuron: NeuronNode = {
        id: `neuron-${Date.now()}`,
        label: req.content.substring(0, 30),
        type: req.event_type,
        strength: 1.0,
        elo: 1200,
        emotional_valence: req.impact_score ? (req.impact_score - 0.5) * 2 : 0,
        created_at: new Date().toISOString(),
        content: req.content,
      };
      
      setNetwork(prev => ({
        nodes: [...prev.nodes, newNeuron],
        links: [...prev.links],
      }));
      
      return newNeuron;
    } catch (err) {
      setError('Failed to add memory');
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [apiCall, refreshNetwork]);

  // Retrieve memory
  const retrieveMemory = useCallback(async (req: RetrieveRequest): Promise<RetrievalResult | null> => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await apiCall<RetrievalResult>('/memory/retrieve', {
        method: 'POST',
        body: JSON.stringify(req),
      });
      
      if (result) {
        setRetrievalResult(result);
        // Mark activated neurons
        setNetwork(prev => ({
          ...prev,
          nodes: prev.nodes.map(node => ({
            ...node,
            is_active: result.activated_neurons.includes(node.id),
          })),
        }));
        return result;
      }
      
      // Mock retrieval
      const mockResult: RetrievalResult = {
        id: `retrieval-${Date.now()}`,
        content: `Found memories related to: ${req.query}`,
        score: 0.85,
        activated_neurons: network.nodes.slice(0, 3).map(n => n.id),
        retrieval_path: network.nodes.slice(0, 3).map(n => n.id),
      };
      
      setRetrievalResult(mockResult);
      setNetwork(prev => ({
        ...prev,
        nodes: prev.nodes.map(node => ({
          ...node,
          is_active: mockResult.activated_neurons.includes(node.id),
        })),
      }));
      
      return mockResult;
    } catch (err) {
      setError('Failed to retrieve memory');
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [apiCall, network]);

  // Trigger DMN
  const triggerDMN = useCallback(async (): Promise<DMNResult | null> => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await apiCall<DMNResult>('/memory/dmn', {
        method: 'POST',
      });
      
      if (result) {
        setDmnResult(result);
        await refreshNetwork();
        return result;
      }
      
      // Mock DMN result
      const mockResult: DMNResult = {
        success: true,
        consolidations: Math.floor(Math.random() * 5),
        prunings: Math.floor(Math.random() * 3),
        new_associations: Math.floor(Math.random() * 4),
        messages: ['Memory consolidation complete'],
        activated_neurons: network.nodes.slice(0, 5).map(n => n.id),
      };
      
      setDmnResult(mockResult);
      return mockResult;
    } catch (err) {
      setError('Failed to trigger DMN');
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [apiCall, refreshNetwork, network]);

  // Trigger Elo competition
  const triggerEloCompetition = useCallback(async () => {
    setIsLoading(true);
    
    try {
      await apiCall('/memory/elo/compete', { method: 'POST' });
      await refreshNetwork();
      await refreshStats();
    } catch {
      // Mock: simulate Elo changes
      setNetwork(prev => ({
        ...prev,
        nodes: prev.nodes.map(node => ({
          ...node,
          elo: node.elo + (Math.random() - 0.5) * 50,
        })),
      }));
    } finally {
      setIsLoading(false);
    }
  }, [apiCall, refreshNetwork, refreshStats]);

  // Update config
  const updateConfig = useCallback(async (newConfig: Partial<SystemConfig>) => {
    try {
      await apiCall('/memory/config', {
        method: 'PUT',
        body: JSON.stringify(newConfig),
      });
      setConfig(prev => ({ ...prev, ...newConfig }));
    } catch {
      setConfig(prev => ({ ...prev, ...newConfig }));
    }
  }, [apiCall]);

  // Select neuron
  const selectNeuron = useCallback((neuron: NeuronNode | null) => {
    setSelectedNeuron(neuron);
  }, []);

  // Clear retrieval result
  const clearRetrievalResult = useCallback(() => {
    setRetrievalResult(null);
    setNetwork(prev => ({
      ...prev,
      nodes: prev.nodes.map(node => ({
        ...node,
        is_active: false,
      })),
    }));
  }, []);

  // Initial load
  useEffect(() => {
    refreshNetwork();
    refreshStats();
  }, []);

  // Update Elo ranking when network changes
  useEffect(() => {
    const ranking = [...network.nodes]
      .sort((a, b) => b.elo - a.elo)
      .slice(0, 10)
      .map((node) => ({
        id: node.id,
        label: node.label,
        type: node.type,
        elo: node.elo,
        strength: node.strength,
        rank_change: 0,
      }));
    setEloRanking(ranking);
  }, [network.nodes]);

  return {
    network,
    stats,
    eloRanking,
    decayHistory,
    config,
    isLoading,
    error,
    selectedNeuron,
    retrievalResult,
    dmnResult,
    refreshNetwork,
    refreshStats,
    addMemory,
    retrieveMemory,
    triggerDMN,
    triggerEloCompetition,
    updateConfig,
    selectNeuron,
    clearRetrievalResult,
  };
}

// Mock data generators
function generateMockNetwork(): MemoryNetwork {
  const nodes: NeuronNode[] = [
    { id: 'n1', label: 'User introduced themselves', type: 'chat', strength: 0.9, elo: 1400, emotional_valence: 0.5, is_consolidated: true },
    { id: 'n2', label: 'Discussed project deadline', type: 'thought', strength: 0.7, elo: 1250, emotional_valence: -0.3 },
    { id: 'n3', label: 'Morning coffee routine', type: 'experience', strength: 0.5, elo: 1100, emotional_valence: 0.8 },
    { id: 'n4', label: 'AI reflection on learning', type: 'reflection', strength: 0.8, elo: 1350, emotional_valence: 0.2, is_consolidated: true },
    { id: 'n5', label: 'User mentioned favorite color', type: 'perception', strength: 0.6, elo: 1200, emotional_valence: 0.1 },
    { id: 'n6', label: 'Code review discussion', type: 'chat', strength: 0.75, elo: 1300, emotional_valence: 0.4 },
    { id: 'n7', label: 'Bug fix memory trace', type: 'experience', strength: 0.85, elo: 1380, emotional_valence: 0.6, is_consolidated: true },
    { id: 'n8', label: 'Team meeting notes', type: 'thought', strength: 0.65, elo: 1150, emotional_valence: -0.1 },
  ];

  const links = [
    { source: 'n1', target: 'n5', weight: 0.8 },
    { source: 'n2', target: 'n6', weight: 0.7 },
    { source: 'n3', target: 'n7', weight: 0.6 },
    { source: 'n4', target: 'n8', weight: 0.5 },
    { source: 'n1', target: 'n4', weight: 0.9 },
    { source: 'n6', target: 'n7', weight: 0.75 },
    { source: 'n2', target: 'n8', weight: 0.65 },
  ];

  return { nodes, links };
}

function generateMockStats(network: MemoryNetwork): NetworkStats {
  const neuronsByType: Record<string, number> = {};
  network.nodes.forEach(n => {
    neuronsByType[n.type] = (neuronsByType[n.type] || 0) + 1;
  });

  return {
    total_neurons: network.nodes.length,
    total_connections: network.links.length,
    neurons_by_type: neuronsByType as Record<'chat' | 'perception' | 'thought' | 'reflection' | 'experience', number>,
    avg_connections_per_neuron: network.links.length / Math.max(network.nodes.length, 1),
    avg_strength: network.nodes.reduce((sum, n) => sum + n.strength, 0) / Math.max(network.nodes.length, 1),
    avg_elo: network.nodes.reduce((sum, n) => sum + n.elo, 0) / Math.max(network.nodes.length, 1),
    active_neurons: network.nodes.filter(n => n.is_active).length,
    consolidated_neurons: network.nodes.filter(n => n.is_consolidated).length,
  };
}

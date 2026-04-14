import { useState, useCallback, useMemo } from 'react';
import { useMemorySystem } from './hooks/useMemorySystem';
import MemoryNetworkGraph from './components/MemoryNetwork';
import EloRanking from './components/EloRanking';
import Timeline from './components/Timeline';
import DecayChart from './components/DecayChart';
import ControlPanel from './components/ControlPanel';
import NeuronNodeDetail from './components/NeuronNode';
import type { GraphNode, MemoryItem, NeuronNode } from './types/memory';
import './styles/index.css';

function App() {
  const {
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
    addMemory,
    retrieveMemory,
    triggerDMN,
    triggerEloCompetition,
    updateConfig,
    selectNeuron,
    refreshNetwork,
  } = useMemorySystem();

  const [detailNeuron, setDetailNeuron] = useState<NeuronNode | null>(null);

  // Convert network nodes to timeline items
  const timelineItems: MemoryItem[] = useMemo(() => {
    return network.nodes.map(node => ({
      id: node.id,
      content: node.content || node.label,
      type: node.type,
      timestamp: node.created_at || new Date().toISOString(),
      strength: node.strength,
      elo: node.elo,
      actor: 'system',
    })).sort((a, b) => 
      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  }, [network.nodes]);

  const handleNodeClick = useCallback((node: GraphNode) => {
    setDetailNeuron(node as NeuronNode);
    selectNeuron(node as NeuronNode);
  }, [selectNeuron]);

  const handleCloseDetail = useCallback(() => {
    setDetailNeuron(null);
    selectNeuron(null);
  }, [selectNeuron]);

  return (
    <div className="h-screen flex flex-col bg-dark-bg text-white overflow-hidden">
      {/* Header */}
      <header className="flex-shrink-0 bg-dark-card border-b border-dark-border px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 
                          flex items-center justify-center text-xl">
              🧠
            </div>
            <div>
              <h1 className="text-xl font-bold">Engram Memory Visualizer</h1>
              <p className="text-xs text-gray-400">
                Neural Memory Network & DMN Consolidation
              </p>
            </div>
          </div>

          {/* Stats summary */}
          <div className="flex items-center gap-6">
            {stats && (
              <>
                <div className="text-center">
                  <p className="text-2xl font-bold text-blue-400">{stats.total_neurons}</p>
                  <p className="text-xs text-gray-500">Neurons</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-purple-400">{stats.total_connections}</p>
                  <p className="text-xs text-gray-500">Connections</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-emerald-400">{stats.consolidated_neurons}</p>
                  <p className="text-xs text-gray-500">Consolidated</p>
                </div>
              </>
            )}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => refreshNetwork()}
              disabled={isLoading}
              className="px-3 py-1.5 bg-dark-bg hover:bg-gray-700 rounded-lg text-sm transition-colors
                       disabled:opacity-50 flex items-center gap-1"
            >
              🔄 Refresh
            </button>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 flex overflow-hidden">
        {/* Left sidebar - Network visualization */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 p-4 min-h-0">
            <div className="h-full bg-dark-card/50 rounded-xl overflow-hidden border border-dark-border">
              <MemoryNetworkGraph
                data={network}
                onNodeClick={handleNodeClick}
                selectedNodeId={selectedNeuron?.id}
              />
            </div>
          </div>

          {/* Bottom section - Charts */}
          <div className="flex-shrink-0 p-4 pt-0">
            <div className="grid grid-cols-2 gap-4">
              <DecayChart history={decayHistory} decayRate={config.decay_rate} />
              <div className="bg-dark-card rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <span>📊</span> Network Legend
                </h3>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {Object.entries({
                    chat: '💬',
                    thought: '💭',
                    reflection: '🔄',
                    perception: '👁️',
                    experience: '⭐',
                  }).map(([type, icon]) => (
                    <div key={type} className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{
                          backgroundColor: {
                            chat: '#60A5FA',
                            thought: '#A78BFA',
                            reflection: '#34D399',
                            perception: '#FBBF24',
                            experience: '#F87171',
                          }[type],
                        }}
                      />
                      <span>{icon} {type}</span>
                    </div>
                  ))}
                </div>
                <div className="mt-3 pt-3 border-t border-dark-border">
                  <p className="text-xs text-gray-500">
                    Node size = Memory strength<br/>
                    Glow = Active during retrieval
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right sidebar - Controls and details */}
        <div className="w-80 flex-shrink-0 flex flex-col gap-4 p-4 overflow-y-auto border-l border-dark-border">
          {/* Control Panel */}
          <div className="flex-shrink-0">
            <ControlPanel
              onAddMemory={addMemory}
              onRetrieve={retrieveMemory}
              onTriggerDMN={triggerDMN}
              onTriggerElo={triggerEloCompetition}
              onUpdateConfig={updateConfig}
              config={config}
              isLoading={isLoading}
              dmnResult={dmnResult}
            />
          </div>

          {/* Elo Ranking */}
          <div className="flex-shrink-0">
            <EloRanking
              rankings={eloRanking}
              onItemClick={(item) => {
                const node = network.nodes.find(n => n.id === item.id);
                if (node) setDetailNeuron(node);
              }}
            />
          </div>

          {/* Timeline */}
          <div className="flex-1 min-h-0">
            <Timeline
              items={timelineItems}
              onItemClick={(item) => {
                const node = network.nodes.find(n => n.id === item.id);
                if (node) setDetailNeuron(node);
              }}
            />
          </div>
        </div>
      </main>

      {/* Retrieval result overlay */}
      {retrievalResult && (
        <div className="fixed bottom-4 left-4 right-4 md:left-auto md:right-4 md:w-96 
                       bg-dark-card border border-purple-500/50 rounded-lg p-4 shadow-2xl
                       animate-slide-up z-40">
          <div className="flex items-start justify-between mb-2">
            <h4 className="font-semibold flex items-center gap-2">
              <span>🔍</span> Retrieval Result
            </h4>
          </div>
          <p className="text-sm text-gray-300 mb-3">{retrievalResult.content}</p>
          <div className="flex items-center gap-4 text-xs">
            <span className="text-purple-400">
              Score: {(retrievalResult.score * 100).toFixed(0)}%
            </span>
            <span className="text-gray-500">
              {retrievalResult.activated_neurons.length} neurons activated
            </span>
          </div>
        </div>
      )}

      {/* Error toast */}
      {error && (
        <div className="fixed top-20 left-1/2 transform -translate-x-1/2 
                       bg-red-500/90 text-white px-4 py-2 rounded-lg shadow-lg z-50">
          {error}
        </div>
      )}

      {/* Loading overlay */}
      {isLoading && (
        <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-30">
          <div className="spinner w-12 h-12 border-4 border-blue-500" />
        </div>
      )}

      {/* Neuron detail modal */}
      {detailNeuron && (
        <NeuronNodeDetail
          neuron={detailNeuron}
          onClose={handleCloseDetail}
        />
      )}
    </div>
  );
}

export default App;

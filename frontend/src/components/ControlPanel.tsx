import React, { useState, useCallback } from 'react';
import type { AddMemoryRequest, RetrieveRequest, SystemConfig, EventType } from '../types/memory';

interface ControlPanelProps {
  onAddMemory: (req: AddMemoryRequest) => Promise<unknown>;
  onRetrieve: (req: RetrieveRequest) => Promise<unknown>;
  onTriggerDMN: () => Promise<unknown>;
  onTriggerElo: () => Promise<unknown>;
  onUpdateConfig: (config: Partial<SystemConfig>) => void;
  config: SystemConfig;
  isLoading: boolean;
  dmnResult: { success: boolean; messages: string[] } | null;
}

const EVENT_TYPES: EventType[] = ['chat', 'thought', 'reflection', 'perception', 'experience'];

export const ControlPanel: React.FC<ControlPanelProps> = ({
  onAddMemory,
  onRetrieve,
  onTriggerDMN,
  onTriggerElo,
  onUpdateConfig,
  config,
  isLoading,
  dmnResult,
}) => {
  const [activeTab, setActiveTab] = useState<'add' | 'retrieve' | 'config'>('add');
  
  // Add memory form
  const [memoryContent, setMemoryContent] = useState('');
  const [memoryType, setMemoryType] = useState<EventType>('chat');
  const [memoryActor, setMemoryActor] = useState('user');
  const [impactScore, setImpactScore] = useState(0.5);

  // Retrieve form
  const [queryText, setQueryText] = useState('');
  const [topK, setTopK] = useState(5);
  const [threshold, setThreshold] = useState(0.3);

  const handleAddMemory = useCallback(async () => {
    if (!memoryContent.trim()) return;
    await onAddMemory({
      content: memoryContent,
      event_type: memoryType,
      actor: memoryActor,
      impact_score: impactScore,
    });
    setMemoryContent('');
  }, [memoryContent, memoryType, memoryActor, impactScore, onAddMemory]);

  const handleRetrieve = useCallback(async () => {
    if (!queryText.trim()) return;
    await onRetrieve({
      query: queryText,
      top_k: topK,
      threshold: threshold,
    });
  }, [queryText, topK, threshold, onRetrieve]);

  return (
    <div className="bg-dark-card rounded-lg p-4 h-full flex flex-col">
      <h3 className="text-lg font-semibold mb-4">🎛️ Control Panel</h3>

      {/* Tabs */}
      <div className="flex gap-1 mb-4">
        {(['add', 'retrieve', 'config'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`
              px-3 py-1.5 text-sm rounded-lg transition-colors
              ${activeTab === tab 
                ? 'bg-blue-600 text-white' 
                : 'bg-dark-bg text-gray-400 hover:text-white'}
            `}
          >
            {tab === 'add' ? '➕ Add' : tab === 'retrieve' ? '🔍 Retrieve' : '⚙️ Config'}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === 'add' && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Memory Content</label>
              <textarea
                value={memoryContent}
                onChange={(e) => setMemoryContent(e.target.value)}
                placeholder="Enter memory content..."
                className="w-full px-3 py-2 bg-dark-bg rounded-lg border border-dark-border 
                         focus:border-blue-500 focus:outline-none resize-none"
                rows={3}
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Event Type</label>
              <div className="flex flex-wrap gap-2">
                {EVENT_TYPES.map((type) => (
                  <button
                    key={type}
                    onClick={() => setMemoryType(type)}
                    className={`
                      px-2 py-1 text-xs rounded-lg transition-colors
                      ${memoryType === type 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-dark-bg text-gray-400 hover:text-white'}
                    `}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Actor</label>
              <input
                type="text"
                value={memoryActor}
                onChange={(e) => setMemoryActor(e.target.value)}
                placeholder="user, assistant, system..."
                className="w-full px-3 py-2 bg-dark-bg rounded-lg border border-dark-border 
                         focus:border-blue-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Impact Score: {impactScore.toFixed(2)}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={impactScore}
                onChange={(e) => setImpactScore(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <button
              onClick={handleAddMemory}
              disabled={isLoading || !memoryContent.trim()}
              className="w-full py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600
                       rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <div className="spinner w-4 h-4" />
                  Adding...
                </>
              ) : (
                <>➕ Add Memory</>
              )}
            </button>
          </div>
        )}

        {activeTab === 'retrieve' && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Query</label>
              <textarea
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                placeholder="Enter search query..."
                className="w-full px-3 py-2 bg-dark-bg rounded-lg border border-dark-border 
                         focus:border-blue-500 focus:outline-none resize-none"
                rows={2}
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Top K: {topK}</label>
              <input
                type="range"
                min="1"
                max="20"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Threshold: {threshold.toFixed(2)}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <button
              onClick={handleRetrieve}
              disabled={isLoading || !queryText.trim()}
              className="w-full py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600
                       rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <div className="spinner w-4 h-4" />
                  Retrieving...
                </>
              ) : (
                <>🔍 Retrieve Memory</>
              )}
            </button>
          </div>
        )}

        {activeTab === 'config' && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Decay Rate: {config.decay_rate.toFixed(3)}
              </label>
              <input
                type="range"
                min="0.9"
                max="1"
                step="0.001"
                value={config.decay_rate}
                onChange={(e) => onUpdateConfig({ decay_rate: parseFloat(e.target.value) })}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Activation Threshold: {config.activation_threshold.toFixed(2)}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={config.activation_threshold}
                onChange={(e) => onUpdateConfig({ activation_threshold: parseFloat(e.target.value) })}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Elo K-Factor: {config.elo_k_factor}
              </label>
              <input
                type="range"
                min="8"
                max="64"
                step="4"
                value={config.elo_k_factor}
                onChange={(e) => onUpdateConfig({ elo_k_factor: parseInt(e.target.value) })}
                className="w-full"
              />
            </div>

            <div className="flex flex-col gap-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.auto_decay}
                  onChange={(e) => onUpdateConfig({ auto_decay: e.target.checked })}
                  className="w-4 h-4 rounded"
                />
                <span className="text-sm">Auto Decay</span>
              </label>
              
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.auto_consolidation}
                  onChange={(e) => onUpdateConfig({ auto_consolidation: e.target.checked })}
                  className="w-4 h-4 rounded"
                />
                <span className="text-sm">Auto Consolidation</span>
              </label>
            </div>

            <div className="pt-2 border-t border-dark-border space-y-2">
              <button
                onClick={onTriggerDMN}
                disabled={isLoading}
                className="w-full py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-600
                         rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
              >
                🧠 Trigger DMN Consolidation
              </button>

              <button
                onClick={onTriggerElo}
                disabled={isLoading}
                className="w-full py-2 bg-orange-600 hover:bg-orange-700 disabled:bg-gray-600
                         rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
              >
                ⚔️ Trigger Elo Competition
              </button>
            </div>
          </div>
        )}
      </div>

      {/* DMN Result Toast */}
      {dmnResult && dmnResult.success && (
        <div className="mt-4 p-3 bg-emerald-500/20 border border-emerald-500/50 rounded-lg">
          <p className="text-sm text-emerald-400 font-medium">✓ DMN Complete</p>
          <ul className="text-xs text-gray-400 mt-1">
            {dmnResult.messages.map((msg, i) => (
              <li key={i}>{msg}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default ControlPanel;

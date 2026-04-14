import React from 'react';
import type { NeuronNode } from '../types/memory';

const COLORS: Record<string, string> = {
  chat: '#60A5FA',
  thought: '#A78BFA',
  reflection: '#34D399',
  perception: '#FBBF24',
  experience: '#F87171',
};

interface NeuronNodeDetailProps {
  neuron: NeuronNode;
  onClose: () => void;
  onHighlight?: (connectedIds: string[]) => void;
}

export const NeuronNodeDetail: React.FC<NeuronNodeDetailProps> = ({
  neuron,
  onClose,
  onHighlight,
}) => {
  const color = COLORS[neuron.type] || '#6B7280';

  const getEmotionLabel = (valence: number): string => {
    if (valence > 0.5) return 'Very Positive';
    if (valence > 0.2) return 'Positive';
    if (valence < -0.5) return 'Very Negative';
    if (valence < -0.2) return 'Negative';
    return 'Neutral';
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm animate-fade-in">
      <div 
        className="bg-dark-card rounded-xl shadow-2xl w-full max-w-md mx-4 overflow-hidden animate-slide-up"
        style={{ borderTop: `4px solid ${color}` }}
      >
        {/* Header */}
        <div className="p-4 border-b border-dark-border">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div
                className="w-12 h-12 rounded-full flex items-center justify-center text-xl"
                style={{ backgroundColor: `${color}20` }}
              >
                {getTypeEmoji(neuron.type)}
              </div>
              <div>
                <h3 className="font-semibold">{neuron.label}</h3>
                <span
                  className="text-xs px-2 py-0.5 rounded-full"
                  style={{ backgroundColor: `${color}20`, color }}
                >
                  {neuron.type}
                </span>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-1 hover:bg-dark-bg rounded-lg transition-colors"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* Strength */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-400">Strength</span>
              <span>{(neuron.strength * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2 bg-dark-bg rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all"
                style={{ width: `${neuron.strength * 100}%`, backgroundColor: color }}
              />
            </div>
          </div>

          {/* Elo */}
          <div className="flex justify-between items-center">
            <span className="text-gray-400">Elo Rating</span>
            <span className="text-lg font-bold">{neuron.elo.toFixed(0)}</span>
          </div>

          {/* Stats grid */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-dark-bg rounded-lg p-3">
              <p className="text-xs text-gray-500">Activation Threshold</p>
              <p className="text-lg font-semibold">
                {((neuron.activation_threshold ?? 0.3) * 100).toFixed(0)}%
              </p>
            </div>
            <div className="bg-dark-bg rounded-lg p-3">
              <p className="text-xs text-gray-500">Consolidated</p>
              <p className="text-lg font-semibold">
                {neuron.is_consolidated ? '✓ Yes' : '✗ No'}
              </p>
            </div>
          </div>

          {/* Emotional state */}
          {neuron.emotional_valence !== undefined && (
            <div className="bg-dark-bg rounded-lg p-3">
              <p className="text-xs text-gray-500 mb-2">Emotional State</p>
              <div className="flex items-center justify-between">
                <span>{getEmotionEmoji(neuron.emotional_valence)}</span>
                <span className="text-sm">{getEmotionLabel(neuron.emotional_valence)}</span>
              </div>
              <div className="mt-2 flex items-center gap-2 text-xs text-gray-500">
                <span>Arousal: {((neuron.emotional_arousal ?? 0.5) * 100).toFixed(0)}%</span>
              </div>
            </div>
          )}

          {/* Content preview */}
          {neuron.content && (
            <div className="bg-dark-bg rounded-lg p-3">
              <p className="text-xs text-gray-500 mb-1">Content</p>
              <p className="text-sm line-clamp-3">{neuron.content}</p>
            </div>
          )}

          {/* Created time */}
          {neuron.created_at && (
            <p className="text-xs text-gray-500 text-center">
              Created: {new Date(neuron.created_at).toLocaleString()}
            </p>
          )}
        </div>

        {/* Footer actions */}
        <div className="p-4 border-t border-dark-border flex gap-2">
          <button
            onClick={onClose}
            className="flex-1 py-2 bg-dark-bg hover:bg-gray-700 rounded-lg transition-colors"
          >
            Close
          </button>
          {onHighlight && (
            <button
              onClick={() => onHighlight([neuron.id])}
              className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
            >
              Highlight
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

function getTypeEmoji(type: string): string {
  const icons: Record<string, string> = {
    chat: '💬',
    thought: '💭',
    reflection: '🔄',
    perception: '👁️',
    experience: '⭐',
  };
  return icons[type] || '•';
}

function getEmotionEmoji(valence: number): string {
  if (valence > 0.5) return '😄';
  if (valence > 0.2) return '🙂';
  if (valence < -0.5) return '😢';
  if (valence < -0.2) return '😔';
  return '😐';
}

export default NeuronNodeDetail;

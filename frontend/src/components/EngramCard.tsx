import React from 'react';
import type { NeuronNode } from '../types/memory';

const COLORS: Record<string, string> = {
  chat: '#60A5FA',
  thought: '#A78BFA',
  reflection: '#34D399',
  perception: '#FBBF24',
  experience: '#F87171',
};

interface EngramCardProps {
  neuron: NeuronNode;
  isSelected?: boolean;
  isActive?: boolean;
  onClick?: () => void;
}

export const EngramCard: React.FC<EngramCardProps> = ({
  neuron,
  isSelected = false,
  isActive = false,
  onClick,
}) => {
  const color = COLORS[neuron.type] || '#6B7280';
  
  return (
    <div
      onClick={onClick}
      className={`
        relative p-4 rounded-lg border cursor-pointer
        transition-all duration-200 card-hover
        ${isSelected 
          ? 'bg-dark-card border-white/50 shadow-lg' 
          : 'bg-dark-card/50 border-dark-border hover:border-gray-500'}
        ${isActive ? 'ring-2 ring-offset-2 ring-offset-dark-bg' : ''}
      `}
      style={{
        borderColor: isActive ? color : undefined,
        boxShadow: isActive ? `0 0 20px ${color}40` : undefined,
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <span
          className="px-2 py-0.5 text-xs font-medium rounded-full"
          style={{ backgroundColor: `${color}20`, color }}
        >
          {neuron.type}
        </span>
        {neuron.is_consolidated && (
          <span className="text-xs text-emerald-400">✓ Consolidated</span>
        )}
      </div>

      {/* Content */}
      <h4 className="font-medium text-sm mb-2 line-clamp-2">
        {neuron.label}
      </h4>

      {/* Stats */}
      <div className="flex items-center gap-4 text-xs text-gray-400">
        <div className="flex items-center gap-1">
          <span>Strength:</span>
          <div className="w-16 h-1.5 bg-dark-bg rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{
                width: `${neuron.strength * 100}%`,
                backgroundColor: color,
              }}
            />
          </div>
          <span>{(neuron.strength * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Elo badge */}
      <div className="mt-2 flex items-center justify-between">
        <span className="text-xs text-gray-500">
          Elo: <span className="text-gray-300">{neuron.elo.toFixed(0)}</span>
        </span>
        {neuron.emotional_valence !== undefined && (
          <span className="text-xs">
            {neuron.emotional_valence > 0 ? '😊' : neuron.emotional_valence < 0 ? '😔' : '😐'}
          </span>
        )}
      </div>

      {/* Active glow effect */}
      {isActive && (
        <div
          className="absolute inset-0 rounded-lg animate-pulse pointer-events-none"
          style={{
            background: `radial-gradient(circle at center, ${color}20 0%, transparent 70%)`,
          }}
        />
      )}
    </div>
  );
};

export default EngramCard;

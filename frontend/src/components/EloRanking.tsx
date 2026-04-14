import React from 'react';
import type { EloRankingItem } from '../types/memory';

const COLORS: Record<string, string> = {
  chat: '#60A5FA',
  thought: '#A78BFA',
  reflection: '#34D399',
  perception: '#FBBF24',
  experience: '#F87171',
};

interface EloRankingProps {
  rankings: EloRankingItem[];
  onItemClick?: (item: EloRankingItem) => void;
}

export const EloRanking: React.FC<EloRankingProps> = ({ rankings, onItemClick }) => {
  const getRankIcon = (index: number): string => {
    if (index === 0) return '🥇';
    if (index === 1) return '🥈';
    if (index === 2) return '🥉';
    return `${index + 1}`;
  };

  const getRankChangeColor = (change: number): string => {
    if (change > 0) return 'text-emerald-400';
    if (change < 0) return 'text-red-400';
    return 'text-gray-400';
  };

  return (
    <div className="bg-dark-card rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <span>🏆</span> Elo Ranking
      </h3>
      
      <div className="space-y-2">
        {rankings.length === 0 ? (
          <p className="text-gray-500 text-sm text-center py-4">No rankings yet</p>
        ) : (
          rankings.map((item, index) => (
            <div
              key={item.id}
              onClick={() => onItemClick?.(item)}
              className={`
                flex items-center gap-3 p-2 rounded-lg
                hover:bg-dark-bg/50 cursor-pointer transition-colors
                ${index < 3 ? 'bg-yellow-500/10' : ''}
              `}
            >
              {/* Rank */}
              <div className="w-8 text-center font-medium">
                {typeof getRankIcon(index) === 'string' ? (
                  <span className="text-gray-400">{getRankIcon(index)}</span>
                ) : (
                  <span>{getRankIcon(index)}</span>
                )}
              </div>

              {/* Type indicator */}
              <div
                className="w-2 h-8 rounded-full"
                style={{ backgroundColor: COLORS[item.type] || '#6B7280' }}
              />

              {/* Info */}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">
                  {item.label}
                </p>
                <p className="text-xs text-gray-500">{item.type}</p>
              </div>

              {/* Elo */}
              <div className="text-right">
                <p className="text-sm font-bold text-white">
                  {item.elo.toFixed(0)}
                </p>
                {item.rank_change !== 0 && (
                  <p className={`text-xs ${getRankChangeColor(item.rank_change)}`}>
                    {item.rank_change > 0 ? '+' : ''}{item.rank_change}
                  </p>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default EloRanking;

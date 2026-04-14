import React from 'react';
import type { MemoryItem } from '../types/memory';

const COLORS: Record<string, string> = {
  chat: '#60A5FA',
  thought: '#A78BFA',
  reflection: '#34D399',
  perception: '#FBBF24',
  experience: '#F87171',
};

interface TimelineProps {
  items: MemoryItem[];
  onItemClick?: (item: MemoryItem) => void;
}

export const Timeline: React.FC<TimelineProps> = ({ items, onItemClick }) => {
  const formatTime = (timestamp: string): string => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      hour12: false 
    });
  };

  const formatDate = (timestamp: string): string => {
    const date = new Date(timestamp);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    if (date.toDateString() === today.toDateString()) {
      return 'Today';
    } else if (date.toDateString() === yesterday.toDateString()) {
      return 'Yesterday';
    }
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  // Group items by date
  const groupedItems = items.reduce((acc, item) => {
    const dateKey = formatDate(item.timestamp);
    if (!acc[dateKey]) {
      acc[dateKey] = [];
    }
    acc[dateKey].push(item);
    return acc;
  }, {} as Record<string, MemoryItem[]>);

  return (
    <div className="bg-dark-card rounded-lg p-4 h-full overflow-hidden flex flex-col">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <span>📜</span> Memory Timeline
      </h3>

      <div className="flex-1 overflow-y-auto">
        {items.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            <div className="text-3xl mb-2">📭</div>
            <p>No memories recorded yet</p>
          </div>
        ) : (
          <div className="relative">
            {/* Timeline line */}
            <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-dark-border" />

            {Object.entries(groupedItems).map(([date, dateItems]) => (
              <div key={date} className="mb-4">
                {/* Date header */}
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-3 h-3 rounded-full bg-blue-500 z-10" />
                  <span className="text-sm font-medium text-gray-400">{date}</span>
                </div>

                {/* Items for this date */}
                <div className="ml-12 space-y-3">
                  {dateItems.map((item) => (
                    <div
                      key={item.id}
                      onClick={() => onItemClick?.(item)}
                      className={`
                        p-3 rounded-lg bg-dark-bg/50 hover:bg-dark-bg
                        cursor-pointer transition-colors
                        border border-transparent hover:border-dark-border
                      `}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm line-clamp-2">{item.content}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <span
                              className="text-xs px-1.5 py-0.5 rounded"
                              style={{
                                backgroundColor: `${COLORS[item.type]}20`,
                                color: COLORS[item.type],
                              }}
                            >
                              {item.type}
                            </span>
                            <span className="text-xs text-gray-500">
                              {item.actor}
                            </span>
                          </div>
                        </div>
                        <div className="text-xs text-gray-500">
                          {formatTime(item.timestamp)}
                        </div>
                      </div>

                      {/* Strength indicator */}
                      <div className="mt-2 flex items-center gap-2">
                        <div className="flex-1 h-1 bg-dark-border rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all"
                            style={{
                              width: `${item.strength * 100}%`,
                              backgroundColor: COLORS[item.type],
                            }}
                          />
                        </div>
                        <span className="text-xs text-gray-500 w-16 text-right">
                          ELO: {item.elo.toFixed(0)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Timeline;

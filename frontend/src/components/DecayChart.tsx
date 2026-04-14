import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import type { DecayHistoryItem } from '../types/memory';

interface DecayChartProps {
  history: DecayHistoryItem[];
  decayRate?: number;
}

export const DecayChart: React.FC<DecayChartProps> = ({ history, decayRate = 0.995 }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = 200;
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    svg.attr('width', width).attr('height', height);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Calculate decay curve points
    const maxTime = 24; // hours
    const points = d3.range(0, maxTime + 1).map(hour => ({
      hour,
      strength: Math.pow(decayRate, hour),
    }));

    // Scales
    const xScale = d3.scaleLinear()
      .domain([0, maxTime])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([innerHeight, 0]);

    // Area generator
    const area = d3.area<{ hour: number; strength: number }>()
      .x(d => xScale(d.hour))
      .y0(innerHeight)
      .y1(d => yScale(d.strength))
      .curve(d3.curveMonotoneX);

    // Line generator
    const line = d3.line<{ hour: number; strength: number }>()
      .x(d => xScale(d.hour))
      .y(d => yScale(d.strength))
      .curve(d3.curveMonotoneX);

    // Gradient
    const gradient = svg.append('defs')
      .append('linearGradient')
      .attr('id', 'decayGradient')
      .attr('x1', '0%')
      .attr('y1', '0%')
      .attr('x2', '0%')
      .attr('y2', '100%');

    gradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#60A5FA')
      .attr('stop-opacity', 0.4);

    gradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#60A5FA')
      .attr('stop-opacity', 0);

    // Draw area
    g.append('path')
      .datum(points)
      .attr('fill', 'url(#decayGradient)')
      .attr('d', area);

    // Draw line
    g.append('path')
      .datum(points)
      .attr('fill', 'none')
      .attr('stroke', '#60A5FA')
      .attr('stroke-width', 2)
      .attr('d', line);

    // X axis
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .attr('color', '#64748B')
      .selectAll('text')
      .attr('fill', '#64748B');

    // Y axis
    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5).tickFormat(d => `${(d as number * 100).toFixed(0)}%`))
      .attr('color', '#64748B')
      .selectAll('text')
      .attr('fill', '#64748B');

    // Half-life marker
    const halfLife = Math.log(0.5) / Math.log(decayRate);
    if (halfLife <= maxTime) {
      g.append('line')
        .attr('x1', xScale(halfLife))
        .attr('x2', xScale(halfLife))
        .attr('y1', 0)
        .attr('y2', innerHeight)
        .attr('stroke', '#FBBF24')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4,4');

      g.append('text')
        .attr('x', xScale(halfLife))
        .attr('y', -5)
        .attr('text-anchor', 'middle')
        .attr('fill', '#FBBF24')
        .attr('font-size', '10px')
        .text(`t½=${halfLife.toFixed(1)}h`);
    }

    // Grid lines
    g.append('g')
      .attr('class', 'grid')
      .attr('opacity', 0.1)
      .call(d3.axisLeft(yScale)
        .ticks(5)
        .tickSize(-innerWidth)
        .tickFormat(() => '')
      );

  }, [decayRate]);

  return (
    <div className="bg-dark-card rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <span>📉</span> Decay Curve
        </h3>
        <span className="text-xs text-gray-400">
          Rate: {decayRate.toFixed(3)} | Half-life: {halfLife(decayRate).toFixed(1)}h
        </span>
      </div>
      
      <div ref={containerRef} className="w-full">
        <svg ref={svgRef} />
      </div>

      {history.length > 0 && (
        <div className="mt-4 border-t border-dark-border pt-2">
          <p className="text-xs text-gray-400 mb-2">Recent Decay Events:</p>
          <div className="space-y-1 max-h-20 overflow-y-auto">
            {history.slice(-5).reverse().map((item) => (
              <div key={item.neuron_id} className="text-xs flex justify-between">
                <span className="text-gray-500 truncate max-w-[150px]">{item.label}</span>
                <span className="text-gray-400">
                  {(item.previous_strength * 100).toFixed(0)}% → {(item.current_strength * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

function halfLife(decayRate: number): number {
  return Math.log(0.5) / Math.log(decayRate);
}

export default DecayChart;

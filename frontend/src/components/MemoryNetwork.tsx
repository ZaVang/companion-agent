import React, { useRef, useEffect, useCallback } from 'react';
import * as d3 from 'd3';
import type { MemoryNetwork, GraphNode, GraphLink } from '../types/memory';

const COLORS: Record<string, string> = {
  chat: '#60A5FA',
  thought: '#A78BFA',
  reflection: '#34D399',
  perception: '#FBBF24',
  experience: '#F87171',
};

interface MemoryNetworkProps {
  data: MemoryNetwork;
  onNodeClick: (node: GraphNode) => void;
  selectedNodeId?: string;
  width?: number;
  height?: number;
}

export const MemoryNetworkGraph: React.FC<MemoryNetworkProps> = ({
  data,
  onNodeClick,
  selectedNodeId,
  width = 800,
  height = 600,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const simulationRef = useRef<d3.Simulation<GraphNode, GraphLink> | null>(null);

  const drawGraph = useCallback(() => {
    if (!svgRef.current || !data.nodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create container group for zoom
    const container = svg.append('g');

    // Setup zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Process data
    const nodes: GraphNode[] = data.nodes.map(n => ({ ...n }));
    const links: GraphLink[] = data.links.map(l => ({
      source: l.source,
      target: l.target,
      weight: l.weight,
    }));

    // Create force simulation
    const simulation = d3.forceSimulation<GraphNode>(nodes)
      .force('link', d3.forceLink<GraphNode, GraphLink>(links).id(d => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    simulationRef.current = simulation;

    // Draw links
    const link = container.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('stroke', '#475569')
      .attr('stroke-width', d => Math.max(1, d.weight * 3))
      .attr('stroke-opacity', 0.6);

    // Draw nodes
    const node = container.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'neuron-node')
      .call(d3.drag<SVGGElement, GraphNode>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }));

    // Node circles
    node.append('circle')
      .attr('r', d => 10 + d.strength * 15)
      .attr('fill', d => COLORS[d.type] || '#6B7280')
      .attr('stroke', d => d.id === selectedNodeId ? '#fff' : 'transparent')
      .attr('stroke-width', 3)
      .attr('class', d => d.is_active ? 'node-active' : '')
      .style('filter', d => d.is_active ? `drop-shadow(0 0 10px ${COLORS[d.type]})` : 'none');

    // Node labels
    node.append('text')
      .text(d => d.label.length > 15 ? d.label.substring(0, 15) + '...' : d.label)
      .attr('x', 0)
      .attr('y', d => 10 + d.strength * 15 + 15)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94A3B8')
      .attr('font-size', '11px')
      .style('pointer-events', 'none');

    // Node type icon
    node.append('text')
      .text(d => getTypeIcon(d.type))
      .attr('x', 0)
      .attr('y', 4)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .style('pointer-events', 'none');

    // Click handler
    node.on('click', (event, d) => {
      event.stopPropagation();
      onNodeClick(d);
    });

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as GraphNode).x || 0)
        .attr('y1', d => (d.source as GraphNode).y || 0)
        .attr('x2', d => (d.target as GraphNode).x || 0)
        .attr('y2', d => (d.target as GraphNode).y || 0);

      node.attr('transform', d => `translate(${d.x || 0},${d.y || 0})`);
    });

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [data, width, height, onNodeClick, selectedNodeId]);

  useEffect(() => {
    const cleanup = drawGraph();
    return () => {
      cleanup?.();
      simulationRef.current?.stop();
    };
  }, [drawGraph]);

  return (
    <div className="relative w-full h-full bg-dark-bg rounded-lg overflow-hidden">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="w-full h-full"
      />
      {data.nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center text-gray-500">
            <div className="text-4xl mb-2">🧠</div>
            <p>No memories yet</p>
            <p className="text-sm">Add a memory to see the network</p>
          </div>
        </div>
      )}
    </div>
  );
};

function getTypeIcon(type: string): string {
  const icons: Record<string, string> = {
    chat: '💬',
    thought: '💭',
    reflection: '🔄',
    perception: '👁️',
    experience: '⭐',
  };
  return icons[type] || '•';
}

export default MemoryNetworkGraph;

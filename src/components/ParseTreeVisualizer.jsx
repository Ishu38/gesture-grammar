/**
 * ParseTreeVisualizer.jsx
 * Renders a constituency parse tree diagram using SVG.
 * Takes the JSON parse tree from the Nearley parser and displays
 * a hierarchical tree (S at top, branching to NP/VP).
 *
 * The tree follows standard linguistics notation:
 *   - Internal nodes show category labels (S, NP, VP)
 *   - Leaf nodes show category + a terminal word node below
 */

import { useMemo } from 'react';
import { LEXICON } from '../utils/GrammarEngine';

// ── Node color scheme (matches existing BLOCK_COLORS) ───────────────────────

const NODE_STYLES = {
  S:    { fill: '#1e293b', stroke: '#64748b', text: '#e2e8f0' },
  NP:   { fill: '#1e3a8a', stroke: '#3b82f6', text: '#bfdbfe' },
  VP:   { fill: '#581c87', stroke: '#a855f7', text: '#e9d5ff' },
  VT:   { fill: '#7f1d1d', stroke: '#ef4444', text: '#fecaca' },
  VI:   { fill: '#7f1d1d', stroke: '#ef4444', text: '#fecaca' },
  OBJ:  { fill: '#14532d', stroke: '#22c55e', text: '#bbf7d0' },
  WORD: { fill: 'none',    stroke: 'none',    text: '#fbbf24' },
};

// ── Layout constants ────────────────────────────────────────────────────────

const LEVEL_HEIGHT = 80;
const MIN_LEAF_WIDTH = 120;
const NODE_HEIGHT = 30;
const NODE_BORDER_RADIUS = 6;
const NODE_PADDING_X = 14;

// ── Tree transformation ─────────────────────────────────────────────────────

/**
 * Transform the Nearley parse tree into a normalized form where every
 * category leaf gets a WORD child node (standard constituency tree style).
 *
 * Input:  { type: 'NP', value: 'SUBJECT_HE', person: 3, number: 'singular' }
 * Output: { type: 'NP', children: [{ type: 'WORD', displayWord: 'He', children: [] }] }
 */
function normalizeTree(node) {
  if (node.children && node.children.length > 0) {
    return {
      type: node.type,
      children: node.children.map(normalizeTree),
    };
  }

  // Leaf category node -> add a terminal word child
  const entry = node.value ? LEXICON[node.value] : null;
  const displayWord = entry?.display || node.value || '?';

  return {
    type: node.type,
    children: [{
      type: 'WORD',
      displayWord,
      children: [],
    }],
  };
}

// ── Layout computation ──────────────────────────────────────────────────────

/**
 * Compute (x, y) positions for every node and edge in the tree.
 * Uses a bottom-up leaf count to determine horizontal spacing,
 * then a top-down pass to assign coordinates.
 */
function computeLayout(tree) {
  // Use a WeakMap to avoid mutating the input tree nodes
  const leafCounts = new WeakMap();

  // Pass 1: count leaves bottom-up (determines subtree width)
  function countLeaves(node) {
    if (!node.children || node.children.length === 0) {
      leafCounts.set(node, 1);
    } else {
      leafCounts.set(node, node.children.reduce(
        (sum, child) => sum + countLeaves(child), 0
      ));
    }
    return leafCounts.get(node);
  }
  countLeaves(tree);

  const totalWidth = leafCounts.get(tree) * MIN_LEAF_WIDTH;
  const nodes = [];
  const edges = [];
  let maxDepth = 0;

  // Pass 2: assign positions top-down
  function layout(node, x, width, depth) {
    const cx = x + width / 2;
    const cy = 30 + depth * LEVEL_HEIGHT;
    if (depth > maxDepth) maxDepth = depth;

    nodes.push({
      type: node.type,
      displayWord: node.displayWord || null,
      x: cx,
      y: cy,
      depth,
    });

    if (node.children && node.children.length > 0) {
      let childX = x;
      for (const child of node.children) {
        const childWidth = (leafCounts.get(child) / leafCounts.get(node)) * width;
        const childCx = childX + childWidth / 2;
        const childCy = 30 + (depth + 1) * LEVEL_HEIGHT;

        edges.push({
          x1: cx, y1: cy + NODE_HEIGHT / 2,
          x2: childCx, y2: childCy - NODE_HEIGHT / 2,
        });

        layout(child, childX, childWidth, depth + 1);
        childX += childWidth;
      }
    }
  }

  layout(tree, 0, totalWidth, 0);

  return {
    nodes,
    edges,
    width: totalWidth,
    height: 30 + maxDepth * LEVEL_HEIGHT + NODE_HEIGHT + 20,
  };
}

// ── SVG sub-components ──────────────────────────────────────────────────────

function TreeEdge({ edge, index }) {
  return (
    <line
      x1={edge.x1}
      y1={edge.y1}
      x2={edge.x2}
      y2={edge.y2}
      stroke="rgba(148, 163, 184, 0.4)"
      strokeWidth={2}
      strokeLinecap="round"
      style={{
        animation: `tree-edge-appear 0.3s ease-out ${index * 0.06}s both`,
      }}
    />
  );
}

function TreeNode({ node, index }) {
  const style = NODE_STYLES[node.type] || NODE_STYLES.S;
  const isWord = node.type === 'WORD';
  const label = isWord ? node.displayWord : node.type;

  // Estimate box width from label length
  const charWidth = isWord ? 9 : 11;
  const textWidth = label.length * charWidth + NODE_PADDING_X * 2;
  const boxWidth = Math.max(textWidth, 44);

  return (
    <g
      style={{
        animation: `tree-node-appear 0.4s ease-out ${index * 0.06}s both`,
      }}
    >
      {/* Category nodes get a rounded rect; word nodes are bare text */}
      {!isWord && (
        <rect
          x={node.x - boxWidth / 2}
          y={node.y - NODE_HEIGHT / 2}
          width={boxWidth}
          height={NODE_HEIGHT}
          rx={NODE_BORDER_RADIUS}
          fill={style.fill}
          stroke={style.stroke}
          strokeWidth={2}
        />
      )}
      <text
        x={node.x}
        y={node.y + (isWord ? 2 : 1)}
        textAnchor="middle"
        dominantBaseline="middle"
        fill={style.text}
        fontSize={isWord ? 14 : 13}
        fontWeight={isWord ? 400 : 700}
        fontStyle={isWord ? 'italic' : 'normal'}
        fontFamily="ui-monospace, SFMono-Regular, monospace"
      >
        {label}
      </text>
    </g>
  );
}

// ── Legend ───────────────────────────────────────────────────────────────────

const LEGEND_ITEMS = [
  { type: 'S', label: 'Sentence' },
  { type: 'NP', label: 'Noun Phrase' },
  { type: 'VP', label: 'Verb Phrase' },
  { type: 'VT', label: 'Verb (trans.)' },
  { type: 'OBJ', label: 'Object' },
];

function TreeLegend() {
  return (
    <div className="parse-tree-legend">
      {LEGEND_ITEMS.map(({ type, label }) => {
        const style = NODE_STYLES[type];
        return (
          <div key={type} className="legend-item">
            <span
              className="legend-swatch"
              style={{
                background: style.fill,
                borderColor: style.stroke,
              }}
            />
            <span className="legend-label">
              <strong>{type}</strong> = {label}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ── Main component ──────────────────────────────────────────────────────────

function ParseTreeVisualizer({ parseTree }) {
  const layout = useMemo(() => {
    if (!parseTree) return null;
    const normalized = normalizeTree(parseTree);
    return computeLayout(normalized);
  }, [parseTree]);

  if (!layout) return null;

  return (
    <div className="parse-tree-container">
      <div className="parse-tree-header">
        <span className="parse-tree-title">Parse Tree</span>
        <span className="parse-tree-subtitle">Constituency Structure</span>
      </div>

      <div className="parse-tree-svg-wrapper">
        <svg
          viewBox={`0 0 ${layout.width} ${layout.height}`}
          width="100%"
          preserveAspectRatio="xMidYMid meet"
          className="parse-tree-svg"
        >
          {/* Edges (drawn behind nodes) */}
          {layout.edges.map((edge, i) => (
            <TreeEdge key={`e-${i}`} edge={edge} index={i} />
          ))}
          {/* Nodes */}
          {layout.nodes.map((node, i) => (
            <TreeNode key={`n-${i}`} node={node} index={i} />
          ))}
        </svg>
      </div>

      <TreeLegend />
    </div>
  );
}

export default ParseTreeVisualizer;

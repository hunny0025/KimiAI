/**
 * XAIWaterfallChart.jsx – Interactive Explainability Waterfall Chart
 * SkillGenome X
 *
 * Renders all XAI factors (positive + negative) as a unified sorted horizontal
 * bar chart. Green = score boosters, red = score reducers.
 * Hovering shows a rich tooltip with a plain-English explanation.
 *
 * Props:
 *   topPositive  – array of { feature: string, impact: number }  (positive)
 *   topNegative  – array of { feature: string, impact: number }  (negative, impact < 0)
 *   totalScore   – number (0-100) shown in the header
 */

import React, { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Cell, ResponsiveContainer, ReferenceLine,
} from 'recharts';
import { Brain, TrendingUp, TrendingDown, Info } from 'lucide-react';

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Convert a raw feature name like "digital_presence" → "Digital Presence"
 */
const fmt = (name) =>
  String(name)
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());

/**
 * Generate a plain-English explanation for a factor.
 * Positive impact → "increased", negative → "reduced"
 */
const buildExplanation = (feature, impact) => {
  const label = fmt(feature);
  const pts = Math.abs(impact).toFixed(1);
  const dir = impact >= 0 ? 'increased' : 'reduced';
  const strength = Math.abs(impact) > 8 ? 'significantly ' : Math.abs(impact) > 4 ? '' : 'slightly ';

  const CONTEXT = {
    digital_presence:
      'Digital presence reflects how actively this person uses online platforms for work or learning.',
    learning_behavior:
      'Learning behavior captures commitment to continuous skill development and training hours.',
    creation_output:
      'Creation output measures tangible work delivered—projects, products, or creative output.',
    experience_consistency:
      'Experience consistency shows steady professional activity over time without long gaps.',
    economic_activity:
      'Economic activity indicates engagement in income-generating or trade activities.',
    innovation_problem_solving:
      'Innovation score reflects ability to devise novel solutions and adapt to new challenges.',
    collaboration_community:
      'Collaboration score measures participation in team work and community initiatives.',
    offline_capability:
      'Offline capability shows skill and productivity without digital infrastructure dependence.',
    opportunity_level:
      `Opportunity level reflects the socio-economic environment available in this person's region.`,
    digital_access:
      `Digital access shows quality of internet and device connectivity in this person's area.`,
    literacy_rate:
      'The regional literacy rate impacts baseline knowledge access and workforce readiness.',
    unemployment_rate:
      'High regional unemployment makes it harder to convert skills into economic value.',
    internet_penetration:
      'Internet penetration in the region affects digital skilling and market reach opportunities.',
    per_capita_income:
      'Local per-capita income sets the economic ceiling for skill monetization.',
  };

  const context = CONTEXT[feature] || `${label} is a key feature in the SkillGenome AI model.`;
  return `${context} This factor ${strength}${dir} this person\u2019s AI score by ${pts} points.`;
};

// ─── Custom Tooltip ───────────────────────────────────────────────────────────

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div
      className="max-w-xs rounded-xl border border-gray-700 shadow-2xl backdrop-blur-md"
      style={{ background: 'rgba(15,23,42,0.97)', padding: '12px 14px' }}
    >
      <div className="flex items-center gap-2 mb-2">
        <div
          className="w-2.5 h-2.5 rounded-full flex-shrink-0"
          style={{ background: d.color }}
        />
        <span className="text-xs font-bold text-white">{fmt(d.feature)}</span>
        <span
          className="ml-auto text-xs font-mono font-bold"
          style={{ color: d.color }}
        >
          {d.impact >= 0 ? '+' : ''}
          {d.impact.toFixed(1)} pts
        </span>
      </div>
      <p className="text-[11px] text-gray-300 leading-relaxed">
        {buildExplanation(d.feature, d.impact)}
      </p>
    </div>
  );
};

// ─── Main Component ───────────────────────────────────────────────────────────

const XAIWaterfallChart = ({ topPositive = [], topNegative = [], totalScore }) => {
  const [hovered, setHovered] = useState(null);

  // Merge, normalise sign, and sort by |impact| descending
  const allFactors = [
    ...topPositive.map((f) => ({ ...f, impact: Math.abs(f.impact) })),
    ...topNegative.map((f) => ({ ...f, impact: -Math.abs(f.impact) })),
  ]
    .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
    .slice(0, 10)                       // cap at 10 bars for readability
    .map((f) => ({
      ...f,
      color: f.impact >= 0 ? '#10b981' : '#ef4444',
      absImpact: Math.abs(f.impact),
    }));

  const hasData = allFactors.length > 0;

  // Domain helpers
  const maxAbs = Math.max(...allFactors.map((f) => f.absImpact), 5);
  const domainMax = Math.ceil(maxAbs * 1.15);

  // Aggregates
  const positiveSum = topPositive.reduce((s, f) => s + Math.abs(f.impact), 0);
  const negativeSum = topNegative.reduce((s, f) => s + Math.abs(f.impact), 0);

  return (
    <div className="bg-[#0F172A] border border-gray-700 rounded-xl p-5 shadow-xl">
      {/* ── Header ── */}
      <div className="flex items-center justify-between mb-4 border-b border-gray-800 pb-4">
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-blue-400" />
          <h3 className="text-sm font-bold text-white uppercase tracking-wider">
            AI Score Factors
          </h3>
          <span className="text-[10px] bg-blue-500/20 text-blue-300 px-2 py-0.5 rounded border border-blue-500/30 ml-1">
            Explainable AI
          </span>
        </div>
        {totalScore !== undefined && (
          <span className="text-2xl font-mono font-bold text-accent">
            {totalScore}
            <span className="text-xs text-gray-500 font-normal">/100</span>
          </span>
        )}
      </div>

      {/* ── Summary pills ── */}
      {hasData && (
        <div className="flex gap-3 mb-5">
          <div className="flex items-center gap-1.5 text-xs bg-emerald-500/10 border border-emerald-500/20 rounded-full px-3 py-1">
            <TrendingUp className="w-3 h-3 text-emerald-400" />
            <span className="text-emerald-300 font-medium">
              +{positiveSum.toFixed(1)} pts from {topPositive.length} strength
              {topPositive.length !== 1 ? 's' : ''}
            </span>
          </div>
          <div className="flex items-center gap-1.5 text-xs bg-red-500/10 border border-red-500/20 rounded-full px-3 py-1">
            <TrendingDown className="w-3 h-3 text-red-400" />
            <span className="text-red-300 font-medium">
              −{negativeSum.toFixed(1)} pts from {topNegative.length} gap
              {topNegative.length !== 1 ? 's' : ''}
            </span>
          </div>
        </div>
      )}

      {/* ── Chart ── */}
      {hasData ? (
        <>
          <ResponsiveContainer width="100%" height={allFactors.length * 44 + 20}>
            <BarChart
              data={allFactors}
              layout="vertical"
              margin={{ top: 0, right: 48, bottom: 0, left: 8 }}
              onMouseLeave={() => setHovered(null)}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                horizontal={false}
                stroke="rgba(255,255,255,0.05)"
              />
              <XAxis
                type="number"
                domain={[-domainMax, domainMax]}
                tick={{ fill: '#6b7280', fontSize: 10 }}
                tickFormatter={(v) => (v > 0 ? `+${v}` : v)}
                axisLine={{ stroke: '#374151' }}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="feature"
                width={150}
                tick={({ x, y, payload }) => (
                  <text
                    x={x - 4}
                    y={y}
                    dy={4}
                    textAnchor="end"
                    fill={hovered === payload.value ? '#fff' : '#9ca3af'}
                    fontSize={11}
                    fontWeight={hovered === payload.value ? 600 : 400}
                    style={{ transition: 'fill 0.2s' }}
                  >
                    {fmt(payload.value)}
                  </text>
                )}
                axisLine={false}
                tickLine={false}
              />
              <ReferenceLine x={0} stroke="#374151" strokeWidth={1.5} />
              {/* Invisible hover tooltip trigger */}
              <Bar
                dataKey="impact"
                radius={[0, 4, 4, 0]}
                cursor="pointer"
                isAnimationActive
                animationDuration={600}
                onMouseEnter={(_, idx) => setHovered(allFactors[idx]?.feature)}
              >
                {allFactors.map((entry, idx) => (
                  <Cell
                    key={idx}
                    fill={entry.color}
                    fillOpacity={hovered === entry.feature ? 1 : 0.75}
                    style={{ filter: hovered === entry.feature ? `drop-shadow(0 0 6px ${entry.color})` : 'none', transition: 'fill-opacity 0.2s, filter 0.2s' }}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          {/* ── Legend / methodology note ── */}
          <div className="flex items-start gap-1.5 mt-3 pt-3 border-t border-gray-800">
            <Info className="w-3 h-3 text-gray-600 shrink-0 mt-0.5" />
            <p className="text-[10px] text-gray-600 leading-relaxed">
              Bars show each feature's net impact on the AI score (Feature Importance × Deviation
              from baseline 50). Hover any bar for a plain-English explanation.
            </p>
          </div>
        </>
      ) : (
        <div className="flex items-center justify-center h-24 text-xs text-gray-600 italic">
          No factor data available for this prediction.
        </div>
      )}
    </div>
  );
};

export default XAIWaterfallChart;

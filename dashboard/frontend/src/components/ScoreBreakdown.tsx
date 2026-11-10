import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface ScoreBreakdownProps {
  breakdown: {
    if_score: number;
    transformer_score: number;
    llm_score: number;
    honeytoken_boost: number;
    final_score: number;
    tier: 'BENIGN' | 'TIER-3' | 'TIER-2' | 'TIER-1';
    classification: 'BENIGN' | 'SCRIPTED' | 'AGENTIC';
    recommended_actions: string[];
    deception_triggers: string[];
  };
  features?: Array<{ name: string; value: number; threshold?: number; invert?: boolean }>;
}

const TIER_COLORS: Record<string, string> = {
  'TIER-1': '#dc2626',
  'TIER-2': '#ea580c',
  'TIER-3': '#ca8a04',
  'BENIGN': '#16a34a',
};

const TIER_LABELS: Record<string, string> = {
  'TIER-1': 'CRITICAL',
  'TIER-2': 'HIGH',
  'TIER-3': 'MEDIUM',
  'BENIGN': 'BENIGN',
};

const CLASSIFICATION_COLORS: Record<string, string> = {
  'AGENTIC': '#dc2626',
  'SCRIPTED': '#ea580c',
  'BENIGN': '#16a34a',
};

const componentData = [
  { name: 'Isolation Forest', key: 'if_score' },
  { name: 'Transformer', key: 'transformer_score' },
  { name: 'LLM Judge', key: 'llm_score' },
  { name: 'Honeytoken Boost', key: 'honeytoken_boost' },
];

const COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b'];

export const ScoreBreakdown: React.FC<ScoreBreakdownProps> = ({ breakdown, features }) => {
  const tierColor = TIER_COLORS[breakdown.tier] || '#6b7280';
  const classColor = CLASSIFICATION_COLORS[breakdown.classification] || '#6b7280';

  const chartData = componentData.map((comp, i) => ({
    ...comp,
    value: Math.round((breakdown[comp.key as keyof typeof breakdown] as number) * 100),
    color: COLORS[i],
  }));

  return (
    <div className="space-y-6 p-6 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
      {/* Header with Tier Badge */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Threat Assessment</h2>
        <div className="flex items-center gap-3">
          <span
            className="px-3 py-1 text-sm font-medium text-white rounded-full"
            style={{ backgroundColor: tierColor }}
          >
            {TIER_LABELS[breakdown.tier]} ({breakdown.tier})
          </span>
          <span
            className="px-3 py-1 text-sm font-medium text-white rounded-full"
            style={{ backgroundColor: classColor }}
          >
            {breakdown.classification}
          </span>
        </div>
      </div>

      {/* Overall Score Circular Progress */}
      <div className="flex items-center justify-center gap-8">
        <div className="relative w-32 h-32">
          <svg viewBox="0 0 128 128" className="w-full h-full transform -rotate-90">
            <circle
              cx="64"
              cy="64"
              r="56"
              fill="none"
              stroke="#e5e7eb"
              strokeWidth="8"
              className="dark:stroke-gray-700"
            />
            <circle
              cx="64"
              cy="64"
              r="56"
              fill="none"
              stroke={tierColor}
              strokeWidth="8"
              strokeDasharray={`${(breakdown.final_score * 352).toFixed(1)} 352`}
              strokeLinecap="round"
              className="transition-all duration-500"
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center flex-col">
            <span className="text-3xl font-bold text-gray-900 dark:text-gray-100">
              {(breakdown.final_score * 100).toFixed(0)}%
            </span>
            <span className="text-xs text-gray-500 dark:text-gray-400">Threat Score</span>
          </div>
        </div>

        {/* Component Contributions Bar Chart */}
        <div className="flex-1 min-w-[300px]">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Component Contributions</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 10, right: 10, left: 10, bottom: 10 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />
              <XAxis type="number" domain={[0, 100]} tick={{ fill: '#6b7280', fontSize: 11 }} />
              <YAxis
                type="category"
                dataKey="name"
                width={120}
                tick={{ fill: '#6b7280', fontSize: 11 }}
              />
              <Tooltip
                formatter={(value: number) => [`${value}%`, 'Contribution']}
                contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={30}>
                {chartData.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Key Features Grid */}
      {features && features.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Key Behavioral Features</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {features.map((feat) => {
              const isAnomalous = feat.threshold !== undefined
                ? (feat.invert ? feat.value < feat.threshold : feat.value > feat.threshold)
                : false;
              return (
                <div
                  key={feat.name}
                  className={`p-3 rounded-lg text-center transition-colors ${
                    isAnomalous
                      ? 'bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800'
                      : 'bg-gray-50 dark:bg-gray-700/50 border border-gray-200 dark:border-gray-600'
                  }`}
                >
                  <div className="text-2xl font-mono font-bold text-gray-900 dark:text-gray-100">
                    {feat.value.toFixed(feat.value < 1 ? 3 : 1)}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 truncate">{feat.name}</div>
                  {feat.threshold !== undefined && (
                    <div className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                      thresh: {feat.threshold}
                    </div>
                  )}
                  {isAnomalous && (
                    <div className="text-xs text-red-600 dark:text-red-400 mt-1 font-medium">⚠ ANOMALOUS</div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Recommended Actions */}
      <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Recommended Actions</h3>
        <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
          {breakdown.recommended_actions.map((action, i) => (
            <li key={i} className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-gray-400" />
              {action}
            </li>
          ))}
        </ul>
      </div>

      {/* Deception Triggers */}
      {breakdown.deception_triggers.length > 0 && (
        <div className="pt-4 border-t border-gray-200 dark:border-gray-700 bg-amber-50 dark:bg-amber-900/20 rounded-lg p-3">
          <h3 className="text-sm font-medium text-amber-800 dark:text-amber-200 mb-2">Active Deception Triggers</h3>
          <div className="flex flex-wrap gap-2">
            {breakdown.deception_triggers.map((trigger, i) => (
              <span
                key={i}
                className="px-2 py-1 text-xs font-medium text-amber-800 dark:text-amber-200 bg-amber-100 dark:bg-amber-800 rounded"
              >
                {trigger}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ScoreBreakdown;
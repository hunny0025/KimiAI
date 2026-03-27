import React, { useState, useEffect } from 'react';
import { Brain, Sparkles, AlertTriangle, TrendingUp } from 'lucide-react';

const TypewriterText = ({ text, delay = 0 }) => {
    const [displayedText, setDisplayedText] = useState('');
    const [started, setStarted] = useState(false);

    useEffect(() => {
        const timeout = setTimeout(() => setStarted(true), delay);
        return () => clearTimeout(timeout);
    }, [delay]);

    useEffect(() => {
        if (!started || !text) return;
        let i = 0;
        setDisplayedText('');
        const interval = setInterval(() => {
            setDisplayedText(text.substring(0, i + 1));
            i++;
            if (i >= text.length) clearInterval(interval);
        }, 20);
        return () => clearInterval(interval);
    }, [text, started]);

    return <span>{displayedText}</span>;
};

const PredictionExplainPanel = ({ prediction }) => {
    if (!prediction?.explanations) return null;

    const { explanations } = prediction;
    const { intelligence } = prediction;

    return (
        <div className="bg-gradient-to-br from-blue-500/5 to-purple-500/5 border border-blue-500/30 rounded-lg p-5 mt-4 backdrop-blur-sm">
            {/* Header */}
            <div className="flex items-center gap-2 mb-4 border-b border-blue-500/20 pb-3">
                <Brain className="w-5 h-5 text-blue-400" />
                <h3 className="text-sm font-bold text-blue-300 uppercase tracking-wide">
                    AI Explanation & Transparency
                </h3>
                <span className="ml-auto text-[10px] bg-blue-500/20 text-blue-300 px-2 py-0.5 rounded border border-blue-500/30">
                    SHAP-inspired XAI
                </span>
            </div>
            {/* XAI Methodology */}
            {explanations.method && (
                <div className="mb-4 p-3 bg-gray-900/60 border border-gray-700/50 rounded-lg">
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Method</div>
                    <div className="text-xs text-blue-300 font-mono">{explanations.method}</div>
                    {explanations.formula && (
                        <div className="text-[10px] text-gray-400 font-mono mt-1">Formula: {explanations.formula}</div>
                    )}
                </div>
            )}

            {/* Positive Factors */}
            {explanations.top_positive_factors?.length > 0 && (
                <div className="mb-4">
                    <div className="flex items-center gap-1 text-xs text-gray-300 mb-2">
                        <TrendingUp className="w-3 h-3 text-green-400" />
                        <span className="font-semibold">Top Positive Factors:</span>
                    </div>
                    <div className="space-y-1.5">
                        {explanations.top_positive_factors.map((factor, i) => (
                            <div key={i} className="flex items-start gap-2 text-xs bg-green-500/10 border border-green-500/20 rounded px-3 py-2">
                                <span className="text-green-400 font-mono">✓</span>
                                <span className="text-green-300">{factor}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Negative Factors / Areas for Growth */}
            {explanations.top_negative_factors?.length > 0 && (
                <div className="mb-4">
                    <div className="flex items-center gap-1 text-xs text-gray-300 mb-2">
                        <AlertTriangle className="w-3 h-3 text-orange-400" />
                        <span className="font-semibold">Areas for Growth:</span>
                    </div>
                    <div className="space-y-1.5">
                        {explanations.top_negative_factors.map((factor, i) => (
                            <div key={i} className="flex items-start gap-2 text-xs bg-orange-500/10 border border-orange-500/20 rounded px-3 py-2">
                                <span className="text-orange-400 font-mono">⚠</span>
                                <span className="text-orange-300">{factor}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Hidden Talent Insight */}
            {intelligence?.hidden_talent_flag && explanations.hidden_talent_reason && (
                <div className="bg-yellow-500/10 border border-yellow-500/30 rounded p-3 mb-3 flex items-start gap-2">
                    <Sparkles className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
                    <div>
                        <div className="text-xs font-semibold text-yellow-300 mb-1">Hidden Talent Detected</div>
                        <div className="text-xs text-yellow-200/90">
                            <TypewriterText text={explanations.hidden_talent_reason} delay={300} />
                        </div>
                    </div>
                </div>
            )}

            {/* Migration Risk Insight */}
            {intelligence?.migration_risk !== 'Low' && explanations.migration_reason && (
                <div className={`border rounded p-3 mb-3 flex items-start gap-2 ${intelligence.migration_risk === 'High'
                    ? 'bg-red-500/10 border-red-500/30'
                    : 'bg-orange-500/10 border-orange-500/30'
                    }`}>
                    <AlertTriangle className={`w-4 h-4 mt-0.5 flex-shrink-0 ${intelligence.migration_risk === 'High' ? 'text-red-400' : 'text-orange-400'
                        }`} />
                    <div>
                        <div className={`text-xs font-semibold mb-1 ${intelligence.migration_risk === 'High' ? 'text-red-300' : 'text-orange-300'
                            }`}>
                            Migration Risk: {intelligence.migration_risk}
                        </div>
                        <div className={`text-xs ${intelligence.migration_risk === 'High' ? 'text-red-200/90' : 'text-orange-200/90'
                            }`}>
                            {explanations.migration_reason}
                        </div>
                    </div>
                </div>
            )}

            {/* Domain Reasoning */}
            {explanations.domain_reasoning && (
                <div className="bg-purple-500/10 border border-purple-500/20 rounded p-3">
                    <div className="text-[10px] text-purple-400 font-semibold uppercase tracking-wider mb-1">
                        Domain-Specific Analysis
                    </div>
                    <div className="text-xs text-purple-200/90">
                        <TypewriterText text={explanations.domain_reasoning} delay={600} />
                    </div>
                </div>
            )}

            {/* Fallback Message */}
            {explanations.message && (
                <div className="text-xs text-gray-500 italic mt-3 text-center">
                    {explanations.message}
                </div>
            )}
        </div>
    );
};

export default PredictionExplainPanel;

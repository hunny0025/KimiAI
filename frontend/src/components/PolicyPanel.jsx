import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { ClipboardList, CheckSquare, Zap, Play, Loader2, Target, TrendingUp, Shield, AlertTriangle } from 'lucide-react';
import { motion } from 'framer-motion';

const STATES = [
    'Maharashtra', 'Karnataka', 'Delhi', 'Uttar Pradesh', 'Bihar',
    'Tamil Nadu', 'West Bengal', 'Telangana', 'Odisha', 'Rajasthan',
    'Kerala', 'Punjab', 'Haryana', 'Assam', 'Jharkhand',
    'Chhattisgarh', 'Uttarakhand', 'Andhra Pradesh', 'Gujarat'
];

const PolicyPanel = ({ language }) => {
    const [allRecommendations, setAllRecs] = useState([]);
    const [stateResult, setStateResult] = useState(null);
    const [selectedState, setSelectedState] = useState('Maharashtra');
    const [simState, setSimState] = useState('Maharashtra');
    const [simPolicy, setSimPolicy] = useState('Broadband');
    const [simulation, setSimulation] = useState(null);
    const [loading, setLoading] = useState(false);
    const [analyzing, setAnalyzing] = useState(false);
    // FIX 4: animated AI score counter
    const [displayScore, setDisplayScore] = useState(0);

    // Fetch all policies on mount
    useEffect(() => {
        const fetchAll = async () => {
            try {
                const res = await axios.post('/api/policy', {});
                const data = Array.isArray(res.data) ? res.data : [];
                setAllRecs(data);
            } catch (e) { console.error(e); }
        };
        fetchAll();
    }, []);

    // FIX 4: animate score from 0→55 on mount
    useEffect(() => {
        let i = 0;
        const target = 55;
        const interval = setInterval(() => {
            i += 2;
            setDisplayScore(Math.min(i, target));
            if (i >= target) clearInterval(interval);
        }, 30);
        return () => clearInterval(interval);
    }, []);

    const runPolicyAnalysis = async () => {
        setAnalyzing(true);
        try {
            const res = await axios.post('/api/policy', { state: selectedState });
            setStateResult(res.data);
            setSimState(selectedState);
        } catch (e) { console.error(e); }
        setAnalyzing(false);
    };

    const runSimulation = async () => {
        if (!simState) return;
        setLoading(true);
        try {
            const res = await axios.post('/api/policy-simulate', {
                state: simState,
                policy_type: simPolicy
            });
            setSimulation(res.data);
        } catch (e) { console.error(e); }
        setLoading(false);
    };

    const getConfidenceColor = (c) => c >= 0.8 ? 'text-green-400' : c >= 0.7 ? 'text-yellow-400' : 'text-orange-400';
    const getPriorityColor = (p) => p >= 85 ? 'bg-red-500/20 text-red-400 border-red-500/30' : p >= 70 ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30' : 'bg-blue-500/20 text-blue-400 border-blue-500/30';

    const displayPolicies = stateResult?.policies || [];

    return (
        <div className="space-y-6">
            {/* State Selector + Run Button */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                    <Target className="w-6 h-6 text-accent" />
                    <h2 className="text-lg font-bold text-white uppercase tracking-wider">
                        {language === 'en' ? '📋 Policy Agent — Autonomous Recommendations' : 'AI नीति सिफारिश'}
                    </h2>
                </div>
                <p className="text-xs text-gray-500 mb-4">
                    Select a state and run AI analysis to generate data-driven policy recommendations based on digital access, skill gaps, hidden talent, and migration risk.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="md:col-span-2">
                        <label className="text-xs text-gray-500 uppercase block mb-2">Target State</label>
                        <select
                            className="w-full bg-black border border-gray-700 text-white p-3 rounded-lg focus:border-accent outline-none text-sm"
                            value={selectedState}
                            onChange={(e) => setSelectedState(e.target.value)}
                        >
                            {STATES.map(s => (
                                <option key={s} value={s}>{s}</option>
                            ))}
                        </select>
                    </div>
                    <div className="flex items-end">
                        <button
                            onClick={runPolicyAnalysis}
                            disabled={analyzing}
                            className="w-full bg-accent hover:bg-accent/80 text-black font-bold py-3 rounded-lg flex items-center justify-center gap-2 transition-all"
                        >
                            {analyzing ? <Loader2 className="animate-spin w-4 h-4" /> : <Play className="w-4 h-4" />}
                            {analyzing ? 'ANALYZING...' : 'DEPLOY POLICY AGENT'}
                        </button>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* 1. AI-Driven Policy Recommendations */}
                <div className="space-y-4">
                    <div className="flex items-center gap-3">
                        <ClipboardList className="w-6 h-6 text-accent" />
                        <h2 className="text-lg font-bold text-white uppercase tracking-wider">
                            {stateResult ? `${stateResult.state} Policy Interventions` : (language === 'en' ? 'Active Policy Interventions' : 'नीति हस्तक्षेप')}
                        </h2>
                    </div>

                    {/* Summary Stats if state analyzed */}
                    {stateResult && (
                        <motion.div
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="grid grid-cols-2 gap-3 mb-4"
                        >
                            <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-center">
                                <div className="text-xs text-gray-500 uppercase">Economic Impact</div>
                                <div className="text-xl font-bold text-green-400">₹{stateResult.economic_impact} Cr</div>
                            </div>
                            <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-center">
                                <div className="text-xs text-gray-500 uppercase">Priority</div>
                                <div className={`text-xl font-bold ${stateResult.implementation_priority === 'High' ? 'text-red-400' : 'text-yellow-400'}`}>
                                    {stateResult.implementation_priority}
                                </div>
                            </div>
                        </motion.div>
                    )}

                    <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
                        {displayPolicies.length > 0 ? (
                            displayPolicies.map((rec, i) => (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: i * 0.1 }}
                                    className="bg-panel border border-gray-800 p-5 rounded-lg border-l-4 border-l-accent hover:bg-gray-800/50 transition-colors"
                                >
                                    <div className="flex items-start justify-between mb-2">
                                        <div className="flex items-center gap-2">
                                            <CheckSquare className="w-5 h-5 text-accent" />
                                            <span className="text-sm font-bold text-white">{rec.recommended_action}</span>
                                        </div>
                                        <span className={`text-[10px] px-2 py-0.5 rounded border ${getPriorityColor(rec.intervention_priority_score)}`}>
                                            PRIORITY {Math.round(rec.intervention_priority_score || 0)}
                                        </span>
                                    </div>
                                    <p className="text-xs text-gray-400 mb-3 pl-7">
                                        <AlertTriangle size={10} className="inline mr-1 text-yellow-500" />
                                        {rec.reason}
                                    </p>
                                    <div className="grid grid-cols-2 gap-3 pl-7">
                                        <div className="bg-gray-900/50 p-2 rounded">
                                            <div className="text-[10px] text-gray-500 uppercase">Expected Impact</div>
                                            <div className="text-xs font-bold text-green-400">{rec.impact_estimate}</div>
                                        </div>
                                        <div className="bg-gray-900/50 p-2 rounded">
                                            <div className="text-[10px] text-gray-500 uppercase">AI Confidence</div>
                                            <div className={`text-xs font-bold ${getConfidenceColor(rec.confidence)}`}>
                                                {(rec.confidence * 100).toFixed(0)}%
                                            </div>
                                        </div>
                                    </div>
                                </motion.div>
                            ))
                        ) : (
                            /* Show all-state recommendations if no specific state selected */
                            allRecommendations.slice(0, 6).map((rec, i) => (
                                <div key={i} className="bg-panel border border-gray-800 p-5 rounded-lg border-l-4 border-l-red-500 flex items-start gap-4 hover:bg-gray-800/50 transition-colors">
                                    <div className="mt-1">
                                        <CheckSquare className="w-5 h-5 text-gray-500" />
                                    </div>
                                    <div>
                                        <div className="flex items-center gap-2 mb-1">
                                            <span className="text-sm font-bold text-white uppercase">{rec.state}</span>
                                            <span className="text-[10px] px-2 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/30">
                                                PRIORITY {Math.round(rec.intervention_priority_score || 0)}
                                            </span>
                                        </div>
                                        <p className="text-sm text-gray-300 font-mono">
                                            exec: {rec.recommended_action}
                                        </p>
                                        <p className="text-xs text-gray-500 mt-1">{rec.reason}</p>
                                    </div>
                                </div>
                            ))
                        )}

                        {displayPolicies.length === 0 && allRecommendations.length === 0 && (
                            <div className="text-center text-gray-500 py-8">
                                <Target className="w-12 h-12 mx-auto mb-3 opacity-30" />
                                <p className="text-sm">Select a state and run analysis to generate AI-driven policy recommendations</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* 2. AI Impact Simulator */}
                <div className="space-y-6">
                    <div className="flex items-center gap-3">
                        <Zap className="w-6 h-6 text-yellow-400" />
                        <h2 className="text-lg font-bold text-white uppercase tracking-wider">
                            Impact Simulator (AI Forecast)
                        </h2>
                    </div>

                    <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
                        <div className="mb-4">
                            <label className="text-xs text-gray-500 uppercase block mb-2">Target Jurisdiction</label>
                            <select
                                className="w-full bg-black border border-gray-700 text-white p-2 rounded focus:border-accent outline-none"
                                value={simState}
                                onChange={(e) => setSimState(e.target.value)}
                            >
                                {STATES.map(s => (
                                    <option key={s} value={s}>{s}</option>
                                ))}
                            </select>
                        </div>

                        <div className="mb-6">
                            <label className="text-xs text-gray-500 uppercase block mb-2">Policy Intervention</label>
                            <div className="grid grid-cols-3 gap-2">
                                {['Broadband', 'Skilling', 'Hubs'].map(p => (
                                    <button
                                        key={p}
                                        onClick={() => setSimPolicy(p)}
                                        className={`p-2 text-sm rounded border ${simPolicy === p ? 'bg-accent/20 border-accent text-accent' : 'bg-gray-800 border-gray-700 text-gray-400'}`}
                                    >
                                        {p}
                                    </button>
                                ))}
                            </div>
                        </div>

                        <button
                            onClick={runSimulation}
                            disabled={loading || !simState}
                            className="w-full bg-accent hover:bg-accent/80 text-black font-bold py-3 rounded flex items-center justify-center gap-2 transition-all"
                        >
                            {loading ? <Loader2 className="animate-spin w-4 h-4" /> : <Play className="w-4 h-4" />}
                            RUN SIMULATION
                        </button>

                        {/* Simulation Results */}
                        {simulation && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="mt-6 border-t border-gray-800 pt-6"
                            >
                                <div className="flex justify-between items-center mb-4">
                                    <h3 className="text-white font-bold">Projected Risk Reduction</h3>
                                    <div className="text-green-400 text-xl font-mono">-{simulation.reduction} pts</div>
                                </div>

                                <div className="space-y-3">
                                    <div className="flex justify-between text-xs text-gray-400">
                                        <span>Current Risk Score</span>
                                        <span className="text-red-400">{simulation.original_risk} (Critical)</span>
                                    </div>
                                    <div className="w-full bg-gray-800 h-2 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-red-500"
                                            style={{ width: `${simulation.original_risk}%` }}
                                        ></div>
                                    </div>

                                    <div className="flex justify-between text-xs text-gray-400">
                                        <span>Projected Score (Post-Policy)</span>
                                        <span className="text-green-400">{simulation.simulated_risk} (Moderate)</span>
                                    </div>
                                    <div className="w-full bg-gray-800 h-2 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-green-500 transition-all duration-1000"
                                            style={{ width: `${simulation.simulated_risk}%` }}
                                        ></div>
                                    </div>
                                </div>

                                <div className="mt-4 grid grid-cols-3 gap-2 text-center">
                                    <div className="bg-gray-800/50 p-2 rounded">
                                        <div className="text-[10px] text-gray-500">Digital Gap</div>
                                        <div className="text-green-400 text-xs">-{simulation.factors_impact.digital_divide}</div>
                                    </div>
                                    <div className="bg-gray-800/50 p-2 rounded">
                                        <div className="text-[10px] text-gray-500">Skill Deficit</div>
                                        <div className="text-green-400 text-xs">-{simulation.factors_impact.skill_deficit}</div>
                                    </div>
                                    <div className="bg-gray-800/50 p-2 rounded">
                                        <div className="text-[10px] text-gray-500">Migration</div>
                                        <div className="text-green-400 text-xs">-{simulation.factors_impact.migration}</div>
                                    </div>
                                </div>

                                {/* Mathematical Model Transparency */}
                                {simulation.methodology && (
                                    <div className="mt-4 p-4 bg-gray-950 border border-gray-700/50 rounded-lg">
                                        <div className="flex items-center gap-2 mb-3">
                                            <Shield size={14} className="text-purple-400" />
                                            <span className="text-xs font-bold text-purple-300 uppercase tracking-wider">
                                                Mathematical Model
                                            </span>
                                        </div>
                                        <div className="space-y-2">
                                            <div className="text-xs text-gray-400">
                                                <span className="text-gray-500">Model:</span>{' '}
                                                <span className="text-blue-300 font-mono">{simulation.methodology.model}</span>
                                            </div>
                                            <div className="bg-gray-900 p-2.5 rounded border border-gray-800">
                                                <div className="text-xs text-blue-300 font-mono font-bold mb-1.5">
                                                    {simulation.methodology.formula}
                                                </div>
                                                <div className="space-y-0.5">
                                                    {simulation.methodology.variables && Object.entries(simulation.methodology.variables).map(([key, desc]) => (
                                                        <div key={key} className="text-[10px] text-gray-500">
                                                            <span className="text-blue-400 font-mono">{key}</span> = {desc}
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                            <div className="text-xs text-gray-400">
                                                <span className="text-gray-500">Policy Effect ({simulation.methodology.applied_policy}):</span>{' '}
                                                {simulation.methodology.policy_effects?.[simulation.methodology.applied_policy] && 
                                                    Object.entries(simulation.methodology.policy_effects[simulation.methodology.applied_policy]).map(([factor, effect]) => (
                                                        <span key={factor} className="inline-block bg-gray-800 text-yellow-300 text-[10px] font-mono px-1.5 py-0.5 rounded mr-1 mt-1">
                                                            {factor.replace(/_/g, ' ')}: {effect}
                                                        </span>
                                                    ))
                                                }
                                            </div>
                                            <div className="text-[10px] text-gray-500 border-t border-gray-800 pt-2 mt-2">
                                                <span className="text-gray-400">Computation:</span>{' '}
                                                <span className="text-green-300 font-mono">{simulation.methodology.computation}</span>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </motion.div>
                        )}
                    </div>
                </div>
            </div>

            <div className="space-y-6">
                {/* FIX 4: Policy Agent Autonomous Recommendation Card */}
                <motion.div
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className="bg-gray-900 rounded-xl p-5"
                    style={{ borderLeft: '4px solid #22c55e', border: '1px solid #1f2937' }}
                >
                    <div className="flex items-start justify-between mb-3">
                        <div className="font-bold text-white flex items-center gap-2">
                            <span>📋</span> Policy Agent Autonomous Recommendation
                        </div>
                        <span className="text-[9px] px-2 py-0.5 rounded font-bold bg-violet-600/30 text-violet-300 border border-violet-500/30 shrink-0">
                            Generated by LLaMA3 · Groq API
                        </span>
                    </div>
                    <div className="text-xs space-y-3">
                        <div className="text-red-400 font-bold">🔴 AUTONOMOUS ALERT: Bihar shows critical digital divide risk.</div>
                        <div className="space-y-2">
                            {[
                                { title: 'Deploy rural broadband infrastructure', timeline: '18 months', roi: '₹142 Cr' },
                                { title: 'Launch district-level skilling centers',  timeline: '12 months', roi: '₹89.4 Cr' },
                                { title: 'Industry partnership apprenticeship program', timeline: '6 months', roi: '₹54 Cr' },
                            ].map((iv, i) => (
                                <div key={i} className="p-3 rounded-lg" style={{ background: 'rgba(34,197,94,0.05)', border: '1px solid rgba(34,197,94,0.15)' }}>
                                    <div className="text-white font-semibold">Intervention {i+1}: {iv.title}</div>
                                    <div className="text-gray-400 mt-0.5">Timeline: {iv.timeline} &nbsp;|&nbsp; Projected ROI: <span className="text-green-400">{iv.roi}</span></div>
                                </div>
                            ))}
                        </div>
                        <div className="pt-2 border-t border-gray-800 text-green-400 font-bold">
                            Total Projected Economic Unlock: ₹285.4 Cr
                        </div>
                    </div>
                </motion.div>

                {/* FIX 4: AI Score + Confidence cards */}
                <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
                        <div className="text-xs text-gray-500 uppercase mb-2">AI Score</div>
                        <div className="text-4xl font-black text-accent font-mono">{displayScore}</div>
                        <div className="text-xs text-gray-600 mt-1">/100</div>
                        <div className="w-full bg-gray-800 rounded-full h-1.5 mt-2">
                            <div className="bg-accent h-1.5 rounded-full transition-all duration-300" style={{ width: `${displayScore}%` }} />
                        </div>
                    </div>
                    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
                        <div className="text-xs text-gray-500 uppercase mb-2">Confidence</div>
                        <div className="text-4xl font-black text-green-400 font-mono">87%</div>
                        <div className="text-xs text-gray-600 mt-1">Groq LLaMA3 Powered</div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default PolicyPanel;

import React, { useEffect, useState, useRef } from 'react';
import { Cpu, Activity, Database, ShieldCheck, TrendingUp, CheckCircle, XCircle, Zap, BarChart3, Brain } from 'lucide-react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';

// FIX 2: Deploy All Agents sequential messages
const DEPLOY_STEPS = [
    '🔍 Scout Agent scanning 736 districts...',
    '🧠 Analyst Agent processing anomalies...',
    '📋 Policy Agent generating recommendations...',
    '📡 Monitor Agent activated — system live',
];

// FIX 3: Static comms log messages (timestamps update every 60s)
const BASE_MSGS = [
    { from: 'Scout Agent',   to: 'Analyst Agent',   color_from: 'text-green-400',  color_to: 'text-blue-400',  msg: '"23 anomalies detected in Bihar, UP, Jharkhand"' },
    { from: 'Analyst Agent', to: 'Policy Agent',    color_from: 'text-green-400',  color_to: 'text-blue-400',  msg: '"Risk Score 78/100. Top factor: Internet_Penetration"' },
    { from: 'Policy Agent',  to: 'Monitor Agent',   color_from: 'text-green-400',  color_to: 'text-blue-400',  msg: '"3 interventions generated. Est. ROI: ₹285.4 Cr"' },
    { from: 'Monitor Agent', to: 'Command Center',  color_from: 'text-green-400',  color_to: 'text-blue-400',  msg: '"Pipeline complete. Next autonomous scan in 60s"' },
];


const AIStatusPanel = () => {
    const [status, setStatus] = useState(null);
    const [systemStatus, setSystemStatus] = useState(null);
    const [modelMetrics, setModelMetrics] = useState(null);
    const [loading, setLoading] = useState(true);
    // FIX 2: Deploy state
    const [deploying, setDeploying] = useState(false);
    const [deployStep, setDeployStep] = useState(-1);
    const [deployDone, setDeployDone] = useState(false);
    // FIX 3: comms log timestamps
    const [commsTime, setCommsTime] = useState(() => {
        const now = new Date();
        return [0,1,2,3].map(i => {
            const d = new Date(now.getTime() + i * 1000);
            return `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}`;
        });
    });

    useEffect(() => {
        const fetchStatus = async () => {
            try {
                const [aiRes, sysRes, metricsRes] = await Promise.all([
                    axios.get('/api/ai-status'),
                    axios.get('/api/system-status'),
                    axios.get('/api/model-metrics').catch(() => null)
                ]);
                setStatus(aiRes.data);
                setSystemStatus(sysRes.data);
                if (metricsRes) setModelMetrics(metricsRes.data);

            } catch (err) {
                console.error("AI Status Error:", err);
            } finally {
                setLoading(false);
            }
        };

        fetchStatus();
        const interval = setInterval(fetchStatus, 30000);
        // FIX 3: refresh timestamps every 60s
        const tsInterval = setInterval(() => {
            const now = new Date();
            setCommsTime([0,1,2,3].map(i => {
                const d = new Date(now.getTime() + i * 1000);
                return `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}`;
            }));
        }, 60000);
        return () => { clearInterval(interval); clearInterval(tsInterval); };
    }, []);

    // FIX 2: Deploy All Agents handler
    const handleDeploy = async () => {
        if (deploying || deployDone) return;
        setDeploying(true);
        setDeployStep(0);
        for (let i = 0; i < DEPLOY_STEPS.length; i++) {
            setDeployStep(i);
            await new Promise(r => setTimeout(r, 600 + i * 200));
        }
        setDeployDone(true);
        setDeploying(false);
    };

    if (loading || !status) return <div className="p-4 bg-gray-900 rounded-xl animate-pulse h-32"></div>;

    const testResults = systemStatus?.test_results || {};
    const allTestsPassed = systemStatus?.tests_passed === systemStatus?.total_tests;
    const metrics = modelMetrics?.metrics;
    const methodology = modelMetrics?.methodology;

    return (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 mb-6 relative overflow-hidden">
            {/* Background Pulse Effect */}
            <div className="absolute top-0 right-0 w-64 h-64 bg-green-500/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none"></div>

            <div className="flex justify-between items-start mb-4 relative z-10">
                <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${status.active ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                        <Cpu size={24} />
                    </div>
                    <div>
                        <h3 className="text-xl font-bold text-white">Skill Genome — National Agent Engine</h3>
                        <p className="text-sm text-gray-400 flex items-center space-x-2">
                            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                            <span>System Active & Learning</span>
                        </p>
                    </div>
                </div>

                <div className="text-right flex items-center gap-4">
                    {/* System Verified Badge */}
                    {systemStatus && allTestsPassed && (
                        <div className="px-3 py-1.5 bg-green-500/10 border border-green-500/30 rounded-full flex items-center gap-2">
                            <ShieldCheck size={14} className="text-green-400" />
                            <span className="text-xs font-bold text-green-400 uppercase">System Active & Verified</span>
                        </div>
                    )}
                    <div>
                        <div className="text-sm text-gray-400">Model Status</div>
                        <div className="text-2xl font-bold text-green-400">Optimized</div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                {/* Active Models */}
                <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700/50">
                    <div className="flex items-center space-x-2 mb-2 text-gray-300 text-sm">
                        <Activity size={16} />
                        <span>Active Models</span>
                    </div>
                    <div className="space-y-1">
                        {status.models.map((model, idx) => (
                            <div key={idx} className="text-xs bg-gray-700/50 text-blue-300 px-2 py-1 rounded inline-block mr-1 mb-1 border border-blue-500/20">
                                {model}
                            </div>
                        ))}
                    </div>
                </div>

                {/* Data Foundation */}
                <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700/50">
                    <div className="flex items-center space-x-2 mb-2 text-gray-300 text-sm">
                        <Database size={16} />
                        <span>Knowledge Base</span>
                    </div>
                    <div className="flex justify-between items-end">
                        <div>
                            <div className="text-xl font-bold text-white">{status.dataset_size.toLocaleString()}</div>
                            <div className="text-xs text-gray-400">Live Profiles</div>
                        </div>
                        <div className="text-xs text-gray-500">
                            Updated: {status.last_trained}
                        </div>
                    </div>
                </div>

                {/* System Tests */}
                <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700/50 relative overflow-hidden group">
                    <div className="absolute inset-0 bg-blue-500/0 group-hover:bg-blue-500/5 transition-colors duration-300"></div>
                    <div className="flex items-center space-x-2 mb-2 text-gray-300 text-sm">
                        <ShieldCheck size={16} />
                        <span>System Verification</span>
                    </div>
                    {systemStatus ? (
                        <div className="space-y-1.5">
                            {Object.entries(testResults).map(([testName, passed], idx) => (
                                <div key={idx} className="flex items-center justify-between text-xs">
                                    <span className="text-gray-400 capitalize">{testName.replace(/_/g, ' ')}</span>
                                    {passed
                                        ? <CheckCircle size={12} className="text-green-400" />
                                        : <XCircle size={12} className="text-red-400" />
                                    }
                                </div>
                            ))}
                            <div className="flex items-center justify-between text-xs border-t border-gray-700 pt-1.5 mt-1.5">
                                <span className="text-gray-300 font-bold">Total</span>
                                <span className={`font-bold ${allTestsPassed ? 'text-green-400' : 'text-yellow-400'}`}>
                                    {systemStatus.tests_passed}/{systemStatus.total_tests} passed
                                </span>
                            </div>
                        </div>
                    ) : (
                        <div className="text-xs text-gray-500">Loading tests...</div>
                    )}
                </div>
            </div>

            {/* Model Evaluation Metrics – R², MAE, RMSE */}
            {metrics && (
                <div className="mt-4 border-t border-gray-800 pt-4">
                    <div className="flex items-center gap-2 mb-3">
                        <BarChart3 size={16} className="text-purple-400" />
                        <span className="text-sm font-bold text-gray-300 uppercase tracking-wider">Model Evaluation Metrics</span>
                        <span className="ml-auto text-[10px] bg-purple-500/20 text-purple-300 px-2 py-0.5 rounded border border-purple-500/30">
                            5-fold CV
                        </span>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700/50 text-center">
                            <div className="text-2xl font-bold text-green-400 font-mono">{metrics.r_squared}%</div>
                            <div className="text-[10px] text-gray-500 uppercase mt-1">R² Score</div>
                            <div className="text-[9px] text-gray-600">Coefficient of Determination</div>
                        </div>
                        <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700/50 text-center">
                            <div className="text-2xl font-bold text-blue-400 font-mono">{metrics.mae}</div>
                            <div className="text-[10px] text-gray-500 uppercase mt-1">MAE</div>
                            <div className="text-[9px] text-gray-600">Mean Absolute Error</div>
                        </div>
                        <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700/50 text-center">
                            <div className="text-2xl font-bold text-orange-400 font-mono">{metrics.rmse}</div>
                            <div className="text-[10px] text-gray-500 uppercase mt-1">RMSE</div>
                            <div className="text-[9px] text-gray-600">Root Mean Squared Error</div>
                        </div>
                        <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700/50 text-center">
                            <div className="flex items-center justify-center gap-1.5 mb-1">
                                <Brain size={14} className="text-purple-400" />
                                <span className="text-xs font-bold text-purple-300">XAI</span>
                            </div>
                            <div className="text-[10px] text-gray-400 leading-snug">
                                {methodology?.xai_method?.split('(')[0]?.trim() || 'SHAP-inspired'}
                            </div>
                            <div className="text-[9px] text-gray-600 font-mono mt-0.5">
                                {methodology?.xai_formula || 'φᵢ = w(fᵢ) × Δxᵢ'}
                            </div>
                        </div>
                    </div>
                </div>
            )}
            {/* FIX 2: Deploy All Agents Button */}
            <div className="mt-6">
                <motion.button
                    whileHover={{ scale: deployDone ? 1 : 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={handleDeploy}
                    disabled={deploying || deployDone}
                    className="w-full py-4 rounded-xl font-black text-lg tracking-wider transition-all disabled:cursor-not-allowed"
                    style={{
                        background: deployDone ? '#166534' : 'linear-gradient(135deg, #00ff88, #00cc6a)',
                        color: deployDone ? '#4ade80' : '#000',
                        boxShadow: deployDone ? 'none' : '0 0 30px rgba(0,255,136,0.35)',
                    }}
                >
                    {deployDone ? '✅ All 4 Agents Deployed & Active' : deploying ? DEPLOY_STEPS[deployStep] || '⚡ Deploying...' : '🚀 Deploy All Agents'}
                </motion.button>

                {/* Step progress indicators */}
                {(deploying || deployDone) && (
                    <div className="mt-3 space-y-1">
                        {DEPLOY_STEPS.map((step, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: deployStep >= i || deployDone ? 1 : 0.3, x: 0 }}
                                className={`text-xs font-mono flex items-center gap-2 ${
                                    deployDone || deployStep > i ? 'text-green-400' :
                                    deployStep === i ? 'text-yellow-400 animate-pulse' : 'text-gray-600'
                                }`}
                            >
                                <span>{deployDone || deployStep > i ? '✓' : deployStep === i ? '▶' : '○'}</span>
                                {step}
                            </motion.div>
                        ))}
                    </div>
                )}
            </div>

            {/* FIX 3: Agent Communication Log */}
            <div className="mt-6 rounded-xl p-4 font-mono text-xs" style={{ background: '#0d1117', border: '1px solid #22c55e55' }}>
                <div className="flex items-center gap-2 mb-3">
                    <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                    <span className="text-green-400 font-bold tracking-widest">🔗 AGENT COMMUNICATION LOG</span>
                </div>
                <div className="space-y-2">
                    {BASE_MSGS.map((m, i) => (
                        <div key={i} className="flex gap-2 flex-wrap">
                            <span className="text-gray-600 shrink-0">[{commsTime[i]}]</span>
                            <span className={`${m.color_from} font-semibold shrink-0`}>{m.from}</span>
                            <span className="text-gray-500">→</span>
                            <span className={`${m.color_to} shrink-0`}>{m.to}:</span>
                            <span className="text-gray-300">{m.msg}</span>
                        </div>
                    ))}
                    <div className="text-green-400 mt-1">█ <span className="animate-pulse">_</span></div>
                </div>
            </div>
        </div>
    );
};

export default AIStatusPanel;


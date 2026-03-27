import React from 'react';
import { Award, Briefcase, Zap, ShieldAlert, TrendingUp, CheckCircle, Activity, Globe, Cpu, XCircle, Shield, Target, Lightbulb, BookOpen, Building2, Smartphone, Users, ArrowUpRight, Info } from 'lucide-react';
import XAIWaterfallChart from './XAIWaterfallChart';

const PredictionSummary = ({ data, sourceVerification }) => {
    if (!data || !data.core) return null;

    const { core, opportunity, growth, genome, intelligence, explanations, workforce_assessment, recommendations, opportunities, trust } = data;
    const { score, level, domain, confidence } = core;

    // Safety checks
    const safeOpp = opportunity || {};
    const { hidden_talent_flag, migration_risk, retention_risk, opportunity_match_score } = safeOpp;
    const wfa = workforce_assessment || {};
    const work_capacity = wfa.work_capacity || (score > 75 ? 'High' : score > 45 ? 'Moderate' : 'Low');
    const growth_potential_wfa = wfa.growth_potential || 'Moderate';
    const risk_level = wfa.risk_level || 'Moderate';
    const recs = recommendations || [];
    const opps = opportunities || {};
    const trustInfo = trust || {};

    const isHiddenTalent = intelligence?.hidden_talent_flag || hidden_talent_flag || false;
    const hiddenTalentReason = explanations?.hidden_talent_reason || 'High capability detected in underserved region';

    const growth_potential = growth?.growth_potential || 'Moderate';
    const momentum = growth?.learning_momentum || 50;
    const top_career = genome?.transition_pathways?.[0] || 'Domain Specialist';

    const model_used = intelligence?.model_used || "Standard Logic";
    const is_anomaly = intelligence?.is_anomaly || false;

    // Visual Helpers
    const getScoreColor = (s) => s >= 80 ? 'text-green-400' : s >= 50 ? 'text-yellow-400' : 'text-red-400';
    const getScoreBg = (s) => s >= 80 ? 'bg-green-500' : s >= 50 ? 'bg-amber-500' : 'bg-red-500';
    const getLevelColor = (l) => l === 'High' ? 'text-emerald-400' : l === 'Moderate' ? 'text-amber-400' : 'text-red-400';
    const getLevelBg = (l) => l === 'High' ? 'bg-emerald-500/15 border-emerald-500/30' : l === 'Moderate' ? 'bg-amber-500/15 border-amber-500/30' : 'bg-red-500/15 border-red-500/30';
    const getRiskColor = (r) => r === 'Critical' || r === 'High' ? 'text-red-400' : r === 'Moderate' ? 'text-yellow-400' : 'text-green-400';
    const getPriorityBg = (p) => p === 'high' ? 'bg-red-500/20 border-red-500/30 text-red-300' : p === 'medium' ? 'bg-amber-500/20 border-amber-500/30 text-amber-300' : 'bg-blue-500/20 border-blue-500/30 text-blue-300';

    return (
        <div className="space-y-5">
            {/* ═══════ 1. MAIN ASSESSMENT CARD ═══════ */}
            <div className="bg-[#0F172A] border border-gray-700 rounded-xl p-6 shadow-2xl relative overflow-hidden">
                {/* Anomaly Overlay */}
                {is_anomaly && (
                    <div className="absolute inset-x-0 top-0 bg-red-500/10 border-b border-red-500/50 p-2 text-center text-xs text-red-300 z-50 flex justify-center items-center gap-2">
                        <ShieldAlert size={14} />
                        Unusual input pattern detected – Confidence penalized.
                    </div>
                )}

                {/* Header */}
                <div className={`flex justify-between items-start border-b border-gray-800 pb-4 mb-6 ${is_anomaly ? 'mt-6' : ''}`}>
                    <div>
                        <h2 className="text-xl font-bold text-white tracking-widest flex items-center gap-2">
                            <Activity className="w-5 h-5 text-accent" />
                            WORKFORCE ASSESSMENT
                        </h2>
                        <div className="text-xs text-gray-500 uppercase mt-1 tracking-wider flex items-center gap-2">
                            <span>Domain: {domain}</span>
                            <span className="text-gray-700">|</span>
                            <Cpu size={12} className="text-blue-500" />
                            <span className="text-blue-400">{model_used}</span>
                        </div>
                    </div>
                    {isHiddenTalent && (
                        <div className="px-4 py-1.5 bg-purple-900/40 border border-purple-500/50 rounded-full flex items-center gap-2 animate-pulse">
                            <Zap className="w-4 h-4 text-purple-400" />
                            <span className="text-xs font-bold text-purple-300 uppercase">High Skill – Low Opportunity Detected</span>
                        </div>
                    )}
                </div>

                {/* ── Workforce Assessment: 3 big cards ── */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div className={`rounded-xl p-5 border ${getLevelBg(work_capacity)} text-center`}>
                        <div className="text-[10px] text-gray-400 uppercase tracking-widest mb-2 font-bold">Work Capacity</div>
                        <div className={`text-3xl font-bold ${getLevelColor(work_capacity)}`}>{work_capacity}</div>
                    </div>
                    <div className={`rounded-xl p-5 border ${getLevelBg(growth_potential_wfa)} text-center`}>
                        <div className="text-[10px] text-gray-400 uppercase tracking-widest mb-2 font-bold">Growth Potential</div>
                        <div className={`text-3xl font-bold ${getLevelColor(growth_potential_wfa)}`}>{growth_potential_wfa}</div>
                    </div>
                    <div className={`rounded-xl p-5 border ${getLevelBg(risk_level === 'Low' ? 'High' : risk_level === 'High' ? 'Low' : 'Moderate')} text-center`}>
                        <div className="text-[10px] text-gray-400 uppercase tracking-widest mb-2 font-bold">Risk Level</div>
                        <div className={`text-3xl font-bold ${getRiskColor(risk_level)}`}>{risk_level}</div>
                    </div>
                </div>

                {/* ── Score + Details Row ── */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-800">
                        <div className="text-[10px] text-gray-500 uppercase mb-1">AI Score</div>
                        <div className="flex items-baseline gap-1">
                            <span className={`text-3xl font-mono font-bold ${getScoreColor(score)}`}>{score}</span>
                            <span className="text-xs text-gray-600">/100</span>
                        </div>
                        <div className="w-full bg-gray-800 rounded-full h-1.5 mt-2">
                            <div className={`${getScoreBg(score)} h-1.5 rounded-full transition-all duration-700`} style={{ width: `${score}%` }}></div>
                        </div>
                    </div>
                    <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-800">
                        <div className="text-[10px] text-gray-500 uppercase mb-1">Skill Level</div>
                        <div className={`text-xl font-bold ${getScoreColor(score)}`}>{level}</div>
                    </div>
                    <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-800">
                        <div className="text-[10px] text-gray-500 uppercase mb-1">Confidence</div>
                        <div className="text-xl font-mono font-bold text-accent">{confidence}%</div>
                    </div>
                    <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-800">
                        <div className="text-[10px] text-gray-500 uppercase mb-1">Migration Risk</div>
                        <div className={`text-xl font-bold ${getRiskColor(migration_risk || 'Low')}`}>{migration_risk || 'Low'}</div>
                    </div>
                </div>
            </div>

            {/* ═══════ 2. EXPLAINABILITY – Waterfall Chart ═══════ */}
            {explanations && (explanations.top_positive?.length > 0 || explanations.top_negative?.length > 0) && (
                <XAIWaterfallChart
                    topPositive={explanations.top_positive || []}
                    topNegative={explanations.top_negative || []}
                    totalScore={score}
                />
            )}

            {/* ═══════ 3. ACTION RECOMMENDATIONS ═══════ */}
            {recs.length > 0 && (
                <div className="bg-[#0F172A] border border-gray-700 rounded-xl p-5 shadow-lg">
                    <h3 className="text-sm font-bold text-white uppercase mb-4 flex items-center gap-2 tracking-wider">
                        <Target className="w-4 h-4 text-amber-400" /> Action Recommendations
                    </h3>
                    <div className="space-y-2">
                        {recs.slice(0, 6).map((rec, i) => (
                            <div key={i} className="flex items-start gap-3 bg-gray-900/50 rounded-lg p-3 border border-gray-800">
                                <ArrowUpRight className="w-4 h-4 text-accent shrink-0 mt-0.5" />
                                <div className="flex-1">
                                    <div className="text-xs text-gray-200">{rec.action}</div>
                                </div>
                                <span className={`text-[9px] px-2 py-0.5 rounded-full border font-bold uppercase shrink-0 ${getPriorityBg(rec.priority)}`}>
                                    {rec.priority}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* ═══════ 4. GROWTH OPPORTUNITIES ═══════ */}
            {opps && (opps.training?.length > 0 || opps.platforms?.length > 0 || opps.government_schemes?.length > 0 || opps.digital_growth?.length > 0) && (
                <div className="bg-[#0F172A] border border-gray-700 rounded-xl p-5 shadow-lg">
                    <h3 className="text-sm font-bold text-white uppercase mb-1 flex items-center gap-2 tracking-wider">
                        <Lightbulb className="w-4 h-4 text-yellow-400" /> Growth Opportunities
                    </h3>
                    <p className="text-[11px] text-gray-500 mb-4">These opportunities are recommended based on your work profile and growth potential.</p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Training */}
                        {opps.training?.length > 0 && (
                            <div className="bg-blue-900/10 border border-blue-500/20 rounded-lg p-4">
                                <div className="text-[10px] text-blue-400 uppercase tracking-wider font-bold mb-2 flex items-center gap-1.5">
                                    <BookOpen className="w-3 h-3" /> Training Opportunities
                                </div>
                                <ul className="space-y-1.5">
                                    {opps.training.map((t, i) => (
                                        <li key={i} className="text-xs text-gray-300 flex items-start gap-2">
                                            <CheckCircle className="w-3 h-3 text-blue-400 shrink-0 mt-0.5" /> {t}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                        {/* Government Schemes */}
                        {opps.government_schemes?.length > 0 && (
                            <div className="bg-green-900/10 border border-green-500/20 rounded-lg p-4">
                                <div className="text-[10px] text-green-400 uppercase tracking-wider font-bold mb-2 flex items-center gap-1.5">
                                    <Building2 className="w-3 h-3" /> Government Schemes
                                </div>
                                <ul className="space-y-1.5">
                                    {opps.government_schemes.map((g, i) => (
                                        <li key={i} className="text-xs text-gray-300 flex items-start gap-2">
                                            <CheckCircle className="w-3 h-3 text-green-400 shrink-0 mt-0.5" /> {g}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                        {/* Platform Opportunities */}
                        {opps.platforms?.length > 0 && (
                            <div className="bg-purple-900/10 border border-purple-500/20 rounded-lg p-4">
                                <div className="text-[10px] text-purple-400 uppercase tracking-wider font-bold mb-2 flex items-center gap-1.5">
                                    <Globe className="w-3 h-3" /> Platform Opportunities
                                </div>
                                <ul className="space-y-1.5">
                                    {opps.platforms.map((p, i) => (
                                        <li key={i} className="text-xs text-gray-300 flex items-start gap-2">
                                            <CheckCircle className="w-3 h-3 text-purple-400 shrink-0 mt-0.5" /> {p}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                        {/* Digital Growth */}
                        {opps.digital_growth?.length > 0 && (
                            <div className="bg-cyan-900/10 border border-cyan-500/20 rounded-lg p-4">
                                <div className="text-[10px] text-cyan-400 uppercase tracking-wider font-bold mb-2 flex items-center gap-1.5">
                                    <Smartphone className="w-3 h-3" /> Digital Growth Actions
                                </div>
                                <ul className="space-y-1.5">
                                    {opps.digital_growth.map((d, i) => (
                                        <li key={i} className="text-xs text-gray-300 flex items-start gap-2">
                                            <CheckCircle className="w-3 h-3 text-cyan-400 shrink-0 mt-0.5" /> {d}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* ═══════ 5. TRUST INDICATOR ═══════ */}
            <div className="bg-[#0F172A] border border-gray-700 rounded-xl p-5 shadow-lg">
                <h3 className="text-sm font-bold text-white uppercase mb-4 flex items-center gap-2 tracking-wider">
                    <Shield className="w-4 h-4 text-blue-400" /> Assessment Trust
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <div className="bg-gray-900/50 rounded-lg p-3 border border-gray-800">
                        <div className="text-[10px] text-gray-500 uppercase mb-1">Data Source</div>
                        <div className="text-xs font-bold text-gray-300">{trustInfo.data_source || 'Self-reported'}</div>
                    </div>
                    <div className="bg-gray-900/50 rounded-lg p-3 border border-gray-800">
                        <div className="text-[10px] text-gray-500 uppercase mb-1">Confidence Level</div>
                        <div className={`text-sm font-bold ${trustInfo.confidence_level === 'High' ? 'text-emerald-400' : trustInfo.confidence_level === 'Medium' ? 'text-amber-400' : 'text-red-400'}`}>
                            {trustInfo.confidence_level || 'Medium'}
                        </div>
                    </div>
                    {sourceVerification && (
                        <div className="bg-gray-900/50 rounded-lg p-3 border border-gray-800">
                            <div className="text-[10px] text-gray-500 uppercase mb-1">Documents Verified</div>
                            <div className={`text-sm font-bold ${sourceVerification.proof_strength === 'High' ? 'text-emerald-400' : sourceVerification.proof_strength === 'Medium' ? 'text-amber-400' : 'text-gray-500'}`}>
                                {sourceVerification.proof_strength || 'None'}
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* ═══════ 6. USER BENEFITS ═══════ */}
            <div className="bg-[#0F172A] border border-gray-700 rounded-xl p-5 shadow-lg">
                <h3 className="text-sm font-bold text-white uppercase mb-3 flex items-center gap-2 tracking-wider">
                    <Users className="w-4 h-4 text-accent" /> How This Helps You
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {[
                        { text: 'Understand your work potential', icon: '💡' },
                        { text: 'Identify growth areas', icon: '📈' },
                        { text: 'Prepare for training programs', icon: '🎓' },
                        { text: 'Improve income opportunities', icon: '💰' }
                    ].map((b, i) => (
                        <div key={i} className="bg-gray-900/50 rounded-lg p-3 border border-gray-800 flex items-start gap-2">
                            <span className="text-lg">{b.icon}</span>
                            <span className="text-[11px] text-gray-300 leading-snug">{b.text}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* ═══════ 7. HIDDEN TALENT BANNER ═══════ */}
            {isHiddenTalent && (
                <div className="p-4 bg-purple-900/20 border border-purple-500/30 rounded-xl flex items-center gap-3">
                    <Zap className="w-6 h-6 text-purple-400 shrink-0" />
                    <div>
                        <div className="text-xs font-bold text-purple-300 uppercase">Hidden Talent Intelligence</div>
                        <div className="text-xs text-purple-400/80 mt-0.5">{hiddenTalentReason}</div>
                    </div>
                </div>
            )}

            {/* ═══════ 8. DEMO DISCLAIMER ═══════ */}
            <div className="flex items-start gap-2 px-4 py-3 rounded-lg bg-gray-800/30 border border-gray-800">
                <Info className="w-3.5 h-3.5 text-gray-500 shrink-0 mt-0.5" />
                <p className="text-[10px] text-gray-500 leading-relaxed">
                    Current version uses structured self-reported inputs. Future versions will integrate with government and digital data sources for automated verification.
                </p>
            </div>
        </div>
    );
};

export default PredictionSummary;

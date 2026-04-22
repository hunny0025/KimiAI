import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';

// ─── Region selector options ──────────────────────────────────────────────────
const REGIONS = [
  'National', 'Bihar', 'Rajasthan', 'Uttar Pradesh', 'Madhya Pradesh',
  'Jharkhand', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Kerala', 'Gujarat',
];

// ─── Agent status constants ───────────────────────────────────────────────────
const AGENT_META = [
  { key: 'scout',   icon: '🔍', label: 'Scout Agent',   model: 'IsolationForest', color: 'cyan'   },
  { key: 'analyst', icon: '🧠', label: 'Analyst Agent', model: 'GBR + XAI',       color: 'violet' },
  { key: 'policy',  icon: '📋', label: 'Policy Agent',  model: 'Groq LLaMA3',     color: 'amber'  },
  { key: 'monitor', icon: '📡', label: 'Monitor Agent', model: 'Health Loop',     color: 'green'  },
];

const colorMap = {
  cyan:   { border: '#06b6d4', bg: 'rgba(6,182,212,0.08)',   text: '#22d3ee', dot: 'bg-cyan-400'   },
  violet: { border: '#7c3aed', bg: 'rgba(124,58,237,0.08)',  text: '#a78bfa', dot: 'bg-violet-400' },
  amber:  { border: '#d97706', bg: 'rgba(217,119,6,0.08)',   text: '#fbbf24', dot: 'bg-amber-400'  },
  green:  { border: '#16a34a', bg: 'rgba(22,163,74,0.08)',   text: '#4ade80', dot: 'bg-green-400'  },
};

// ─── Type-writer hook ─────────────────────────────────────────────────────────
function useTypewriter(text, speed = 18, active = true) {
  const [displayed, setDisplayed] = useState('');
  useEffect(() => {
    if (!active || !text) { setDisplayed(text || ''); return; }
    setDisplayed('');
    let i = 0;
    const id = setInterval(() => {
      setDisplayed(text.slice(0, ++i));
      if (i >= text.length) clearInterval(id);
    }, speed);
    return () => clearInterval(id);
  }, [text, active]);
  return displayed;
}

// ─── Agent Card ───────────────────────────────────────────────────────────────
function AgentCard({ meta, data, loading, step, currentStep }) {
  const c = colorMap[meta.color];
  const isActive  = step <= currentStep;
  const isCurrent = step === currentStep && loading;
  const isDone    = step < currentStep;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: isActive ? 1 : 0.35, y: 0 }}
      transition={{ duration: 0.4, delay: step * 0.08 }}
      style={{ border: `1px solid ${isActive ? c.border : '#374151'}`, background: isActive ? c.bg : 'transparent' }}
      className="rounded-xl p-4 relative overflow-hidden"
    >
      {/* Scanning sweep animation */}
      {isCurrent && (
        <motion.div
          className="absolute inset-0 pointer-events-none"
          style={{ background: `linear-gradient(90deg, transparent 0%, ${c.border}22 50%, transparent 100%)` }}
          animate={{ x: ['-100%', '200%'] }}
          transition={{ duration: 1.2, repeat: Infinity, ease: 'linear' }}
        />
      )}

      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-2xl">{meta.icon}</span>
          <div>
            <div className="font-bold text-sm" style={{ color: c.text }}>{meta.label}</div>
            <div className="text-[10px] text-gray-500 font-mono">{meta.model}</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isCurrent && (
            <span className="text-[10px] font-mono text-yellow-400 animate-pulse">RUNNING…</span>
          )}
          {isDone && (
            <span className="text-[10px] font-mono" style={{ color: c.text }}>✓ DONE</span>
          )}
          {!isActive && !loading && (
            <span className="text-[10px] font-mono text-gray-600">IDLE</span>
          )}
          <span className={`w-2 h-2 rounded-full ${isActive ? c.dot : 'bg-gray-700'} ${isCurrent ? 'animate-pulse' : ''}`} />
        </div>
      </div>

      {/* Data output */}
      {data && (
        <div className="space-y-1.5 text-xs font-mono">
          {meta.key === 'scout' && (
            <>
              <Row label="Region" value={data.region} color={c.text} />
              <Row label="Districts Scanned" value={data.districts_scanned} color={c.text} />
              <Row label="Hidden Talent Found" value={data.discovered} color={c.text} bold />
              <Row label="Anomaly Rate" value={data.anomaly_rate} color={c.text} />
              {data.high_risk_zones?.length > 0 && (
                <Row label="High Risk Zones" value={data.high_risk_zones.join(', ')} color="#f87171" />
              )}
            </>
          )}
          {meta.key === 'analyst' && (
            <>
              <Row label="States Analyzed" value={data.states_analyzed} color={c.text} />
              <Row label="Risk Score" value={`${data.risk_score}/100`} color={data.risk_score > 70 ? '#f87171' : data.risk_score > 45 ? '#fbbf24' : '#4ade80'} bold />
              <Row label="Risk Level" value={data.risk_label} color={c.text} />
              {data.top_factors?.map((f, i) => (
                <Row key={i} label={i === 0 ? 'Top Factors' : ''} value={f} color="#e879f9" />
              ))}
            </>
          )}
          {meta.key === 'policy' && (
            <>
              <Row label="Source" value={data.llm_powered ? '🤖 Groq LLaMA3' : '⚙️ Rule Engine'} color={c.text} />
              {data.interventions?.slice(0, 3).map((iv, i) => (
                <div key={i} className="mt-1 p-2 rounded" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid #374151' }}>
                  <div style={{ color: c.text }} className="font-semibold">{iv.policy}</div>
                  <div className="text-gray-400 text-[10px]">{iv.timeline} · ₹{iv.roi_crores} Cr ROI</div>
                </div>
              ))}
            </>
          )}
          {meta.key === 'monitor' && (
            <>
              <Row label="Pipeline Health" value={data.pipeline_health}
                color={data.pipeline_health === 'GREEN' ? '#4ade80' : data.pipeline_health === 'AMBER' ? '#fbbf24' : '#f87171'} bold />
              <Row label="Highest ROI" value={data.highest_roi_crores ? `₹${data.highest_roi_crores} Cr` : 'N/A'} color={c.text} />
              <Row label="Districts Monitored" value={data.districts_monitored} color={c.text} />
              <Row label="Next Scan" value={`${data.next_scan_seconds}s`} color={c.text} />
              {data.alerts?.[0] && (
                <div className={`mt-1 p-1.5 rounded text-[10px] ${
                  data.alerts[0].severity === 'OK' ? 'text-green-400 bg-green-900/20' :
                  data.alerts[0].severity === 'WARNING' ? 'text-yellow-400 bg-yellow-900/20' :
                  'text-red-400 bg-red-900/20'
                }`}>
                  {data.alerts[0].message}
                </div>
              )}
            </>
          )}
        </div>
      )}
      {!data && isActive && isCurrent && (
        <div className="text-xs text-gray-500 font-mono animate-pulse">Processing…</div>
      )}
      {!data && !isActive && (
        <div className="text-xs text-gray-600 font-mono">Waiting for previous agent…</div>
      )}
    </motion.div>
  );
}

function Row({ label, value, color, bold }) {
  return (
    <div className="flex gap-2">
      {label && <span className="text-gray-600 shrink-0 w-32">{label}:</span>}
      {!label && <span className="w-32 shrink-0" />}
      <span style={{ color }} className={bold ? 'font-bold' : ''}>{value}</span>
    </div>
  );
}

// ─── Agent Communication Log ──────────────────────────────────────────────────
function AgentCommsLog({ pipeline, visible }) {
  const bottomRef = useRef(null);

  const messages = !pipeline ? [] : [
    {
      from: 'Scout Agent', to: 'Analyst Agent',
      msg: `Found ${pipeline[0]?.discovered ?? '?'} anomalies in ${pipeline[0]?.region ?? '?'} across ${pipeline[0]?.districts_scanned ?? '?'} districts`,
      time: '00:00', color: '#22d3ee',
    },
    {
      from: 'Analyst Agent', to: 'Policy Agent',
      msg: `Risk score: ${pipeline[1]?.risk_score ?? '?'}/100 (${pipeline[1]?.risk_label ?? '?'}). Top factor: ${pipeline[1]?.top_factors?.[0] ?? '?'}`,
      time: '00:01', color: '#a78bfa',
    },
    {
      from: 'Policy Agent', to: 'Monitor Agent',
      msg: `${pipeline[2]?.interventions?.length ?? 0} interventions generated. Highest ROI: ₹${pipeline[3]?.highest_roi_crores ?? 'N/A'} Cr`,
      time: '00:02', color: '#fbbf24',
    },
    {
      from: 'Monitor Agent', to: 'Command Center',
      msg: `Pipeline complete. Health: ${pipeline[3]?.pipeline_health ?? '?'}. Next autonomous scan in ${pipeline[3]?.next_scan_seconds ?? 60}s.`,
      time: '00:03', color: '#4ade80',
    },
  ];

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages.length]);

  if (!visible) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl p-4 font-mono text-xs"
      style={{ background: '#0d1117', border: '1px solid #22c55e55' }}
    >
      <div className="flex items-center gap-2 mb-3">
        <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
        <span className="text-green-400 font-bold text-sm tracking-widest">🔗 AGENT COMMUNICATION LOG</span>
      </div>
      <div className="space-y-2">
        <AnimatePresence>
          {messages.map((m, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.3 }}
              className="flex gap-2 items-start"
            >
              <span className="text-gray-600 shrink-0">[{m.time}]</span>
              <span style={{ color: m.color }} className="shrink-0 font-semibold">{m.from}</span>
              <span className="text-gray-500">→</span>
              <span className="text-blue-400 shrink-0">{m.to}:</span>
              <span className="text-gray-300 break-all">{m.msg}</span>
            </motion.div>
          ))}
        </AnimatePresence>
        {messages.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.5 }}
            className="text-green-400 mt-2"
          >
            █ <span className="animate-pulse">_</span>
          </motion.div>
        )}
      </div>
      <div ref={bottomRef} />
    </motion.div>
  );
}

// ─── Monitor Ticker ───────────────────────────────────────────────────────────
function MonitorTicker({ pipelineCount }) {
  const [stats, setStats] = useState({ scans: pipelineCount || 1, uptime: 0, lastSync: 'Just now' });

  useEffect(() => {
    const id = setInterval(() => {
      setStats(s => ({ scans: s.scans + 1, uptime: s.uptime + 1, lastSync: 'Just now' }));
    }, 60000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    setStats(s => ({ ...s, scans: pipelineCount || s.scans }));
  }, [pipelineCount]);

  return (
    <div className="flex flex-wrap gap-4 text-xs font-mono border-t border-green-500/30 pt-3 mt-2">
      <span className="text-green-400">🟢 Monitor Agent: ACTIVE</span>
      <span className="text-gray-400">📡 Pipelines Run: <span className="text-white">{stats.scans}</span></span>
      <span className="text-gray-400">⏱ Uptime: <span className="text-white">{stats.uptime}m</span></span>
      <span className="text-gray-400">🗺 Districts Monitored: <span className="text-white">736</span></span>
      <span className="text-gray-400">🔄 Last Sync: <span className="text-white">{stats.lastSync}</span></span>
    </div>
  );
}

// ─── Autonomous Alert Box ─────────────────────────────────────────────────────
function AutonomousAlerts({ data }) {
  if (!data) return null;
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      className="rounded-xl p-4 font-mono text-xs"
      style={{ background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.3)' }}
    >
      <div className="text-red-400 font-bold mb-2 text-sm">⚠️ AUTONOMOUS ALERT — CRITICAL REGIONS DETECTED</div>
      <div className="text-gray-400 mb-2">{data.status}</div>
      <div className="space-y-1">
        {data.triggered?.map((t, i) => (
          <div key={i} className="flex gap-3">
            <span className="text-red-400">{t.region}</span>
            <span className="text-gray-500">Risk: <span className="text-red-300">{t.risk_score}</span></span>
            <span className="text-yellow-400">Pipeline: {t.auto_pipeline}</span>
            <span className="text-green-400">{t.policy_count} policies generated</span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

// ─── Main AgentCommandCenter ──────────────────────────────────────────────────
export default function AgentCommandCenter() {
  const [region, setRegion] = useState('National');
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(-1);
  const [pipeline, setPipeline] = useState(null);
  const [autonomousData, setAutonomousData] = useState(null);
  const [pipelineCount, setPipelineCount] = useState(0);
  const [error, setError] = useState(null);

  // Auto-run autonomous check on mount
  useEffect(() => {
    runAutonomousCheck();
  }, []);

  const runAutonomousCheck = async () => {
    try {
      const res = await axios.get('/api/autonomous-check');
      setAutonomousData(res.data);
    } catch (e) {
      console.warn('Autonomous check unavailable', e);
    }
  };

  const deployAgents = async () => {
    setLoading(true);
    setError(null);
    setPipeline(null);
    setPipelineCount(c => c + 1);

    // Animate step-by-step
    const agentKeys = ['scout', 'analyst', 'policy', 'monitor'];
    const partial = {};

    for (let i = 0; i < agentKeys.length; i++) {
      setCurrentStep(i);
      await new Promise(r => setTimeout(r, 600)); // brief visual delay per step
    }

    try {
      const res = await axios.post('/api/orchestrate', { region });
      const data = res.data;
      const p = data.pipeline || [];
      setPipeline(p);
      setCurrentStep(4); // all done
    } catch (e) {
      setError('Orchestration failed — check backend connection.');
      setCurrentStep(-1);
    }
    setLoading(false);
  };

  const agentData = pipeline
    ? { scout: pipeline[0], analyst: pipeline[1], policy: pipeline[2], monitor: pipeline[3] }
    : {};

  return (
    <div className="space-y-6 p-1">
      {/* ── Header ── */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h2 className="text-2xl font-black text-white tracking-tight">KARM.AI <span className="text-violet-400">Agent Command Center</span></h2>
          <p className="text-gray-400 text-sm mt-1">
            Knowledge-driven Autonomous Regional Mapping · 4 specialized AI agents
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={region}
            onChange={e => setRegion(e.target.value)}
            className="bg-gray-900 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 outline-none focus:border-violet-500"
          >
            {REGIONS.map(r => <option key={r}>{r}</option>)}
          </select>
          <motion.button
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            onClick={deployAgents}
            disabled={loading}
            className="px-5 py-2 rounded-lg font-bold text-sm text-white disabled:opacity-50 transition-all"
            style={{
              background: loading
                ? 'linear-gradient(135deg, #4c1d95, #1e1b4b)'
                : 'linear-gradient(135deg, #7c3aed, #2563eb)',
              boxShadow: loading ? 'none' : '0 0 20px rgba(124,58,237,0.4)'
            }}
          >
            {loading ? '⚡ Deploying…' : '⚡ Deploy Agents'}
          </motion.button>
        </div>
      </div>

      {/* ── Autonomous Alert ── */}
      {autonomousData && autonomousData.pipelines_triggered > 0 && (
        <AutonomousAlerts data={autonomousData} />
      )}

      {/* ── 4-Agent Grid ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {AGENT_META.map((meta, i) => (
          <AgentCard
            key={meta.key}
            meta={meta}
            data={agentData[meta.key]}
            loading={loading}
            step={i}
            currentStep={currentStep}
          />
        ))}
      </div>

      {/* ── Agent Comms Log (shows after pipeline completes) ── */}
      <AgentCommsLog pipeline={pipeline} visible={!!pipeline} />

      {/* ── Monitor Ticker ── */}
      <MonitorTicker pipelineCount={pipelineCount} />

      {/* ── Error ── */}
      {error && (
        <div className="text-red-400 text-sm font-mono p-3 rounded-lg bg-red-900/20 border border-red-800">
          {error}
        </div>
      )}

      {/* ── Pipeline status summary ── */}
      {pipeline && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-xs font-mono text-gray-500 text-center"
        >
          ✓ ORCHESTRATION COMPLETE · {AGENT_META.length} agents · Region: {region} · {new Date().toLocaleTimeString()}
        </motion.div>
      )}
    </div>
  );
}

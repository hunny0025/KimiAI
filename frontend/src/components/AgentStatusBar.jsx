import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

/**
 * AgentStatusBar — Persistent header strip showing live agent health.
 * Replaces the plain title header.
 *
 * ┌────────────────────────────────────────────────────────────┐
 * │  KARM.AI  │ 🔍 Scout ●  │ 🧠 Analyst ● │ 📋 Policy ● │ 📡 Monitor ● │ LIVE │
 * └────────────────────────────────────────────────────────────┘
 */
const AGENTS = [
  { icon: '🔍', label: 'Scout',   color: '#22d3ee' },
  { icon: '🧠', label: 'Analyst', color: '#a78bfa' },
  { icon: '📋', label: 'Policy',  color: '#fbbf24' },
  { icon: '📡', label: 'Monitor', color: '#4ade80' },
];

export default function AgentStatusBar({ activeTab, setActiveTab }) {
  const [tick, setTick] = useState(0);

  // Simulate live pulse every 3 seconds
  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 3000);
    return () => clearInterval(id);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex items-center justify-between px-4 py-2 border-b border-gray-800 bg-gray-950/80 backdrop-blur-sm shrink-0"
      style={{ minHeight: 44 }}
    >
      {/* Brand */}
      <button
        onClick={() => setActiveTab('agents')}
        className="flex items-center gap-2 hover:opacity-80 transition-opacity"
      >
        <div
          className="w-6 h-6 rounded flex items-center justify-center text-xs font-black shrink-0"
          style={{ background: 'linear-gradient(135deg, #7c3aed, #06b6d4)' }}
        >
          K
        </div>
        <span className="font-black text-white text-sm tracking-wider hidden sm:block">KARM.AI</span>
      </button>

      {/* Agent Pills */}
      <div className="flex items-center gap-1 sm:gap-3">
        {AGENTS.map((a, i) => (
          <motion.div
            key={a.label}
            className="flex items-center gap-1 px-2 py-1 rounded-md text-xs font-mono cursor-default"
            style={{ background: 'rgba(255,255,255,0.03)', border: `1px solid ${a.color}33` }}
            animate={{ borderColor: [`${a.color}33`, `${a.color}99`, `${a.color}33`] }}
            transition={{ duration: 3, delay: i * 0.5, repeat: Infinity }}
          >
            <span className="hidden sm:inline">{a.icon}</span>
            <span style={{ color: a.color }} className="hidden md:inline">{a.label}</span>
            <motion.span
              className="w-1.5 h-1.5 rounded-full shrink-0"
              style={{ background: a.color }}
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ duration: 1.5, delay: i * 0.3, repeat: Infinity }}
            />
          </motion.div>
        ))}
      </div>

      {/* Live badge */}
      <div className="flex items-center gap-2">
        <motion.div
          className="flex items-center gap-1.5 px-2 py-1 rounded text-[10px] font-mono font-bold"
          style={{ background: 'rgba(20,83,45,0.5)', border: '1px solid #16a34a55', color: '#4ade80' }}
          animate={{ opacity: [1, 0.7, 1] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
          LIVE
        </motion.div>
      </div>
    </motion.div>
  );
}

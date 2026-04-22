import React from 'react';
import {
    LayoutDashboard, Map, BarChart3, AlertTriangle,
    TrendingUp, FileText, Activity, Zap, Database, BarChart2
} from 'lucide-react';

// FIX 8: Agent health data
const AGENT_HEALTH = [
    { label: 'Scout Agent',   dots: 5, pct: 100, color: '#22c55e' },
    { label: 'Analyst Agent', dots: 5, pct: 100, color: '#22c55e' },
    { label: 'Policy Agent',  dots: 4, pct: 80,  color: '#22c55e' },
    { label: 'Monitor Agent', dots: 5, pct: 100, color: '#22c55e' },
];

// FIX 7: Hindi translations map
const HINDI = {
    'Agent Command Center': 'एजेंट कमांड सेंटर',
    'Regional Intel':       'क्षेत्रीय खुफिया',
    'Market Pulse':         'बाजार के रुझान',
    'Future Forecast':      'भविष्य का पूर्वानुमान',
    '🧠 Analyst Agent':     'विश्लेषक एजेंट',
    '📋 Policy Agent':      'नीति एजेंट',
    'Policy Simulator':     'सिमुलेशन',
    'Dataset & Training':   'डेटासेट प्रशिक्षण',
    '🔍 Scout Agent':       'स्काउट एजेंट',
    'Switch to Hindi (हिंदी)': 'Switch to English (English में बदलें)',
    'Switch to English':    'Switch to Hindi (हिंदी)',
};

const t = (key, lang) => lang === 'hi' ? (HINDI[key] || key) : key;

const Sidebar = ({ activeTab, setActiveTab, language, setLanguage }) => {
    const menuItems = [
        { id: 'dashboard',  label: 'Agent Command Center', icon: LayoutDashboard,  dot: null },
        { id: 'regional',   label: 'Regional Intel',       icon: Map,              dot: null },
        { id: 'market',     label: 'Market Pulse',         icon: Activity,         dot: null },
        { id: 'forecast',   label: 'Future Forecast',      icon: TrendingUp,       dot: null },
        { id: 'risk',       label: '🧠 Analyst Agent',     icon: AlertTriangle,    dot: '#22c55e' }, // FIX 8: green dot
        { id: 'policy',     label: '📋 Policy Agent',      icon: FileText,         dot: '#22c55e' }, // FIX 8: green dot
        { id: 'simulation', label: 'Policy Simulator',     icon: Zap,              dot: null },
        { id: 'dataset',    label: 'Dataset & Training',   icon: Database,         dot: null },
        { id: 'skill-risk', label: '🔍 Scout Agent',       icon: BarChart2,        dot: '#22c55e' }, // FIX 8: green dot
    ];

    return (
        <div className="h-full bg-panel border-r border-gray-800 flex flex-col w-[240px]">
            {/* Logo */}
            <div className="p-6 border-b border-gray-800 flex items-center gap-3">
                <div className="w-8 h-8 rounded bg-gradient-to-br from-accent to-purple-800 flex items-center justify-center shadow-lg shadow-accent/20">
                    <Activity className="text-white w-5 h-5" />
                </div>
                <div>
                    <h1 className="font-bold text-white tracking-wider text-sm">KARM.AI</h1>
                    <div className="text-[10px] text-gray-500 tracking-widest uppercase">
                        {language === 'hi' ? 'अभिनव बुद्धिमत्ता' : 'Agentic Intelligence'}
                    </div>
                </div>
            </div>

            {/* Nav */}
            <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
                {menuItems.map((item) => {
                    const Icon = item.icon;
                    const isActive = activeTab === item.id;
                    const label = t(item.label, language);
                    return (
                        <button
                            key={item.id}
                            onClick={() => setActiveTab(item.id)}
                            className={`flex items-center w-full px-4 py-3 text-sm font-medium rounded-lg transition-all
                                ${isActive
                                    ? 'bg-accent/10 text-accent border border-accent/20 shadow-[0_0_15px_rgba(108,92,231,0.1)]'
                                    : 'text-gray-400 hover:bg-gray-800 hover:text-white'
                                }`}
                        >
                            <Icon className={`w-4 h-4 mr-3 shrink-0 ${isActive ? 'text-accent' : 'text-gray-500'}`} />
                            <span className="truncate flex-1 text-left">{label}</span>
                            {/* FIX 8: colored status dot */}
                            {item.dot && (
                                <span
                                    className="w-2 h-2 rounded-full shrink-0 animate-pulse"
                                    style={{ background: item.dot }}
                                />
                            )}
                        </button>
                    );
                })}
            </nav>

            {/* FIX 8: Agent Health Box */}
            <div className="mx-4 mb-3 p-3 rounded-lg" style={{ background: '#0d1117', border: '1px solid #1f2937' }}>
                <div className="text-[9px] text-gray-500 uppercase tracking-widest font-bold mb-2">
                    {language === 'hi' ? 'एजेंट स्वास्थ्य' : 'AGENT HEALTH'}
                </div>
                {AGENT_HEALTH.map((a) => (
                    <div key={a.label} className="flex items-center justify-between mb-1.5">
                        <span className="text-[10px] text-gray-400 w-[90px] truncate">
                            {language === 'hi' ? t(`${a.label.includes('Scout') ? '🔍 Scout Agent' : a.label.includes('Analyst') ? '🧠 Analyst Agent' : a.label.includes('Policy') ? '📋 Policy Agent' : a.label}`, language) : a.label}
                        </span>
                        <div className="flex items-center gap-0.5">
                            {[1,2,3,4,5].map(d => (
                                <span
                                    key={d}
                                    className="w-1.5 h-1.5 rounded-full"
                                    style={{ background: d <= a.dots ? a.color : '#374151' }}
                                />
                            ))}
                            <span className="text-[9px] text-green-400 ml-1 font-mono">{a.pct}%</span>
                        </div>
                    </div>
                ))}
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-gray-800">
                {/* FIX 7: Hindi toggle with proper labels */}
                <button
                    onClick={() => setLanguage(language === 'en' ? 'hi' : 'en')}
                    className="w-full py-2 px-4 bg-gray-900 hover:bg-gray-800 border border-gray-700 rounded text-xs text-gray-400 flex items-center justify-center transition-colors"
                >
                    {language === 'en' ? 'Switch to Hindi (हिंदी)' : 'Switch to English (English में बदलें)'}
                </button>
                <div className="mt-4 flex items-center justify-between text-[10px] text-gray-600">
                    <span>v2.0.0-AGENTIC</span>
                    <div className="flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                        ONLINE
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Sidebar;

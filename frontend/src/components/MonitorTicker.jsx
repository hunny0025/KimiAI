import React, { useState, useEffect } from 'react';

// FIX 6: Global fixed bottom Monitor Ticker
const MonitorTicker = ({ language }) => {
    const [scans, setScans] = useState(1);
    const [uptime, setUptime] = useState(847);
    const [countdown, setCountdown] = useState(60);

    useEffect(() => {
        const id = setInterval(() => {
            setScans(s => s + 1);
            setUptime(u => u + 1);
            setCountdown(60);
        }, 60000);
        const cdId = setInterval(() => {
            setCountdown(c => c > 0 ? c - 1 : 60);
        }, 1000);
        return () => { clearInterval(id); clearInterval(cdId); };
    }, []);

    return (
        <div
            className="fixed bottom-0 left-0 w-full z-50 flex items-center justify-between px-4"
            style={{ background: '#0a0a0a', borderTop: '1px solid #22c55e55', height: 32 }}
        >
            <span className="text-green-400 font-mono text-xs flex items-center gap-3">
                <span className="flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                    {language === 'hi' ? 'मॉनिटर एजेंट: सक्रिय' : 'Monitor Agent: ACTIVE'}
                </span>
                <span className="text-gray-600 hidden sm:inline">|</span>
                <span className="hidden sm:inline">📡 {language === 'hi' ? 'स्कैन:' : 'Scans:'} <span className="text-white">{scans}</span></span>
                <span className="text-gray-600 hidden md:inline">|</span>
                <span className="hidden md:inline">⏱ {language === 'hi' ? 'अपटाइम:' : 'Uptime:'} <span className="text-white">{uptime}m</span></span>
            </span>
            <span className="text-green-400 font-mono text-xs flex items-center gap-3">
                <span className="hidden sm:inline">🔄 {language === 'hi' ? 'अगला स्कैन:' : 'Next Scan:'} <span className="text-white">{countdown}s</span></span>
                <span className="text-gray-600 hidden md:inline">|</span>
                <span className="hidden md:inline">736 {language === 'hi' ? 'जिले देख रहे हैं' : 'Districts Under Watch'}</span>
                <span className="text-gray-600 hidden lg:inline">|</span>
                <span className="hidden lg:inline">v2.0.0-AGENTIC <span className="text-green-400">🟢 ONLINE</span></span>
            </span>
        </div>
    );
};

export default MonitorTicker;

import React, { useEffect, useState } from 'react';
import { ComposableMap, Geographies, Geography } from 'react-simple-maps';
import { scaleQuantile } from 'd3-scale';
import axios from 'axios';
import { MapPin, TrendingUp, Users, Wifi, Zap, ChevronRight, Info, X } from 'lucide-react';

const geoUrl = "/maps/india_states.geojson";

// Colour scales for two views
const INNOVATION_COLORS = ["#1e3a5f","#1a56db","#6366f1","#a78bfa","#c7d2fe"];
const RISK_COLORS       = ["#14532d","#15803d","#eab308","#dc2626","#7f1d1d"];

// Map metric → display label + colour in legend
const VIEW_CONFIG = {
  innovation: {
    label: 'Innovation Intensity',
    sublabel: 'Composite score derived from digital adoption, creation output, and economic activity per state.',
    field: 'innovation_intensity',
    colors: INNOVATION_COLORS,
    legend: ['Very Low','Low','Medium','High','Very High'],
    unit: '/100',
  },
  risk: {
    label: 'Workforce Risk Index',
    sublabel: 'Composite risk score combining digital divide, skill deficit, and migration pressure (0 = safest).',
    field: 'hidden_talent_density',
    colors: RISK_COLORS,
    legend: ['Safe','Low','Moderate','High','Critical'],
    unit: '%',
  },
};

const StatCard = ({ icon: Icon, label, value, color }) => (
  <div className="flex items-center gap-2.5 bg-gray-800/60 rounded-lg p-2.5 min-w-0">
    <div className={`w-7 h-7 rounded flex items-center justify-center flex-shrink-0 ${color}`}>
      <Icon size={14} />
    </div>
    <div className="min-w-0">
      <div className="text-[10px] text-gray-500 uppercase truncate">{label}</div>
      <div className="text-sm font-bold text-white truncate">{value}</div>
    </div>
  </div>
);

const RegionalMap = ({ language }) => {
  const [data, setData]               = useState([]);
  const [mapError, setMapError]       = useState(false);
  const [loading, setLoading]         = useState(true);
  const [view, setView]               = useState('innovation');
  const [selected, setSelected]       = useState(null);   // selected state data
  const [hovered, setHovered]         = useState(null);   // hovered state name

  useEffect(() => {
    axios.get('/api/regional-analysis')
      .then(res => setData(Array.isArray(res.data) ? res.data : []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const cfg = VIEW_CONFIG[view];

  const colorScale = scaleQuantile()
    .domain(data.map(d => d[cfg.field] || 0))
    .range(cfg.colors);

  const getFill = (cur) => {
    if (!cur) return '#1f2937';
    const v = cur[cfg.field];
    if (v == null) return '#1f2937';
    return colorScale(v);
  };

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
      {/* ── Header ── */}
      <div className="px-5 pt-5 pb-3 border-b border-gray-800">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <MapPin size={16} className="text-blue-400" />
              <h2 className="text-base font-bold text-white uppercase tracking-wider">
                {language === 'en' ? 'Regional Talent Intelligence Heatmap' : 'क्षेत्रीय प्रतिभा खुफिया'}
              </h2>
              <span className="text-[10px] bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full border border-green-500/30">LIVE</span>
            </div>
            <p className="text-xs text-gray-500 max-w-xl">{cfg.sublabel}</p>
          </div>
          {/* View toggle */}
          <div className="flex gap-1 bg-gray-800 p-1 rounded-lg flex-shrink-0">
            {Object.entries(VIEW_CONFIG).map(([key, v]) => (
              <button
                key={key}
                onClick={() => setView(key)}
                className={`px-3 py-1 rounded text-xs font-semibold transition-all ${
                  view === key
                    ? 'bg-blue-600 text-white shadow'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                {v.label.split(' ')[0]}
              </button>
            ))}
          </div>
        </div>

        {/* ── Currently Showing ── */}
        <div className="mt-3 flex items-center gap-2">
          <Info size={12} className="text-gray-500 flex-shrink-0" />
          <span className="text-[11px] text-gray-500">
            Showing: <span className="text-blue-300 font-semibold">{cfg.label}</span> — click any state to see a detailed breakdown
          </span>
        </div>
      </div>

      {/* ── Body: Map + Side Panel ── */}
      <div className="flex flex-col lg:flex-row">
        {/* Map area */}
        <div className="relative flex-1 min-h-[460px] bg-gray-950">
          {loading && !mapError && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-sm animate-pulse">
              Loading India Map...
            </div>
          )}

          {!loading && !mapError && (
            <>
              <ComposableMap
                projection="geoMercator"
                projectionConfig={{ scale: 1050, center: [82, 22] }}
                width={700}
                height={560}
                style={{ width: '100%', height: '100%' }}
              >
                <Geographies geography={geoUrl} onError={() => setMapError(true)}>
                  {({ geographies }) => geographies.map((geo) => {
                    const stateName = geo.properties.ST_NM || geo.properties.st_nm || geo.properties.NAME_1;
                    const cur = data.find(s => s.state === stateName);
                    const isSelected = selected?.state === stateName;
                    const isHovered  = hovered === stateName;

                    return (
                      <Geography
                        key={geo.rsmKey}
                        geography={geo}
                        fill={isSelected ? '#f59e0b' : getFill(cur)}
                        stroke={isSelected ? '#fbbf24' : '#0B0F1A'}
                        strokeWidth={isSelected ? 1.5 : 0.5}
                        style={{
                          default: { outline: 'none', transition: 'fill 0.15s ease, opacity 0.15s ease', opacity: isHovered && !isSelected ? 0.75 : 1 },
                          hover:   { outline: 'none', cursor: 'pointer' },
                          pressed: { outline: 'none' },
                        }}
                        onMouseEnter={() => setHovered(stateName)}
                        onMouseLeave={() => setHovered(null)}
                        onClick={() => {
                          if (isSelected) { setSelected(null); return; }
                          if (cur) setSelected(cur);
                        }}
                      />
                    );
                  })}
                </Geographies>
              </ComposableMap>

              {/* Floating hover label */}
              {hovered && (
                <div className="absolute top-3 left-3 bg-gray-900/90 border border-gray-700 rounded-lg px-3 py-1.5 text-xs text-white pointer-events-none flex items-center gap-2">
                  <MapPin size={10} className="text-blue-400" />
                  {hovered}
                  {data.find(s => s.state === hovered) == null && (
                    <span className="text-gray-500 ml-1">· No data</span>
                  )}
                </div>
              )}
            </>
          )}

          {mapError && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-red-500 text-xs gap-2">
              <span>Failed to load GeoJSON map</span>
              <span className="text-gray-600">{geoUrl}</span>
            </div>
          )}

          {/* ── Color Legend ── */}
          <div className="absolute bottom-3 left-3 bg-gray-900/90 border border-gray-800 rounded-lg p-3">
            <div className="text-[10px] text-gray-500 uppercase mb-2 tracking-wider">{cfg.label}</div>
            <div className="flex items-center gap-1">
              {cfg.colors.map((color, i) => (
                <div key={i} className="flex flex-col items-center gap-1">
                  <div
                    className="w-5 h-5 rounded-sm border border-gray-700/50"
                    style={{ backgroundColor: color }}
                  />
                  <span className="text-[8px] text-gray-500">{cfg.legend[i]}</span>
                </div>
              ))}
            </div>
            <div className="text-[9px] text-gray-600 mt-1.5">↑ Click a state to explore</div>
          </div>
        </div>

        {/* ── Right: State Detail Panel ── */}
        <div className="w-full lg:w-72 flex-shrink-0 border-t lg:border-t-0 lg:border-l border-gray-800 bg-gray-900/50">
          {selected ? (
            <div className="p-4 h-full flex flex-col gap-4">
              {/* State name */}
              <div className="flex items-start justify-between">
                <div>
                  <div className="text-[10px] text-gray-500 uppercase mb-0.5">Selected State</div>
                  <div className="text-lg font-bold text-white">{selected.state}</div>
                  <div className="text-xs text-blue-300 mt-0.5">
                    {selected.specialization || 'General Economy'}
                  </div>
                </div>
                <button
                  onClick={() => setSelected(null)}
                  className="text-gray-600 hover:text-gray-300 transition-colors p-1"
                >
                  <X size={14} />
                </button>
              </div>

              {/* KPI cards */}
              <div className="grid grid-cols-1 gap-2">
                <StatCard icon={TrendingUp} label="Innovation Intensity" value={`${selected.innovation_intensity ?? '—'}/100`} color="bg-blue-500/20 text-blue-400" />
                <StatCard icon={Users}      label="Hidden Talent Density" value={`${selected.hidden_talent_density ?? '—'}%`}  color="bg-purple-500/20 text-purple-400" />
                <StatCard icon={Zap}         label="Ecosystem Balance"    value={`${selected.ecosystem_balance_score ?? '—'}/10`} color="bg-green-500/20 text-green-400" />
                <StatCard icon={Wifi}        label="Digital Access Score" value={selected.digital_access != null ? `${selected.digital_access}%` : 'N/A'} color="bg-cyan-500/20 text-cyan-400" />
              </div>

              {/* Specialization bar */}
              {selected.innovation_intensity != null && (
                <div>
                  <div className="flex justify-between text-[10px] text-gray-500 mb-1">
                    <span>Innovation Score</span>
                    <span className="text-white font-bold">{selected.innovation_intensity}%</span>
                  </div>
                  <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-blue-500 transition-all duration-700"
                      style={{ width: `${Math.min(selected.innovation_intensity, 100)}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Raw JSON toggle */}
              <div className="mt-auto border-t border-gray-800 pt-3">
                <div className="text-[10px] text-gray-600 font-mono">
                  Source: PLFS 2023-24 + MoSPI Regional Estimates
                </div>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full py-12 px-6 text-center gap-3">
              <MapPin size={28} className="text-gray-700" />
              <p className="text-sm text-gray-500 font-medium">Select a State</p>
              <p className="text-xs text-gray-600">
                Click any state on the map to view its Innovation Intensity, Hidden Talent Density, Ecosystem Balance, and more.
              </p>
              <div className="text-xs text-gray-700 border border-gray-800 rounded px-2 py-1">
                {data.length} states loaded
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Footer: what is this ── */}
      <div className="border-t border-gray-800 px-5 py-3 flex items-center justify-between bg-gray-950/50">
        <div className="text-[10px] text-gray-600">
          Data: National Skill Census v2.4 · PLFS 2023-24 (MoSPI) · Wheebox India Skills Report 2025
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
          <span className="text-[10px] text-gray-500">Live intelligence layer active</span>
        </div>
      </div>
    </div>
  );
};

export default RegionalMap;

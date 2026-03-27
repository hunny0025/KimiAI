/**
 * src/components/DataBadge.jsx – Fix 3: Data Provenance Badge
 * Shows "Simulated · PLFS-calibrated" on all KPI cards
 */
import React, { useState } from 'react';

const PROVENANCE = `Data source: PLFS-calibrated synthetic population (PLFS 2023-24 / MoSPI / TRAI). 
All scores are AI estimates for planning purposes — not verified individual records. 
Model: GradientBoostingRegressor · Calibrated to Wheebox 2025 (mean=42.6).`;

const DataBadge = ({ compact = false }) => {
  const [showTip, setShowTip] = useState(false);

  return (
    <span className="relative inline-flex items-center">
      <span
        onMouseEnter={() => setShowTip(true)}
        onMouseLeave={() => setShowTip(false)}
        className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold cursor-help select-none"
        style={{ background: '#FAEEDA', color: '#854F0B' }}
      >
        <span>⚙</span>
        {!compact && <span>Simulated · PLFS-calibrated</span>}
        {compact && <span>Simulated</span>}
      </span>

      {showTip && (
        <span
          className="absolute z-50 bottom-full left-0 mb-2 w-72 rounded-lg border border-amber-500/30 p-3 shadow-2xl text-[11px] leading-relaxed text-amber-100"
          style={{ background: '#1C1208' }}
        >
          {PROVENANCE}
        </span>
      )}
    </span>
  );
};

export default DataBadge;

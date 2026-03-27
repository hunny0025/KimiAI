/**
 * src/components/ProvenanceBanner.jsx – Fix 3: Dismissable provenance banner
 * Framer Motion slide-down 0.3s. Dismissed via React state only (not localStorage).
 */
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Info } from 'lucide-react';

const ProvenanceBanner = () => {
  const [visible, setVisible] = useState(true);

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          key="provenance-banner"
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{ duration: 0.3 }}
          className="overflow-hidden"
        >
          <div
            className="flex items-start gap-3 px-4 py-3 text-[11px] leading-relaxed"
            style={{
              background: 'rgba(250,238,218,0.05)',
              borderBottom: '1px solid rgba(133,79,11,0.35)',
              color: '#E8C97A',
            }}
          >
            <Info className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
            <span className="flex-1">
              <strong>Planning tool — not verified records.</strong>{' '}
              This platform uses AI predictions on PLFS-calibrated synthetic data.
              All scores are estimates for planning purposes, not verified individual records.
              Source: PLFS 2023-24 (MoSPI) · Wheebox 2025 · TRAI.
            </span>
            <button
              onClick={() => setVisible(false)}
              aria-label="Dismiss"
              className="text-amber-400/60 hover:text-amber-300 transition-colors shrink-0"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default ProvenanceBanner;

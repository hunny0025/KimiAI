/**
 * src/hooks/useApi.js – Fix 4: Unified API hook with retries + skeleton loading
 *
 * Usage:
 *   const { data, loading, error, refetch } = useApi('/api/national-distribution');
 *
 * Features:
 *   - Exponential backoff: 3 retries at 1s → 2s → 4s delay
 *   - Global API status tracking via a shared atom (simple module-level signal)
 *   - Returns { data, loading, error, refetch }
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';

// Simple module-level status emitter so the nav dot can subscribe
let _statusListeners = [];
export const subscribeApiStatus = (fn) => {
  _statusListeners.push(fn);
  return () => { _statusListeners = _statusListeners.filter(l => l !== fn); };
};
const _emit = (status) => _statusListeners.forEach(fn => fn(status));

const RETRY_DELAYS = [1000, 2000, 4000]; // ms

const useApi = (url, options = {}) => {
  const { method = 'GET', body = null, immediate = true } = options;
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const abortRef = useRef(null);

  const execute = useCallback(async (overrideBody) => {
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();

    setLoading(true);
    setError(null);
    _emit('loading');

    let attempt = 0;
    while (attempt <= RETRY_DELAYS.length) {
      try {
        const cfg = {
          method,
          url,
          signal: abortRef.current.signal,
          ...(body || overrideBody
            ? { data: overrideBody ?? body }
            : {}),
        };
        const res = await axios(cfg);
        setData(res.data);
        setLoading(false);
        _emit('ok');
        return res.data;
      } catch (err) {
        if (axios.isCancel(err)) return;
        attempt++;
        if (attempt > RETRY_DELAYS.length) {
          setError(err?.response?.data?.message ?? err.message ?? 'Request failed');
          _emit('error');
          break;
        }
        _emit('retrying');
        await new Promise(r => setTimeout(r, RETRY_DELAYS[attempt - 1]));
      }
    }
    setLoading(false);
  }, [url, method, body]);

  useEffect(() => {
    if (immediate && method === 'GET') execute();
    return () => { if (abortRef.current) abortRef.current.abort(); };
  }, [url, immediate, method]);

  return { data, loading, error, refetch: execute };
};

export default useApi;

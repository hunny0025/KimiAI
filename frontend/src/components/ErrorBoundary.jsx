// src/components/ErrorBoundary.jsx
// Fix 4 – React error boundary wrapping the whole app
import React from 'react';
import { motion } from 'framer-motion';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error('[ErrorBoundary] Caught render error:', error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-[#0A0F1E] flex items-center justify-center p-8">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4 }}
            className="max-w-lg w-full bg-[#0F172A] border border-red-500/30 rounded-2xl p-8 text-center shadow-2xl"
          >
            <div className="text-4xl mb-4">⚠️</div>
            <h2 className="text-xl font-bold text-white mb-2">Something went wrong</h2>
            <p className="text-sm text-gray-400 mb-6 leading-relaxed">
              The AI engine may be restarting. This is usually temporary — please try reloading the page.
            </p>
            {this.state.error && (
              <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 mb-6 text-left">
                <p className="text-xs text-red-300 font-mono break-words">
                  {this.state.error.toString()}
                </p>
              </div>
            )}
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold rounded-lg transition-colors"
            >
              Reload Page
            </button>
          </motion.div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;

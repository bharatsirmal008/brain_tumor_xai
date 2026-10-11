import React, { useRef, useState } from 'react';
import { Sparkles, LoaderCircle, Check } from 'lucide-react';
import './button.css';

/** onAnalyze must resolve only when real analysis finishes, and reject on failure. */
export function AIAnalyzeButton({ onAnalyze, disabled = false,
  loadingText = 'Analyzing MRI views and clinical inputs…', initialSuccess = false,
  onError, className = '' }) {
  const [state, setState] = useState(initialSuccess ? 'success' : 'idle');
  const [error, setError] = useState('');
  const busy = useRef(false);
  async function analyze() {
    if (disabled || busy.current) return;
    busy.current = true;
    setError('');
    setState('loading');
    try {
      await onAnalyze();
      setState('success');
    } catch (cause) {
      setState('idle');
      setError('Analysis could not finish. Please try again.');
      onError?.(cause);
    } finally {
      busy.current = false;
    }
  }
  const label = state === 'loading' ? loadingText : state === 'success' ? 'Analysis complete' : 'Analyze MRI';
  return <div className={`ai-control w-full ${className}`}>
    <button type="button" onClick={analyze} disabled={disabled || state === 'loading'}
      aria-busy={state === 'loading'} data-state={state}
      className="ai-analyze relative isolate grid w-full items-center overflow-hidden rounded-xl border text-sm font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-purple-300">
      <span className="ai-shimmer" aria-hidden="true"/>
      <span className="ai-icon relative z-10" aria-hidden="true">
        {state === 'success' ? <Check className="ai-check" size={20}/> : <Sparkles className="ai-sparkle" size={20}/>}
      </span>
      <span className="ai-label relative z-10" aria-live="polite" aria-atomic="true">{label}</span>
      <span className="ai-trailing relative z-10" aria-hidden="true">
        {state === 'loading' && <LoaderCircle size={19} className="ai-spinner"/>}
      </span>
    </button>
    {error && <p role="alert" className="mt-2 text-xs text-red-300">{error}</p>}
  </div>;
}

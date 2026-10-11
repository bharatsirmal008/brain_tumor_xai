import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { AIAnalyzeButton } from './AIAnalyzeButton';

const send = (type, data = {}) => window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, '*');
function StreamlitButton() {
  const [args, setArgs] = useState(null);
  const pending = useRef(null);
  useEffect(() => {
    function receive(event) {
      if (event.source !== window.parent || event.data?.type !== 'streamlit:render') return;
      const next = event.data.args;
      setArgs(next);
      if (pending.current && next.request_id === pending.current.id && next.outcome) {
        const task = pending.current;
        pending.current = null;
        if (next.outcome === 'success') task.resolve();
        else task.reject(new Error('Analysis failed'));
      }
    }
    window.addEventListener('message', receive);
    send('streamlit:componentReady', { apiVersion: 1 });
    const observer = new ResizeObserver(() => send('streamlit:setFrameHeight', { height: Math.ceil(document.getElementById('root').getBoundingClientRect().height) }));
    observer.observe(document.getElementById('root'));
    return () => { window.removeEventListener('message', receive); observer.disconnect(); };
  }, []);
  if (!args) return null;
  function onAnalyze() {
    return new Promise((resolve, reject) => {
      const id = crypto.randomUUID();
      pending.current = { id, resolve, reject };
      send('streamlit:setComponentValue', { value: id, dataType: 'json' });
    });
  }
  return <AIAnalyzeButton key={args.input_key} disabled={!args.ready}
    initialSuccess={args.outcome === 'success'} loadingText="Analyzing MRI views and clinical inputs…" onAnalyze={onAnalyze}/>;
}
function Demo() {
  const [runs, setRuns] = useState(0);
  const [fail, setFail] = useState(false);
  return <main className="min-h-screen bg-[#0b1729] px-6 py-24 text-[#e7e7ef]">
    <div className="mx-auto max-w-xl">
      <p className="mb-3 text-xs tracking-[.2em] text-purple-300">MRI ANALYSIS BUTTON DEMO</p>
      <h1 className="mb-3 text-3xl font-semibold">BraintumorXAI</h1>
      <p className="mb-9 text-sm leading-6 text-zinc-400">A simulated 3.5-second analysis. Click again after completion to restart.</p>
      <AIAnalyzeButton onAnalyze={async () => { await new Promise(r => setTimeout(r, 3500)); if (fail) throw new Error('Demo failure'); setRuns(n => n + 1); }}/>
      <p className="mt-5 text-sm text-zinc-400" data-testid="runs">Completed analyses: {runs}</p>
      <label className="mt-5 flex items-center gap-2 text-sm text-zinc-400"><input type="checkbox" checked={fail} onChange={e => setFail(e.target.checked)}/> Simulate a failed request</label>
    </div>
  </main>;
}
const demo = new URLSearchParams(location.search).has('demo') || window.parent === window;
createRoot(document.getElementById('root')).render(demo ? <Demo/> : <StreamlitButton/>);

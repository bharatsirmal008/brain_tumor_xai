# AIAnalyzeButton

Reusable React + Tailwind CSS + Lucide button with idle, loading, and success states.

## Demo

From this directory:

```sh
npm ci
npm run dev
```

Open http://127.0.0.1:8512/?demo=1. The demo resolves in 3.5 seconds and includes a failure toggle.

## React use

```jsx
import { AIAnalyzeButton } from './src/AIAnalyzeButton';

<AIAnalyzeButton
  disabled={!inputsReady}
  loadingText="Analyzing MRI views and clinical inputs…"
  onAnalyze={async () => {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) throw new Error('Analysis failed');
    const result = await response.json();
    setResult(result);
  }}
/>
```

The example endpoint must be supplied by the consuming application. `onAnalyze` must resolve after analysis finishes or reject on failure. Optional `onError(error)` supports application-specific handling. The default loading label is “Analyzing MRI views and clinical inputs…”.

## Streamlit integration

`app_with_no_tumor.py` serves `dist/` through a local Streamlit component. The component sends one opaque request ID per click and awaits a matching completion response from Python. MRI images and clinical inputs are not sent to the component. An input change resets the button. A failed analysis restores retry behavior; a negative screen that correctly skips later stages counts as completed screening.

```sh
npm run build
```

Keep the generated `dist/` assets with the Python app when deploying. Streamlit Community Cloud needs no Node build step when these assets are included. Rebuild after editing React or CSS source.

## Motion and accessibility

- Sparkle: 2 seconds, ease-in-out (1 second while loading).
- Shimmer: 1.5 seconds, left to right, linear loop while loading.
- Spinner: 0.75 seconds, linear loop.
- Hover: 200 milliseconds.
- Fixed 68px button height, keyboard focus outline, live status label, disabled state during loading.
- Reduced-motion preference disables animation and preserves status text.

Verified in a real Chromium browser: moving shimmer, fixed dimensions, repeat-click blocking, success/restart, error recovery, 320px width, reduced motion, Streamlit request matching, and changed-input reset. No clinical inference was simulated in the production dashboard; the timer belongs only to the demo.

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import { App } from './ui/App';
import { VisionTarget } from './ui/VisionTarget';

// ?view=target is the bare game page the Vision Bridge drives: canvas only, official input only.
const target = new URLSearchParams(window.location.search).get('view') === 'target';

createRoot(document.getElementById('root')!).render(
  <StrictMode>{target ? <VisionTarget /> : <App />}</StrictMode>,
);

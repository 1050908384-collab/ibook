import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import './styles.css';
import { GenerationProgressProvider } from './components/GenerationProgress';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <GenerationProgressProvider>
        <App />
      </GenerationProgressProvider>
    </BrowserRouter>
  </React.StrictMode>,
);

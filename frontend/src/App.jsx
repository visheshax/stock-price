import React, { useState } from "react";
import SearchInput from "./components/SearchInput";
import StockColumn from "./components/StockColumn";
import { TrendingUp, Calendar, AlertCircle } from "lucide-react";

// Dynamically determine the backend API URL (handles Render and dev)
const BACKEND_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [tickers, setTickers] = useState(["", "", ""]);
  const [labels, setLabels] = useState(["", "", ""]);
  const [targetDate, setTargetDate] = useState(() => {
    // Tomorrow formatted as YYYY-MM-DD
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    return tomorrow.toISOString().split("T")[0];
  });
  
  const [loadings, setLoadings] = useState([false, false, false]);
  const [results, setResults] = useState([null, null, null]);
  const [errors, setErrors] = useState([null, null, null]);

  // Handle stock selection
  const handleSelectStock = (index, symbol, label) => {
    const updatedTickers = [...tickers];
    const updatedLabels = [...labels];
    updatedTickers[index] = symbol;
    updatedLabels[index] = label;
    setTickers(updatedTickers);
    setLabels(updatedLabels);
    
    // Clear old result for this slot
    const updatedResults = [...results];
    const updatedErrors = [...errors];
    updatedResults[index] = null;
    updatedErrors[index] = null;
    setResults(updatedResults);
    setErrors(updatedErrors);
  };

  // Handle stock slot reset
  const handleClearStock = (index) => {
    const updatedTickers = [...tickers];
    const updatedLabels = [...labels];
    updatedTickers[index] = "";
    updatedLabels[index] = "";
    setTickers(updatedTickers);
    setLabels(updatedLabels);

    const updatedLoadings = [...loadings];
    const updatedResults = [...results];
    const updatedErrors = [...errors];
    updatedLoadings[index] = false;
    updatedResults[index] = null;
    updatedErrors[index] = null;
    setLoadings(updatedLoadings);
    setResults(updatedResults);
    setErrors(updatedErrors);
  };

  // Execute Parallel Concurrent Predictions
  const handlePredictAll = async () => {
    const activeIndices = tickers.map((t, idx) => t ? idx : -1).filter(idx => idx !== -1);
    if (activeIndices.length === 0) return;

    // Trigger loader state in parallel
    const initialLoadings = [...loadings];
    const initialErrors = [...errors];
    const initialResults = [...results];
    activeIndices.forEach(idx => {
      initialLoadings[idx] = true;
      initialErrors[idx] = null;
      initialResults[idx] = null;
    });
    setLoadings(initialLoadings);
    setErrors(initialErrors);
    setResults(initialResults);

    // Launch async fetch requests concurrently
    const predictionPromises = activeIndices.map(async (idx) => {
      const symbol = tickers[idx];
      try {
        const response = await fetch(`${BACKEND_URL}/api/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ticker: symbol, target_date: targetDate })
        });
        
        if (!response.ok) {
          const errMsg = await response.text();
          throw new Error(errMsg || "Internal server error");
        }
        
        const data = await response.json();
        
        // Update results state individually as they arrive
        setResults(prev => {
          const updated = [...prev];
          updated[idx] = data;
          return updated;
        });
      } catch (err) {
        console.error(`Error predicting ${symbol}:`, err);
        setErrors(prev => {
          const updated = [...prev];
          updated[idx] = err.message || "Failed to fetch predictions";
          return updated;
        });
      } finally {
        setLoadings(prev => {
          const updated = [...prev];
          updated[idx] = false;
          return updated;
        });
      }
    });

    // Wait for all processes to fully resolve concurrently
    await Promise.all(predictionPromises);
  };

  const hasActiveStocks = tickers.some(t => t !== "");

  return (
    <div className="min-h-screen px-4 py-8 sm:px-6 lg:px-8 max-w-7xl mx-auto flex flex-col justify-between">
      <div>
        {/* Header Block */}
        <header className="mb-8 text-center sm:text-left flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-900 pb-6">
          <div>
            <h1 className="text-2xl font-black bg-gradient-to-r from-indigo-400 to-emerald-400 bg-clip-text text-transparent flex items-center justify-center sm:justify-start gap-2.5">
              <TrendingUp size={28} className="text-indigo-400" />
              HYBRID STOCK PRICE PREDICTOR
            </h1>
            <p className="text-xs text-slate-500 mt-1 uppercase tracking-wider font-semibold">
              Facebook Prophet macro-trends + Gradient Boosting residual volatility
            </p>
          </div>
        </header>

        {/* Comparison Settings Dashboard */}
        <section className="glass-panel p-6 mb-8">
          <h2 className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-4">Comparison Grid Settings</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            {[0, 1, 2].map((idx) => (
              <div key={idx} className="bg-slate-950/40 border border-slate-900/60 rounded-xl p-4">
                <label className="text-[10px] text-slate-500 font-bold uppercase block mb-2">
                  Stock Slot {idx + 1} {idx > 0 && "(Optional)"}
                </label>
                <SearchInput
                  value={labels[idx] ? `${labels[idx]} (${tickers[idx]})` : ""}
                  index={idx + 1}
                  onSelect={(symbol, label) => handleSelectStock(idx, symbol, label)}
                  onClear={() => handleClearStock(idx)}
                  backendUrl={BACKEND_URL}
                />
              </div>
            ))}
          </div>

          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 border-t border-slate-950 pt-5">
            <div className="w-full sm:max-w-xs">
              <label className="text-[10px] text-slate-400 flex items-center gap-1.5 font-bold uppercase mb-2">
                <Calendar size={14} className="text-indigo-400" />
                Target Date (Forecast Horizon)
              </label>
              <input
                type="date"
                className="glass-input w-full py-2 px-3 text-xs"
                value={targetDate}
                min={new Date(Date.now() + 86400000).toISOString().split("T")[0]}
                onChange={(e) => setTargetDate(e.target.value)}
              />
            </div>

            <button
              onClick={handlePredictAll}
              disabled={!hasActiveStocks || loadings.some(l => l)}
              className="glass-button-primary w-full sm:w-auto uppercase tracking-wide text-xs"
            >
              {loadings.some(l => l) ? "Computing Forecasts..." : "Predict Prices"}
            </button>
          </div>
        </section>

        {/* Dynamic Grid Results */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[0, 1, 2].map((idx) => (
            <StockColumn
              key={idx}
              label={labels[idx] ? `${labels[idx]} (${tickers[idx]})` : `Stock Slot ${idx + 1}`}
              loading={loadings[idx]}
              data={results[idx]}
              error={errors[idx]}
              targetDate={targetDate}
            />
          ))}
        </section>
      </div>

      <footer className="text-center text-[10px] text-slate-600 mt-12 pt-6 border-t border-slate-900/40">
        <p className="uppercase tracking-wider font-semibold">Institutional Grade Machine Learning Analytics Engine</p>
        <p className="mt-1">Powered by React, Recharts, FastAPI, Facebook Prophet, and Scikit-Learn</p>
      </footer>
    </div>
  );
}

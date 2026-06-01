import React, { useState } from "react";
import SearchInput from "./components/SearchInput";
import StockColumn from "./components/StockColumn";
import { TrendingUp, Calendar } from "lucide-react";

// Dynamically determine the backend API URL (handles Render and dev)
const BACKEND_URL = import.meta.env.VITE_API_URL || "https://stock-price-gm21.onrender.com";

export default function App() {
  const [tickers, setTickers] = useState(["", "", ""]);
  const [labels, setLabels] = useState(["", "", ""]);
  const [targetDates, setTargetDates] = useState(() => {
    // Default tomorrow formatted as YYYY-MM-DD for each slot
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    const tomorrowStr = tomorrow.toISOString().split("T")[0];
    return [tomorrowStr, tomorrowStr, tomorrowStr];
  });
  
  const [loadings, setLoadings] = useState([false, false, false]);
  const [results, setResults] = useState([null, null, null]);
  const [errors, setErrors] = useState([null, null, null]);

  // Handle stock selection for a slot
  const handleSelectStock = (index, symbol, label) => {
    const updatedTickers = [...tickers];
    const updatedLabels = [...labels];
    updatedTickers[index] = symbol;
    updatedLabels[index] = label;
    setTickers(updatedTickers);
    setLabels(updatedLabels);
    
    // Clear old state for this slot
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

  // Update target date for a specific stock slot
  const handleUpdateDate = (index, dateStr) => {
    const updatedDates = [...targetDates];
    updatedDates[index] = dateStr;
    setTargetDates(updatedDates);
  };

  // Execute predictions for a single stock slot (completely decoupled)
  const handlePredictSlot = async (index) => {
    const symbol = tickers[index];
    const targetDate = targetDates[index];
    if (!symbol || !targetDate) return;

    // Set loading state for this index only
    setLoadings(prev => {
      const updated = [...prev];
      updated[index] = true;
      return updated;
    });
    setErrors(prev => {
      const updated = [...prev];
      updated[index] = null;
      return updated;
    });
    setResults(prev => {
      const updated = [...prev];
      updated[index] = null;
      return updated;
    });

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
      
      setResults(prev => {
        const updated = [...prev];
        updated[index] = data;
        return updated;
      });
    } catch (err) {
      console.error(`Error predicting slot ${index} (${symbol}):`, err);
      setErrors(prev => {
        const updated = [...prev];
        updated[index] = err.message || "Failed to fetch predictions";
        return updated;
      });
    } finally {
      setLoadings(prev => {
        const updated = [...prev];
        updated[index] = false;
        return updated;
      });
    }
  };

  return (
    <div className="min-h-screen px-4 py-8 sm:px-6 lg:px-8 max-w-7xl mx-auto flex flex-col justify-between">
      <div>
        {/* Header Block */}
        <header className="mb-8 text-center sm:text-left flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-200 pb-6">
          <div>
            <h1 className="text-2xl font-black bg-gradient-to-r from-indigo-600 to-emerald-600 bg-clip-text text-transparent flex items-center justify-center sm:justify-start gap-2.5">
              <TrendingUp size={28} className="text-indigo-600" />
              HYBRID STOCK PRICE PREDICTOR
            </h1>
            <p className="text-xs text-slate-500 mt-1 uppercase tracking-wider font-semibold">
              Proprietary Macro-Trend Modeling + Quantitative Residual Volatility Engine
            </p>
          </div>
        </header>

        {/* Dynamic Decoupled Grid Results */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[0, 1, 2].map((idx) => (
            <StockColumn
              key={idx}
              index={idx}
              ticker={tickers[idx]}
              label={labels[idx]}
              loading={loadings[idx]}
              data={results[idx]}
              error={errors[idx]}
              targetDate={targetDates[idx]}
              onSelect={(symbol, label) => handleSelectStock(idx, symbol, label)}
              onClear={() => handleClearStock(idx)}
              onUpdateDate={(dateStr) => handleUpdateDate(idx, dateStr)}
              onPredict={() => handlePredictSlot(idx)}
              backendUrl={BACKEND_URL}
            />
          ))}
        </section>
      </div>
    </div>
  );
}

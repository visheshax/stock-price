import React, { useState } from "react";
import ForecastChart from "./ForecastChart";
import { TrendingUp, TrendingDown, Target, HelpCircle, Calendar, MessageSquare } from "lucide-react";

export default function StockColumn({ label, data, loading, error, targetDate }) {
  const [goalPrice, setGoalPrice] = useState("");
  const [seekResult, setSeekResult] = useState(null);

  // Run the client-side Goal Seek logic using the forecasted dataset
  const handleGoalSeek = () => {
    if (!goalPrice || !data || !data.chart_data) return;
    const target = parseFloat(goalPrice);
    if (isNaN(target)) return;

    const forecastPts = data.chart_data.filter(pt => pt.type === "Forecast");
    const lastPrice = data.last_price;

    let hitRow = null;
    if (target > lastPrice) {
      hitRow = forecastPts.find(pt => pt.price >= target);
    } else {
      hitRow = forecastPts.find(pt => pt.price <= target);
    }

    if (hitRow) {
      const hitDate = new Date(hitRow.date);
      const daysToHit = Math.ceil((hitDate - new Date()) / (1000 * 60 * 60 * 24));
      const yearsToHit = (daysToHit / 365).toFixed(1);
      
      setSeekResult({
        success: true,
        message: `Projected to hit $${target.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})} on ${hitDate.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })} (${yearsToHit} years).`
      });
    } else {
      setSeekResult({
        success: false,
        message: "Not projected to hit this price within the 20-year macro forecast horizon."
      });
    }
  };

  // 1. Loading State (Skeletal pulsing screen for parallel fetch feel)
  if (loading) {
    return (
      <div className="glass-panel p-6 flex flex-col h-[650px] animate-pulse">
        <div className="h-6 w-32 bg-slate-800 rounded-lg mb-8"></div>
        <div className="grid grid-cols-3 gap-4 mb-8">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-16 bg-slate-800 rounded-xl"></div>
          ))}
        </div>
        <div className="h-[220px] bg-slate-800 rounded-xl mb-8"></div>
        <div className="h-16 bg-slate-800 rounded-xl mt-auto"></div>
      </div>
    );
  }

  // 2. Empty State
  if (!data && !error) {
    return (
      <div className="glass-panel p-6 flex flex-col items-center justify-center h-[650px] border-dashed border-slate-800 text-center">
        <HelpCircle size={40} className="text-slate-600 mb-3" />
        <p className="text-slate-400 font-medium text-sm">Waiting for prediction triggers</p>
        <p className="text-xs text-slate-600 max-w-[200px] mt-1">Select a symbol above and hit predict to run simulations</p>
      </div>
    );
  }

  // 3. Error State
  if (error) {
    return (
      <div className="glass-panel p-6 flex flex-col items-center justify-center h-[650px] border-rose-950 text-center">
        <div className="bg-rose-950/50 p-3 rounded-full text-rose-500 mb-3 border border-rose-900/50">
          <TrendingDown size={30} />
        </div>
        <p className="text-rose-400 font-semibold text-sm">Error Analyzing Stock</p>
        <p className="text-xs text-slate-500 max-w-[250px] mt-1.5 line-clamp-3">{error}</p>
      </div>
    );
  }

  const isUp = data.projected_move_pct >= 0;
  const sentiment = data.qualitative_context?.sentiment_score || 0;

  return (
    <div className="glass-panel p-6 flex flex-col justify-between h-[680px]">
      <div>
        {/* Title */}
        <h3 className="text-lg font-bold text-slate-100 truncate mb-6">{label}</h3>

        {/* Dynamic Metric Grid */}
        <div className="grid grid-cols-3 gap-2 text-center mb-6">
          <div className="bg-slate-950/40 border border-slate-900 rounded-xl p-2.5">
            <span className="text-[10px] text-slate-500 block uppercase font-semibold">Last Close</span>
            <span className="text-sm font-bold text-slate-200 block mt-0.5">${data.last_price.toFixed(2)}</span>
            <span className="text-[9px] text-slate-500 block mt-0.5 truncate">{data.last_date}</span>
          </div>
          
          <div className="bg-slate-950/40 border border-slate-900 rounded-xl p-2.5">
            <span className="text-[10px] text-slate-500 block uppercase font-semibold">Prediction</span>
            <span className="text-sm font-bold text-slate-200 block mt-0.5">${data.predicted_price.toFixed(2)}</span>
            <span className="text-[9px] text-slate-500 block mt-0.5 truncate">{targetDate}</span>
          </div>

          <div className={`bg-slate-950/40 border rounded-xl p-2.5 ${isUp ? 'border-emerald-950/30' : 'border-rose-950/30'}`}>
            <span className="text-[10px] text-slate-500 block uppercase font-semibold">Projected Move</span>
            <span className={`text-sm font-bold block mt-0.5 ${isUp ? 'text-emerald-400' : 'text-rose-400'}`}>
              {isUp ? '+' : ''}{data.projected_move_val.toFixed(2)}
            </span>
            <span className={`text-[10px] font-medium block mt-0.5 ${isUp ? 'text-emerald-500' : 'text-rose-500'}`}>
              {isUp ? '+' : ''}{data.projected_move_pct.toFixed(2)}%
            </span>
          </div>
        </div>

        {/* Qualitative AI Block */}
        {data.qualitative_context && data.qualitative_context.news_count > 0 && (
          <div className="bg-indigo-950/15 border border-indigo-900/30 rounded-xl p-3 text-[11px] mb-6 flex items-start gap-2.5">
            <MessageSquare size={16} className="text-indigo-400 shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="font-semibold text-indigo-300">🤖 AI Qualitative Context</p>
              <p className="text-slate-400 mt-0.5 leading-relaxed">
                Sentiment: <span className="text-slate-200 font-medium">{sentiment >= 0 ? '+' : ''}{sentiment.toFixed(2)}</span> | 
                Margin: <span className="text-slate-200 font-medium">{data.qualitative_context.profit_margins}</span> | 
                Rev: <span className="text-slate-200 font-medium">{data.qualitative_context.revenue_growth}</span>
              </p>
            </div>
          </div>
        )}

        {/* Visualization */}
        <ForecastChart chartData={data.chart_data} />
      </div>

      {/* Goal Seek Interactive Expandable */}
      <div className="border-t border-slate-900 pt-5 mt-4">
        <h4 className="text-[11px] text-slate-400 flex items-center gap-1.5 font-bold uppercase mb-2">
          <Target size={14} className="text-indigo-400" />
          Goal Seek (Reverse Forecast)
        </h4>
        <div className="flex gap-2">
          <input
            type="number"
            className="glass-input flex-1 py-1.5 px-3 text-xs"
            placeholder="Target Price..."
            value={goalPrice}
            onChange={(e) => setGoalPrice(e.target.value)}
          />
          <button 
            onClick={handleGoalSeek}
            className="glass-button-secondary py-1.5 px-3 text-xs hover:bg-slate-700 active:scale-95"
          >
            Calculate
          </button>
        </div>

        {seekResult && (
          <div className={`mt-3 p-2.5 rounded-lg text-[10px] leading-normal border ${
            seekResult.success 
              ? 'bg-emerald-950/20 border-emerald-900/30 text-emerald-400' 
              : 'bg-amber-950/20 border-amber-900/30 text-amber-400'
          }`}>
            {seekResult.message}
          </div>
        )}
      </div>
    </div>
  );
}

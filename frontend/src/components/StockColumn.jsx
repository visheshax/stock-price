import React, { useState } from "react";
import SearchInput from "./SearchInput";
import ForecastChart from "./ForecastChart";
import { TrendingUp, TrendingDown, Target, HelpCircle, Calendar, MessageSquare, Loader2 } from "lucide-react";

export default function StockColumn({
  index,
  ticker,
  label,
  loading,
  data,
  error,
  targetDate,
  onSelect,
  onClear,
  onUpdateDate,
  onPredict,
  backendUrl
}) {
  const [goalPrice, setGoalPrice] = useState("");
  const [seekResult, setSeekResult] = useState(null);

  // Run client-side Goal Seek scan
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
        message: "Not projected to hit this price within the 10-year macro forecast horizon."
      });
    }
  };

  const isUp = data?.projected_move_pct >= 0;
  const sentiment = data?.qualitative_context?.sentiment_score || 0;

  return (
    <div className="glass-panel p-6 flex flex-col justify-between min-h-[720px] transition-all duration-300">
      <div className="flex flex-col flex-1">
        {/* Column Header & Search Autocomplete */}
        <div className="mb-6">
          <label className="text-[10px] text-slate-500 font-bold uppercase block mb-2">
            Stock Column {index + 1} {index > 0 && "(Optional)"}
          </label>
          <SearchInput
            value={label ? `${label} (${ticker})` : ""}
            index={index + 1}
            onSelect={onSelect}
            onClear={onClear}
            backendUrl={backendUrl}
          />
        </div>

        {/* Decoder body state based on whether a stock is active */}
        {!ticker ? (
          // 1. Empty State (No stock selected)
          <div className="flex flex-col items-center justify-center flex-1 min-h-[450px] border-2 border-dashed border-slate-200 rounded-2xl text-center p-6 bg-slate-50/20">
            <HelpCircle size={36} className="text-slate-400 mb-3 animate-pulse" />
            <p className="text-slate-600 font-semibold text-sm">Select Stock Symbol</p>
            <p className="text-xs text-slate-500 max-w-[200px] mt-1 leading-normal">
              Search for a company or symbol in the search bar above to begin predictions.
            </p>
          </div>
        ) : (
          <div className="flex flex-col flex-1 justify-between">
            {/* Decoupled Date Selector & Prediction Button Row (Visible when stock is selected) */}
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-3.5 flex items-center justify-between gap-3 mb-6">
              <div className="flex-1 flex gap-2">
                <div className="flex-1">
                  <label className="text-[9px] text-slate-500 flex items-center gap-1 font-bold uppercase mb-1">
                    <Calendar size={12} className="text-indigo-600" />
                    Month
                  </label>
                  <select
                    className="glass-input w-full py-1 px-2 text-xs text-slate-900 border-slate-200/80 bg-white cursor-pointer focus:outline-none"
                    value={(() => {
                      const d = new Date(targetDate);
                      return isNaN(d.getTime()) ? new Date().getMonth() : d.getMonth();
                    })()}
                    onChange={(e) => {
                      const newMonth = parseInt(e.target.value);
                      const d = new Date(targetDate);
                      const year = isNaN(d.getTime()) ? new Date().getFullYear() + 1 : d.getFullYear();
                      
                      const lastDay = new Date(year, newMonth + 1, 0).getDate();
                      const monthStr = String(newMonth + 1).padStart(2, "0");
                      const dayStr = String(lastDay).padStart(2, "0");
                      onUpdateDate(`${year}-${monthStr}-${dayStr}`);
                    }}
                    disabled={loading}
                  >
                    {["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].map((m, idx) => (
                      <option key={m} value={idx}>{m}</option>
                    ))}
                  </select>
                </div>

                <div className="w-[80px]">
                  <label className="text-[9px] text-slate-500 flex items-center gap-1 font-bold uppercase mb-1">
                    Year
                  </label>
                  <select
                    className="glass-input w-full py-1 px-2 text-xs text-slate-900 border-slate-200/80 bg-white cursor-pointer focus:outline-none"
                    value={(() => {
                      const d = new Date(targetDate);
                      return isNaN(d.getTime()) ? new Date().getFullYear() + 1 : d.getFullYear();
                    })()}
                    onChange={(e) => {
                      const newYear = parseInt(e.target.value);
                      const d = new Date(targetDate);
                      const month = isNaN(d.getTime()) ? new Date().getMonth() : d.getMonth();
                      
                      const lastDay = new Date(newYear, month + 1, 0).getDate();
                      const monthStr = String(month + 1).padStart(2, "0");
                      const dayStr = String(lastDay).padStart(2, "0");
                      onUpdateDate(`${newYear}-${monthStr}-${dayStr}`);
                    }}
                    disabled={loading}
                  >
                    {Array.from({ length: 11 }, (_, i) => new Date().getFullYear() + i).map((y) => (
                      <option key={y} value={y}>{y}</option>
                    ))}
                  </select>
                </div>
              </div>
              <button
                onClick={onPredict}
                disabled={loading}
                className="glass-button-primary py-2 px-4 text-xs tracking-wider uppercase h-[34px] flex items-center justify-center gap-1.5 self-end cursor-pointer disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <Loader2 size={12} className="animate-spin" />
                    Fitting...
                  </>
                ) : (
                  "Predict"
                )}
              </button>
            </div>

            {/* Result body conditional on load status */}
            {loading ? (
              // 2. Loading Skeletal Pulsing
              <div className="flex flex-col flex-1 justify-between min-h-[380px] animate-pulse">
                <div className="grid grid-cols-3 gap-3 mb-6">
                  {[1, 2, 3].map(i => (
                    <div key={i} className="h-16 bg-slate-200 rounded-xl"></div>
                  ))}
                </div>
                <div className="h-[200px] bg-slate-200 rounded-xl mb-6"></div>
                <div className="h-14 bg-slate-200 rounded-xl mt-auto"></div>
              </div>
            ) : error ? (
              // 3. Error Card
              <div className="flex flex-col items-center justify-center flex-1 min-h-[380px] border border-rose-200 rounded-2xl text-center p-6 bg-rose-50/10">
                <div className="bg-rose-50 p-2.5 rounded-full text-rose-600 mb-3 border border-rose-200">
                  <TrendingDown size={24} />
                </div>
                <p className="text-rose-600 font-bold text-sm">Error Analyzing Stock</p>
                <p className="text-xs text-slate-500 max-w-[220px] mt-1.5 leading-normal line-clamp-3">{error}</p>
              </div>
            ) : data ? (
              // 4. Clean Results Grid
              <div className="flex flex-col flex-1 justify-between">
                <div>
                  {/* Dynamic Metrics */}
                  <div className="grid grid-cols-3 gap-2 text-center mb-6 animate-fadeIn">
                    <div className="bg-slate-50 border border-slate-200 rounded-xl p-2">
                      <span className="text-[9px] text-slate-500 block uppercase font-bold">Last Close</span>
                      <span className="text-xs font-bold text-slate-900 block mt-0.5">${data.last_price.toFixed(2)}</span>
                      <span className="text-[8px] text-slate-500 block mt-0.5 truncate">{data.last_date}</span>
                    </div>
                    
                    <div className="bg-slate-50 border border-slate-200 rounded-xl p-2">
                      <span className="text-[9px] text-slate-500 block uppercase font-bold">Prediction</span>
                      <span className="text-xs font-bold text-slate-900 block mt-0.5">${data.predicted_price.toFixed(2)}</span>
                      <span className="text-[8px] text-slate-500 block mt-0.5 truncate">{targetDate}</span>
                    </div>

                    <div className={`bg-slate-50 border rounded-xl p-2 ${isUp ? 'border-emerald-200' : 'border-rose-200'}`}>
                      <span className="text-[9px] text-slate-500 block uppercase font-bold">Move</span>
                      <span className={`text-xs font-black block mt-0.5 ${isUp ? 'text-emerald-600' : 'text-rose-600'}`}>
                        {isUp ? '+' : ''}{data.projected_move_val.toFixed(2)}
                      </span>
                      <span className={`text-[9px] font-bold block mt-0.5 ${isUp ? 'text-emerald-700' : 'text-rose-700'}`}>
                        {isUp ? '+' : ''}{data.projected_move_pct.toFixed(2)}%
                      </span>
                    </div>
                  </div>

                  {/* AI qualitative context block */}
                  {data.qualitative_context && data.qualitative_context.news_count > 0 && (
                    <div className="bg-indigo-50/50 border border-indigo-200/60 rounded-xl p-3 text-[11px] mb-6 flex items-start gap-2.5 animate-fadeIn">
                      <MessageSquare size={16} className="text-indigo-600 shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <p className="font-bold text-indigo-900">💼 Analyst Context & Sentiment</p>
                        <p className="text-slate-700 mt-0.5 leading-relaxed">
                          Sentiment: <span className="text-slate-900 font-semibold">{sentiment >= 0 ? '+' : ''}{sentiment.toFixed(2)}</span> | 
                          Margin: <span className="text-slate-900 font-semibold">{data.qualitative_context.profit_margins}</span> | 
                          Rev: <span className="text-slate-900 font-semibold">{data.qualitative_context.revenue_growth}</span>
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Chart Renderer */}
                  <ForecastChart chartData={data.chart_data} />
                </div>

                {/* Goal Seek Scanner */}
                <div className="border-t border-slate-200 pt-5 mt-4">
                  <h4 className="text-[11px] text-slate-600 flex items-center gap-1.5 font-bold uppercase mb-2">
                    <Target size={14} className="text-indigo-600" />
                    Goal Seek (Reverse Forecast)
                  </h4>
                  <div className="flex gap-2">
                    <input
                      type="number"
                      className="glass-input flex-1 py-1.5 px-3 text-xs bg-white border-slate-200"
                      placeholder="Target Price..."
                      value={goalPrice}
                      onChange={(e) => setGoalPrice(e.target.value)}
                    />
                    <button 
                      onClick={handleGoalSeek}
                      className="glass-button-secondary py-1.5 px-3 text-xs hover:bg-slate-200 active:scale-95 cursor-pointer"
                    >
                      Calculate
                    </button>
                  </div>

                  {seekResult && (
                    <div className={`mt-3 p-2.5 rounded-lg text-[10px] leading-normal border ${
                      seekResult.success 
                        ? 'bg-emerald-50 border-emerald-200 text-emerald-700 font-bold' 
                        : 'bg-amber-50 border-amber-200 text-amber-700 font-bold'
                    }`}>
                      {seekResult.message}
                    </div>
                  )}
                </div>
              </div>
            ) : (
              // 5. Active Stock but not run yet
              <div className="flex flex-col items-center justify-center flex-1 min-h-[380px] border border-dashed border-slate-200 rounded-2xl text-center p-6 bg-slate-50/10">
                <HelpCircle size={32} className="text-slate-400 mb-2.5" />
                <p className="text-slate-600 font-semibold text-xs">Awaiting Prediction</p>
                <p className="text-[10px] text-slate-500 max-w-[180px] mt-1 leading-normal">
                  Select a forecast horizon date above and click predict to launch hybrid simulations.
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

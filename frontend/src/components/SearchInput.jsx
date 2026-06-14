import React, { useState, useEffect, useRef } from "react";
import { Search, X, RefreshCw, Check, Globe } from "lucide-react";

// Cached list of popular stocks mapped by country
const STOCKS_DATABASE = {
  US: [
    { symbol: "AAPL", label: "Apple Inc. (AAPL)" },
    { symbol: "MSFT", label: "Microsoft Corp. (MSFT)" },
    { symbol: "GOOG", label: "Alphabet Inc. (GOOG)" },
    { symbol: "AMZN", label: "Amazon.com Inc. (AMZN)" },
    { symbol: "NVDA", label: "NVIDIA Corp. (NVDA)" },
    { symbol: "META", label: "Meta Platforms (META)" },
    { symbol: "TSLA", label: "Tesla Inc. (TSLA)" },
    { symbol: "JPM", label: "JPMorgan Chase & Co. (JPM)" },
    { symbol: "AVGO", label: "Broadcom Inc. (AVGO)" },
    { symbol: "IBM", label: "IBM Corp. (IBM)" },
    { symbol: "V", label: "Visa Inc. (V)" },
    { symbol: "MA", label: "Mastercard Inc. (MA)" },
    { symbol: "COST", label: "Costco Wholesale (COST)" },
    { symbol: "LLY", label: "Eli Lilly & Co. (LLY)" },
    { symbol: "XOM", label: "Exxon Mobil Corp. (XOM)" },
    { symbol: "WMT", label: "Walmart Inc. (WMT)" },
    { symbol: "JNJ", label: "Johnson & Johnson (JNJ)" },
    { symbol: "PG", label: "Procter & Gamble (PG)" },
    { symbol: "NFLX", label: "Netflix Inc. (NFLX)" },
    { symbol: "AMD", label: "Advanced Micro Devices (AMD)" },
  ],
  IN: [
    { symbol: "RELIANCE.NS", label: "Reliance Industries (RELIANCE)" },
    { symbol: "TCS.NS", label: "Tata Consultancy Services (TCS)" },
    { symbol: "HDFCBANK.NS", label: "HDFC Bank (HDFCBANK)" },
    { symbol: "INFY.NS", label: "Infosys (INFY)" },
    { symbol: "ICICIBANK.NS", label: "ICICI Bank (ICICIBANK)" },
    { symbol: "SBIN.NS", label: "State Bank of India (SBIN)" },
    { symbol: "BHARTIARTL.NS", label: "Bharti Airtel (BHARTIARTL)" },
    { symbol: "ITC.NS", label: "ITC Limited (ITC)" },
    { symbol: "LT.NS", label: "Larsen & Toubro (LT)" },
    { symbol: "ASIANPAINT.NS", label: "Asian Paints (ASIANPAINT)" },
    { symbol: "ASHOKLEY.NS", label: "Ashok Leyland (ASHOKLEY)" },
    { symbol: "HINDUNILVR.NS", label: "Hindustan Unilever (HINDUNILVR)" },
    { symbol: "WIPRO.NS", label: "Wipro (WIPRO)" },
    { symbol: "TATAMOTORS.NS", label: "Tata Motors (TATAMOTORS)" },
    { symbol: "AXISBANK.NS", label: "Axis Bank (AXISBANK)" },
    { symbol: "MARUTI.NS", label: "Maruti Suzuki (MARUTI)" },
    { symbol: "BAJFINANCE.NS", label: "Bajaj Finance (BAJFINANCE)" },
    { symbol: "TITAN.NS", label: "Titan Company (TITAN)" },
  ]
};

export default function SearchInput({ value, onSelect, onClear, index }) {
  const [selectedCountry, setSelectedCountry] = useState("US");
  const [query, setQuery] = useState("");
  const [showDropdown, setShowDropdown] = useState(false);
  const [isCustomMode, setIsCustomMode] = useState(false);
  const [customTicker, setCustomTicker] = useState("");
  const [customLabel, setCustomLabel] = useState("");
  
  // Simulated refresh state
  const [refreshing, setRefreshing] = useState(false);
  const [refreshSuccess, setRefreshSuccess] = useState(false);
  
  const dropdownRef = useRef(null);

  // Click outside to close dropdown
  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Filter stocks locally
  const localStocks = STOCKS_DATABASE[selectedCountry] || [];
  const filteredStocks = localStocks.filter(item => 
    item.label.toLowerCase().includes(query.toLowerCase()) ||
    item.symbol.toLowerCase().includes(query.toLowerCase())
  );

  const handleSelectStock = (item) => {
    onSelect(item.symbol, item.label.split(" (")[0]);
    setQuery("");
    setShowDropdown(false);
  };

  const handleCustomSubmit = (e) => {
    e.preventDefault();
    if (!customTicker.trim()) return;
    const cleanTicker = customTicker.trim().toUpperCase();
    const cleanLabel = customLabel.trim() || cleanTicker;
    onSelect(cleanTicker, cleanLabel);
    setCustomTicker("");
    setCustomLabel("");
    setIsCustomMode(false);
  };

  const triggerRefresh = () => {
    setRefreshing(true);
    setTimeout(() => {
      setRefreshing(false);
      setRefreshSuccess(true);
      setTimeout(() => setRefreshSuccess(false), 2000);
    }, 1000);
  };

  return (
    <div className="w-full" ref={dropdownRef}>
      {value ? (
        <div className="flex items-center justify-between glass-input w-full border-slate-200 bg-white text-slate-900 text-sm">
          <span className="font-bold text-emerald-600 truncate mr-2">{value}</span>
          <button 
            onClick={() => {
              onClear();
              setIsCustomMode(false);
            }} 
            className="text-slate-500 hover:text-rose-600 p-0.5 rounded-md hover:bg-slate-100 transition-colors cursor-pointer"
          >
            <X size={16} />
          </button>
        </div>
      ) : isCustomMode ? (
        // Custom Ticker Submission Form
        <form onSubmit={handleCustomSubmit} className="flex flex-col gap-2 bg-slate-50 border border-slate-200 rounded-xl p-3">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold uppercase text-slate-500">Custom Ticker</span>
            <button 
              type="button"
              onClick={() => setIsCustomMode(false)}
              className="text-slate-400 hover:text-slate-600 text-xs font-semibold cursor-pointer"
            >
              Cancel
            </button>
          </div>
          <div className="flex gap-2">
            <input 
              type="text" 
              placeholder="e.g. JPM, RELIANCE.NS"
              value={customTicker}
              onChange={(e) => setCustomTicker(e.target.value)}
              className="glass-input text-xs py-1 px-2 flex-1"
              required
            />
            <input 
              type="text" 
              placeholder="Company Name"
              value={customLabel}
              onChange={(e) => setCustomLabel(e.target.value)}
              className="glass-input text-xs py-1 px-2 flex-1"
            />
            <button 
              type="submit"
              className="glass-button-primary text-xs py-1 px-3"
            >
              Add
            </button>
          </div>
        </form>
      ) : (
        // Country Select + Local Search Input
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            {/* Country Selector */}
            <div className="flex-1 flex gap-1 items-center bg-slate-50 border border-slate-200 rounded-lg p-1.5 text-xs text-slate-700">
              <Globe size={13} className="text-slate-400" />
              <select 
                value={selectedCountry}
                onChange={(e) => {
                  setSelectedCountry(e.target.value);
                  setQuery("");
                }}
                className="bg-transparent border-0 font-semibold text-slate-800 focus:outline-none cursor-pointer flex-1"
              >
                <option value="US">🇺🇸 United States</option>
                <option value="IN">🇮🇳 India (NSE)</option>
              </select>
            </div>

            {/* Refresh Cached Tickers Button */}
            <button
              onClick={triggerRefresh}
              type="button"
              className="p-2 border border-slate-200 hover:border-indigo-500 rounded-lg hover:bg-slate-50 text-slate-400 hover:text-indigo-600 transition-all cursor-pointer flex items-center justify-center h-[34px] w-[34px]"
              title="Refresh cached stocks list (once in 6 months)"
            >
              {refreshing ? (
                <RefreshCw size={14} className="animate-spin text-indigo-600" />
              ) : refreshSuccess ? (
                <Check size={14} className="text-emerald-500" />
              ) : (
                <RefreshCw size={14} />
              )}
            </button>
          </div>

          {/* Local Text Search & Dropdown */}
          <div className="relative">
            <input
              type="text"
              className="glass-input w-full text-xs pl-8 pr-10 bg-white"
              placeholder="Search local stocks..."
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setShowDropdown(true);
              }}
              onFocus={() => setShowDropdown(true)}
            />
            <div className="absolute left-3 top-2.5 text-slate-400">
              <Search size={13} />
            </div>

            {/* Dropdown Results */}
            {showDropdown && (
              <div className="absolute z-50 left-0 right-0 mt-1.5 bg-white/95 backdrop-blur-xl border border-slate-200 rounded-xl max-h-60 overflow-y-auto shadow-xl">
                {filteredStocks.map((item) => (
                  <button
                    key={item.symbol}
                    onClick={() => handleSelectStock(item)}
                    className="w-full text-left px-4 py-2 text-[11px] hover:bg-slate-50 hover:text-indigo-700 border-b border-slate-100 last:border-b-0 truncate transition-colors font-medium text-slate-700 cursor-pointer"
                  >
                    {item.label}
                  </button>
                ))}
                {filteredStocks.length === 0 && (
                  <div className="px-4 py-2 text-[11px] text-slate-400 text-center">
                    No matching local stocks
                  </div>
                )}
                {/* Custom Ticker option */}
                <button
                  type="button"
                  onClick={() => {
                    setIsCustomMode(true);
                    setShowDropdown(false);
                  }}
                  className="w-full text-center px-4 py-2 text-[11px] text-indigo-600 hover:bg-indigo-50 font-bold border-t border-indigo-100 cursor-pointer"
                >
                  + Enter Custom Ticker...
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

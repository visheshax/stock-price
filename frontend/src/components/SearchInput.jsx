import React, { useState, useEffect, useRef } from "react";
import { Search, X, Loader2 } from "lucide-react";

export default function SearchInput({ value, onSelect, onClear, index, backendUrl }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef(null);

  // Debouncing logic for fuzzy search
  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      return;
    }

    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const response = await fetch(`${backendUrl}/api/search?q=${encodeURIComponent(query)}`);
        if (response.ok) {
          const data = await response.json();
          setResults(data);
        }
      } catch (err) {
        console.error("Search API error:", err);
      } finally {
        setLoading(false);
      }
    }, 300); // 300ms debounce

    return () => clearTimeout(timer);
  }, [query, backendUrl]);

  // Click outside listener to close dropdown
  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelectResult = (item) => {
    onSelect(item.symbol, item.label.split(" (")[0]);
    setQuery("");
    setResults([]);
    setShowDropdown(false);
  };

  return (
    <div className="relative w-full" ref={dropdownRef}>
      {value ? (
        <div className="flex items-center justify-between glass-input w-full border-slate-700 bg-slate-900/40 text-sm">
          <span className="font-semibold text-emerald-400 truncate mr-2">{value}</span>
          <button 
            onClick={onClear} 
            className="text-slate-400 hover:text-rose-400 p-0.5 rounded-md hover:bg-slate-800/80 transition-colors"
          >
            <X size={16} />
          </button>
        </div>
      ) : (
        <div className="relative">
          <input
            type="text"
            className="glass-input w-full text-sm pl-10 pr-10"
            placeholder={`Search Company ${index}...`}
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setShowDropdown(true);
            }}
            onFocus={() => setShowDropdown(true)}
          />
          <div className="absolute left-3.5 top-2.5 text-slate-500">
            <Search size={16} />
          </div>
          {loading && (
            <div className="absolute right-3.5 top-2.5 text-slate-500 animate-spin">
              <Loader2 size={16} />
            </div>
          )}
          
          {showDropdown && results.length > 0 && (
            <div className="absolute z-50 left-0 right-0 mt-2 bg-slate-950/95 backdrop-blur-xl border border-slate-800 rounded-xl max-h-60 overflow-y-auto shadow-2xl">
              {results.map((item) => (
                <button
                  key={item.symbol}
                  onClick={() => handleSelectResult(item)}
                  className="w-full text-left px-4 py-2.5 text-xs hover:bg-indigo-600/20 hover:text-indigo-200 border-b border-slate-900/50 last:border-b-0 truncate transition-colors"
                >
                  {item.label}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

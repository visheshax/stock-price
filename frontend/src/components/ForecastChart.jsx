import React from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

export default function ForecastChart({ chartData }) {
  if (!chartData || chartData.length === 0) return null;

  // Stitch them seamlessly:
  // We want to create two fields for each point so Recharts draws two connecting lines of different colors
  let lastHistPt = null;
  const processedData = chartData.map((pt) => {
    const isHist = pt.type === "Historical";
    if (isHist) {
      lastHistPt = pt;
    }
    return {
      date: pt.date,
      Historical: isHist ? pt.price : null,
      Forecast: !isHist ? pt.price : null,
      ForecastUpper: !isHist ? pt.price_upper : null,
      ForecastLower: !isHist ? pt.price_lower : null
    };
  });

  // Inject the last historical point as the starting forecast point to stitch them together seamlessly
  if (lastHistPt) {
    const firstForecastIdx = processedData.findIndex(p => p.Forecast !== null);
    if (firstForecastIdx !== -1) {
      processedData[firstForecastIdx].Historical = lastHistPt.price;
      processedData[firstForecastIdx].ForecastUpper = lastHistPt.price;
      processedData[firstForecastIdx].ForecastLower = lastHistPt.price;
    }
  }

  // Custom tooltips for a premium look
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const isForecast = data.Forecast !== null;
      
      return (
        <div className="bg-slate-950/95 backdrop-blur-md border border-slate-800 p-3 rounded-xl shadow-2xl text-[11px]">
          <p className="text-slate-400 font-medium mb-1.5">{data.date}</p>
          {!isForecast ? (
            <p className="font-semibold flex items-center gap-1.5 text-white">
              <span className="w-2 h-2 rounded-full bg-indigo-500"></span>
              Historical: <span className="font-bold">${data.Historical.toFixed(2)}</span>
            </p>
          ) : (
            <div className="space-y-1 text-slate-300">
              <p className="flex items-center justify-between gap-4">
                <span className="flex items-center gap-1.5 text-emerald-400 font-medium">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                  Best Case:
                </span>
                <span className="font-bold text-white">${data.ForecastUpper?.toFixed(2)}</span>
              </p>
              <p className="flex items-center justify-between gap-4">
                <span className="flex items-center gap-1.5 text-slate-300 font-medium">
                  <span className="w-1.5 h-1.5 rounded-full bg-slate-300"></span>
                  Expected:
                </span>
                <span className="font-bold text-white">${data.Forecast?.toFixed(2)}</span>
              </p>
              <p className="flex items-center justify-between gap-4">
                <span className="flex items-center gap-1.5 text-rose-400 font-medium">
                  <span className="w-1.5 h-1.5 rounded-full bg-rose-400"></span>
                  Worst Case:
                </span>
                <span className="font-bold text-white">${data.ForecastLower?.toFixed(2)}</span>
              </p>
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-[220px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={processedData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity={0.3} vertical={false} />
          <XAxis 
            dataKey="date" 
            tick={false} 
            axisLine={false} 
            stroke="#64748b" 
          />
          <YAxis 
            domain={["auto", "auto"]} 
            tick={{ fill: "#64748b", fontSize: 10 }} 
            axisLine={false} 
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line
            type="monotone"
            dataKey="Historical"
            stroke="#6366f1"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, strokeWidth: 0 }}
          />
          <Line
            type="monotone"
            dataKey="Forecast"
            stroke="#10b981"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, strokeWidth: 0 }}
          />
          <Line
            type="monotone"
            dataKey="ForecastUpper"
            stroke="#10b981"
            strokeWidth={1.5}
            strokeDasharray="3 3"
            dot={false}
            opacity={0.4}
          />
          <Line
            type="monotone"
            dataKey="ForecastLower"
            stroke="#ef4444"
            strokeWidth={1.5}
            strokeDasharray="3 3"
            dot={false}
            opacity={0.4}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

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
      Forecast: !isHist ? pt.price : null
    };
  });

  // Inject the last historical point as the starting forecast point to stitch them together seamlessly
  if (lastHistPt) {
    const firstForecastIdx = processedData.findIndex(p => p.Forecast !== null);
    if (firstForecastIdx !== -1) {
      processedData[firstForecastIdx].Historical = lastHistPt.price;
    }
  }

  // Custom tooltips for a premium look
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const val = payload[0].value || payload[1]?.value;
      const type = payload[0].value ? "Historical" : "Forecast";
      return (
        <div className="bg-slate-950/95 backdrop-blur-md border border-slate-800 p-3 rounded-xl shadow-2xl text-xs">
          <p className="text-slate-400 font-medium mb-1">{data.date}</p>
          <p className="font-semibold flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${type === "Historical" ? "bg-indigo-500" : "bg-emerald-400"}`}></span>
            {type}: <span className="text-white">${val.toFixed(2)}</span>
          </p>
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
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

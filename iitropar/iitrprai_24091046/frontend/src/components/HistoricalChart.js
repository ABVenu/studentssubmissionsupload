import React from 'react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

const HistoricalChart = ({ data }) => {
  const formattedData = data.map((item, index) => ({
    ...item,
    date: new Date(item.date).toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric' 
    }),
    index
  }));

  // Show only every 10th label to avoid crowding
  const tickFormatter = (value, index) => {
    return index % 10 === 0 ? value : '';
  };

  return (
    <ResponsiveContainer width="100%" height={400}>
      <AreaChart data={formattedData} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
        <defs>
          <linearGradient id="historicalGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#10f2c5" stopOpacity={0.8}/>
            <stop offset="95%" stopColor="#10f2c5" stopOpacity={0.1}/>
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge> 
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.2)" />
        <XAxis 
          dataKey="date" 
          stroke="white" 
          fontSize={11}
          tick={{ fill: 'white' }}
          tickFormatter={tickFormatter}
          interval="preserveStartEnd"
        />
        <YAxis 
          stroke="white" 
          fontSize={11}
          tick={{ fill: 'white' }}
          domain={['dataMin - 50', 'dataMax + 50']}
          tickFormatter={(value) => `₹${value.toFixed(0)}`}
        />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: 'rgba(0,0,0,0.9)', 
            border: '1px solid rgba(16,242,197,0.5)',
            borderRadius: '12px',
            color: 'white',
            boxShadow: '0 10px 25px rgba(0,0,0,0.3)'
          }}
          formatter={(value) => [`₹${Number(value).toFixed(2)}`, 'Close Price']}
          labelStyle={{ color: '#10f2c5' }}
        />
        <Area
          type="monotone"
          dataKey="price"
          stroke="#1ca187ff"
          strokeWidth={3}
          fill="url(#historicalGradient)"
          dot={false}
          activeDot={{ 
            r: 6, 
            fill: '#10f2c5', 
            strokeWidth: 2, 
            stroke: 'white',
            filter: 'url(#glow)'
          }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
};

export default HistoricalChart;
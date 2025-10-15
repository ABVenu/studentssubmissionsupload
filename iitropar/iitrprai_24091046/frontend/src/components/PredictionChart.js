import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

const PredictionChart = ({ data }) => {
  const formattedData = data.map(item => ({
    ...item,
    date: new Date(item.date).toLocaleDateString('en-US', { 
      weekday: 'short',
      month: 'short', 
      day: 'numeric'
    })
  }));

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={formattedData} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
        <defs>
          <filter id="glowPurple">
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
            border: '1px solid rgba(168,85,247,0.5)',
            borderRadius: '12px',
            color: 'white',
            boxShadow: '0 10px 25px rgba(0,0,0,0.3)'
          }}
          formatter={(value) => [`₹${Number(value).toFixed(2)}`, 'Predicted Close']}
          labelStyle={{ color: '#a855f7' }}
        />
        <Line
          type="monotone"
          dataKey="price"
          stroke="#522e8dff"
          strokeWidth={3}
          strokeDasharray="8 4"
          dot={{ 
            fill: '#522e8dff', 
            strokeWidth: 2, 
            r: 6,
            stroke: 'white'
          }}
          activeDot={{ 
            r: 8, 
            fill: '#a855f7', 
            strokeWidth: 2,
            stroke: 'white',
            filter: 'url(#glowPurple)'
          }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default PredictionChart;
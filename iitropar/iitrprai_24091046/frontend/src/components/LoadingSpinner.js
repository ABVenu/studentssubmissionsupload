import React from 'react';
import { BarChart3, TrendingUp } from 'lucide-react';

const LoadingSpinner = ({ message = "Loading market data..." }) => {
  return (
    <div className="flex flex-col items-center justify-center h-64 space-y-4">
      <div className="relative">
        <div className="w-16 h-16 border-4 border-purple-200 border-solid rounded-full animate-spin border-t-purple-500"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
          <BarChart3 className="h-6 w-6 text-purple-500 animate-pulse" />
        </div>
      </div>
      
      <div className="text-center">
        <p className="text-white/80 text-lg font-medium">{message}</p>
        <div className="flex items-center justify-center space-x-1 mt-2">
          <TrendingUp className="h-4 w-4 text-green-400 animate-bounce" />
          <span className="text-white/60 text-sm">Analyzing trends...</span>
        </div>
      </div>
      
      <div className="flex space-x-2">
        <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></div>
        <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
        <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
      </div>
    </div>
  );
};

export default LoadingSpinner;
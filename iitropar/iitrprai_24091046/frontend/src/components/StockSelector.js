import React from 'react';
import { ChevronDown } from 'lucide-react';

const StockSelector = ({ stocks, selectedStock, onStockChange }) => {
  return (
    <div className="mb-8 animate-fade-in">
      <div className="relative max-w-md mx-auto">
        <select
          value={selectedStock}
          onChange={(e) => onStockChange(e.target.value)}
          className="w-full appearance-none bg-gradient-to-r from-indigo-500 to-purple-600 text-white text-xl font-bold py-4 px-6 pr-12 rounded-2xl shadow-lg border-0 focus:outline-none focus:ring-4 focus:ring-purple-300 transition-all duration-300 hover:shadow-2xl hover:scale-105 transform cursor-pointer"
        >
          {stocks.map((stock) => (
            <option key={stock} value={stock} className="bg-gray-800 text-white py-2">
              {stock.replace('.NS', '')} {stock.includes('.NS') ? '(NSE)' : ''}
            </option>
          ))}
        </select>
        <ChevronDown className="absolute right-4 top-1/2 transform -translate-y-1/2 h-6 w-6 text-white pointer-events-none" />
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-600 to-purple-700 rounded-2xl opacity-0 hover:opacity-20 transition-opacity duration-300 pointer-events-none"></div>
      </div>
    </div>
  );
};

export default StockSelector;
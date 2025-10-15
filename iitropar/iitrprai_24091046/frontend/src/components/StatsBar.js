import React from 'react';
import { IndianRupee, Activity, TrendingUp, TrendingDown } from 'lucide-react';

const StatsBar = ({ historicalData, predictedData }) => {
  const currentPrice = historicalData.length > 0 ? historicalData[historicalData.length - 1].price : 0;
  const predictedPrice = predictedData.length > 0 ? predictedData[predictedData.length - 1].price : 0;
  const change = predictedPrice - currentPrice;
  const changePercent = currentPrice !== 0 ? (change / currentPrice) * 100 : 0;
  const isPositive = change >= 0;

  const stats = [
    {
      title: 'Current Close',
      value: `₹${currentPrice.toFixed(2)}`,
      icon: IndianRupee,
      gradient: 'from-green-500 to-emerald-600',
      bgClass: 'bg-gradient-to-br from-green-500 to-emerald-600'
    },
    {
      title: 'Predicted Close',
      value: `₹${predictedPrice.toFixed(2)}`,
      icon: Activity,
      gradient: 'from-purple-500 to-pink-600',
      bgClass: 'bg-gradient-to-br from-purple-500 to-pink-600'
    },
    {
      title: 'Expected Change',
      value: `${isPositive ? '+' : ''}${changePercent.toFixed(2)}%`,
      icon: isPositive ? TrendingUp : TrendingDown,
      gradient: isPositive ? 'from-green-500 to-emerald-600' : 'from-red-500 to-pink-600',
      bgClass: `bg-gradient-to-br ${isPositive ? 'from-green-500 to-emerald-600' : 'from-red-500 to-pink-600'}`
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
      {stats.map((stat, index) => (
        <div 
          key={stat.title}
          className={`${stat.bgClass} p-6 rounded-2xl shadow-xl card-hover animate-fade-in`}
          style={{ animationDelay: `${index * 0.1}s` }}
        >
          <div className="flex items-center space-x-4">
            <div className="bg-white/20 backdrop-blur-sm p-3 rounded-xl">
              <stat.icon className="h-8 w-8 text-white" />
            </div>
            <div>
              <p className="text-white/90 text-sm font-medium">{stat.title}</p>
              <p className="text-3xl font-bold text-white">{stat.value}</p>
            </div>
          </div>
          <div className="mt-3 bg-white/10 h-1 rounded-full">
            <div 
              className="h-full bg-white/30 rounded-full transition-all duration-1000 ease-out"
              style={{ width: '100%' }}
            ></div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default StatsBar;
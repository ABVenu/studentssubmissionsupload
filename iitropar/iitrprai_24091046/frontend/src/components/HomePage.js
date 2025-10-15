import React from 'react';
import { useStockData } from '../hooks/useStockData';
import StockSelector from './StockSelector';
import StatsBar from './StatsBar';
import ChartCard from './ChartCard';
import HistoricalChart from './HistoricalChart';
import PredictionChart from './PredictionChart';
import LoadingSpinner from './LoadingSpinner';
import { BarChart3, Calendar, AlertCircle } from 'lucide-react';

const HomePage = () => {
  const {
    stocks,
    selectedStock,
    setSelectedStock,
    historicalData,
    predictedData,
    loading,
    error
  } = useStockData();

  return (
    <div className="animate-fade-in">
      {error && (
        <div className="bg-yellow-500/20 border border-yellow-500/50 rounded-xl p-4 mb-6 animate-fade-in">
          <div className="flex items-center space-x-3">
            <AlertCircle className="h-5 w-5 text-yellow-400" />
            <p className="text-yellow-100 text-sm">{error}</p>
          </div>
        </div>
      )}
      
      <StockSelector 
        stocks={stocks}
        selectedStock={selectedStock}
        onStockChange={setSelectedStock}
      />
      
      <StatsBar 
        historicalData={historicalData}
        predictedData={predictedData}
      />
      
      {loading ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-white/10 backdrop-blur-sm rounded-3xl p-8 border border-white/20">
            <LoadingSpinner message="Loading historical data..." />
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-3xl p-8 border border-white/20">
            <LoadingSpinner message="Generating predictions..." />
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <ChartCard
            title="Historical Data"
            subtitle="Last 60 days performance"
            icon={BarChart3}
            gradient="from-cyan-500 to-teal-600"
          >
            <HistoricalChart data={historicalData} />
          </ChartCard>
          
          <ChartCard
            title="AI Predictions"
            subtitle="Next 5 days forecast"
            icon={Calendar}
            gradient="from-purple-500 to-indigo-600"
          >
            <PredictionChart data={predictedData} />
          </ChartCard>
        </div>
      )}
      
      {/* Footer */}
      <div className="mt-12 text-center animate-fade-in">
        <div className="bg-gradient-to-r from-purple-600/20 to-cyan-600/20 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
          <p className="text-white/80 text-sm mb-2">
            🚀 Powered by Advanced AI • 📊 Real-time Market Analysis • ⚡ Live Data Updates
          </p>
          <p className="text-white/60 text-xs">
            Disclaimer: This is for educational purposes only. Not financial advice.
          </p>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
import { useState, useEffect } from 'react';
import ApiService from '../utils/api';

export const useStockData = () => {
  const [stocks, setStocks] = useState([]);
  const [selectedStock, setSelectedStock] = useState('');
  const [historicalData, setHistoricalData] = useState([]);
  const [predictedData, setPredictedData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch stock list on component mount
  useEffect(() => {
    const fetchStocks = async () => {
      try {
        setLoading(true);
        const response = await ApiService.getStockList();
        const stockList = response.stocks || [];
        
        setStocks(stockList);
        if (stockList.length > 0) {
          setSelectedStock(stockList[0]);
        }
        setError(null);
      } catch (error) {
        console.error('Error fetching stocks:', error);
        // Fallback to demo stocks
        const demoStocks = ['TCS.NS', 'RELIANCE.NS', 'SBIN.NS', 'INFY.NS', 'HDFC.NS', 'WIPRO.NS'];
        setStocks(demoStocks);
        setSelectedStock(demoStocks[0]);
        setError('Using demo data - API unavailable');
      } finally {
        setLoading(false);
      }
    };

    fetchStocks();
  }, []);

  // Fetch stock data when selected stock changes
  useEffect(() => {
    if (!selectedStock) return;

    const fetchStockData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch both historical and predicted data concurrently
        const [historicalResponse, predictedResponse] = await Promise.allSettled([
          ApiService.getHistoricalData(selectedStock),
          ApiService.getPredictedData(selectedStock)
        ]);

        // Handle historical data
        if (historicalResponse.status === 'fulfilled' && historicalResponse.value.length > 0) {
          setHistoricalData(historicalResponse.value);
        } else {
          console.warn('Historical data fetch failed, using demo data');
          setHistoricalData(ApiService.generateDemoHistoricalData());
        }

        // Handle predicted data
        if (predictedResponse.status === 'fulfilled' && predictedResponse.value.length > 0) {
          setPredictedData(predictedResponse.value);
        } else {
          console.warn('Predicted data fetch failed, using demo data');
          const lastPrice = historicalResponse.status === 'fulfilled' && 
            historicalResponse.value?.length > 0 ? 
            historicalResponse.value[historicalResponse.value.length - 1].price : 2900;
          setPredictedData(ApiService.generateDemoPredictedData(lastPrice));
        }

        if (historicalResponse.status === 'rejected' && predictedResponse.status === 'rejected') {
          setError('API unavailable - showing demo data');
        }

      } catch (error) {
        console.error('Error fetching stock data:', error);
        setHistoricalData(ApiService.generateDemoHistoricalData());
        setPredictedData(ApiService.generateDemoPredictedData());
        setError('Using demo data - API error');
      } finally {
        setLoading(false);
      }
    };

    fetchStockData();
  }, [selectedStock]);

  return {
    stocks,
    selectedStock,
    setSelectedStock,
    historicalData,
    predictedData,
    loading,
    error
  };
};
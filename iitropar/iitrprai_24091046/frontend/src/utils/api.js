// API utility functions
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:8000';

class ApiService {
  static async fetchWithErrorHandling(url, options = {}) {
    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`API Error for ${url}:`, error);
      throw error;
    }
  }

  static async getStockList() {
    return this.fetchWithErrorHandling(`${API_BASE_URL}/api/stocklist`);
  }

  static async getHistoricalData(stockName) {
    const response = await this.fetchWithErrorHandling(`${API_BASE_URL}/api/history/${stockName}`);
    // Transform the response to match our expected format
    return response.history?.map(item => ({
      date: item.date,
      price: item.close
    })) || [];
  }

  static async getPredictedData(stockName) {
    const response = await this.fetchWithErrorHandling(`${API_BASE_URL}/api/predict/${stockName}`);
    // Transform the response to match our expected format
    return response.predictions?.map(item => ({
      date: item.date,
      price: item.predicted_close
    })) || [];
  }

  // Demo data generators for fallback
  static generateDemoHistoricalData() {
    const data = [];
    const basePrice = 2800 + Math.random() * 400; // More realistic base price for Indian stocks
    
    for (let i = 59; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      
      const trend = Math.sin(i / 10) * 100;
      const noise = (Math.random() - 0.5) * 150;
      const price = Math.max(basePrice + trend + noise, 1000);
      
      data.push({
        date: date.toISOString().split('T')[0],
        price: Number(price.toFixed(2))
      });
    }
    
    return data;
  }

  static generateDemoPredictedData(lastPrice = 2900) {
    const data = [];
    
    for (let i = 1; i <= 5; i++) {
      const date = new Date();
      date.setDate(date.getDate() + i);
      
      const change = (Math.random() - 0.5) * 100;
      const price = Math.max(lastPrice + change, 1000);
      
      data.push({
        date: date.toISOString().split('T')[0],
        price: Number(price.toFixed(2))
      });
    }
    
    return data;
  }
}

export default ApiService;
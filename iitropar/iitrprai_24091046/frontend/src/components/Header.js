import React from 'react';
import { TrendingUp } from 'lucide-react';

const Header = ({ currentPage, setCurrentPage }) => {
  return (
    <div className="bg-gradient-to-r from-purple-600 via-blue-600 to-cyan-500 p-6 rounded-3xl shadow-2xl mb-8 animate-slide-up">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="bg-white/20 backdrop-blur-sm p-3 rounded-xl floating-animation">
            <TrendingUp className="h-8 w-8 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white tracking-tight gradient-text-white">
              Stock Market Prediction
            </h1>
            <p className="text-blue-100 text-lg mt-1">
              AI-Powered Market Analytics Dashboard
            </p>
          </div>
        </div>
        <div className="flex space-x-6">
          <button 
            onClick={() => setCurrentPage('home')}
            className={`text-lg font-medium transition-all duration-300 hover:scale-105 transform px-4 py-2 rounded-lg ${
              currentPage === 'home' 
                ? 'text-white bg-white/20 shadow-lg' 
                : 'text-white hover:text-blue-200 hover:bg-white/10'
            }`}
          >
            Home
          </button>
          <button 
            onClick={() => setCurrentPage('about')}
            className={`text-lg font-medium transition-all duration-300 hover:scale-105 transform px-4 py-2 rounded-lg ${
              currentPage === 'about' 
                ? 'text-white bg-white/20 shadow-lg' 
                : 'text-white hover:text-blue-200 hover:bg-white/10'
            }`}
          >
            About
          </button>
        </div>
      </div>
    </div>
  );
};

export default Header;
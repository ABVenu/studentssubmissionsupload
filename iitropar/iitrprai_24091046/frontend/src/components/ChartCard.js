import React from 'react';

const ChartCard = ({ title, subtitle, children, icon: Icon, gradient }) => {
  return (
    <div className={`bg-gradient-to-br ${gradient} p-6 rounded-3xl shadow-2xl card-hover backdrop-blur-sm border border-white/20 animate-slide-up`}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="flex items-center space-x-3 mb-2">
            <div className="bg-white/20 backdrop-blur-sm p-2 rounded-lg">
              <Icon className="h-6 w-6 text-white" />
            </div>
            <h2 className="text-2xl font-bold text-white">{title}</h2>
          </div>
          <p className="text-white/80 text-lg">{subtitle}</p>
        </div>
      </div>
      <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-4 border border-white/20">
        {children}
      </div>
    </div>
  );
};

export default ChartCard;
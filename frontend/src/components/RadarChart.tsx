import React from 'react';
import { Radar } from 'react-chartjs-2';
import { Chart as ChartJS, RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend } from 'chart.js';

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

interface RadarChartProps {
  data: {
    labels: string[];
    values: number[];
  };
}

const RadarChart: React.FC<RadarChartProps> = ({ data }) => {
  const chartData = {
    labels: data.labels,
    datasets: [{
      label: '知识掌握度',
      data: data.values,
      backgroundColor: 'rgba(99, 102, 241, 0.2)',
      borderColor: 'rgba(99, 102, 241, 1)',
      borderWidth: 2,
      pointBackgroundColor: 'rgba(99, 102, 241, 1)',
      pointBorderColor: '#fff',
      pointHoverBackgroundColor: '#fff',
      pointHoverBorderColor: 'rgba(99, 102, 241, 1)'
    }]
  };

  const options = {
    scales: {
      r: {
        angleLines: { display: true },
        suggestedMin: 0,
        suggestedMax: 100
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'bottom' as const
      }
    }
  };

  return (
    <div className="radar-chart">
      <h3>知识盲区雷达图</h3>
      <Radar data={chartData} options={options} />
    </div>
  );
};

export default RadarChart;

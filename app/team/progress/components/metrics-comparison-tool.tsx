'use client';

import { useState } from 'react';
import { 
  BarChart3, 
  LineChart, 
  ArrowUpRight, 
  ArrowDownRight, 
  Calendar, 
  Filter,
  Download,
  Info,
  RefreshCw,
  Maximize2,
  Crosshair,
  Layers
} from 'lucide-react';
import { format, subDays, subMonths, eachDayOfInterval, isAfter } from 'date-fns';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@/components/ui/tabs';

// Metric types for comparison
type MetricType = 'weight' | 'bodyFat' | 'calories' | 'steps' | 'sleep' | 'workoutDuration' | 'strength' | 'heartRate' | 'recoveryScore';

// Time range options
type TimeRange = '7d' | '30d' | '90d' | '6m' | '1y';

interface MetricData {
  id: string;
  name: string;
  type: MetricType;
  unit: string;
  color: string;
  data: Array<{
    date: Date;
    value: number;
  }>;
  trend: 'up' | 'down' | 'neutral';
  percentChange: number;
}

interface MetricsComparisonToolProps {
  initialMetrics?: MetricType[];
}

export default function MetricsComparisonTool({ 
  initialMetrics = ['weight', 'bodyFat'] 
}: MetricsComparisonToolProps) {
  const [selectedMetrics, setSelectedMetrics] = useState<MetricType[]>(initialMetrics);
  const [timeRange, setTimeRange] = useState<TimeRange>('30d');
  const [chartType, setChartType] = useState<'line' | 'bar'>('line');
  const [isNormalized, setIsNormalized] = useState<boolean>(true);
  
  // Available metrics for selection
  const availableMetrics: Record<MetricType, { name: string; unit: string; color: string }> = {
    weight: { name: 'Weight', unit: 'kg', color: '#3b82f6' },
    bodyFat: { name: 'Body Fat', unit: '%', color: '#ec4899' },
    calories: { name: 'Calories Burned', unit: 'kcal', color: '#f97316' },
    steps: { name: 'Daily Steps', unit: 'steps', color: '#10b981' },
    sleep: { name: 'Sleep Duration', unit: 'hours', color: '#8b5cf6' },
    workoutDuration: { name: 'Workout Time', unit: 'min', color: '#f43f5e' },
    strength: { name: 'Strength Score', unit: 'points', color: '#fbbf24' },
    heartRate: { name: 'Resting HR', unit: 'bpm', color: '#ef4444' },
    recoveryScore: { name: 'Recovery Score', unit: '%', color: '#14b8a6' }
  };
  
  // Generate mock data for each metric
  const generateMockData = (metric: MetricType): MetricData => {
    const today = new Date();
    let startDate;
    
    // Set start date based on selected time range
    switch (timeRange) {
      case '7d':
        startDate = subDays(today, 7);
        break;
      case '30d':
        startDate = subDays(today, 30);
        break;
      case '90d':
        startDate = subDays(today, 90);
        break;
      case '6m':
        startDate = subMonths(today, 6);
        break;
      case '1y':
        startDate = subMonths(today, 12);
        break;
      default:
        startDate = subDays(today, 30);
    }
    
    // Generate dates for the selected range
    const dateRange = eachDayOfInterval({ start: startDate, end: today });
    
    // Generate values based on metric type
    let baseValue = 0;
    let variability = 0;
    let trend: 'up' | 'down' | 'neutral' = 'neutral';
    
    switch (metric) {
      case 'weight':
        baseValue = 75;
        variability = 0.5;
        trend = 'down';
        break;
      case 'bodyFat':
        baseValue = 18;
        variability = 0.3;
        trend = 'down';
        break;
      case 'calories':
        baseValue = 2200;
        variability = 300;
        trend = 'up';
        break;
      case 'steps':
        baseValue = 8000;
        variability = 2000;
        trend = 'up';
        break;
      case 'sleep':
        baseValue = 7;
        variability = 1;
        trend = 'up';
        break;
      case 'workoutDuration':
        baseValue = 45;
        variability = 15;
        trend = 'up';
        break;
      case 'strength':
        baseValue = 100;
        variability = 5;
        trend = 'up';
        break;
      case 'heartRate':
        baseValue = 65;
        variability = 3;
        trend = 'down';
        break;
      case 'recoveryScore':
        baseValue = 70;
        variability = 10;
        trend = 'up';
        break;
    }
    
    // Generate random values with a slight trend in the specified direction
    const data = dateRange.map((date, index) => {
      const trendFactor = trend === 'up' ? 0.1 : trend === 'down' ? -0.1 : 0;
      const randomVariation = (Math.random() - 0.5) * variability;
      const trendValue = trendFactor * (index / dateRange.length) * baseValue;
      
      let value = baseValue + randomVariation + trendValue;
      
      // Ensure value stays within reasonable ranges
      if (metric === 'bodyFat') value = Math.max(5, Math.min(30, value));
      if (metric === 'recoveryScore') value = Math.max(0, Math.min(100, value));
      if (metric === 'sleep') value = Math.max(3, Math.min(10, value));
      
      return {
        date,
        value: Number(value.toFixed(1))
      };
    });
    
    // Calculate percent change
    const firstValue = data[0].value;
    const lastValue = data[data.length - 1].value;
    const percentChange = ((lastValue - firstValue) / firstValue) * 100;
    
    // Determine trend based on actual change
    const actualTrend = percentChange > 1 ? 'up' : percentChange < -1 ? 'down' : 'neutral';
    
    return {
      id: `metric-${metric}`,
      name: availableMetrics[metric].name,
      type: metric,
      unit: availableMetrics[metric].unit,
      color: availableMetrics[metric].color,
      data,
      trend: actualTrend,
      percentChange: Math.abs(percentChange)
    };
  };
  
  // Generate data for all selected metrics
  const metricsData = selectedMetrics.map(metric => generateMockData(metric));
  
  // Get the appropriate date format based on the time range
  const getDateFormat = (): string => {
    switch (timeRange) {
      case '7d':
      case '30d':
        return 'MMM d';
      case '90d':
      case '6m':
        return 'MMM';
      case '1y':
        return 'MMM yyyy';
      default:
        return 'MMM d';
    }
  };
  
  // Helper function to normalize data (0-100%) for better comparison
  const getNormalizedValue = (value: number, metric: MetricData): number => {
    if (!isNormalized) return value;
    
    const values = metric.data.map(d => d.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    return ((value - min) / (max - min)) * 100;
  };
  
  // Get formatted value with unit
  const getFormattedValue = (value: number, unit: string): string => {
    if (unit === 'kg') return `${value.toFixed(1)} kg`;
    if (unit === '%') return `${value.toFixed(1)}%`;
    if (unit === 'kcal') return `${value.toLocaleString()} kcal`;
    if (unit === 'steps') return `${value.toLocaleString()}`;
    if (unit === 'hours') return `${value.toFixed(1)} hrs`;
    if (unit === 'min') return `${value} min`;
    if (unit === 'points') return value.toFixed(0);
    if (unit === 'bpm') return `${value.toFixed(0)} bpm`;
    
    return `${value} ${unit}`;
  };
  
  // Handle metric selection changes
  const toggleMetric = (metric: MetricType) => {
    setSelectedMetrics(prev => {
      if (prev.includes(metric)) {
        return prev.filter(m => m !== metric);
      } else {
        return [...prev, metric];
      }
    });
  };
  
  // Determine if a metric should improve up or down
  const isImprovementUp = (metric: MetricType): boolean => {
    return !['weight', 'bodyFat', 'heartRate'].includes(metric);
  };

  return (
    <div className="metrics-comparison-tool bg-gray-900 rounded-xl p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-white text-xl font-medium">Metrics Comparison</h2>
          <p className="text-gray-400">Compare different fitness metrics over time</p>
        </div>
        
        <div className="flex gap-3">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="outline" size="icon" onClick={() => setChartType(chartType === 'line' ? 'bar' : 'line')}>
                  {chartType === 'line' ? <LineChart className="h-4 w-4" /> : <BarChart3 className="h-4 w-4" />}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Switch to {chartType === 'line' ? 'Bar' : 'Line'} Chart</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
          
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="outline" size="icon" onClick={() => setIsNormalized(!isNormalized)}>
                  {isNormalized ? <Layers className="h-4 w-4" /> : <Crosshair className="h-4 w-4" />}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>{isNormalized ? 'Show Absolute Values' : 'Normalize Values (0-100%)'}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
          
          <Select value={timeRange} onValueChange={(value: TimeRange) => setTimeRange(value)}>
            <SelectTrigger className="w-[120px]">
              <SelectValue placeholder="Time Range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="7d">Last 7 Days</SelectItem>
              <SelectItem value="30d">Last 30 Days</SelectItem>
              <SelectItem value="90d">Last 90 Days</SelectItem>
              <SelectItem value="6m">Last 6 Months</SelectItem>
              <SelectItem value="1y">Last Year</SelectItem>
            </SelectContent>
          </Select>
          
          <Button variant="outline">
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
        </div>
      </div>
      
      {/* Metrics Selection */}
      <div className="flex flex-wrap gap-2 mb-6">
        {(Object.keys(availableMetrics) as MetricType[]).map(metric => (
          <Badge
            key={metric}
            onClick={() => toggleMetric(metric)}
            className={`py-1 px-3 cursor-pointer ${
              selectedMetrics.includes(metric)
                ? `bg-${availableMetrics[metric].color.replace('#', '')} hover:bg-${availableMetrics[metric].color.replace('#', '')}/90`
                : 'bg-gray-700 hover:bg-gray-600'
            }`}
            style={{
              backgroundColor: selectedMetrics.includes(metric) 
                ? availableMetrics[metric].color 
                : undefined
            }}
          >
            {availableMetrics[metric].name}
          </Badge>
        ))}
        
        {selectedMetrics.length === 0 && (
          <div className="text-gray-400 text-sm p-2">
            Select metrics to compare
          </div>
        )}
      </div>
      
      {/* Chart Visualization */}
      <Card className="bg-gray-800 border-gray-700 mb-6">
        <CardContent className="p-6">
          {selectedMetrics.length > 0 ? (
            <div>
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-white font-medium">
                  {isNormalized 
                    ? 'Normalized Comparison (%)' 
                    : 'Metric Comparison'}
                </h3>
                
                <div className="flex gap-2">
                  <Badge variant="outline" className="gap-1">
                    <Calendar className="h-3 w-3" />
                    {format(new Date(), getDateFormat())}
                  </Badge>
                  
                  <Button variant="ghost" size="icon" className="h-7 w-7">
                    <Maximize2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              
              {/* Chart placeholder - In a real app, use a charting library like Chart.js or Recharts */}
              <div className="relative">
                <div className="w-full h-[350px] bg-gray-850 rounded-lg border border-gray-700 p-4">
                  <div className="absolute inset-0 p-6">
                    {/* This would be replaced by actual chart in a real implementation */}
                    <div className="h-full flex items-end">
                      {metricsData.map((metric, metricIndex) => (
                        <div 
                          key={metric.id}
                          className="flex-1 h-full flex items-end"
                        >
                          {metric.data.map((dataPoint, i) => {
                            // Only show a sample of points for visualization
                            if (i % Math.max(1, Math.floor(metric.data.length / 10)) !== 0) return null;
                            
                            const value = isNormalized 
                              ? getNormalizedValue(dataPoint.value, metric)
                              : dataPoint.value;
                            
                            // Scale the value for display
                            const displayHeight = isNormalized 
                              ? `${value}%`
                              : `${(value / (metric.type === 'steps' ? 15000 : metric.type === 'calories' ? 3000 : 150)) * 100}%`;
                            
                            return (
                              <div 
                                key={i} 
                                className="flex-1 px-1 flex items-end"
                                style={{ height: '100%' }}
                              >
                                <div 
                                  className="w-full rounded-sm"
                                  style={{ 
                                    height: displayHeight, 
                                    backgroundColor: metric.color,
                                    opacity: chartType === 'bar' ? 0.7 : 0.5,
                                    // Create a line chart effect with a border
                                    borderTop: chartType === 'line' ? `2px solid ${metric.color}` : 'none',
                                  }}
                                ></div>
                              </div>
                            );
                          })}
                        </div>
                      ))}
                    </div>
                    
                    {/* X-axis labels */}
                    <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-gray-500">
                      {metricsData[0]?.data
                        .filter((_, i) => i % Math.max(1, Math.floor(metricsData[0].data.length / 5)) === 0)
                        .map((dataPoint, i) => (
                          <div key={i}>{format(dataPoint.date, getDateFormat())}</div>
                        ))}
                    </div>
                  </div>
                  
                  <div className="absolute inset-0 flex items-center justify-center">
                    {selectedMetrics.length === 0 && (
                      <div className="text-gray-500">
                        Select metrics to visualize
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              {/* Legend */}
              <div className="flex flex-wrap gap-4 mt-4">
                {metricsData.map(metric => (
                  <div key={metric.id} className="flex items-center gap-2">
                    <div 
                      className="w-3 h-3 rounded-sm" 
                      style={{ backgroundColor: metric.color }}
                    ></div>
                    <span className="text-gray-300 text-sm">{metric.name}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="h-[350px] flex items-center justify-center text-gray-500">
              <div className="text-center">
                <Filter className="h-10 w-10 mx-auto mb-3 opacity-50" />
                <p>Select at least one metric to visualize data</p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
      
      {/* Metric Details */}
      {metricsData.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {metricsData.map(metric => {
            const firstValue = metric.data[0].value;
            const lastValue = metric.data[metric.data.length - 1].value;
            const isPositive = isImprovementUp(metric.type) 
              ? lastValue > firstValue 
              : lastValue < firstValue;
            
            return (
              <Card 
                key={metric.id} 
                className="border-gray-700 overflow-hidden"
                style={{ backgroundColor: `${metric.color}10` }}
              >
                <CardContent className="p-4">
                  <div className="flex justify-between items-start">
                    <div>
                      <h4 className="text-white font-medium">{metric.name}</h4>
                      <div className="text-gray-400 text-sm">
                        Current: {getFormattedValue(lastValue, metric.unit)}
                      </div>
                    </div>
                    
                    <div className={`flex items-center gap-1 px-2 py-1 rounded ${
                      isPositive ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'
                    }`}>
                      {isPositive ? (
                        <ArrowUpRight className="h-4 w-4" />
                      ) : (
                        <ArrowDownRight className="h-4 w-4" />
                      )}
                      <span>{metric.percentChange.toFixed(1)}%</span>
                    </div>
                  </div>
                  
                  <div className="mt-4">
                    <div className="flex justify-between text-xs text-gray-400 mb-1">
                      <span>{format(metric.data[0].date, getDateFormat())}</span>
                      <span>{format(metric.data[metric.data.length - 1].date, getDateFormat())}</span>
                    </div>
                    
                    <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div 
                        className="h-full rounded-full"
                        style={{ 
                          width: `${isPositive ? metric.percentChange : 0}%`,
                          backgroundColor: metric.color,
                          maxWidth: '100%'
                        }}
                      ></div>
                    </div>
                    
                    <div className="flex justify-between text-xs mt-1">
                      <span className="text-gray-400">
                        Start: {getFormattedValue(firstValue, metric.unit)}
                      </span>
                      <span className="text-white">
                        Now: {getFormattedValue(lastValue, metric.unit)}
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>
      )}
    </div>
  );
} 
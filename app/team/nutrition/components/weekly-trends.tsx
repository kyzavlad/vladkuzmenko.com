'use client';

import { LineChart, BarChart, TrendingUp, Calendar, ArrowRight, ChevronRight } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { DailyNutritionLog } from '../types';

interface WeeklyTrendsProps {
  nutritionData: DailyNutritionLog;
  weeklyData?: DailyNutritionLog[];
}

export default function WeeklyTrends({ nutritionData, weeklyData = [] }: WeeklyTrendsProps) {
  // Mock data for the weekly trends
  const mockWeeklyData = [
    { day: 'Mon', calories: 2300, protein: 180, carbs: 220, fat: 65, quality: 82 },
    { day: 'Tue', calories: 2100, protein: 165, carbs: 200, fat: 60, quality: 85 },
    { day: 'Wed', calories: 2400, protein: 190, carbs: 230, fat: 70, quality: 80 },
    { day: 'Thu', calories: 2150, protein: 175, carbs: 210, fat: 58, quality: 88 },
    { day: 'Fri', calories: 2250, protein: 185, carbs: 215, fat: 62, quality: 84 },
    { day: 'Sat', calories: 2500, protein: 195, carbs: 250, fat: 73, quality: 78 },
    { day: 'Sun', calories: 2200, protein: 170, carbs: 210, fat: 65, quality: 85 },
  ];
  
  // Calculate averages
  const avgCalories = Math.round(mockWeeklyData.reduce((acc, day) => acc + day.calories, 0) / mockWeeklyData.length);
  const avgProtein = Math.round(mockWeeklyData.reduce((acc, day) => acc + day.protein, 0) / mockWeeklyData.length);
  const avgCarbs = Math.round(mockWeeklyData.reduce((acc, day) => acc + day.carbs, 0) / mockWeeklyData.length);
  const avgFat = Math.round(mockWeeklyData.reduce((acc, day) => acc + day.fat, 0) / mockWeeklyData.length);
  const avgQuality = Math.round(mockWeeklyData.reduce((acc, day) => acc + day.quality, 0) / mockWeeklyData.length);
  
  // Calculate protein/carbs/fat percentages
  const totalCals = (avgProtein * 4) + (avgCarbs * 4) + (avgFat * 9);
  const proteinPct = Math.round((avgProtein * 4 / totalCals) * 100);
  const carbsPct = Math.round((avgCarbs * 4 / totalCals) * 100);
  const fatPct = Math.round((avgFat * 9 / totalCals) * 100);
  
  // Calculate week-over-week changes (mock data)
  const calorieChange = +5; // 5% increase from previous week
  const qualityChange = +3; // 3 point increase from previous week
  
  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardHeader className="pb-2">
        <div className="flex items-center">
          <TrendingUp className="h-5 w-5 text-blue-400 mr-2" />
          <CardTitle className="text-white">Weekly Nutrition Trends</CardTitle>
        </div>
        <CardDescription className="text-gray-400">
          Analysis of your past 7 days of nutrition data
        </CardDescription>
      </CardHeader>
      <CardContent>
        {/* Calories Chart */}
        <div className="mb-6">
          <div className="flex justify-between items-center mb-3">
            <h4 className="text-white font-medium">Calorie Intake</h4>
            <Badge variant="outline" className="text-xs">
              <span className={calorieChange >= 0 ? "text-green-400" : "text-red-400"}>
                {calorieChange >= 0 ? "+" : ""}{calorieChange}%
              </span>
              <span className="text-gray-400 ml-1">week over week</span>
            </Badge>
          </div>
          
          <div className="bg-gray-750 rounded-md p-4 h-48 flex flex-col">
            {/* In a real app, this would be a chart component */}
            <div className="flex-grow flex items-end">
              {mockWeeklyData.map((day, i) => (
                <div key={i} className="flex-1 flex flex-col items-center">
                  <div 
                    className="w-full max-w-[30px] bg-blue-500 rounded-sm mx-1" 
                    style={{ 
                      height: `${(day.calories / 3000) * 100}%`,
                      opacity: i === 6 ? 1 : 0.7 // Highlight the most recent day
                    }}
                  />
                </div>
              ))}
            </div>
            <div className="flex justify-between mt-2 text-xs text-gray-400">
              {mockWeeklyData.map((day, i) => (
                <div key={i} className="text-center">
                  {day.day}
                </div>
              ))}
            </div>
          </div>
        </div>
        
        {/* Summary Stats */}
        <div className="grid grid-cols-2 gap-3 mb-6">
          <div className="bg-gray-700 p-3 rounded-md">
            <div className="text-gray-400 text-xs mb-1">Avg. Daily Calories</div>
            <div className="text-white font-medium">{avgCalories} kcal</div>
            <div className="text-xs text-gray-400 mt-1">
              Target: {nutritionData.calorieTarget} kcal
            </div>
          </div>
          <div className="bg-gray-700 p-3 rounded-md">
            <div className="text-gray-400 text-xs mb-1">Avg. Nutrient Quality</div>
            <div className="text-white font-medium">{avgQuality} / 100</div>
            <div className="text-xs text-green-400 mt-1">
              {qualityChange > 0 ? "+" : ""}{qualityChange} from last week
            </div>
          </div>
          <div className="bg-gray-700 p-3 rounded-md">
            <div className="text-gray-400 text-xs mb-1">Macro Balance</div>
            <div className="text-white font-medium">
              {proteinPct}% P • {carbsPct}% C • {fatPct}% F
            </div>
            <div className="text-xs text-gray-400 mt-1">
              Target: 30% P • 45% C • 25% F
            </div>
          </div>
          <div className="bg-gray-700 p-3 rounded-md">
            <div className="text-gray-400 text-xs mb-1">Consistency</div>
            <div className="text-white font-medium">92%</div>
            <div className="text-xs text-gray-400 mt-1">
              Logging streak: {nutritionData.streak} days
            </div>
          </div>
        </div>
        
        {/* Weekly Insights */}
        <div className="bg-gray-700 rounded-md p-4">
          <h4 className="text-white font-medium mb-3">Weekly Insights</h4>
          
          <div className="space-y-3 text-sm">
            <div className="flex">
              <div className="w-2 h-2 rounded-full bg-green-500 mt-1.5 mr-2"></div>
              <p className="text-gray-300">
                Your protein intake has been consistently high, supporting your muscle building goals.
              </p>
            </div>
            <div className="flex">
              <div className="w-2 h-2 rounded-full bg-yellow-500 mt-1.5 mr-2"></div>
              <p className="text-gray-300">
                Weekend calorie intake tends to be 10-15% higher than weekdays. Consider balancing meals on these days.
              </p>
            </div>
            <div className="flex">
              <div className="w-2 h-2 rounded-full bg-blue-500 mt-1.5 mr-2"></div>
              <p className="text-gray-300">
                Your fiber intake has improved by 15% this week, great job incorporating more whole foods!
              </p>
            </div>
          </div>
          
          <Button variant="outline" size="sm" className="w-full mt-4">
            View Full Analysis Report
          </Button>
        </div>
      </CardContent>
      <CardFooter className="pt-0">
        <Button variant="ghost" size="sm" className="ml-auto text-blue-400">
          Compare to Previous Weeks <ChevronRight className="ml-1 h-4 w-4" />
        </Button>
      </CardFooter>
    </Card>
  );
} 
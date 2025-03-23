'use client';

import { useState } from 'react';
import { Droplet, Plus, Minus, LineChart } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { DailyNutritionLog } from '../types';

interface HydrationTrackerProps {
  nutritionData: DailyNutritionLog;
  onWaterUpdate?: (newAmount: number) => void;
}

export default function HydrationTracker({ 
  nutritionData, 
  onWaterUpdate 
}: HydrationTrackerProps) {
  const [waterAmount, setWaterAmount] = useState(nutritionData.water);
  const waterTarget = 2500; // ml, could be customized per user
  const waterPercentage = Math.round((waterAmount / waterTarget) * 100);
  
  // Quick add buttons for common amounts
  const waterIncrements = [
    { label: '100ml', value: 100 },
    { label: '250ml', value: 250 },
    { label: '500ml', value: 500 },
  ];
  
  const handleAddWater = (amount: number) => {
    const newAmount = waterAmount + amount;
    setWaterAmount(newAmount);
    onWaterUpdate && onWaterUpdate(newAmount);
  };
  
  const handleReduceWater = (amount: number) => {
    const newAmount = Math.max(0, waterAmount - amount);
    setWaterAmount(newAmount);
    onWaterUpdate && onWaterUpdate(newAmount);
  };
  
  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardHeader className="pb-2">
        <div className="flex items-center">
          <Droplet className="h-5 w-5 text-blue-400 mr-2" />
          <CardTitle className="text-white">Hydration Tracker</CardTitle>
        </div>
        <CardDescription className="text-gray-400">
          Track your daily water intake
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-center mb-6">
          <div className="relative w-32 h-32">
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="text-blue-400 text-2xl font-bold">{waterAmount}ml</div>
                <div className="text-gray-400 text-xs">of {waterTarget}ml</div>
              </div>
            </div>
            <svg className="w-full h-full" viewBox="0 0 100 100">
              <circle 
                className="text-gray-700 stroke-current" 
                strokeWidth="10" 
                fill="transparent" 
                r="40" 
                cx="50" 
                cy="50" 
              />
              <circle 
                className="text-blue-400 stroke-current" 
                strokeWidth="10" 
                fill="transparent" 
                r="40" 
                cx="50" 
                cy="50" 
                strokeDasharray={`${2 * Math.PI * 40}`}
                strokeDashoffset={`${2 * Math.PI * 40 * (1 - waterPercentage / 100)}`}
                strokeLinecap="round"
                transform="rotate(-90 50 50)"
              />
            </svg>
          </div>
        </div>
        
        <div className="mb-4">
          <div className="flex justify-between text-xs mb-1">
            <span className="text-gray-400">Daily Goal</span>
            <span className="text-white">{waterPercentage}%</span>
          </div>
          <Progress value={waterPercentage} className="h-2 bg-gray-700" indicatorClassName="bg-blue-400" />
        </div>
        
        <div className="flex justify-between mb-6">
          {waterIncrements.map((increment) => (
            <Button 
              key={increment.value}
              variant="outline" 
              size="sm"
              className="flex-1 mx-1"
              onClick={() => handleAddWater(increment.value)}
            >
              + {increment.label}
            </Button>
          ))}
        </div>
        
        <div className="flex items-center justify-between bg-gray-700 p-3 rounded-md mb-4">
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={() => handleReduceWater(100)}
            disabled={waterAmount <= 0}
          >
            <Minus className="h-4 w-4 text-gray-400" />
          </Button>
          
          <div className="text-center">
            <div className="text-white font-medium">{waterAmount} ml</div>
            <div className="text-gray-400 text-xs">Current total</div>
          </div>
          
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={() => handleAddWater(100)}
          >
            <Plus className="h-4 w-4 text-gray-400" />
          </Button>
        </div>
        
        <div className="bg-gray-700 rounded-md p-3">
          <div className="flex justify-between items-center mb-2">
            <h4 className="text-white text-sm font-medium">Hydration Insights</h4>
            <LineChart className="h-4 w-4 text-gray-400" />
          </div>
          
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-400">7-day average:</span>
              <span className="text-white">2,150 ml</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Today's trend:</span>
              <span className="text-green-400">On track</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Recommended:</span>
              <span className="text-white">
                {waterAmount < waterTarget / 2 ? 'Drink more water' : 
                 waterAmount >= waterTarget ? 'Great job!' : 'Keep drinking regularly'}
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 
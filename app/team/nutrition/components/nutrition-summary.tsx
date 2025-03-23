'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { DailyNutritionLog } from '../types';

interface NutritionSummaryProps {
  nutritionData: DailyNutritionLog;
}

export default function NutritionSummary({ nutritionData }: NutritionSummaryProps) {
  // Calculate percentages for the progress bars
  const caloriePercentage = Math.round((nutritionData.totalNutrition.calories / nutritionData.calorieTarget) * 100);
  const proteinPercentage = Math.round((nutritionData.totalNutrition.protein / nutritionData.macroTargets.protein) * 100);
  const carbsPercentage = Math.round((nutritionData.totalNutrition.carbs / nutritionData.macroTargets.carbs) * 100);
  const fatPercentage = Math.round((nutritionData.totalNutrition.fat / nutritionData.macroTargets.fat) * 100);
  const fiberPercentage = Math.round((nutritionData.totalNutrition.fiber / nutritionData.macroTargets.fiber) * 100);
  
  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardHeader className="pb-2">
        <CardTitle className="text-white">Daily Nutrition Summary</CardTitle>
        <CardDescription className="text-gray-400">
          {nutritionData.workoutDay ? 'Workout day' : 'Rest day'} • Target: {nutritionData.calorieTarget} calories
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Calories */}
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Calories</span>
              <span className="text-white">
                {nutritionData.totalNutrition.calories} / {nutritionData.calorieTarget} kcal
              </span>
            </div>
            <div className="flex items-center">
              <Progress 
                value={caloriePercentage > 100 ? 100 : caloriePercentage} 
                className="h-2 flex-grow bg-gray-700" 
              />
              <span className="text-gray-400 text-xs ml-2 min-w-[45px] text-right">
                {caloriePercentage}%
              </span>
            </div>
          </div>
          
          {/* Protein */}
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Protein</span>
              <span className="text-white">
                {nutritionData.totalNutrition.protein} / {nutritionData.macroTargets.protein} g
              </span>
            </div>
            <div className="flex items-center">
              <Progress 
                value={proteinPercentage > 100 ? 100 : proteinPercentage} 
                className="h-2 flex-grow bg-gray-700" 
                indicatorClassName={proteinPercentage > 100 ? "bg-yellow-500" : "bg-blue-500"}
              />
              <span className="text-gray-400 text-xs ml-2 min-w-[45px] text-right">
                {proteinPercentage}%
              </span>
            </div>
          </div>
          
          {/* Carbs */}
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Carbohydrates</span>
              <span className="text-white">
                {nutritionData.totalNutrition.carbs} / {nutritionData.macroTargets.carbs} g
              </span>
            </div>
            <div className="flex items-center">
              <Progress 
                value={carbsPercentage > 100 ? 100 : carbsPercentage} 
                className="h-2 flex-grow bg-gray-700" 
                indicatorClassName={carbsPercentage > 100 ? "bg-yellow-500" : "bg-green-500"}
              />
              <span className="text-gray-400 text-xs ml-2 min-w-[45px] text-right">
                {carbsPercentage}%
              </span>
            </div>
          </div>
          
          {/* Fats */}
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Fats</span>
              <span className="text-white">
                {nutritionData.totalNutrition.fat} / {nutritionData.macroTargets.fat} g
              </span>
            </div>
            <div className="flex items-center">
              <Progress 
                value={fatPercentage > 100 ? 100 : fatPercentage} 
                className="h-2 flex-grow bg-gray-700" 
                indicatorClassName={fatPercentage > 100 ? "bg-yellow-500" : "bg-yellow-400"}
              />
              <span className="text-gray-400 text-xs ml-2 min-w-[45px] text-right">
                {fatPercentage}%
              </span>
            </div>
          </div>
          
          {/* Fiber */}
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Fiber</span>
              <span className="text-white">
                {nutritionData.totalNutrition.fiber} / {nutritionData.macroTargets.fiber} g
              </span>
            </div>
            <div className="flex items-center">
              <Progress 
                value={fiberPercentage > 100 ? 100 : fiberPercentage} 
                className="h-2 flex-grow bg-gray-700" 
                indicatorClassName={fiberPercentage > 100 ? "bg-yellow-500" : "bg-purple-500"}
              />
              <span className="text-gray-400 text-xs ml-2 min-w-[45px] text-right">
                {fiberPercentage}%
              </span>
            </div>
          </div>
        </div>
        
        <div className="mt-4 pt-4 border-t border-gray-700">
          <h4 className="text-white font-medium mb-2">Macro Ratio</h4>
          <div className="flex h-4 rounded-full overflow-hidden">
            <div 
              className="bg-blue-500" 
              style={{ width: `${(nutritionData.totalNutrition.protein * 4 / (nutritionData.totalNutrition.calories || 1)) * 100}%` }}
            />
            <div 
              className="bg-green-500" 
              style={{ width: `${(nutritionData.totalNutrition.carbs * 4 / (nutritionData.totalNutrition.calories || 1)) * 100}%` }}
            />
            <div 
              className="bg-yellow-400" 
              style={{ width: `${(nutritionData.totalNutrition.fat * 9 / (nutritionData.totalNutrition.calories || 1)) * 100}%` }}
            />
          </div>
          <div className="flex justify-between mt-2 text-xs">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-blue-500 rounded-full mr-1" />
              <span className="text-gray-400">
                Protein ({Math.round((nutritionData.totalNutrition.protein * 4 / (nutritionData.totalNutrition.calories || 1)) * 100)}%)
              </span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-green-500 rounded-full mr-1" />
              <span className="text-gray-400">
                Carbs ({Math.round((nutritionData.totalNutrition.carbs * 4 / (nutritionData.totalNutrition.calories || 1)) * 100)}%)
              </span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-yellow-400 rounded-full mr-1" />
              <span className="text-gray-400">
                Fat ({Math.round((nutritionData.totalNutrition.fat * 9 / (nutritionData.totalNutrition.calories || 1)) * 100)}%)
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 
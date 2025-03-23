'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { DailyNutritionLog } from '../types';

interface MacroProgressProps {
  nutritionData: DailyNutritionLog;
}

export default function MacroProgress({ nutritionData }: MacroProgressProps) {
  // Calculate percentages
  const proteinPercentage = Math.round((nutritionData.totalNutrition.protein / nutritionData.macroTargets.protein) * 100);
  const carbsPercentage = Math.round((nutritionData.totalNutrition.carbs / nutritionData.macroTargets.carbs) * 100);
  const fatPercentage = Math.round((nutritionData.totalNutrition.fat / nutritionData.macroTargets.fat) * 100);
  
  // Calculate percentage of daily calories from each macro
  const proteinCalories = nutritionData.totalNutrition.protein * 4;
  const carbsCalories = nutritionData.totalNutrition.carbs * 4;
  const fatCalories = nutritionData.totalNutrition.fat * 9;
  const totalCalories = nutritionData.totalNutrition.calories || 1; // Prevent division by zero
  
  const proteinCaloriePercentage = Math.round((proteinCalories / totalCalories) * 100);
  const carbsCaloriePercentage = Math.round((carbsCalories / totalCalories) * 100);
  const fatCaloriePercentage = Math.round((fatCalories / totalCalories) * 100);
  
  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardContent className="pt-6">
        <h3 className="text-white font-medium mb-4">Macronutrient Balance</h3>
        
        <div className="space-y-6">
          {/* Protein */}
          <div className="space-y-2">
            <div className="flex justify-between items-end">
              <div>
                <span className="text-gray-400 text-sm">Protein</span>
                <div className="text-white font-medium">
                  {nutritionData.totalNutrition.protein}g / {nutritionData.macroTargets.protein}g
                </div>
              </div>
              <div className="text-right">
                <div className="text-blue-500 font-medium">{proteinCaloriePercentage}%</div>
                <div className="text-gray-400 text-xs">{proteinCalories} kcal</div>
              </div>
            </div>
            <Progress 
              value={proteinPercentage > 100 ? 100 : proteinPercentage} 
              className="h-2 bg-gray-700" 
              indicatorClassName="bg-blue-500"
            />
          </div>
          
          {/* Carbs */}
          <div className="space-y-2">
            <div className="flex justify-between items-end">
              <div>
                <span className="text-gray-400 text-sm">Carbohydrates</span>
                <div className="text-white font-medium">
                  {nutritionData.totalNutrition.carbs}g / {nutritionData.macroTargets.carbs}g
                </div>
              </div>
              <div className="text-right">
                <div className="text-green-500 font-medium">{carbsCaloriePercentage}%</div>
                <div className="text-gray-400 text-xs">{carbsCalories} kcal</div>
              </div>
            </div>
            <Progress 
              value={carbsPercentage > 100 ? 100 : carbsPercentage} 
              className="h-2 bg-gray-700" 
              indicatorClassName="bg-green-500"
            />
          </div>
          
          {/* Fats */}
          <div className="space-y-2">
            <div className="flex justify-between items-end">
              <div>
                <span className="text-gray-400 text-sm">Fats</span>
                <div className="text-white font-medium">
                  {nutritionData.totalNutrition.fat}g / {nutritionData.macroTargets.fat}g
                </div>
              </div>
              <div className="text-right">
                <div className="text-yellow-400 font-medium">{fatCaloriePercentage}%</div>
                <div className="text-gray-400 text-xs">{fatCalories} kcal</div>
              </div>
            </div>
            <Progress 
              value={fatPercentage > 100 ? 100 : fatPercentage} 
              className="h-2 bg-gray-700" 
              indicatorClassName="bg-yellow-400"
            />
          </div>
        </div>
        
        {/* Visual representation of macro balance */}
        <div className="mt-6">
          <div className="text-gray-400 text-sm mb-2">Calorie Distribution</div>
          <div className="flex h-6 rounded-md overflow-hidden">
            <div 
              className="bg-blue-500 flex items-center justify-center text-xs text-white font-medium"
              style={{ width: `${proteinCaloriePercentage}%` }}
            >
              {proteinCaloriePercentage >= 10 ? `${proteinCaloriePercentage}%` : ""}
            </div>
            <div 
              className="bg-green-500 flex items-center justify-center text-xs text-white font-medium"
              style={{ width: `${carbsCaloriePercentage}%` }}
            >
              {carbsCaloriePercentage >= 10 ? `${carbsCaloriePercentage}%` : ""}
            </div>
            <div 
              className="bg-yellow-400 flex items-center justify-center text-xs text-white font-medium"
              style={{ width: `${fatCaloriePercentage}%` }}
            >
              {fatCaloriePercentage >= 10 ? `${fatCaloriePercentage}%` : ""}
            </div>
          </div>
          
          <div className="flex justify-between mt-2 text-xs">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-blue-500 rounded-full mr-1"></div>
              <span className="text-gray-400">Protein</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-green-500 rounded-full mr-1"></div>
              <span className="text-gray-400">Carbs</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-yellow-400 rounded-full mr-1"></div>
              <span className="text-gray-400">Fat</span>
            </div>
          </div>
        </div>
        
        {/* Target ratio vs. actual ratio */}
        <div className="mt-6 bg-gray-700 p-3 rounded-md">
          <div className="text-white text-sm font-medium mb-1">Target vs. Actual</div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div>
              <div className="text-gray-400">Protein</div>
              <div className="text-white">
                Target: {Math.round((nutritionData.macroTargets.protein * 4 / nutritionData.calorieTarget) * 100)}%
              </div>
              <div className="text-blue-500">
                Actual: {proteinCaloriePercentage}%
              </div>
            </div>
            <div>
              <div className="text-gray-400">Carbs</div>
              <div className="text-white">
                Target: {Math.round((nutritionData.macroTargets.carbs * 4 / nutritionData.calorieTarget) * 100)}%
              </div>
              <div className="text-green-500">
                Actual: {carbsCaloriePercentage}%
              </div>
            </div>
            <div>
              <div className="text-gray-400">Fat</div>
              <div className="text-white">
                Target: {Math.round((nutritionData.macroTargets.fat * 9 / nutritionData.calorieTarget) * 100)}%
              </div>
              <div className="text-yellow-400">
                Actual: {fatCaloriePercentage}%
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 
'use client';

import { useState } from 'react';
import Link from 'next/link';
import { 
  Utensils, 
  ChevronRight, 
  ChevronDown, 
  PlusCircle, 
  MoreHorizontal, 
  Edit, 
  Trash, 
  Camera
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { DailyNutritionLog, MealLogEntry } from '../types';

interface MealListProps {
  nutritionData: DailyNutritionLog;
  onAddMeal?: () => void;
  onEditMeal?: (meal: MealLogEntry) => void;
  onDeleteMeal?: (mealId: string) => void;
}

export default function MealList({ 
  nutritionData, 
  onAddMeal, 
  onEditMeal, 
  onDeleteMeal 
}: MealListProps) {
  const [expandedMeals, setExpandedMeals] = useState<Record<string, boolean>>({});
  
  const toggleMealExpansion = (mealId: string) => {
    setExpandedMeals((prev) => ({
      ...prev,
      [mealId]: !prev[mealId]
    }));
  };
  
  // Function to get the meal icon based on meal type
  const getMealIcon = (mealType: string) => {
    switch (mealType) {
      case 'breakfast':
        return '🍳';
      case 'lunch':
        return '🥗';
      case 'dinner':
        return '🍽️';
      case 'snack':
        return '🍎';
      case 'pre_workout':
        return '🏋️';
      case 'post_workout':
        return '🥤';
      default:
        return '🍽️';
    }
  };
  
  // Format the meal type for display
  const formatMealType = (mealType: string) => {
    return mealType
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };
  
  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardHeader className="pb-2 flex flex-row items-center justify-between">
        <div>
          <CardTitle className="text-white">Today's Meals</CardTitle>
          <CardDescription className="text-gray-400">
            {nutritionData.meals.length} logged meals • {nutritionData.totalNutrition.calories} / {nutritionData.calorieTarget} kcal
          </CardDescription>
        </div>
        <Button size="sm" onClick={onAddMeal}>
          <PlusCircle className="mr-2 h-4 w-4" />
          Add Meal
        </Button>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {nutritionData.meals.length > 0 ? (
            nutritionData.meals.map((meal) => (
              <div key={meal.id} className="bg-gray-700 rounded-lg overflow-hidden">
                <div 
                  className="p-4 flex items-center justify-between cursor-pointer"
                  onClick={() => toggleMealExpansion(meal.id)}
                >
                  <div className="flex items-center">
                    <div className="w-10 h-10 rounded-full bg-gray-600 flex items-center justify-center mr-4">
                      <span className="text-lg">{getMealIcon(meal.mealType)}</span>
                    </div>
                    <div>
                      <h4 className="text-white font-medium">
                        {formatMealType(meal.mealType)}
                      </h4>
                      <p className="text-gray-400 text-sm">
                        {meal.time} • {meal.totalNutrition.calories} kcal
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center">
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon" onClick={(e) => e.stopPropagation()}>
                          <MoreHorizontal className="h-4 w-4 text-gray-400" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent className="bg-gray-800 border-gray-700 text-white">
                        <DropdownMenuItem 
                          className="cursor-pointer flex items-center"
                          onClick={(e) => {
                            e.stopPropagation();
                            onEditMeal && onEditMeal(meal);
                          }}
                        >
                          <Edit className="mr-2 h-4 w-4" />
                          <span>Edit Meal</span>
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          className="cursor-pointer flex items-center"
                          onClick={(e) => {
                            e.stopPropagation();
                            // Link to food analysis
                          }}
                        >
                          <Camera className="mr-2 h-4 w-4" />
                          <span>Analyze with Camera</span>
                        </DropdownMenuItem>
                        <DropdownMenuSeparator className="bg-gray-700" />
                        <DropdownMenuItem 
                          className="cursor-pointer text-red-400 flex items-center"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteMeal && onDeleteMeal(meal.id);
                          }}
                        >
                          <Trash className="mr-2 h-4 w-4" />
                          <span>Delete Meal</span>
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                    
                    <ChevronDown 
                      className={`h-5 w-5 text-gray-400 transform transition-transform ${
                        expandedMeals[meal.id] ? 'rotate-180' : ''
                      }`} 
                    />
                  </div>
                </div>
                
                {expandedMeals[meal.id] && (
                  <div className="px-4 pb-4 border-t border-gray-600 pt-3">
                    <h5 className="text-white text-sm font-medium mb-2">Food Items:</h5>
                    <div className="space-y-2">
                      {meal.items.map((item, index) => (
                        <div 
                          key={`${meal.id}-item-${index}`}
                          className="flex justify-between items-center bg-gray-750 p-2 rounded"
                        >
                          <div className="flex items-center">
                            <div className="w-6 h-6 rounded bg-gray-600 flex items-center justify-center mr-2">
                              <Utensils className="h-3 w-3 text-gray-400" />
                            </div>
                            <div>
                              <div className="text-white text-sm">
                                {/* This would be the actual food or recipe name in a real app */}
                                {item.itemType === 'recipe' ? 'Recipe Item' : 'Food Item'} 
                                <Badge variant="outline" className="ml-2 text-xs">
                                  {item.itemType}
                                </Badge>
                              </div>
                              <div className="text-gray-400 text-xs">
                                {item.servingSize} {item.servingUnit}
                              </div>
                            </div>
                          </div>
                          <div className="text-white text-sm">
                            {/* This would be the actual calories in a real app */}
                            {item.itemType === 'recipe' ? '350' : '120'} kcal
                          </div>
                        </div>
                      ))}
                    </div>
                    
                    <div className="mt-4 bg-gray-750 p-3 rounded">
                      <div className="grid grid-cols-3 gap-2 text-center">
                        <div>
                          <div className="text-xs text-gray-400">Protein</div>
                          <div className="text-white">{meal.totalNutrition.protein}g</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-400">Carbs</div>
                          <div className="text-white">{meal.totalNutrition.carbs}g</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-400">Fat</div>
                          <div className="text-white">{meal.totalNutrition.fat}g</div>
                        </div>
                      </div>
                    </div>
                    
                    {meal.notes && (
                      <div className="mt-3">
                        <div className="text-gray-400 text-xs">Notes:</div>
                        <div className="text-white text-sm">{meal.notes}</div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))
          ) : (
            <div className="text-center py-6 bg-gray-750 rounded-lg">
              <Utensils className="h-12 w-12 text-gray-600 mx-auto mb-3" />
              <h4 className="text-white font-medium mb-1">No meals logged today</h4>
              <p className="text-gray-400 text-sm mb-4">Start tracking your nutrition by adding a meal</p>
              <Button onClick={onAddMeal}>
                <PlusCircle className="mr-2 h-4 w-4" />
                Add Your First Meal
              </Button>
            </div>
          )}
          
          {/* Suggested next meal (if applicable) */}
          {nutritionData.meals.length > 0 && nutritionData.meals.length < 4 && (
            <div className="bg-gray-750 rounded-lg p-4 border border-dashed border-gray-600">
              <h4 className="text-white font-medium mb-2">Suggested Next Meal</h4>
              <p className="text-gray-400 text-sm mb-3">
                Based on your meal plan and nutrition goals
              </p>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center mr-3">
                    <span className="text-lg">
                      {nutritionData.meals.length === 3 ? '🍽️' : 
                       nutritionData.meals.length === 2 ? '🥗' : 
                       nutritionData.meals.length === 1 ? '🥤' : '🍎'}
                    </span>
                  </div>
                  <div>
                    <h5 className="text-white">
                      {nutritionData.meals.length === 3 ? 'Dinner' : 
                       nutritionData.meals.length === 2 ? 'Afternoon Snack' : 
                       nutritionData.meals.length === 1 ? 'Lunch' : 'Breakfast'}
                    </h5>
                    <p className="text-gray-400 text-sm">~500 kcal recommended</p>
                  </div>
                </div>
                <Button variant="outline" size="sm" onClick={onAddMeal}>
                  Add Meal
                </Button>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
} 
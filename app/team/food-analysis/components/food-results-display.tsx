'use client';

import { useState } from 'react';
import { 
  Save,
  ArrowRight, 
  PieChart, 
  Utensils, 
  Flame,
  ChevronDown,
  ChevronUp,
  Plus,
  Minus,
  Copy,
  BookmarkPlus,
  Share2,
  BarChart3
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Slider } from '@/components/ui/slider';

interface FoodItem {
  name: string;
  confidence: number;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  image: string;
}

interface AnalysisResults {
  foods: FoodItem[];
  totalCalories: number;
  totalProtein: number;
  totalCarbs: number;
  totalFat: number;
  mealType: string;
  suggestions: string[];
}

interface FoodResultsDisplayProps {
  results: AnalysisResults;
  onSaveToLog: () => void;
  onAdjustPortion: (foodIndex: number, portion: number) => void;
}

export function FoodResultsDisplay({ 
  results, 
  onSaveToLog, 
  onAdjustPortion 
}: FoodResultsDisplayProps) {
  const [expandedFood, setExpandedFood] = useState<number | null>(null);
  const [adjustedPortions, setAdjustedPortions] = useState<{[key: number]: number}>(
    Object.fromEntries(results.foods.map((_, i) => [i, 1]))
  );
  const [mealType, setMealType] = useState(results.mealType);
  const mealOptions = ['breakfast', 'lunch', 'dinner', 'snack'];
  
  // Toggle expanded food details
  const toggleExpandFood = (index: number) => {
    setExpandedFood(expandedFood === index ? null : index);
  };
  
  // Handle portion adjustment
  const handlePortionChange = (foodIndex: number, portion: number) => {
    setAdjustedPortions(prev => ({
      ...prev,
      [foodIndex]: portion
    }));
    
    onAdjustPortion(foodIndex, portion);
  };
  
  // Increment portion
  const incrementPortion = (foodIndex: number) => {
    const newPortion = Math.min(adjustedPortions[foodIndex] + 0.25, 3);
    handlePortionChange(foodIndex, newPortion);
  };
  
  // Decrement portion
  const decrementPortion = (foodIndex: number) => {
    const newPortion = Math.max(adjustedPortions[foodIndex] - 0.25, 0.25);
    handlePortionChange(foodIndex, newPortion);
  };
  
  // Calculate adjusted nutritional values based on portion sizes
  const getAdjustedValues = () => {
    let calories = 0;
    let protein = 0;
    let carbs = 0;
    let fat = 0;
    
    results.foods.forEach((food, index) => {
      const portionFactor = adjustedPortions[index] || 1;
      calories += food.calories * portionFactor;
      protein += food.protein * portionFactor;
      carbs += food.carbs * portionFactor;
      fat += food.fat * portionFactor;
    });
    
    return {
      calories: Math.round(calories),
      protein: Math.round(protein),
      carbs: Math.round(carbs),
      fat: Math.round(fat)
    };
  };
  
  const adjustedValues = getAdjustedValues();
  
  // Calculate macro percentages
  const totalGrams = adjustedValues.protein + adjustedValues.carbs + adjustedValues.fat;
  const proteinPercentage = totalGrams ? Math.round((adjustedValues.protein / totalGrams) * 100) : 0;
  const carbsPercentage = totalGrams ? Math.round((adjustedValues.carbs / totalGrams) * 100) : 0;
  const fatPercentage = totalGrams ? Math.round((adjustedValues.fat / totalGrams) * 100) : 0;

  return (
    <div className="food-results-display">
      {/* Summary Card */}
      <Card className="bg-gray-800 border-gray-700 text-white mb-6">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center text-blue-400">
            <Flame className="mr-2 h-5 w-5" />
            Nutritional Analysis
          </CardTitle>
          <CardDescription className="text-gray-400">
            {results.foods.length} item{results.foods.length !== 1 ? 's' : ''} detected
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              {/* Total calories */}
              <div className="mb-4">
                <div className="flex justify-between mb-1">
                  <span className="text-gray-300">Total Calories</span>
                  <span className="text-white font-bold">{adjustedValues.calories} kcal</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="relative h-3 w-full bg-gray-700 rounded-full overflow-hidden">
                    {/* This would typically be based on user's daily goals */}
                    <div 
                      className="absolute left-0 top-0 bottom-0 bg-gradient-to-r from-green-500 to-blue-500"
                      style={{ width: `${Math.min(100, (adjustedValues.calories / 2000) * 100)}%` }}
                    ></div>
                  </div>
                  <span className="text-xs text-gray-400 whitespace-nowrap">of 2000 daily goal</span>
                </div>
              </div>
              
              {/* Macronutrients distribution */}
              <div className="mb-4">
                <div className="flex justify-between mb-2">
                  <span className="text-gray-300">Macronutrients</span>
                  <span className="text-gray-400 text-xs">{totalGrams}g total</span>
                </div>
                <div className="h-4 w-full rounded-full overflow-hidden flex">
                  <div 
                    className="bg-blue-500 h-full flex items-center justify-center text-xs text-white"
                    style={{ width: `${proteinPercentage}%` }}
                  >
                    {proteinPercentage > 10 ? `${proteinPercentage}%` : ''}
                  </div>
                  <div 
                    className="bg-purple-500 h-full flex items-center justify-center text-xs text-white"
                    style={{ width: `${carbsPercentage}%` }}
                  >
                    {carbsPercentage > 10 ? `${carbsPercentage}%` : ''}
                  </div>
                  <div 
                    className="bg-yellow-500 h-full flex items-center justify-center text-xs text-white"
                    style={{ width: `${fatPercentage}%` }}
                  >
                    {fatPercentage > 10 ? `${fatPercentage}%` : ''}
                  </div>
                </div>
                <div className="flex justify-between text-xs mt-2">
                  <div className="flex items-center">
                    <div className="w-2 h-2 rounded-full bg-blue-500 mr-1"></div>
                    <span>Protein: {adjustedValues.protein}g</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-2 h-2 rounded-full bg-purple-500 mr-1"></div>
                    <span>Carbs: {adjustedValues.carbs}g</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-2 h-2 rounded-full bg-yellow-500 mr-1"></div>
                    <span>Fat: {adjustedValues.fat}g</span>
                  </div>
                </div>
              </div>
              
              {/* Meal type selector */}
              <div>
                <label className="text-sm text-gray-300 mb-2 block">Meal Type</label>
                <div className="flex space-x-2">
                  {mealOptions.map(option => (
                    <Button 
                      key={option}
                      size="sm"
                      variant={mealType === option ? "default" : "outline"}
                      className={mealType === option ? "bg-blue-600 text-white" : "text-gray-300 border-gray-600"}
                      onClick={() => setMealType(option)}
                    >
                      {option.charAt(0).toUpperCase() + option.slice(1)}
                    </Button>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <div className="p-4 rounded-lg bg-gray-700/50 border border-gray-700">
                <h3 className="text-sm font-medium text-gray-300 mb-2">AI Suggestions</h3>
                <ul className="space-y-2">
                  {results.suggestions.map((suggestion, i) => (
                    <li key={i} className="text-xs text-gray-400 flex">
                      <span className="mr-2">•</span>
                      <span>{suggestion}</span>
                    </li>
                  ))}
                </ul>
              </div>
              
              <div className="flex justify-center">
                <div className="flex flex-col items-center">
                  <div className="relative w-24 h-24">
                    <svg className="w-full h-full" viewBox="0 0 36 36">
                      <path
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                        fill="none"
                        stroke="#4B5563"
                        strokeWidth="2"
                      />
                      <path
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                        fill="none"
                        stroke="#3B82F6"
                        strokeWidth="2"
                        strokeDasharray={`${adjustedValues.calories / 20}, 100`}
                      />
                    </svg>
                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                      <div className="text-xl font-bold text-white">{adjustedValues.calories}</div>
                      <div className="text-xs text-gray-400">calories</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="border-t border-gray-700 mt-4 pt-4 flex justify-between">
            <Button 
              variant="outline" 
              size="sm" 
              className="text-gray-300 border-gray-600"
            >
              <BarChart3 className="h-4 w-4 mr-2" />
              Nutrition Facts
            </Button>
            <div className="space-x-2">
              <Button 
                variant="outline" 
                size="sm" 
                className="text-gray-300 border-gray-600"
              >
                <BookmarkPlus className="h-4 w-4 mr-2" />
                Save
              </Button>
              <Button 
                variant="outline" 
                size="sm" 
                className="text-gray-300 border-gray-600"
              >
                <Share2 className="h-4 w-4 mr-2" />
                Share
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Detected Foods List */}
      <h3 className="text-lg font-medium text-white mb-3">Detected Food Items</h3>
      {results.foods.map((food, index) => (
        <Card 
          key={index} 
          className={`bg-gray-800 border-gray-700 text-white mb-3 transition-all ${expandedFood === index ? 'ring-1 ring-blue-500' : ''}`}
        >
          <div className="flex items-center p-4">
            <div className="w-16 h-16 rounded-md overflow-hidden flex-shrink-0 mr-4 bg-gray-700">
              <img 
                src={food.image || `https://placehold.co/64x64/333/888?text=${encodeURIComponent(food.name)}`} 
                alt={food.name}
                className="w-full h-full object-cover"
              />
            </div>
            <div className="flex-grow">
              <div className="flex justify-between">
                <div>
                  <h4 className="font-medium text-white">{food.name}</h4>
                  <div className="text-sm text-gray-400">
                    {Math.round(food.calories * (adjustedPortions[index] || 1))} calories
                  </div>
                </div>
                <Button 
                  variant="ghost" 
                  size="icon" 
                  className="text-gray-400"
                  onClick={() => toggleExpandFood(index)}
                >
                  {expandedFood === index ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
                </Button>
              </div>
              
              {/* Confidence level indicator */}
              <div className="mt-1 flex items-center">
                <div className="text-xs text-gray-400 mr-2">Confidence:</div>
                <div className="w-24 bg-gray-700 h-1.5 rounded-full overflow-hidden">
                  <div 
                    className={`h-full ${food.confidence > 0.9 ? 'bg-green-500' : food.confidence > 0.8 ? 'bg-yellow-500' : 'bg-red-500'}`}
                    style={{ width: `${food.confidence * 100}%` }}
                  ></div>
                </div>
                <div className="text-xs text-gray-400 ml-2">{Math.round(food.confidence * 100)}%</div>
              </div>
            </div>
          </div>
          
          {/* Expanded details */}
          {expandedFood === index && (
            <div className="px-4 pb-4 pt-0">
              <div className="border-t border-gray-700 mt-2 pt-4">
                {/* Portion control */}
                <div className="mb-3">
                  <div className="flex justify-between mb-1">
                    <label className="text-sm text-gray-300">Portion Size</label>
                    <div className="flex items-center">
                      <Button 
                        variant="outline" 
                        size="icon" 
                        className="h-6 w-6 text-gray-400 border-gray-600"
                        onClick={() => decrementPortion(index)}
                        disabled={adjustedPortions[index] <= 0.25}
                      >
                        <Minus className="h-3 w-3" />
                      </Button>
                      <span className="mx-2 text-sm text-white min-w-[40px] text-center">
                        {adjustedPortions[index] || 1}x
                      </span>
                      <Button 
                        variant="outline" 
                        size="icon" 
                        className="h-6 w-6 text-gray-400 border-gray-600"
                        onClick={() => incrementPortion(index)}
                        disabled={adjustedPortions[index] >= 3}
                      >
                        <Plus className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                  <Slider
                    value={[adjustedPortions[index] * 100]}
                    min={25}
                    max={300}
                    step={25}
                    onValueChange={(value: number[]) => handlePortionChange(index, value[0] / 100)}
                    className="mt-1"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>0.25x</span>
                    <span>1x</span>
                    <span>3x</span>
                  </div>
                </div>
                
                {/* Nutrition breakdown */}
                <div className="space-y-2">
                  <div className="grid grid-cols-3 gap-2 text-xs text-center">
                    <div className="bg-gray-700 rounded-md p-2">
                      <div className="text-gray-400">Proteins</div>
                      <div className="text-white font-medium">{Math.round(food.protein * (adjustedPortions[index] || 1))}g</div>
                    </div>
                    <div className="bg-gray-700 rounded-md p-2">
                      <div className="text-gray-400">Carbs</div>
                      <div className="text-white font-medium">{Math.round(food.carbs * (adjustedPortions[index] || 1))}g</div>
                    </div>
                    <div className="bg-gray-700 rounded-md p-2">
                      <div className="text-gray-400">Fat</div>
                      <div className="text-white font-medium">{Math.round(food.fat * (adjustedPortions[index] || 1))}g</div>
                    </div>
                  </div>
                </div>
                
                {/* Action buttons */}
                <div className="flex justify-between mt-4">
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="text-gray-300 border-gray-600"
                  >
                    <Copy className="h-3 w-3 mr-1" />
                    Copy
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="text-gray-300 border-gray-600"
                  >
                    <PieChart className="h-3 w-3 mr-1" />
                    Alternatives
                  </Button>
                </div>
              </div>
            </div>
          )}
        </Card>
      ))}
      
      {/* Save button */}
      <Button 
        className="w-full mt-6 bg-blue-600 hover:bg-blue-700 text-white"
        onClick={onSaveToLog}
      >
        <Save className="mr-2 h-5 w-5" />
        Save to Meal Log
      </Button>
      
      {/* Navigate to log button */}
      <Button 
        variant="outline" 
        className="w-full mt-3 border-gray-700 text-gray-300"
      >
        <Utensils className="mr-2 h-5 w-5" />
        Go to Meal Log <ArrowRight className="ml-2 h-4 w-4" />
      </Button>
      
      {/* Disclaimer */}
      <p className="mt-4 text-xs text-gray-500 text-center">
        Nutritional values are estimates based on AI analysis and may vary. 
        Values are per portion size as adjusted above.
      </p>
    </div>
  );
} 
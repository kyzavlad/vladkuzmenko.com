'use client';

import { useState } from 'react';
import Link from 'next/link';
import { 
  Activity, 
  Calendar, 
  LayoutGrid, 
  ArrowRight, 
  Camera, 
  Utensils, 
  ClipboardList, 
  ChevronRight, 
  BarChart3,
  Cookie,
  Droplets,
  GraduationCap,
  Award,
  PlusCircle,
  Coffee
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';

// Import our component types
import { DailyNutritionLog, MealType } from './types';

// Import our components
import HydrationTracker from './components/hydration-tracker';
import NutrientQualityScore from './components/nutrient-quality-score';
import WeeklyTrends from './components/weekly-trends';
import MealList from './components/meal-list';

// Mock data for daily nutrition
const mockNutritionData: DailyNutritionLog = {
  date: new Date(),
  meals: [
    {
      id: 'meal-1',
      type: MealType.BREAKFAST,
      name: 'Morning Protein Oats',
      time: new Date().setHours(8, 30),
      items: [
        {
          item: {
            id: 'food-1',
            name: 'Oatmeal',
            category: 'Grains',
            servingSize: { amount: 50, unit: 'g' },
            calories: 180,
            macros: { protein: 5, carbs: 32, fat: 3, fiber: 5, sugar: 1 },
            micros: {},
            tags: ['grain', 'fiber'],
          },
          quantity: 1
        },
        {
          item: {
            id: 'food-2',
            name: 'Whey Protein Powder',
            category: 'Protein',
            servingSize: { amount: 30, unit: 'g' },
            calories: 120,
            macros: { protein: 24, carbs: 3, fat: 2, fiber: 0, sugar: 2 },
            micros: {},
            tags: ['protein', 'supplement'],
          },
          quantity: 1
        },
        {
          item: {
            id: 'food-3',
            name: 'Banana',
            category: 'Fruits',
            servingSize: { amount: 1, unit: 'medium' },
            calories: 105,
            macros: { protein: 1.3, carbs: 27, fat: 0.4, fiber: 3.1, sugar: 14 },
            micros: {},
            tags: ['fruit', 'potassium'],
          },
          quantity: 1
        }
      ],
      notes: 'Added cinnamon and a bit of honey',
      totalNutrition: {
        calories: 405,
        macros: { protein: 30.3, carbs: 62, fat: 5.4, fiber: 8.1, sugar: 17 }
      }
    },
    {
      id: 'meal-2',
      type: MealType.LUNCH,
      name: 'Chicken Salad',
      time: new Date().setHours(13, 0),
      items: [
        {
          item: {
            id: 'food-4',
            name: 'Grilled Chicken Breast',
            category: 'Protein',
            servingSize: { amount: 100, unit: 'g' },
            calories: 165,
            macros: { protein: 31, carbs: 0, fat: 3.6, fiber: 0, sugar: 0 },
            micros: {},
            tags: ['meat', 'high-protein'],
          },
          quantity: 1
        },
        {
          item: {
            id: 'food-5',
            name: 'Mixed Greens',
            category: 'Vegetables',
            servingSize: { amount: 100, unit: 'g' },
            calories: 25,
            macros: { protein: 2, carbs: 5, fat: 0.3, fiber: 2.8, sugar: 0.8 },
            micros: {},
            tags: ['vegetable', 'fiber'],
          },
          quantity: 1
        },
        {
          item: {
            id: 'food-6',
            name: 'Olive Oil Dressing',
            category: 'Oils & Fats',
            servingSize: { amount: 15, unit: 'ml' },
            calories: 120,
            macros: { protein: 0, carbs: 0, fat: 14, fiber: 0, sugar: 0 },
            micros: {},
            tags: ['oil', 'healthy-fat'],
          },
          quantity: 1
        }
      ],
      notes: '',
      totalNutrition: {
        calories: 310,
        macros: { protein: 33, carbs: 5, fat: 17.9, fiber: 2.8, sugar: 0.8 }
      }
    },
    {
      id: 'meal-3',
      type: MealType.PRE_WORKOUT,
      name: 'Pre-Workout Snack',
      time: new Date().setHours(16, 30),
      items: [
        {
          item: {
            id: 'food-7',
            name: 'Apple',
            category: 'Fruits',
            servingSize: { amount: 1, unit: 'medium' },
            calories: 95,
            macros: { protein: 0.5, carbs: 25, fat: 0.3, fiber: 4.4, sugar: 19 },
            micros: {},
            tags: ['fruit', 'fiber'],
          },
          quantity: 1
        },
        {
          item: {
            id: 'food-8',
            name: 'Almond Butter',
            category: 'Nuts & Seeds',
            servingSize: { amount: 15, unit: 'g' },
            calories: 98,
            macros: { protein: 3.4, carbs: 3, fat: 8.9, fiber: 1.6, sugar: 0.7 },
            micros: {},
            tags: ['nut', 'healthy-fat'],
          },
          quantity: 1
        }
      ],
      notes: 'Eaten 1 hour before workout',
      totalNutrition: {
        calories: 193,
        macros: { protein: 3.9, carbs: 28, fat: 9.2, fiber: 6, sugar: 19.7 }
      }
    }
  ],
  waterIntake: 1800, // ml
  goals: {
    calories: 2400,
    macros: { protein: 180, carbs: 250, fat: 80, fiber: 30, sugar: 40 },
    water: 2500 // ml
  },
  actualTotals: {
    calories: 908, // Calculated from meals so far
    macros: { 
      protein: 67.2, 
      carbs: 95, 
      fat: 32.5, 
      fiber: 16.9, 
      sugar: 37.5 
    },
    water: 1800 // ml
  },
  nutrientQualityScore: 83 // 0-100
};

// Progress percentages
const caloriePercentage = Math.round((mockNutritionData.actualTotals.calories / mockNutritionData.goals.calories) * 100);
const proteinPercentage = Math.round((mockNutritionData.actualTotals.macros.protein / mockNutritionData.goals.macros.protein) * 100);
const carbsPercentage = Math.round((mockNutritionData.actualTotals.macros.carbs / mockNutritionData.goals.macros.carbs) * 100);
const fatPercentage = Math.round((mockNutritionData.actualTotals.macros.fat / mockNutritionData.goals.macros.fat) * 100);
const waterPercentage = Math.round((mockNutritionData.waterIntake / mockNutritionData.goals.water) * 100);

// Insights mock data
const mockInsights = [
  {
    id: 'insight-1',
    type: 'tip',
    title: 'More protein needed',
    description: 'You\'re currently at 37% of your daily protein goal. Consider adding a protein-rich snack after your workout.',
    priority: 'medium'
  },
  {
    id: 'insight-2',
    type: 'alert',
    title: 'Sugar intake approaching limit',
    description: 'You\'ve consumed 94% of your daily sugar target. Try to limit added sugars for the rest of the day.',
    priority: 'high'
  },
  {
    id: 'insight-3',
    type: 'suggestion',
    title: 'Missing essential nutrients',
    description: 'Your meals today are low in vitamin D and omega-3 fatty acids. Consider adding fatty fish or a supplement.',
    priority: 'medium'
  }
];

export default function NutritionDashboard() {
  const [activeTab, setActiveTab] = useState('overview');
  
  // Format date for display
  const formatDate = (date: Date | string) => {
    const d = new Date(date);
    return d.toLocaleDateString('en-US', { 
      weekday: 'short', 
      month: 'short', 
      day: 'numeric' 
    });
  };
  
  // Calculate remaining calories and macros
  const remainingCalories = mockNutritionData.goals.calories - mockNutritionData.actualTotals.calories;
  const remainingProtein = mockNutritionData.goals.macros.protein - mockNutritionData.actualTotals.macros.protein;
  const remainingCarbs = mockNutritionData.goals.macros.carbs - mockNutritionData.actualTotals.macros.carbs;
  const remainingFat = mockNutritionData.goals.macros.fat - mockNutritionData.actualTotals.macros.fat;
  
  return (
    <div className="nutrition-dashboard p-4 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-1">Nutrition Dashboard</h1>
        <p className="text-gray-400">Track, analyze, and optimize your nutrition</p>
      </div>
      
      {/* Quick Actions */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        <Button variant="outline" className="flex flex-col h-24 p-2 justify-center" asChild>
          <Link href="/team/food-analysis">
            <Camera className="h-6 w-6 mb-2" />
            <span>Analyze Food</span>
          </Link>
        </Button>
        
        <Button variant="outline" className="flex flex-col h-24 p-2 justify-center" asChild>
          <Link href="/team/nutrition/meals/add">
            <Utensils className="h-6 w-6 mb-2" />
            <span>Log Meal</span>
          </Link>
        </Button>
        
        <Button variant="outline" className="flex flex-col h-24 p-2 justify-center" asChild>
          <Link href="/team/nutrition/meal-planner">
            <ClipboardList className="h-6 w-6 mb-2" />
            <span>Meal Planner</span>
          </Link>
        </Button>
        
        <Button variant="outline" className="flex flex-col h-24 p-2 justify-center" asChild>
          <Link href="/team/nutrition/recipes">
            <Cookie className="h-6 w-6 mb-2" />
            <span>Recipes</span>
          </Link>
        </Button>
      </div>
      
      {/* Date Selector and Status */}
      <div className="flex flex-wrap justify-between items-center mb-6">
        <div className="flex items-center">
          <h2 className="text-xl font-bold text-white">{formatDate(mockNutritionData.date)}</h2>
          <Button variant="ghost" size="sm" className="ml-2">
            <Calendar className="h-4 w-4 mr-1" />
            <span>Change</span>
          </Button>
        </div>
        
        <div className="flex items-center mt-3 sm:mt-0">
          <Badge className="mr-3">
            <Activity className="h-3 w-3 mr-1" />
            Rest Day
          </Badge>
          <Button variant="outline" size="sm" asChild>
            <Link href="/team/nutrition/insights">
              <BarChart3 className="h-4 w-4 mr-1" />
              <span>Insights</span>
            </Link>
          </Button>
        </div>
      </div>
      
      <Tabs defaultValue="overview" className="mb-8" onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-4 mb-6">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="meals">Meals</TabsTrigger>
          <TabsTrigger value="trends">Trends</TabsTrigger>
          <TabsTrigger value="insights">Insights</TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview" className="space-y-6">
          {/* Macros and Calories Summary */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-white">Daily Summary</CardTitle>
              <CardDescription className="text-gray-400">Your nutrition targets for today</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-1">
                    <div className="text-sm text-gray-400">Calories</div>
                    <div className="text-sm text-white">
                      {mockNutritionData.actualTotals.calories} / {mockNutritionData.goals.calories} kcal
                    </div>
                  </div>
                  <Progress value={caloriePercentage} className="h-2" />
                  <div className="mt-1 text-xs text-right text-gray-400">
                    {remainingCalories > 0 ? `${remainingCalories} kcal remaining` : 'Goal reached'}
                  </div>
                </div>
                
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <div className="flex justify-between mb-1">
                      <div className="text-sm text-gray-400">Protein</div>
                      <div className="text-sm text-white">
                        {mockNutritionData.actualTotals.macros.protein}g
                      </div>
                    </div>
                    <Progress value={proteinPercentage} className="h-2" color="blue" />
                    <div className="mt-1 text-xs text-right text-gray-400">
                      {remainingProtein > 0 ? `${remainingProtein}g left` : 'Goal reached'}
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between mb-1">
                      <div className="text-sm text-gray-400">Carbs</div>
                      <div className="text-sm text-white">
                        {mockNutritionData.actualTotals.macros.carbs}g
                      </div>
                    </div>
                    <Progress value={carbsPercentage} className="h-2" color="amber" />
                    <div className="mt-1 text-xs text-right text-gray-400">
                      {remainingCarbs > 0 ? `${remainingCarbs}g left` : 'Goal reached'}
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between mb-1">
                      <div className="text-sm text-gray-400">Fat</div>
                      <div className="text-sm text-white">
                        {mockNutritionData.actualTotals.macros.fat}g
                      </div>
                    </div>
                    <Progress value={fatPercentage} className="h-2" color="pink" />
                    <div className="mt-1 text-xs text-right text-gray-400">
                      {remainingFat > 0 ? `${remainingFat}g left` : 'Goal reached'}
                    </div>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4 mt-2">
                  <div className="bg-gray-750 rounded-md p-3">
                    <div className="text-sm font-medium text-white mb-1">Macronutrient Ratio</div>
                    <div className="flex items-center">
                      <div style={{ width: '40%' }} className="bg-blue-600 h-3 rounded-l-sm"></div>
                      <div style={{ width: '35%' }} className="bg-amber-500 h-3"></div>
                      <div style={{ width: '25%' }} className="bg-pink-500 h-3 rounded-r-sm"></div>
                    </div>
                    <div className="flex justify-between mt-1 text-xs">
                      <div className="text-blue-400">40% Protein</div>
                      <div className="text-amber-400">35% Carbs</div>
                      <div className="text-pink-400">25% Fat</div>
                    </div>
                  </div>
                  
                  <div className="bg-gray-750 rounded-md p-3">
                    <div className="text-sm font-medium text-white mb-1">Water Intake</div>
                    <div className="flex justify-between mb-1">
                      <div className="text-sm text-gray-400">Today's Goal</div>
                      <div className="text-sm text-white">
                        {mockNutritionData.waterIntake} / {mockNutritionData.goals.water} ml
                      </div>
                    </div>
                    <Progress value={waterPercentage} className="h-2" color="cyan" />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Hydration Tracker */}
          <HydrationTracker nutritionData={mockNutritionData} />
          
          {/* Nutrient Quality Score */}
          <NutrientQualityScore nutritionData={mockNutritionData} />
          
          {/* Weekly Trends */}
          <WeeklyTrends nutritionData={mockNutritionData} />
        </TabsContent>
        
        <TabsContent value="meals" className="space-y-6">
          {/* Meal List */}
          <MealList nutritionData={mockNutritionData} />
          
          {/* Meal Suggestions */}
          <Card className="bg-gray-800 border-gray-700 mt-6">
            <CardHeader className="pb-2">
              <CardTitle className="text-white">Meal Suggestions</CardTitle>
              <CardDescription className="text-gray-400">
                Recommended meals based on your goals
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center p-3 bg-gray-750 rounded-md">
                  <div className="w-12 h-12 rounded bg-gray-700 flex items-center justify-center mr-3">
                    <Coffee className="h-5 w-5 text-gray-500" />
                  </div>
                  <div className="flex-grow">
                    <h4 className="text-white font-medium">Protein Smoothie</h4>
                    <div className="flex items-center text-gray-400 text-xs">
                      <span className="mr-2">320 kcal</span>
                      <span className="mr-2">30g protein</span>
                      <Badge variant="outline" className="text-xs">Perfect for dinner</Badge>
                    </div>
                  </div>
                  <Button variant="ghost" size="sm" asChild>
                    <Link href="/team/nutrition/recipes/123">
                      <span className="sr-only">View Recipe</span>
                      <ChevronRight className="h-5 w-5" />
                    </Link>
                  </Button>
                </div>
                
                <div className="flex items-center p-3 bg-gray-750 rounded-md">
                  <div className="w-12 h-12 rounded bg-gray-700 flex items-center justify-center mr-3">
                    <Utensils className="h-5 w-5 text-gray-500" />
                  </div>
                  <div className="flex-grow">
                    <h4 className="text-white font-medium">Salmon with Roasted Vegetables</h4>
                    <div className="flex items-center text-gray-400 text-xs">
                      <span className="mr-2">480 kcal</span>
                      <span className="mr-2">35g protein</span>
                      <Badge variant="outline" className="text-xs">Fits macros</Badge>
                    </div>
                  </div>
                  <Button variant="ghost" size="sm" asChild>
                    <Link href="/team/nutrition/recipes/456">
                      <span className="sr-only">View Recipe</span>
                      <ChevronRight className="h-5 w-5" />
                    </Link>
                  </Button>
                </div>
                
                <Button variant="outline" className="w-full" asChild>
                  <Link href="/team/nutrition/recipes">
                    <PlusCircle className="h-4 w-4 mr-2" />
                    View All Recipes
                  </Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="trends" className="space-y-6">
          {/* Weekly Trends (Full View) */}
          <WeeklyTrends nutritionData={mockNutritionData} />
          
          {/* Additional Trend Analysis */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-white">Long-Term Trends</CardTitle>
              <CardDescription className="text-gray-400">
                Your nutrition patterns over time
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8">
                <BarChart3 className="h-12 w-12 text-gray-500 mx-auto mb-4" />
                <h3 className="text-white text-lg font-medium mb-2">Detailed Trends Coming Soon</h3>
                <p className="text-gray-400 mb-4">We're building a comprehensive trends analysis feature to help you understand your nutrition patterns.</p>
                <Button variant="outline" asChild>
                  <Link href="/team/nutrition/insights">
                    View Available Insights
                  </Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="insights" className="space-y-6">
          {/* Nutrition Insights */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-white">Nutrition Insights</CardTitle>
              <CardDescription className="text-gray-400">
                Personalized guidance based on your data
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {mockInsights.map(insight => (
                  <div key={insight.id} className={`p-4 rounded-md border flex ${
                    insight.priority === 'high' ? 'bg-red-950/20 border-red-800' :
                    insight.priority === 'medium' ? 'bg-amber-950/20 border-amber-800' :
                    'bg-blue-950/20 border-blue-800'
                  }`}>
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center mr-3 ${
                      insight.priority === 'high' ? 'bg-red-800' :
                      insight.priority === 'medium' ? 'bg-amber-800' :
                      'bg-blue-800'
                    }`}>
                      {insight.type === 'tip' && <GraduationCap className="h-4 w-4 text-white" />}
                      {insight.type === 'alert' && <AlertTriangle className="h-4 w-4 text-white" />}
                      {insight.type === 'suggestion' && <Award className="h-4 w-4 text-white" />}
                    </div>
                    <div>
                      <h4 className="text-white font-medium">{insight.title}</h4>
                      <p className="text-gray-400 text-sm">{insight.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
            <CardFooter className="border-t border-gray-700 pt-4">
              <Button className="w-full" variant="outline" asChild>
                <Link href="/team/nutrition/insights">
                  View All Insights
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>
          
          {/* Educational Content */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-white">Nutrition Education</CardTitle>
              <CardDescription className="text-gray-400">
                Learn about nutrition to make better choices
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center p-3 bg-gray-750 rounded-md">
                  <div className="w-12 h-12 rounded bg-gray-700 flex items-center justify-center mr-3">
                    <GraduationCap className="h-5 w-5 text-gray-500" />
                  </div>
                  <div className="flex-grow">
                    <h4 className="text-white font-medium">Macronutrients 101</h4>
                    <p className="text-gray-400 text-xs">Understanding proteins, carbs, and fats</p>
                  </div>
                  <Button variant="ghost" size="sm" asChild>
                    <Link href="#">
                      <span className="sr-only">Read Article</span>
                      <ChevronRight className="h-5 w-5" />
                    </Link>
                  </Button>
                </div>
                
                <div className="flex items-center p-3 bg-gray-750 rounded-md">
                  <div className="w-12 h-12 rounded bg-gray-700 flex items-center justify-center mr-3">
                    <GraduationCap className="h-5 w-5 text-gray-500" />
                  </div>
                  <div className="flex-grow">
                    <h4 className="text-white font-medium">Nutrient Timing</h4>
                    <p className="text-gray-400 text-xs">When to eat for optimal performance</p>
                  </div>
                  <Button variant="ghost" size="sm" asChild>
                    <Link href="#">
                      <span className="sr-only">Read Article</span>
                      <ChevronRight className="h-5 w-5" />
                    </Link>
                  </Button>
                </div>
                
                <div className="flex items-center p-3 bg-gray-750 rounded-md">
                  <div className="w-12 h-12 rounded bg-gray-700 flex items-center justify-center mr-3">
                    <GraduationCap className="h-5 w-5 text-gray-500" />
                  </div>
                  <div className="flex-grow">
                    <h4 className="text-white font-medium">How to Read Food Labels</h4>
                    <p className="text-gray-400 text-xs">Decoding nutrition facts and ingredients</p>
                  </div>
                  <Button variant="ghost" size="sm" asChild>
                    <Link href="#">
                      <span className="sr-only">Read Article</span>
                      <ChevronRight className="h-5 w-5" />
                    </Link>
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
} 
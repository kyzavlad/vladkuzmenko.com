'use client';

import { useState } from 'react';
import Link from 'next/link';
import { 
  ArrowLeft, 
  Calendar, 
  PlusCircle, 
  Trash2, 
  Copy, 
  Filter, 
  ChevronRight,
  Clock,
  Utensils,
  ListFilter,
  MoreHorizontal,
  ShoppingBag,
  ArrowRight,
  Edit,
  Heart
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';

import { sampleMealPlan, sampleRecipes } from '../data/sample-data';
import { DietaryPreference, MealType } from '../types';

export default function MealPlanner() {
  const [activeTab, setActiveTab] = useState('current');
  const [activeMealPlan, setActiveMealPlan] = useState(sampleMealPlan);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDay, setSelectedDay] = useState(0); // Default to first day
  
  // Mock data for meal plans
  const mealPlans = [
    {
      id: 'plan-1',
      name: 'Muscle Building Plan',
      description: 'High protein meal plan for muscle growth',
      days: 7,
      calories: 2500,
      dietaryPreferences: [DietaryPreference.GLUTEN_FREE],
      isActive: true,
      isFavorite: false
    },
    {
      id: 'plan-2',
      name: 'Fat Loss Plan',
      description: 'Calorie-controlled plan for fat loss',
      days: 7,
      calories: 1800,
      dietaryPreferences: [DietaryPreference.LOW_CARB],
      isActive: false,
      isFavorite: true
    },
    {
      id: 'plan-3',
      name: 'Maintenance Plan',
      description: 'Balanced nutrition for weight maintenance',
      days: 7,
      calories: 2200,
      dietaryPreferences: [DietaryPreference.MEDITERRANEAN],
      isActive: false,
      isFavorite: false
    }
  ];
  
  // Days of the week for selection
  const daysOfWeek = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  
  // Function to format meal type for display
  const formatMealType = (mealType: MealType) => {
    return mealType
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };
  
  return (
    <div className="meal-planner">
      {/* Header with navigation */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center">
          <Button
            variant="ghost"
            size="icon"
            className="mr-2"
            asChild
          >
            <Link href="/team/nutrition">
              <ArrowLeft className="h-5 w-5" />
            </Link>
          </Button>
          <div>
            <h1 className="text-2xl font-bold text-white">Meal Planner</h1>
            <p className="text-gray-400">Create and manage your meal plans</p>
          </div>
        </div>
        
        <Button size="sm">
          <PlusCircle className="mr-2 h-4 w-4" />
          Create New Plan
        </Button>
      </div>
      
      {/* Main Tabs */}
      <Tabs defaultValue="current" value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="bg-gray-800">
          <TabsTrigger value="current">Current Plan</TabsTrigger>
          <TabsTrigger value="saved">Saved Plans</TabsTrigger>
          <TabsTrigger value="templates">Templates</TabsTrigger>
          <TabsTrigger value="generator">AI Generator</TabsTrigger>
        </TabsList>
        
        {/* Current Plan Content */}
        <TabsContent value="current" className="space-y-6">
          {/* Plan Header Card */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="pb-2">
              <div className="flex justify-between items-start">
                <div>
                  <CardTitle className="text-white">{activeMealPlan.name}</CardTitle>
                  <CardDescription className="text-gray-400">
                    {activeMealPlan.description}
                  </CardDescription>
                </div>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon">
                      <MoreHorizontal className="h-5 w-5 text-gray-400" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent className="bg-gray-800 border-gray-700 text-white">
                    <DropdownMenuItem className="cursor-pointer flex items-center">
                      <Edit className="mr-2 h-4 w-4" />
                      <span>Edit Plan</span>
                    </DropdownMenuItem>
                    <DropdownMenuItem className="cursor-pointer flex items-center">
                      <Copy className="mr-2 h-4 w-4" />
                      <span>Duplicate Plan</span>
                    </DropdownMenuItem>
                    <DropdownMenuItem className="cursor-pointer flex items-center">
                      <ShoppingBag className="mr-2 h-4 w-4" />
                      <span>Generate Grocery List</span>
                    </DropdownMenuItem>
                    <DropdownMenuSeparator className="bg-gray-700" />
                    <DropdownMenuItem className="cursor-pointer text-red-400 flex items-center">
                      <Trash2 className="mr-2 h-4 w-4" />
                      <span>Delete Plan</span>
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2 mb-4">
                <Badge variant="outline">{activeMealPlan.calorieTarget} kcal/day</Badge>
                <Badge variant="outline">
                  {activeMealPlan.macroTargets.protein}g P • {activeMealPlan.macroTargets.carbs}g C • {activeMealPlan.macroTargets.fat}g F
                </Badge>
                {activeMealPlan.dietaryPreferences.map((pref) => (
                  <Badge key={pref} variant="outline">{pref}</Badge>
                ))}
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-gray-700 p-3 rounded-md">
                  <h3 className="text-white font-medium mb-2">Plan Duration</h3>
                  <div className="flex items-center">
                    <Calendar className="h-5 w-5 text-gray-400 mr-2" />
                    <div>
                      <div className="text-white">
                        {activeMealPlan.startDate} - {activeMealPlan.endDate}
                      </div>
                      <div className="text-gray-400 text-xs">
                        {activeMealPlan.days.length} day plan
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-gray-700 p-3 rounded-md">
                  <h3 className="text-white font-medium mb-2">Quick Actions</h3>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" className="flex-1">
                      <ShoppingBag className="mr-1 h-4 w-4" />
                      Grocery List
                    </Button>
                    <Button variant="outline" size="sm" className="flex-1">
                      <Edit className="mr-1 h-4 w-4" />
                      Edit Plan
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Day Selection and Meal Display */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {/* Day Selector */}
            <Card className="bg-gray-800 border-gray-700 md:col-span-1">
              <CardHeader className="pb-2">
                <CardTitle className="text-white text-base">Days</CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <div className="space-y-1 p-1">
                  {daysOfWeek.map((day, index) => (
                    <button
                      key={index}
                      className={`w-full text-left px-3 py-2 rounded-sm ${
                        selectedDay === index 
                          ? 'bg-blue-600 text-white' 
                          : 'text-gray-300 hover:bg-gray-700'
                      }`}
                      onClick={() => setSelectedDay(index)}
                    >
                      {day}
                    </button>
                  ))}
                </div>
              </CardContent>
            </Card>
            
            {/* Meals for Selected Day */}
            <Card className="bg-gray-800 border-gray-700 md:col-span-3">
              <CardHeader className="pb-2">
                <CardTitle className="text-white">
                  {daysOfWeek[selectedDay]} Meals
                </CardTitle>
                <CardDescription className="text-gray-400">
                  {activeMealPlan.days[0]?.totalNutrition.calories || 0} kcal • 
                  {activeMealPlan.days[0]?.totalNutrition.protein || 0}g protein • 
                  {activeMealPlan.days[0]?.totalNutrition.carbs || 0}g carbs • 
                  {activeMealPlan.days[0]?.totalNutrition.fat || 0}g fat
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {/* Display meals for the selected day */}
                  {activeMealPlan.days[0]?.meals.map((meal, index) => (
                    <div key={index} className="bg-gray-700 rounded-lg p-4">
                      <div className="flex justify-between items-center mb-3">
                        <div className="flex items-center">
                          <div className="w-10 h-10 rounded-full bg-gray-600 flex items-center justify-center mr-3">
                            <Utensils className="h-5 w-5 text-gray-400" />
                          </div>
                          <div>
                            <h4 className="text-white font-medium">
                              {formatMealType(meal.mealType)}
                            </h4>
                            <p className="text-gray-400 text-xs">
                              {meal.time}
                            </p>
                          </div>
                        </div>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon">
                              <MoreHorizontal className="h-4 w-4 text-gray-400" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent className="bg-gray-800 border-gray-700 text-white">
                            <DropdownMenuItem className="cursor-pointer flex items-center">
                              <Edit className="mr-2 h-4 w-4" />
                              <span>Edit Meal</span>
                            </DropdownMenuItem>
                            <DropdownMenuItem className="cursor-pointer flex items-center">
                              <Clock className="mr-2 h-4 w-4" />
                              <span>Change Time</span>
                            </DropdownMenuItem>
                            <DropdownMenuSeparator className="bg-gray-700" />
                            <DropdownMenuItem className="cursor-pointer text-red-400 flex items-center">
                              <Trash2 className="mr-2 h-4 w-4" />
                              <span>Remove Meal</span>
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                      
                      {/* Display meal items */}
                      <div className="space-y-2 mb-3">
                        {meal.items.map((item, i) => (
                          <div key={i} className="flex justify-between items-center bg-gray-750 p-2 rounded-md">
                            <div className="text-white">
                              {/* In a real app, this would show the actual food/recipe name */}
                              {item.itemType === 'recipe' ? 'Recipe' : 'Food Item'} ({item.servingSize} {item.servingUnit})
                            </div>
                            <Button variant="ghost" size="icon">
                              <Edit className="h-4 w-4 text-gray-400" />
                            </Button>
                          </div>
                        ))}
                      </div>
                      
                      <Button variant="outline" size="sm" className="w-full">
                        <PlusCircle className="mr-2 h-4 w-4" />
                        Add Food Item
                      </Button>
                      
                      {meal.notes && (
                        <div className="mt-3 text-gray-400 text-sm">
                          <span className="font-medium text-white">Note:</span> {meal.notes}
                        </div>
                      )}
                    </div>
                  ))}
                  
                  <Button variant="outline" className="w-full">
                    <PlusCircle className="mr-2 h-4 w-4" />
                    Add Meal
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
          
          {/* Recipe Suggestions */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="pb-2 flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-white">Recipe Suggestions</CardTitle>
                <CardDescription className="text-gray-400">
                  Personalized recipes that match your dietary preferences
                </CardDescription>
              </div>
              <Button variant="ghost" size="sm" asChild>
                <Link href="/team/nutrition/recipes">
                  View All <ChevronRight className="ml-1 h-4 w-4" />
                </Link>
              </Button>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {sampleRecipes.slice(0, 3).map((recipe) => (
                  <Card key={recipe.id} className="bg-gray-750 border-gray-700">
                    <div className="aspect-video bg-gray-700 rounded-t-lg flex items-center justify-center relative">
                      <Utensils className="h-10 w-10 text-gray-500" />
                      <Button 
                        variant="ghost" 
                        size="icon" 
                        className="absolute top-2 right-2 text-gray-400 hover:text-red-400 bg-gray-800 bg-opacity-75"
                      >
                        <Heart className={`h-4 w-4 ${recipe.isFavorite ? 'text-red-500 fill-red-500' : ''}`} />
                      </Button>
                    </div>
                    <CardContent className="p-3">
                      <h3 className="text-white font-medium">{recipe.name}</h3>
                      <div className="flex items-center gap-2 my-2">
                        <Badge variant="secondary" className="text-xs">
                          {recipe.nutrition.perServing.calories} kcal
                        </Badge>
                        <Badge variant="secondary" className="text-xs">
                          {recipe.prepTime + recipe.cookTime} min
                        </Badge>
                        <Badge variant="secondary" className="text-xs">
                          {recipe.difficulty}
                        </Badge>
                      </div>
                      <p className="text-gray-400 text-xs line-clamp-2 mb-3">
                        {recipe.description}
                      </p>
                      <div className="flex justify-between">
                        <Button variant="outline" size="sm">View</Button>
                        <Button size="sm">Add to Plan</Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
          
          {/* Generate Grocery List */}
          <Card className="bg-gray-800 border-gray-700">
            <CardContent className="p-6 flex flex-col sm:flex-row items-center justify-between">
              <div className="mb-4 sm:mb-0">
                <h3 className="text-white text-xl font-medium mb-1">Generate Shopping List</h3>
                <p className="text-gray-400">Create a grocery list based on your current meal plan</p>
              </div>
              <Button className="sm:ml-4">
                <ShoppingBag className="mr-2 h-4 w-4" />
                Generate Grocery List
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* Saved Plans Tab Content */}
        <TabsContent value="saved" className="space-y-6">
          <div className="flex justify-between items-center mb-4">
            <Input 
              type="text" 
              placeholder="Search meal plans..." 
              className="max-w-sm bg-gray-800 border-gray-700"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm">
                <Filter className="mr-2 h-4 w-4" />
                Filter
              </Button>
              <Button size="sm">
                <PlusCircle className="mr-2 h-4 w-4" />
                New Plan
              </Button>
            </div>
          </div>
          
          <div className="space-y-4">
            {mealPlans.map((plan) => (
              <Card key={plan.id} className="bg-gray-800 border-gray-700">
                <CardContent className="p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between">
                  <div className="mb-3 sm:mb-0">
                    <div className="flex items-center">
                      <h3 className="text-white font-medium">{plan.name}</h3>
                      {plan.isActive && (
                        <Badge className="bg-green-600 ml-2">Active</Badge>
                      )}
                      {plan.isFavorite && (
                        <Heart className="h-4 w-4 text-red-500 fill-red-500 ml-2" />
                      )}
                    </div>
                    <p className="text-gray-400 text-sm mb-2">{plan.description}</p>
                    <div className="flex flex-wrap gap-2">
                      <Badge variant="outline">{plan.calories} kcal/day</Badge>
                      <Badge variant="outline">{plan.days} days</Badge>
                      {plan.dietaryPreferences.map((pref) => (
                        <Badge key={pref} variant="outline">{pref}</Badge>
                      ))}
                    </div>
                  </div>
                  
                  <div className="flex gap-2 sm:flex-col md:flex-row">
                    {!plan.isActive && (
                      <Button variant="outline" size="sm">
                        Set as Active
                      </Button>
                    )}
                    <Button size="sm">View Plan</Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
        
        {/* Templates Tab Content */}
        <TabsContent value="templates" className="space-y-6">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Meal Plan Templates</CardTitle>
              <CardDescription className="text-gray-400">
                Pre-designed meal plans for various goals
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Template cards would go here */}
                <Card className="bg-gray-750 border-gray-700">
                  <CardContent className="p-6 flex flex-col items-center text-center">
                    <div className="w-16 h-16 rounded-full bg-blue-600 flex items-center justify-center mb-4">
                      <Utensils className="h-8 w-8 text-white" />
                    </div>
                    <h3 className="text-white font-medium mb-1">High Protein Plan</h3>
                    <p className="text-gray-400 text-sm mb-4">
                      Optimized for muscle building and recovery
                    </p>
                    <Button className="w-full">Use Template</Button>
                  </CardContent>
                </Card>
                
                <Card className="bg-gray-750 border-gray-700">
                  <CardContent className="p-6 flex flex-col items-center text-center">
                    <div className="w-16 h-16 rounded-full bg-green-600 flex items-center justify-center mb-4">
                      <Utensils className="h-8 w-8 text-white" />
                    </div>
                    <h3 className="text-white font-medium mb-1">Fat Loss Plan</h3>
                    <p className="text-gray-400 text-sm mb-4">
                      Calorie-controlled nutrition for fat loss
                    </p>
                    <Button className="w-full">Use Template</Button>
                  </CardContent>
                </Card>
                
                <Card className="bg-gray-750 border-gray-700">
                  <CardContent className="p-6 flex flex-col items-center text-center">
                    <div className="w-16 h-16 rounded-full bg-purple-600 flex items-center justify-center mb-4">
                      <Utensils className="h-8 w-8 text-white" />
                    </div>
                    <h3 className="text-white font-medium mb-1">Plant-Based Plan</h3>
                    <p className="text-gray-400 text-sm mb-4">
                      Vegan-friendly meals with complete nutrition
                    </p>
                    <Button className="w-full">Use Template</Button>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* AI Generator Tab Content */}
        <TabsContent value="generator" className="space-y-6">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">AI Meal Plan Generator</CardTitle>
              <CardDescription className="text-gray-400">
                Create a customized meal plan based on your preferences
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="goal" className="text-white">Nutrition Goal</Label>
                  <select id="goal" className="w-full p-2 rounded-md bg-gray-700 border-gray-600 text-white">
                    <option value="weight_loss">Weight Loss</option>
                    <option value="muscle_gain">Muscle Gain</option>
                    <option value="maintenance">Maintenance</option>
                    <option value="performance">Athletic Performance</option>
                    <option value="health">General Health</option>
                  </select>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="calories" className="text-white">Daily Calories</Label>
                  <Input 
                    id="calories" 
                    type="number" 
                    placeholder="2000" 
                    className="bg-gray-700 border-gray-600 text-white" 
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="days" className="text-white">Number of Days</Label>
                  <Input 
                    id="days" 
                    type="number" 
                    placeholder="7" 
                    className="bg-gray-700 border-gray-600 text-white" 
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="meals" className="text-white">Meals per Day</Label>
                  <Input 
                    id="meals" 
                    type="number" 
                    placeholder="4" 
                    className="bg-gray-700 border-gray-600 text-white" 
                  />
                </div>
              </div>
              
              <div className="space-y-2">
                <Label className="text-white">Dietary Preferences</Label>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {Object.values(DietaryPreference).map((pref) => (
                    <div key={pref} className="flex items-center space-x-2">
                      <Switch id={`pref-${pref}`} />
                      <Label htmlFor={`pref-${pref}`} className="text-white text-sm">{pref}</Label>
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="space-y-2">
                <Label className="text-white">Food Allergies</Label>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {['Gluten', 'Dairy', 'Nuts', 'Shellfish', 'Eggs', 'Soy', 'Fish'].map((allergy) => (
                    <div key={allergy} className="flex items-center space-x-2">
                      <Switch id={`allergy-${allergy}`} />
                      <Label htmlFor={`allergy-${allergy}`} className="text-white text-sm">{allergy}</Label>
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="space-y-2">
                <Label className="text-white">Additional Options</Label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div className="flex items-center space-x-2">
                    <Switch id="option-mealprep" />
                    <Label htmlFor="option-mealprep" className="text-white text-sm">
                      Optimize for meal prepping
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch id="option-budget" />
                    <Label htmlFor="option-budget" className="text-white text-sm">
                      Budget-friendly options
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch id="option-quick" />
                    <Label htmlFor="option-quick" className="text-white text-sm">
                      Quick and easy recipes (under 20 mins)
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch id="option-variety" />
                    <Label htmlFor="option-variety" className="text-white text-sm">
                      Maximize food variety
                    </Label>
                  </div>
                </div>
              </div>
            </CardContent>
            <CardFooter>
              <Button className="w-full">
                Generate Custom Meal Plan
              </Button>
            </CardFooter>
          </Card>
          
          <Card className="bg-gray-800 border-gray-700">
            <CardContent className="p-6">
              <div className="flex items-center space-x-4 text-yellow-400 bg-yellow-400/10 p-4 rounded-md mb-4">
                <div className="rounded-full p-2 bg-yellow-400/20">
                  <ListFilter className="h-5 w-5" />
                </div>
                <div>
                  <h4 className="font-medium">Premium Feature</h4>
                  <p className="text-yellow-400/70 text-sm">
                    AI-powered meal plans require tokens. You have 3 tokens remaining.
                  </p>
                </div>
              </div>
              
              <h3 className="text-white font-medium mb-2">What's included:</h3>
              <ul className="space-y-2 text-gray-300">
                <li className="flex items-start">
                  <span className="text-green-400 mr-2">✓</span>
                  <span>Fully customized meal plans based on your preferences</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-400 mr-2">✓</span>
                  <span>Intelligent food combinations for optimal nutrition</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-400 mr-2">✓</span>
                  <span>Automated grocery list generation</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-400 mr-2">✓</span>
                  <span>Nutritionally balanced meals that match your goals</span>
                </li>
              </ul>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
} 
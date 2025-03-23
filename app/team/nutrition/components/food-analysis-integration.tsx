'use client';

import { useState } from 'react';
import Link from 'next/link';
import { 
  Camera, 
  Upload, 
  Save, 
  ChevronRight, 
  Utensils, 
  Plus, 
  ArrowRight, 
  Check, 
  X, 
  Edit,
  AlertTriangle
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { 
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogFooter, 
  DialogHeader, 
  DialogTitle, 
  DialogTrigger
} from '@/components/ui/dialog';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '@/components/ui/select';

import { FoodItem, MealType } from '../types';

// Mock analyzed food data
const mockAnalyzedFood: FoodItem[] = [
  {
    id: 'analyzed-1',
    name: 'Grilled Chicken Breast',
    category: 'Protein',
    servingSize: {
      amount: 100,
      unit: 'g'
    },
    calories: 165,
    macros: {
      protein: 31,
      carbs: 0,
      fat: 3.6,
      fiber: 0,
      sugar: 0
    },
    micros: {},
    tags: ['meat', 'high-protein', 'low-carb'],
    verified: false,
    source: 'analyzed'
  },
  {
    id: 'analyzed-2',
    name: 'Brown Rice',
    category: 'Grains',
    servingSize: {
      amount: 100,
      unit: 'g'
    },
    calories: 112,
    macros: {
      protein: 2.6,
      carbs: 24,
      fat: 0.9,
      fiber: 1.8,
      sugar: 0.4
    },
    micros: {},
    tags: ['grain', 'carb'],
    verified: false,
    source: 'analyzed'
  },
  {
    id: 'analyzed-3',
    name: 'Broccoli',
    category: 'Vegetables',
    servingSize: {
      amount: 100,
      unit: 'g'
    },
    calories: 34,
    macros: {
      protein: 2.8,
      carbs: 6.6,
      fat: 0.4,
      fiber: 2.6,
      sugar: 1.7
    },
    micros: {},
    tags: ['vegetable', 'fiber'],
    verified: false,
    source: 'analyzed'
  }
];

interface FoodAnalysisIntegrationProps {
  onAddToLog?: (items: FoodItem[], mealType: MealType, notes?: string) => void;
  onCreateRecipe?: (items: FoodItem[]) => void;
}

export default function FoodAnalysisIntegration({ 
  onAddToLog, 
  onCreateRecipe 
}: FoodAnalysisIntegrationProps) {
  const [analyzedFood, setAnalyzedFood] = useState(mockAnalyzedFood);
  const [selectedItems, setSelectedItems] = useState<string[]>([]);
  const [mealType, setMealType] = useState<MealType>(MealType.LUNCH);
  const [notes, setNotes] = useState('');
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [confidenceLevel, setConfidenceLevel] = useState<'high' | 'medium' | 'low'>('high');
  
  // Toggle item selection
  const toggleItemSelection = (id: string) => {
    setSelectedItems(prev => 
      prev.includes(id) 
        ? prev.filter(item => item !== id) 
        : [...prev, id]
    );
  };
  
  // Check if all items are selected
  const allSelected = analyzedFood.length > 0 && analyzedFood.every(item => 
    selectedItems.includes(item.id)
  );
  
  // Toggle all items
  const toggleAllItems = () => {
    if (allSelected) {
      setSelectedItems([]);
    } else {
      setSelectedItems(analyzedFood.map(item => item.id));
    }
  };
  
  // Get selected food items
  const getSelectedItems = () => {
    return analyzedFood.filter(item => selectedItems.includes(item.id));
  };
  
  // Add selected items to nutrition log
  const handleAddToLog = () => {
    const items = getSelectedItems();
    if (items.length > 0 && onAddToLog) {
      onAddToLog(items, mealType, notes);
      setShowAddDialog(false);
      setSelectedItems([]);
      setNotes('');
    }
  };
  
  // Calculate total nutrition for selected items
  const calculateTotal = () => {
    const selected = getSelectedItems();
    return {
      calories: selected.reduce((total, item) => total + item.calories, 0),
      protein: selected.reduce((total, item) => total + item.macros.protein, 0),
      carbs: selected.reduce((total, item) => total + item.macros.carbs, 0),
      fat: selected.reduce((total, item) => total + item.macros.fat, 0),
      fiber: selected.reduce((total, item) => total + item.macros.fiber, 0)
    };
  };
  
  // Format the meal type for display
  const formatMealType = (type: MealType) => {
    return type
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };
  
  return (
    <div className="food-analysis-integration">
      <Card className="bg-gray-800 border-gray-700 mb-6">
        <CardHeader className="pb-2">
          <div className="flex justify-between items-start">
            <div>
              <CardTitle className="text-white">Analyzed Food Items</CardTitle>
              <CardDescription className="text-gray-400">
                We've identified {analyzedFood.length} items in your meal
              </CardDescription>
            </div>
            <Badge 
              className={confidenceLevel === 'high' ? 'bg-green-600' : 
                         confidenceLevel === 'medium' ? 'bg-yellow-600' : 'bg-red-600'}
            >
              {confidenceLevel === 'high' ? 'High Confidence' : 
               confidenceLevel === 'medium' ? 'Medium Confidence' : 'Low Confidence'}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          {/* Identified Food Items */}
          {confidenceLevel !== 'high' && (
            <div className="bg-yellow-900/20 border border-yellow-800 text-yellow-400 p-3 rounded-md mb-4 text-sm flex items-start">
              <AlertTriangle className="h-5 w-5 mr-2 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-medium">Some items may require adjustment</p>
                <p>Our AI has detected your food with {confidenceLevel} confidence. Please verify the items and make corrections if needed.</p>
              </div>
            </div>
          )}
          
          <div className="space-y-3 mb-4">
            <div className="flex items-center justify-between px-2">
              <button 
                className="text-blue-400 text-sm flex items-center"
                onClick={toggleAllItems}
              >
                {allSelected ? 'Unselect All' : 'Select All'}
              </button>
              <span className="text-gray-400 text-sm">
                {selectedItems.length} of {analyzedFood.length} selected
              </span>
            </div>
            
            {analyzedFood.map((item) => (
              <div 
                key={item.id}
                className={`flex items-center p-3 rounded-md ${
                  selectedItems.includes(item.id) ? 'bg-blue-900/20 border border-blue-800' : 'bg-gray-750 border border-gray-700'
                }`}
              >
                <button
                  className={`w-5 h-5 rounded-md border flex-shrink-0 flex items-center justify-center mr-3 ${
                    selectedItems.includes(item.id) ? 'bg-blue-600 border-blue-600' : 'border-gray-600'
                  }`}
                  onClick={() => toggleItemSelection(item.id)}
                >
                  {selectedItems.includes(item.id) && <Check className="h-3 w-3 text-white" />}
                </button>
                
                <div className="flex-grow">
                  <div className="flex items-center">
                    <h4 className="text-white font-medium">{item.name}</h4>
                    <Badge variant="outline" className="ml-2 text-xs">{item.category}</Badge>
                  </div>
                  <p className="text-gray-400 text-xs">
                    {item.servingSize.amount} {item.servingSize.unit} • {item.calories} kcal •
                    P: {item.macros.protein}g • C: {item.macros.carbs}g • F: {item.macros.fat}g
                  </p>
                </div>
                
                <Button variant="ghost" size="icon">
                  <Edit className="h-4 w-4 text-gray-400" />
                </Button>
              </div>
            ))}
            
            <Button variant="outline" className="w-full">
              <Plus className="mr-2 h-4 w-4" />
              Add Missing Item
            </Button>
          </div>
          
          {/* Total Nutrition */}
          {selectedItems.length > 0 && (
            <div className="bg-gray-700 p-3 rounded-md mb-4">
              <h4 className="text-white font-medium mb-2">Total Nutrition (Selected Items)</h4>
              <div className="grid grid-cols-5 gap-2 text-center">
                <div>
                  <div className="text-gray-400 text-xs">Calories</div>
                  <div className="text-white font-medium">{calculateTotal().calories} kcal</div>
                </div>
                <div>
                  <div className="text-gray-400 text-xs">Protein</div>
                  <div className="text-white font-medium">{calculateTotal().protein}g</div>
                </div>
                <div>
                  <div className="text-gray-400 text-xs">Carbs</div>
                  <div className="text-white font-medium">{calculateTotal().carbs}g</div>
                </div>
                <div>
                  <div className="text-gray-400 text-xs">Fat</div>
                  <div className="text-white font-medium">{calculateTotal().fat}g</div>
                </div>
                <div>
                  <div className="text-gray-400 text-xs">Fiber</div>
                  <div className="text-white font-medium">{calculateTotal().fiber}g</div>
                </div>
              </div>
            </div>
          )}
        </CardContent>
        <CardFooter className="border-t border-gray-700 pt-4 flex flex-wrap gap-2">
          <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
            <DialogTrigger asChild>
              <Button 
                className="flex-grow"
                disabled={selectedItems.length === 0}
              >
                <Plus className="mr-2 h-4 w-4" />
                Add to Log
              </Button>
            </DialogTrigger>
            <DialogContent className="bg-gray-800 border-gray-700 text-white">
              <DialogHeader>
                <DialogTitle>Add to Nutrition Log</DialogTitle>
                <DialogDescription className="text-gray-400">
                  Add these {selectedItems.length} items to your nutrition log
                </DialogDescription>
              </DialogHeader>
              
              <div className="space-y-4 my-4">
                <div className="space-y-2">
                  <label className="text-white text-sm">Select Meal Type</label>
                  <Select value={mealType} onValueChange={(value) => setMealType(value as MealType)}>
                    <SelectTrigger className="bg-gray-700 border-gray-600">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-gray-800 border-gray-700">
                      {Object.values(MealType).map((type) => (
                        <SelectItem key={type} value={type}>
                          {formatMealType(type)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <label className="text-white text-sm">Notes (optional)</label>
                  <Input 
                    className="bg-gray-700 border-gray-600"
                    placeholder="Add any notes about this meal"
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                  />
                </div>
                
                <div className="bg-gray-700 p-3 rounded-md">
                  <h4 className="text-white text-sm font-medium mb-2">Nutrition Summary</h4>
                  <div className="grid grid-cols-3 gap-2 text-center text-sm">
                    <div>
                      <div className="text-gray-400 text-xs">Calories</div>
                      <div className="text-white">{calculateTotal().calories} kcal</div>
                    </div>
                    <div>
                      <div className="text-gray-400 text-xs">Protein</div>
                      <div className="text-white">{calculateTotal().protein}g</div>
                    </div>
                    <div>
                      <div className="text-gray-400 text-xs">Carbs</div>
                      <div className="text-white">{calculateTotal().carbs}g</div>
                    </div>
                  </div>
                </div>
              </div>
              
              <DialogFooter>
                <Button variant="outline" onClick={() => setShowAddDialog(false)}>
                  Cancel
                </Button>
                <Button onClick={handleAddToLog}>
                  Add to {formatMealType(mealType)}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
          
          <Button 
            variant="outline" 
            className="flex-grow" 
            disabled={selectedItems.length === 0}
            onClick={() => onCreateRecipe && onCreateRecipe(getSelectedItems())}
          >
            <Save className="mr-2 h-4 w-4" />
            Create Recipe
          </Button>
        </CardFooter>
      </Card>
      
      {/* Quick Actions */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4 flex items-center">
            <div className="w-10 h-10 rounded-full bg-blue-600 flex items-center justify-center mr-3">
              <Camera className="h-5 w-5 text-white" />
            </div>
            <div className="flex-grow">
              <h3 className="text-white font-medium">Analyze New Photo</h3>
              <p className="text-gray-400 text-xs">Take a new photo of your food</p>
            </div>
            <Button variant="ghost" size="sm" asChild>
              <Link href="/team/food-analysis">
                <ArrowRight className="h-5 w-5" />
              </Link>
            </Button>
          </CardContent>
        </Card>
        
        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4 flex items-center">
            <div className="w-10 h-10 rounded-full bg-green-600 flex items-center justify-center mr-3">
              <Utensils className="h-5 w-5 text-white" />
            </div>
            <div className="flex-grow">
              <h3 className="text-white font-medium">Log Meal Manually</h3>
              <p className="text-gray-400 text-xs">Add food items from database</p>
            </div>
            <Button variant="ghost" size="sm" asChild>
              <Link href="/team/nutrition/meals/add">
                <ArrowRight className="h-5 w-5" />
              </Link>
            </Button>
          </CardContent>
        </Card>
      </div>
      
      {/* Recipe Recommendations */}
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader className="pb-2">
          <CardTitle className="text-white">Similar Meal Recipes</CardTitle>
          <CardDescription className="text-gray-400">
            We found recipes similar to your analyzed meal
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {/* Example recipe recommendations - in a real app, these would be based on the analyzed food */}
            <div className="flex items-center p-3 bg-gray-750 rounded-md">
              <div className="w-12 h-12 rounded bg-gray-700 flex items-center justify-center mr-3">
                <Utensils className="h-5 w-5 text-gray-500" />
              </div>
              <div className="flex-grow">
                <h4 className="text-white font-medium">Healthy Chicken & Rice Bowl</h4>
                <div className="flex items-center text-gray-400 text-xs">
                  <span className="mr-2">430 kcal</span>
                  <span className="mr-2">32g protein</span>
                  <Badge variant="outline" className="text-xs">89% match</Badge>
                </div>
              </div>
              <Button variant="ghost" size="sm" asChild>
                <Link href="#">
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
                <h4 className="text-white font-medium">Chicken & Broccoli Stir Fry</h4>
                <div className="flex items-center text-gray-400 text-xs">
                  <span className="mr-2">385 kcal</span>
                  <span className="mr-2">28g protein</span>
                  <Badge variant="outline" className="text-xs">76% match</Badge>
                </div>
              </div>
              <Button variant="ghost" size="sm" asChild>
                <Link href="#">
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
                <h4 className="text-white font-medium">Brown Rice Bowl with Vegetables</h4>
                <div className="flex items-center text-gray-400 text-xs">
                  <span className="mr-2">320 kcal</span>
                  <span className="mr-2">12g protein</span>
                  <Badge variant="outline" className="text-xs">65% match</Badge>
                </div>
              </div>
              <Button variant="ghost" size="sm" asChild>
                <Link href="#">
                  <span className="sr-only">View Recipe</span>
                  <ChevronRight className="h-5 w-5" />
                </Link>
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 
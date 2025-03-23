'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { 
  ArrowLeft, 
  Search, 
  Filter, 
  Grid, 
  List as ListIcon, 
  ChevronDown,
  PlusCircle,
  Clock,
  Flame,
  Utensils,
  Heart,
  ChevronRight,
  Star
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

import { sampleRecipes } from '../data/sample-data';
import { DietaryPreference, MealType } from '../types';

export default function RecipeBrowser() {
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [filterCategory, setFilterCategory] = useState('all');
  const [filterMealType, setFilterMealType] = useState<MealType | 'all'>('all');
  const [filterDietaryPref, setFilterDietaryPref] = useState<DietaryPreference | 'all'>('all');
  const [filterDifficulty, setFilterDifficulty] = useState<'easy' | 'medium' | 'hard' | 'all'>('all');
  const [filteredRecipes, setFilteredRecipes] = useState(sampleRecipes);
  
  // Apply filters
  useEffect(() => {
    let results = [...sampleRecipes];
    
    // Apply search query filter
    if (searchQuery) {
      results = results.filter(recipe => 
        recipe.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        recipe.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        recipe.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
      );
    }
    
    // Apply meal type filter
    if (filterMealType !== 'all') {
      results = results.filter(recipe => 
        recipe.mealType.includes(filterMealType as MealType)
      );
    }
    
    // Apply dietary preference filter
    if (filterDietaryPref !== 'all') {
      results = results.filter(recipe => 
        recipe.dietaryPreferences.includes(filterDietaryPref as DietaryPreference)
      );
    }
    
    // Apply difficulty filter
    if (filterDifficulty !== 'all') {
      results = results.filter(recipe => 
        recipe.difficulty === filterDifficulty
      );
    }
    
    setFilteredRecipes(results);
  }, [searchQuery, filterMealType, filterDietaryPref, filterDifficulty]);
  
  // Toggle favorite status
  const toggleFavorite = (id: string) => {
    setFilteredRecipes(prev =>
      prev.map(recipe =>
        recipe.id === id ? { ...recipe, isFavorite: !recipe.isFavorite } : recipe
      )
    );
  };
  
  // Reset all filters
  const resetFilters = () => {
    setSearchQuery('');
    setFilterMealType('all');
    setFilterDietaryPref('all');
    setFilterDifficulty('all');
  };
  
  // Format meal type for display
  const formatMealType = (mealType: MealType) => {
    return mealType
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };
  
  return (
    <div className="recipe-browser">
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
            <h1 className="text-2xl font-bold text-white">Recipe Library</h1>
            <p className="text-gray-400">Browse and discover healthy recipes</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button variant="ghost" size="icon" onClick={() => setViewMode('grid')} className={viewMode === 'grid' ? 'text-blue-400' : 'text-gray-400'}>
            <Grid className="h-5 w-5" />
          </Button>
          <Button variant="ghost" size="icon" onClick={() => setViewMode('list')} className={viewMode === 'list' ? 'text-blue-400' : 'text-gray-400'}>
            <ListIcon className="h-5 w-5" />
          </Button>
          <Button>
            <PlusCircle className="mr-2 h-4 w-4" />
            Add Recipe
          </Button>
        </div>
      </div>
      
      {/* Search and Filter Bar */}
      <div className="mb-6 space-y-4">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="relative flex-grow">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <Input 
              type="text" 
              placeholder="Search recipes..." 
              className="pl-10 bg-gray-800 border-gray-700"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          
          <div className="flex gap-2">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="flex-shrink-0">
                  <Clock className="mr-2 h-4 w-4" />
                  Meal Type
                  <ChevronDown className="ml-2 h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="bg-gray-800 border-gray-700 text-white">
                <DropdownMenuItem 
                  className="cursor-pointer"
                  onClick={() => setFilterMealType('all')}
                >
                  All Meal Types
                </DropdownMenuItem>
                <DropdownMenuSeparator className="bg-gray-700" />
                {Object.values(MealType).map((type) => (
                  <DropdownMenuItem 
                    key={type}
                    className="cursor-pointer"
                    onClick={() => setFilterMealType(type)}
                  >
                    {formatMealType(type)}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="flex-shrink-0">
                  <Utensils className="mr-2 h-4 w-4" />
                  Diet
                  <ChevronDown className="ml-2 h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="bg-gray-800 border-gray-700 text-white">
                <DropdownMenuItem 
                  className="cursor-pointer"
                  onClick={() => setFilterDietaryPref('all')}
                >
                  All Diets
                </DropdownMenuItem>
                <DropdownMenuSeparator className="bg-gray-700" />
                {Object.values(DietaryPreference).map((diet) => (
                  <DropdownMenuItem 
                    key={diet}
                    className="cursor-pointer"
                    onClick={() => setFilterDietaryPref(diet)}
                  >
                    {diet}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="flex-shrink-0">
                  <Flame className="mr-2 h-4 w-4" />
                  Difficulty
                  <ChevronDown className="ml-2 h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="bg-gray-800 border-gray-700 text-white">
                <DropdownMenuItem 
                  className="cursor-pointer"
                  onClick={() => setFilterDifficulty('all')}
                >
                  All Levels
                </DropdownMenuItem>
                <DropdownMenuSeparator className="bg-gray-700" />
                <DropdownMenuItem 
                  className="cursor-pointer"
                  onClick={() => setFilterDifficulty('easy')}
                >
                  Easy
                </DropdownMenuItem>
                <DropdownMenuItem 
                  className="cursor-pointer"
                  onClick={() => setFilterDifficulty('medium')}
                >
                  Medium
                </DropdownMenuItem>
                <DropdownMenuItem 
                  className="cursor-pointer"
                  onClick={() => setFilterDifficulty('hard')}
                >
                  Hard
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
        
        {/* Active Filters */}
        <div className="flex flex-wrap items-center gap-2">
          {searchQuery && (
            <Badge variant="secondary" className="flex items-center gap-1">
              Search: {searchQuery}
              <button 
                className="ml-1 hover:text-gray-300"
                onClick={() => setSearchQuery('')}
              >
                ×
              </button>
            </Badge>
          )}
          
          {filterMealType !== 'all' && (
            <Badge variant="secondary" className="flex items-center gap-1">
              Meal: {formatMealType(filterMealType as MealType)}
              <button 
                className="ml-1 hover:text-gray-300"
                onClick={() => setFilterMealType('all')}
              >
                ×
              </button>
            </Badge>
          )}
          
          {filterDietaryPref !== 'all' && (
            <Badge variant="secondary" className="flex items-center gap-1">
              Diet: {filterDietaryPref}
              <button 
                className="ml-1 hover:text-gray-300"
                onClick={() => setFilterDietaryPref('all')}
              >
                ×
              </button>
            </Badge>
          )}
          
          {filterDifficulty !== 'all' && (
            <Badge variant="secondary" className="flex items-center gap-1">
              Difficulty: {filterDifficulty}
              <button 
                className="ml-1 hover:text-gray-300"
                onClick={() => setFilterDifficulty('all')}
              >
                ×
              </button>
            </Badge>
          )}
          
          {(searchQuery || filterMealType !== 'all' || filterDietaryPref !== 'all' || filterDifficulty !== 'all') && (
            <Button 
              variant="ghost" 
              size="sm" 
              className="text-gray-400 text-xs"
              onClick={resetFilters}
            >
              Clear All
            </Button>
          )}
        </div>
      </div>
      
      {/* Recipe Categories Tabs */}
      <Tabs defaultValue="all" className="mb-6">
        <TabsList className="bg-gray-800">
          <TabsTrigger value="all">All Recipes</TabsTrigger>
          <TabsTrigger value="favorites">Favorites</TabsTrigger>
          <TabsTrigger value="recent">Recently Viewed</TabsTrigger>
          <TabsTrigger value="created">Your Recipes</TabsTrigger>
        </TabsList>
      </Tabs>
      
      {/* Recipe Display */}
      {filteredRecipes.length > 0 ? (
        viewMode === 'grid' ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {filteredRecipes.map((recipe) => (
              <Card key={recipe.id} className="bg-gray-800 border-gray-700 h-full flex flex-col">
                <div className="relative aspect-video bg-gray-700 overflow-hidden rounded-t-lg">
                  {/* This would be an actual image in a real implementation */}
                  <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-gray-700 to-gray-900">
                    <Utensils className="h-8 w-8 text-gray-500" />
                  </div>
                  
                  {recipe.premium && (
                    <div className="absolute top-2 left-2 bg-yellow-600 text-white text-xs px-2 py-1 rounded-full">
                      Premium
                    </div>
                  )}
                  
                  <Button 
                    variant="ghost" 
                    size="icon" 
                    className="absolute top-2 right-2 text-gray-400 hover:text-red-400 bg-gray-800 bg-opacity-50"
                    onClick={() => toggleFavorite(recipe.id)}
                  >
                    <Heart className={`h-4 w-4 ${recipe.isFavorite ? 'text-red-500 fill-red-500' : ''}`} />
                  </Button>
                </div>
                
                <CardContent className="flex-grow p-4">
                  <h3 className="text-lg font-medium text-white mb-1">{recipe.name}</h3>
                  
                  <div className="flex flex-wrap gap-1 mb-2">
                    {recipe.mealType.slice(0, 2).map((mealType, i) => (
                      <Badge key={i} variant="secondary" className="bg-gray-700 text-xs">
                        {formatMealType(mealType)}
                      </Badge>
                    ))}
                    {recipe.mealType.length > 2 && (
                      <Badge variant="secondary" className="bg-gray-700 text-xs">
                        +{recipe.mealType.length - 2}
                      </Badge>
                    )}
                  </div>
                  
                  <p className="text-gray-400 text-sm line-clamp-2 mb-3">
                    {recipe.description}
                  </p>
                  
                  <div className="flex items-center gap-3 mb-4 text-sm">
                    <div className="flex items-center">
                      <Clock className="h-4 w-4 text-gray-400 mr-1" />
                      <span className="text-gray-300">{recipe.prepTime + recipe.cookTime} min</span>
                    </div>
                    <div className="flex items-center">
                      <Flame className="h-4 w-4 text-gray-400 mr-1" />
                      <span className="text-gray-300">{recipe.nutrition.perServing.calories} kcal</span>
                    </div>
                    {recipe.rating && (
                      <div className="flex items-center">
                        <Star className="h-4 w-4 text-yellow-400 fill-yellow-400 mr-1" />
                        <span className="text-gray-300">{recipe.rating}</span>
                      </div>
                    )}
                  </div>
                  
                  <div className="mt-auto flex justify-between items-center">
                    <Badge variant="outline" className="text-xs capitalize">
                      {recipe.difficulty}
                    </Badge>
                    <Button variant="ghost" className="text-blue-400 p-0 h-auto" asChild>
                      <Link href={`/team/nutrition/recipes/${recipe.id}`}>
                        View <ChevronRight className="h-4 w-4 ml-1" />
                      </Link>
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          // List View
          <div className="space-y-3">
            {filteredRecipes.map((recipe) => (
              <div 
                key={recipe.id} 
                className="flex items-center p-4 rounded-md bg-gray-800 border border-gray-700"
              >
                <div className="h-16 w-16 rounded-md bg-gray-700 flex items-center justify-center mr-4">
                  <Utensils className="h-6 w-6 text-gray-500" />
                </div>
                
                <div className="flex-grow">
                  <div className="flex items-center">
                    <h3 className="text-white font-medium">{recipe.name}</h3>
                    {recipe.premium && (
                      <Badge className="bg-yellow-600 ml-2 text-xs">Premium</Badge>
                    )}
                  </div>
                  
                  <p className="text-gray-400 text-sm line-clamp-1 mb-1">
                    {recipe.description}
                  </p>
                  
                  <div className="flex flex-wrap gap-2 items-center">
                    <div className="flex items-center text-xs text-gray-300">
                      <Clock className="h-3 w-3 text-gray-400 mr-1" />
                      {recipe.prepTime + recipe.cookTime} min
                    </div>
                    <div className="flex items-center text-xs text-gray-300">
                      <Flame className="h-3 w-3 text-gray-400 mr-1" />
                      {recipe.nutrition.perServing.calories} kcal
                    </div>
                    <Badge variant="outline" className="text-xs capitalize">
                      {recipe.difficulty}
                    </Badge>
                    {recipe.rating && (
                      <div className="flex items-center text-xs text-gray-300">
                        <Star className="h-3 w-3 text-yellow-400 fill-yellow-400 mr-1" />
                        {recipe.rating} ({recipe.reviews})
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="flex items-center gap-2">
                  <Button 
                    variant="ghost" 
                    size="icon" 
                    className="text-gray-400 hover:text-red-400"
                    onClick={() => toggleFavorite(recipe.id)}
                  >
                    <Heart className={`h-4 w-4 ${recipe.isFavorite ? 'text-red-500 fill-red-500' : ''}`} />
                  </Button>
                  
                  <Button variant="ghost" size="icon" className="text-blue-400" asChild>
                    <Link href={`/team/nutrition/recipes/${recipe.id}`}>
                      <ChevronRight className="h-5 w-5" />
                    </Link>
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )
      ) : (
        // No results
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 text-center">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-gray-700 mb-4">
            <Utensils className="h-6 w-6 text-gray-400" />
          </div>
          <h3 className="text-lg font-medium text-white mb-2">No recipes found</h3>
          <p className="text-gray-400 mb-4">Try adjusting your filters or search term</p>
          <Button 
            variant="outline" 
            onClick={resetFilters}
          >
            Reset Filters
          </Button>
        </div>
      )}
      
      {/* Premium Recipe Banner */}
      <Card className="bg-gray-800 border-gray-700 mt-8">
        <CardContent className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
            <div className="md:col-span-2">
              <Badge className="bg-yellow-600 mb-2">Premium Feature</Badge>
              <h3 className="text-xl font-bold text-white mb-2">Unlock Premium Recipes</h3>
              <p className="text-gray-400 mb-4">
                Get access to exclusive chef-created recipes, advanced nutrition plans, and personalized meal recommendations designed for your specific goals.
              </p>
              <Button>
                Upgrade with Tokens
              </Button>
            </div>
            <div className="hidden md:block bg-gradient-to-br from-yellow-600 to-yellow-800 h-40 rounded-lg flex items-center justify-center">
              <Star className="h-16 w-16 text-yellow-200" />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 
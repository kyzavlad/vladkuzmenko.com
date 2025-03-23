// Nutrient unit enum
export enum NutrientUnit {
  GRAMS = 'g',
  MILLIGRAMS = 'mg',
  MICROGRAMS = 'mcg',
  IU = 'IU',
  PERCENT = '%'
}

// Dietary preference enum
export enum DietaryPreference {
  OMNIVORE = 'omnivore',
  VEGETARIAN = 'vegetarian',
  VEGAN = 'vegan',
  PESCATARIAN = 'pescatarian',
  PALEO = 'paleo',
  KETO = 'keto',
  MEDITERRANEAN = 'mediterranean'
}

// Dietary goal enum
export enum DietaryGoal {
  WEIGHT_LOSS = 'weight_loss',
  MUSCLE_GAIN = 'muscle_gain',
  MAINTENANCE = 'maintenance',
  PERFORMANCE = 'performance',
  HEALTH = 'health'
}

// Food allergy enum
export enum FoodAllergy {
  DAIRY = 'dairy',
  EGGS = 'eggs',
  PEANUTS = 'peanuts',
  TREE_NUTS = 'tree_nuts',
  SOY = 'soy',
  WHEAT = 'wheat',
  FISH = 'fish',
  SHELLFISH = 'shellfish'
}

// Meal Type enum
export enum MealType {
  BREAKFAST = 'breakfast',
  LUNCH = 'lunch',
  DINNER = 'dinner',
  SNACK = 'snack',
  PRE_WORKOUT = 'pre_workout',
  POST_WORKOUT = 'post_workout'
}

// Interface for serving size
export interface ServingSize {
  amount: number;
  unit: string;
}

// Interface for macronutrients
export interface Macros {
  protein: number;
  carbs: number;
  fat: number;
  fiber: number;
  sugar: number;
}

// Interface for micronutrients
export interface Micros {
  [key: string]: number; // e.g., "vitamin_a": 800, "calcium": 200, etc.
}

// Interface for food item
export interface FoodItem {
  id: string;
  name: string;
  category: string;
  servingSize: ServingSize;
  calories: number;
  macros: Macros;
  micros: Micros;
  tags: string[];
  verified?: boolean;
  source?: 'database' | 'user' | 'analyzed';
}

// Interface for meal
export interface Meal {
  id: string;
  type: MealType;
  name?: string;
  time: string | Date;
  items: {
    item: FoodItem;
    quantity: number;
  }[];
  notes?: string;
  totalNutrition: {
    calories: number;
    macros: Macros;
  };
}

// Interface for daily nutrition log
export interface DailyNutritionLog {
  date: string | Date;
  meals: Meal[];
  waterIntake: number; // in ml
  goals: {
    calories: number;
    macros: Macros;
    water: number;
  };
  actualTotals: {
    calories: number;
    macros: Macros;
    water: number;
  };
  nutrientQualityScore: number; // 0-100
}

// Interface for weekly nutrition data
export interface WeeklyNutritionData {
  days: {
    date: string | Date;
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
    qualityScore: number;
    consistencyScore: number;
  }[];
  averages: {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
    qualityScore: number;
    consistencyScore: number;
  };
  weekOverWeekChanges: {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
    qualityScore: number;
  };
}

// Interface for recipe
export interface Recipe {
  id: string;
  name: string;
  description: string;
  images: string[];
  prepTime: number; // in minutes
  cookTime: number; // in minutes
  servings: number;
  difficulty: 'easy' | 'medium' | 'hard';
  mealTypes: MealType[];
  ingredients: {
    item: FoodItem;
    quantity: number;
  }[];
  instructions: string[];
  nutritionPerServing: {
    calories: number;
    macros: Macros;
  };
  tags: string[];
  dietaryCategories: string[]; // e.g., "vegetarian", "keto", "gluten-free"
  isFavorite: boolean;
  author: string;
  dateCreated: string | Date;
  dateModified?: string | Date;
}

// Interface for meal plan
export interface MealPlan {
  id: string;
  name: string;
  description: string;
  startDate?: string | Date;
  endDate?: string | Date;
  dailyTargets: {
    calories: number;
    macros: Macros;
  };
  dietaryPreferences: string[];
  days: {
    dayName: string; // e.g., "Monday", "Tuesday", etc.
    meals: {
      type: MealType;
      recipe?: Recipe;
      customMeal?: Meal;
    }[];
  }[];
  isCurrent: boolean;
  createdAt: string | Date;
  updatedAt?: string | Date;
}

// Interface for user's nutrition preferences
export interface NutritionPreferences {
  dietaryApproach: string; // e.g., "balanced", "keto", "vegan"
  dietaryRestrictions: string[];
  allergies: string[];
  dislikedFoods: string[];
  preferredFoods: string[];
  mealFrequency: number;
  trackingPreferences: {
    trackMacros: boolean;
    trackMicros: boolean;
    trackWater: boolean;
    trackQualityScore: boolean;
  };
}

// Interface for nutrition insights
export interface NutritionInsight {
  id: string;
  type: 'tip' | 'alert' | 'suggestion' | 'achievement';
  title: string;
  description: string;
  actionable: boolean;
  action?: {
    label: string;
    link: string;
  };
  createdAt: string | Date;
  expires?: string | Date;
  dismissed: boolean;
  priority: 'low' | 'medium' | 'high';
} 
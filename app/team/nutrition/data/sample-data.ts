import {
  MealType,
  DietaryPreference,
  DietaryGoal,
  NutrientUnit,
  FoodAllergy,
  FoodItem,
  Recipe,
  MealLogEntry,
  DailyNutritionLog,
  MealPlan
} from '../types';

// Sample food items
export const sampleFoodItems: FoodItem[] = [
  {
    id: 'food-1',
    name: 'Chicken Breast',
    brand: 'Organic Valley',
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
    micros: {
      vitamin_b6: {
        amount: 0.5,
        unit: NutrientUnit.MILLIGRAM,
        percentDV: 38
      },
      niacin: {
        amount: 13.7,
        unit: NutrientUnit.MILLIGRAM,
        percentDV: 85
      },
      phosphorus: {
        amount: 196,
        unit: NutrientUnit.MILLIGRAM,
        percentDV: 28
      }
    },
    tags: ['meat', 'high-protein', 'low-carb'],
    imageUrl: '/foods/chicken-breast.jpg',
    verified: true,
    source: 'system'
  },
  {
    id: 'food-2',
    name: 'Quinoa',
    brand: 'Bob\'s Red Mill',
    category: 'Grains',
    servingSize: {
      amount: 100,
      unit: 'g'
    },
    calories: 368,
    macros: {
      protein: 14.1,
      carbs: 64.2,
      fat: 6.1,
      fiber: 7,
      sugar: 0
    },
    micros: {
      magnesium: {
        amount: 197,
        unit: NutrientUnit.MILLIGRAM,
        percentDV: 47
      },
      iron: {
        amount: 4.6,
        unit: NutrientUnit.MILLIGRAM,
        percentDV: 25
      },
      zinc: {
        amount: 3.1,
        unit: NutrientUnit.MILLIGRAM,
        percentDV: 28
      }
    },
    tags: ['grain', 'gluten-free', 'complete-protein', 'whole-food'],
    imageUrl: '/foods/quinoa.jpg',
    verified: true,
    source: 'system'
  },
  {
    id: 'food-3',
    name: 'Avocado',
    category: 'Fruit',
    servingSize: {
      amount: 100,
      unit: 'g'
    },
    calories: 160,
    macros: {
      protein: 2,
      carbs: 8.5,
      fat: 14.7,
      fiber: 6.7,
      sugar: 0.7
    },
    micros: {
      vitamin_k: {
        amount: 21,
        unit: NutrientUnit.MICROGRAM,
        percentDV: 18
      },
      folate: {
        amount: 81,
        unit: NutrientUnit.MICROGRAM,
        percentDV: 20
      },
      potassium: {
        amount: 485,
        unit: NutrientUnit.MILLIGRAM,
        percentDV: 10
      }
    },
    tags: ['fruit', 'healthy-fat', 'fiber'],
    imageUrl: '/foods/avocado.jpg',
    verified: true,
    source: 'system'
  },
  {
    id: 'food-4',
    name: 'Greek Yogurt, Plain',
    brand: 'Fage',
    category: 'Dairy',
    servingSize: {
      amount: 100,
      unit: 'g'
    },
    calories: 59,
    macros: {
      protein: 10.2,
      carbs: 3.6,
      fat: 0.4,
      fiber: 0,
      sugar: 3.6
    },
    micros: {
      calcium: {
        amount: 110,
        unit: NutrientUnit.MILLIGRAM,
        percentDV: 11
      },
      vitamin_b12: {
        amount: 0.5,
        unit: NutrientUnit.MICROGRAM,
        percentDV: 21
      },
      selenium: {
        amount: 9.7,
        unit: NutrientUnit.MICROGRAM,
        percentDV: 18
      }
    },
    tags: ['dairy', 'high-protein', 'probiotic'],
    imageUrl: '/foods/greek-yogurt.jpg',
    verified: true,
    source: 'system'
  },
  {
    id: 'food-5',
    name: 'Spinach',
    category: 'Vegetables',
    servingSize: {
      amount: 100,
      unit: 'g'
    },
    calories: 23,
    macros: {
      protein: 2.9,
      carbs: 3.6,
      fat: 0.4,
      fiber: 2.2,
      sugar: 0.4
    },
    micros: {
      vitamin_a: {
        amount: 9377,
        unit: NutrientUnit.IU,
        percentDV: 188
      },
      vitamin_k: {
        amount: 483,
        unit: NutrientUnit.MICROGRAM,
        percentDV: 403
      },
      folate: {
        amount: 194,
        unit: NutrientUnit.MICROGRAM,
        percentDV: 49
      }
    },
    tags: ['vegetable', 'leafy-green', 'nutrient-dense'],
    imageUrl: '/foods/spinach.jpg',
    verified: true,
    source: 'system'
  },
  {
    id: 'food-6',
    name: 'Sweet Potato',
    category: 'Vegetables',
    servingSize: {
      amount: 100,
      unit: 'g'
    },
    calories: 86,
    macros: {
      protein: 1.6,
      carbs: 20.1,
      fat: 0.1,
      fiber: 3,
      sugar: 4.2
    },
    micros: {
      vitamin_a: {
        amount: 14187,
        unit: NutrientUnit.IU,
        percentDV: 284
      },
      vitamin_c: {
        amount: 2.4,
        unit: NutrientUnit.MILLIGRAM,
        percentDV: 3
      },
      manganese: {
        amount: 0.3,
        unit: NutrientUnit.MILLIGRAM,
        percentDV: 15
      }
    },
    tags: ['vegetable', 'starchy', 'complex-carb'],
    imageUrl: '/foods/sweet-potato.jpg',
    verified: true,
    source: 'system'
  },
  {
    id: 'food-7',
    name: 'Salmon',
    category: 'Protein',
    servingSize: {
      amount: 100,
      unit: 'g'
    },
    calories: 208,
    macros: {
      protein: 20.4,
      carbs: 0,
      fat: 13.4,
      fiber: 0,
      sugar: 0
    },
    micros: {
      vitamin_d: {
        amount: 526,
        unit: NutrientUnit.IU,
        percentDV: 88
      },
      vitamin_b12: {
        amount: 2.8,
        unit: NutrientUnit.MICROGRAM,
        percentDV: 117
      },
      omega_3: {
        amount: 2.3,
        unit: NutrientUnit.GRAM,
        percentDV: 100
      }
    },
    tags: ['fish', 'high-protein', 'omega-3'],
    imageUrl: '/foods/salmon.jpg',
    verified: true,
    source: 'system'
  },
  {
    id: 'food-8',
    name: 'Blueberries',
    category: 'Fruit',
    servingSize: {
      amount: 100,
      unit: 'g'
    },
    calories: 57,
    macros: {
      protein: 0.7,
      carbs: 14.5,
      fat: 0.3,
      fiber: 2.4,
      sugar: 10
    },
    micros: {
      vitamin_c: {
        amount: 9.7,
        unit: NutrientUnit.MILLIGRAM,
        percentDV: 11
      },
      vitamin_k: {
        amount: 19.3,
        unit: NutrientUnit.MICROGRAM,
        percentDV: 16
      },
      manganese: {
        amount: 0.3,
        unit: NutrientUnit.MILLIGRAM,
        percentDV: 17
      }
    },
    tags: ['fruit', 'antioxidant', 'low-glycemic'],
    imageUrl: '/foods/blueberries.jpg',
    verified: true,
    source: 'system'
  }
];

// Sample recipes
export const sampleRecipes: Recipe[] = [
  {
    id: 'recipe-1',
    name: 'Chicken Quinoa Bowl',
    description: 'A protein-packed bowl with lean chicken breast, quinoa, and vegetables',
    mealType: [MealType.LUNCH, MealType.DINNER],
    servings: 2,
    prepTime: 15,
    cookTime: 25,
    ingredients: [
      {
        foodId: 'food-1', // Chicken Breast
        amount: 200,
        unit: 'g'
      },
      {
        foodId: 'food-2', // Quinoa
        amount: 100,
        unit: 'g',
        preparation: 'cooked'
      },
      {
        foodId: 'food-5', // Spinach
        amount: 80,
        unit: 'g'
      },
      {
        foodId: 'food-3', // Avocado
        amount: 50,
        unit: 'g',
        preparation: 'sliced'
      }
    ],
    instructions: [
      'Cook quinoa according to package instructions',
      'Season chicken breast with salt, pepper, and herbs',
      'Grill or bake chicken until internal temperature reaches 165°F',
      'Slice chicken and arrange over quinoa',
      'Top with fresh spinach and sliced avocado'
    ],
    nutrition: {
      perServing: {
        calories: 412,
        protein: 43,
        carbs: 32,
        fat: 12,
        fiber: 6
      },
      total: {
        calories: 824,
        protein: 86,
        carbs: 64,
        fat: 24,
        fiber: 12
      }
    },
    difficulty: 'easy',
    cuisineType: 'Contemporary',
    dietaryPreferences: [DietaryPreference.GLUTEN_FREE, DietaryPreference.DAIRY_FREE],
    imageUrl: '/recipes/chicken-quinoa-bowl.jpg',
    isFavorite: true,
    tags: ['high-protein', 'meal-prep', 'post-workout'],
    author: 'System',
    createdAt: '2023-01-15T12:00:00Z',
    updatedAt: '2023-01-15T12:00:00Z',
    rating: 4.7,
    reviews: 42,
    premium: false
  },
  {
    id: 'recipe-2',
    name: 'Berry Greek Yogurt Parfait',
    description: 'A simple, protein-rich breakfast with layers of yogurt, berries, and nuts',
    mealType: [MealType.BREAKFAST, MealType.SNACK],
    servings: 1,
    prepTime: 5,
    cookTime: 0,
    ingredients: [
      {
        foodId: 'food-4', // Greek Yogurt
        amount: 200,
        unit: 'g'
      },
      {
        foodId: 'food-8', // Blueberries
        amount: 100,
        unit: 'g'
      }
    ],
    instructions: [
      'Layer half of the Greek yogurt in a glass or bowl',
      'Add half of the blueberries on top',
      'Repeat with remaining yogurt and blueberries',
      'Optional: add honey or granola for extra flavor and texture'
    ],
    nutrition: {
      perServing: {
        calories: 175,
        protein: 21,
        carbs: 18,
        fat: 1,
        fiber: 2.4
      },
      total: {
        calories: 175,
        protein: 21,
        carbs: 18,
        fat: 1,
        fiber: 2.4
      }
    },
    difficulty: 'easy',
    cuisineType: 'American',
    dietaryPreferences: [DietaryPreference.VEGETARIAN, DietaryPreference.GLUTEN_FREE],
    imageUrl: '/recipes/yogurt-parfait.jpg',
    isFavorite: false,
    tags: ['breakfast', 'quick', 'high-protein'],
    author: 'System',
    createdAt: '2023-01-20T09:30:00Z',
    updatedAt: '2023-01-20T09:30:00Z',
    rating: 4.5,
    reviews: 28,
    premium: false
  },
  {
    id: 'recipe-3',
    name: 'Salmon with Sweet Potato and Spinach',
    description: 'A nutrient-dense meal featuring omega-3 rich salmon with sweet potato and spinach',
    mealType: [MealType.DINNER],
    servings: 2,
    prepTime: 10,
    cookTime: 25,
    ingredients: [
      {
        foodId: 'food-7', // Salmon
        amount: 200,
        unit: 'g'
      },
      {
        foodId: 'food-6', // Sweet Potato
        amount: 300,
        unit: 'g',
        preparation: 'cubed'
      },
      {
        foodId: 'food-5', // Spinach
        amount: 100,
        unit: 'g'
      }
    ],
    instructions: [
      'Preheat oven to 400°F (200°C)',
      'Season salmon with salt, pepper, and lemon',
      'Bake salmon and sweet potato cubes for 20-25 minutes',
      'Sauté spinach in a pan until wilted',
      'Serve salmon with sweet potato and spinach'
    ],
    nutrition: {
      perServing: {
        calories: 419,
        protein: 29,
        carbs: 30,
        fat: 20,
        fiber: 5
      },
      total: {
        calories: 838,
        protein: 58,
        carbs: 60,
        fat: 40,
        fiber: 10
      }
    },
    difficulty: 'medium',
    cuisineType: 'Contemporary',
    dietaryPreferences: [DietaryPreference.GLUTEN_FREE, DietaryPreference.DAIRY_FREE, DietaryPreference.PALEO],
    imageUrl: '/recipes/salmon-sweet-potato.jpg',
    isFavorite: true,
    tags: ['omega-3', 'nutrient-dense', 'dinner'],
    author: 'System',
    createdAt: '2023-02-05T18:15:00Z',
    updatedAt: '2023-02-05T18:15:00Z',
    rating: 4.9,
    reviews: 37,
    premium: false
  },
  {
    id: 'recipe-4',
    name: 'Avocado Toast with Eggs',
    description: 'A classic breakfast or brunch option that combines healthy fats with protein',
    mealType: [MealType.BREAKFAST, MealType.LUNCH],
    servings: 1,
    prepTime: 5,
    cookTime: 5,
    ingredients: [
      {
        foodId: 'food-3', // Avocado
        amount: 50,
        unit: 'g',
        preparation: 'mashed'
      }
      // Other ingredients would be added here
    ],
    instructions: [
      'Toast bread until golden brown',
      'Mash avocado and spread on toast',
      'Fry or poach eggs to desired doneness',
      'Place eggs on top of avocado toast',
      'Season with salt, pepper, and optional red pepper flakes'
    ],
    nutrition: {
      perServing: {
        calories: 350,
        protein: 15,
        carbs: 25,
        fat: 22,
        fiber: 7
      },
      total: {
        calories: 350,
        protein: 15,
        carbs: 25,
        fat: 22,
        fiber: 7
      }
    },
    difficulty: 'easy',
    cuisineType: 'Contemporary',
    dietaryPreferences: [DietaryPreference.VEGETARIAN],
    imageUrl: '/recipes/avocado-toast.jpg',
    isFavorite: false,
    tags: ['breakfast', 'brunch', 'healthy-fats'],
    author: 'System',
    createdAt: '2023-02-10T08:45:00Z',
    updatedAt: '2023-02-10T08:45:00Z',
    rating: 4.6,
    reviews: 52,
    premium: false
  }
];

// Sample daily log for current day
export const sampleCurrentDayLog: DailyNutritionLog = {
  id: 'log-20230601',
  userId: 'user-123',
  date: new Date().toISOString().split('T')[0],
  meals: [
    {
      id: 'meal-20230601-1',
      userId: 'user-123',
      date: new Date().toISOString().split('T')[0],
      mealType: MealType.BREAKFAST,
      time: '07:30',
      items: [
        {
          itemId: 'recipe-2', // Berry Greek Yogurt Parfait
          itemType: 'recipe',
          servingSize: 1,
          servingUnit: 'serving'
        }
      ],
      totalNutrition: {
        calories: 175,
        protein: 21,
        carbs: 18,
        fat: 1,
        fiber: 2.4
      }
    },
    {
      id: 'meal-20230601-2',
      userId: 'user-123',
      date: new Date().toISOString().split('T')[0],
      mealType: MealType.LUNCH,
      time: '12:15',
      items: [
        {
          itemId: 'recipe-1', // Chicken Quinoa Bowl
          itemType: 'recipe',
          servingSize: 1,
          servingUnit: 'serving'
        }
      ],
      totalNutrition: {
        calories: 412,
        protein: 43,
        carbs: 32,
        fat: 12,
        fiber: 6
      }
    },
    {
      id: 'meal-20230601-3',
      userId: 'user-123',
      date: new Date().toISOString().split('T')[0],
      mealType: MealType.SNACK,
      time: '15:45',
      items: [
        {
          itemId: 'food-8', // Blueberries
          itemType: 'food',
          servingSize: 1,
          servingUnit: 'cup'
        }
      ],
      totalNutrition: {
        calories: 85,
        protein: 1.1,
        carbs: 21.5,
        fat: 0.5,
        fiber: 3.6
      }
    },
    // Dinner is not yet logged
  ],
  water: 1500, // 1.5 liters so far
  supplements: [
    {
      name: 'Multivitamin',
      amount: 1,
      unit: 'tablet',
      timeOfDay: '08:00'
    },
    {
      name: 'Omega-3',
      amount: 1000,
      unit: 'mg',
      timeOfDay: '08:00'
    }
  ],
  totalNutrition: {
    calories: 672,
    protein: 65.1,
    carbs: 71.5,
    fat: 13.5,
    fiber: 12
  },
  calorieTarget: 2200,
  macroTargets: {
    protein: 165, // 30% of calories
    carbs: 275, // 50% of calories
    fat: 49, // 20% of calories
    fiber: 30
  },
  workoutDay: true,
  nutrientQualityScore: 87,
  streak: 14
};

// Sample meal plan
export const sampleMealPlan: MealPlan = {
  id: 'plan-1',
  userId: 'user-123',
  name: 'Muscle Building Plan',
  description: 'A high-protein meal plan designed to support muscle growth and recovery',
  startDate: '2023-06-01',
  endDate: '2023-06-07',
  days: [
    {
      dayOfWeek: 1, // Monday
      meals: [
        {
          mealType: MealType.BREAKFAST,
          time: '07:30',
          items: [
            {
              itemId: 'recipe-2', // Berry Greek Yogurt Parfait
              itemType: 'recipe',
              servingSize: 1,
              servingUnit: 'serving'
            }
          ],
          notes: 'Add a tablespoon of flaxseeds for extra omega-3'
        },
        {
          mealType: MealType.LUNCH,
          time: '12:30',
          items: [
            {
              itemId: 'recipe-1', // Chicken Quinoa Bowl
              itemType: 'recipe',
              servingSize: 1,
              servingUnit: 'serving'
            }
          ]
        },
        {
          mealType: MealType.SNACK,
          time: '15:30',
          items: [
            {
              itemId: 'food-8', // Blueberries
              itemType: 'food',
              servingSize: 1,
              servingUnit: 'cup'
            }
          ]
        },
        {
          mealType: MealType.DINNER,
          time: '19:00',
          items: [
            {
              itemId: 'recipe-3', // Salmon with Sweet Potato and Spinach
              itemType: 'recipe',
              servingSize: 1,
              servingUnit: 'serving'
            }
          ]
        }
      ],
      totalNutrition: {
        calories: 1091,
        protein: 94.1,
        carbs: 101.5,
        fat: 33.5,
        fiber: 17
      }
    },
    // Additional days would be added here
  ],
  dietaryPreferences: [DietaryPreference.GLUTEN_FREE],
  dietaryGoal: DietaryGoal.MUSCLE_GAIN,
  calorieTarget: 2500,
  macroTargets: {
    protein: 188, // 30% of calories
    carbs: 313, // 50% of calories
    fat: 56, // 20% of calories
    fiber: 35
  },
  allergies: [FoodAllergy.PEANUTS],
  isTemplate: false,
  isActive: true,
  isFavorite: false,
  createdAt: '2023-05-25T14:30:00Z',
  updatedAt: '2023-05-25T14:30:00Z',
  premium: false
}; 
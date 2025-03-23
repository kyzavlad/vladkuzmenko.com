import { 
  FitnessProfile, 
  UserGoals, 
  UserPreferences, 
  BehavioralInsights, 
  RecommendationContext,
  Recommendation,
  OnboardingQuestion,
  OnboardingFlow,
  PremiumPersonalization
} from '../types';

// Mock fitness profile
export const mockFitnessProfile: FitnessProfile = {
  id: 'profile1',
  userId: 'user123',
  createdAt: new Date('2023-05-15'),
  updatedAt: new Date(),
  
  // Basic Information
  age: 32,
  gender: 'male',
  height: 178,
  weight: 82,
  
  // Fitness Background
  fitnessLevel: 'intermediate',
  fitnessBackground: ['weightlifting', 'running', 'soccer'],
  activityLevel: 'moderately_active',
  yearsOfExperience: 5,
  
  // Health Information
  medicalConditions: [],
  injuries: ['minor knee pain'],
  limitations: ['avoid high-impact jumping'],
  
  // Lifestyle Factors
  sleepAverage: 7,
  stressLevel: 'moderate',
  occupation: 'office worker',
  workSchedule: 'standard',
  commute: 30,
  
  // Current Fitness Stats
  currentBodyFat: 18,
  muscleGroups: {
    chest: {
      strength: 7,
      experience: 8,
      preference: 9,
      injuries: []
    },
    back: {
      strength: 8,
      experience: 7,
      preference: 8,
      injuries: []
    },
    legs: {
      strength: 6,
      experience: 6,
      preference: 5,
      injuries: ['minor knee pain']
    },
    shoulders: {
      strength: 7,
      experience: 6,
      preference: 7,
      injuries: []
    },
    arms: {
      strength: 7,
      experience: 8,
      preference: 8,
      injuries: []
    },
    core: {
      strength: 6,
      experience: 7,
      preference: 6,
      injuries: []
    }
  },
  
  // Fitness Testing Results
  fitnessAssessments: [
    {
      id: 'assessment1',
      date: new Date('2023-06-10'),
      cardio: {
        vo2Max: 42,
        restingHeartRate: 65,
        timeToExhaustion: 18.5,
        recoveryRate: 35
      },
      strength: {
        'bench press': 95,
        'squat': 130,
        'deadlift': 150,
        'overhead press': 60
      },
      bodyComposition: {
        weight: 82,
        bodyFat: 18,
        muscleMass: 40,
        measurements: {
          chest: 102,
          waist: 84,
          hips: 96,
          biceps: 38
        }
      }
    }
  ],
  
  // Availability
  availableDays: ['monday', 'tuesday', 'thursday', 'saturday'],
  timeAvailability: {
    'monday': [
      { startTime: '06:30', endTime: '07:30', preferred: true },
      { startTime: '18:00', endTime: '19:30', preferred: false }
    ],
    'tuesday': [
      { startTime: '18:00', endTime: '19:30', preferred: true }
    ],
    'thursday': [
      { startTime: '18:00', endTime: '19:30', preferred: true }
    ],
    'saturday': [
      { startTime: '09:00', endTime: '11:00', preferred: true }
    ]
  },
  
  // Equipment Access
  equipmentAccess: ['commercial gym', 'dumbbells', 'barbell', 'bench', 'pull-up bar'],
  gymAccess: true,
  homeEquipment: ['dumbbells', 'resistance bands', 'pull-up bar']
};

// Mock user goals
export const mockUserGoals: UserGoals = {
  id: 'goals1',
  userId: 'user123',
  createdAt: new Date('2023-05-15'),
  updatedAt: new Date(),
  
  // Primary Goals
  primaryGoals: [
    {
      id: 'goal1',
      type: 'weight',
      title: 'Reach target weight',
      description: 'Reduce body weight to 78kg while maintaining muscle mass',
      targetValue: 78,
      currentValue: 82,
      unit: 'kg',
      startDate: new Date('2023-05-15'),
      targetDate: new Date('2023-11-15'),
      status: 'active',
      priority: 'high',
      relatedMetrics: ['weight', 'bodyFat', 'muscleMass'],
      milestones: [
        { value: 81, achieved: true, date: new Date('2023-06-15') },
        { value: 80, achieved: false },
        { value: 79, achieved: false },
        { value: 78, achieved: false }
      ]
    },
    {
      id: 'goal2',
      type: 'strength',
      title: 'Increase bench press',
      description: 'Increase bench press to 100kg for 5 reps',
      targetValue: 100,
      currentValue: 95,
      unit: 'kg',
      startDate: new Date('2023-05-15'),
      targetDate: new Date('2023-10-15'),
      status: 'active',
      priority: 'medium',
      relatedMetrics: ['strength'],
      milestones: [
        { value: 97, achieved: false },
        { value: 100, achieved: false }
      ]
    }
  ],
  activeGoal: 'goal1',
  
  // Body Composition Goals
  targetWeight: 78,
  targetBodyFat: 15,
  
  // Performance Goals
  strengthGoals: [
    {
      id: 'strength1',
      exercise: 'bench press',
      currentMax: 95,
      targetMax: 100,
      unit: 'kg',
      deadline: new Date('2023-10-15')
    },
    {
      id: 'strength2',
      exercise: 'squat',
      currentMax: 130,
      targetMax: 140,
      unit: 'kg',
      deadline: new Date('2023-11-15')
    }
  ],
  enduranceGoals: [
    {
      id: 'endurance1',
      activityType: '5k run',
      currentCapacity: 25,
      targetCapacity: 22,
      unit: 'minutes',
      deadline: new Date('2023-12-15')
    }
  ],
  mobilityGoals: [
    {
      id: 'mobility1',
      jointOrMuscle: 'hamstrings',
      currentRange: 85,
      targetRange: 100,
      unit: 'degrees',
      deadline: new Date('2023-12-15')
    }
  ],
  
  // Timeline
  desiredCompletionDate: new Date('2023-12-15'),
  
  // Priorities
  priorities: {
    strength: 8,
    endurance: 6,
    flexibility: 5,
    balance: 4,
    speed: 5,
    power: 6,
    muscleGain: 7,
    fatLoss: 9
  }
};

// Mock user preferences
export const mockUserPreferences: UserPreferences = {
  id: 'prefs1',
  userId: 'user123',
  updatedAt: new Date(),
  
  // Workout Preferences
  preferredWorkoutDuration: 60,
  preferredWorkoutFrequency: 4,
  preferredExerciseTypes: ['compound', 'free weights', 'HIIT'],
  dislikedExerciseTypes: ['steady-state cardio', 'machines'],
  preferredMuscleGroups: ['chest', 'back', 'arms'],
  musicPreference: 'upbeat/electronic',
  outdoorPreference: true,
  
  // Rest Preferences
  preferredRestPeriods: 90,
  
  // Learning Style
  learningStyle: 'visual',
  instructionDetail: 'moderate',
  
  // Motivation Style
  motivationType: 'achievement',
  motivationFactors: ['visible progress', 'reaching milestones', 'tracking metrics'],
  demotivationFactors: ['plateaus', 'too complex routines', 'not seeing results'],
  
  // Nutrition Preferences
  dietaryPreferences: ['high protein', 'moderate carb'],
  mealFrequency: 4,
  mealPreptTime: 30,
  calorieTarget: 2400,
  macroPreferences: {
    protein: 30,
    carbs: 45,
    fat: 25
  },
  
  // Notification Preferences
  reminderTiming: 'hour_before',
  notificationFrequency: 'moderate',
  
  // Personalization Level
  dataSharing: 'standard',
  recommendationAggressiveness: 'moderate'
};

// Mock behavioral insights
export const mockBehavioralInsights: BehavioralInsights = {
  id: 'insights1',
  userId: 'user123',
  updatedAt: new Date(),
  
  // Adherence Patterns
  workoutAdherence: 85,
  nutritionAdherence: 70,
  recoveryAdherence: 65,
  adherenceTrend: 'increasing',
  
  // Timing Patterns
  optimalWorkoutTimes: ['6:30-7:30', '18:00-19:00'],
  consistentDays: ['monday', 'saturday'],
  inconsistentDays: ['thursday'],
  adherenceByDayOfWeek: {
    'monday': 90,
    'tuesday': 85,
    'wednesday': 60,
    'thursday': 70,
    'friday': 60,
    'saturday': 95,
    'sunday': 50
  },
  
  // Exercise Behavior
  exerciseCompletionRate: {
    'compound': 95,
    'isolation': 85,
    'cardio': 70,
    'flexibility': 60
  },
  averageIntensity: 8,
  intensityPreference: 'higher',
  restBehavior: 'shorter',
  
  // Response to Changes
  adaptationToIntensity: 'positive',
  adaptationToVolume: 'neutral',
  responseToVariety: 'positive',
  
  // Session Behavior
  averageSessionDuration: 65,
  sessionCompletionRate: 92,
  exercisesSkipped: ['lunges', 'calf raises', 'core work'],
  exercisesModified: ['squats', 'burpees'],
  
  // Recovery Patterns
  recoveryNeeds: 'average',
  sleepConsistency: 75,
  stressManagement: 6,
  
  // Identified Habits
  positiveHabits: ['morning workouts', 'protein intake', 'tracking progress'],
  negativeHabits: ['skipping stretching', 'inadequate hydration'],
  habitFormationSpeed: 'average'
};

// Mock recommendation context data
export const mockRecommendationContext = {
  // User Self-Reported Data
  currentEnergySelf: 7,
  currentStressSelf: 5,
  currentMotivationSelf: 8,
  currentSorenessSelf: 3,
  sleepLastNight: 6.5,
  
  // Biometric Data
  heartRateVariability: 68,
  restingHeartRate: 62,
  averageSteps: 9200,
  bodyWeight: 75.2, // kg
  bodyWeightTrend: -0.5, // kg (negative means weight loss)
  
  // Last Workout
  lastWorkout: {
    type: 'Strength',
    date: new Date(Date.now() - 24 * 60 * 60 * 1000), // yesterday
    duration: 65, // minutes
    intensity: 8,
    completionPercentage: 95,
    performanceRating: 'Good',
    muscleGroupsWorked: ['Chest', 'Triceps', 'Shoulders']
  },
  
  // External Factors
  weather: 'Sunny',
  temperature: 22,
  schedule: 'busy',
  travelStatus: false,
  
  // User Goals
  primaryGoal: 'Build Muscle',
  secondaryGoal: 'Improve Mobility',
  weeklyWorkoutTarget: 4,
  
  // Nutrition Data
  calorieTarget: 2600,
  recentProteinAverage: 165, // g
  recentCarbAverage: 240, // g
  recentFatAverage: 85, // g
  hydrationAverage: 2.8, // liters
  
  // Recovery Data
  recoveryScore: 82, // out of 100
  sleepQualityAverage: 7.2, // out of 10
  restDaysSinceLast: 2,
};

// Mock recommendation data
export const mockRecommendations = [
  // WORKOUT RECOMMENDATIONS
  {
    id: 'rec-workout-1',
    type: 'workout',
    title: "Modify today's workout focus",
    description: "Based on your recovery status and recent workouts, we suggest modifying your planned leg workout.",
    reasoning: [
      "You have worked chest, triceps, and shoulders yesterday at high intensity",
      "Your sleep last night was slightly below your optimal range",
      "Your soreness is still moderate from your previous workout",
      "You have hit chest exercises twice this week already"
    ],
    confidenceScore: 86,
    workoutRecommendation: {
      modifications: [
        {
          originalExercise: 'Bench Press (4 sets)',
          newExercise: 'Incline Dumbbell Press (3 sets)',
          reason: 'To reduce overall chest volume while still maintaining stimulus'
        },
        {
          originalExercise: 'Tricep Pushdowns (3 sets)',
          newExercise: 'Tricep Dips (2 sets)',
          reason: 'To shift focus to bodyweight movement for recovery'
        }
      ],
      intensityAdjustment: -15,
      focusAreas: ['Upper back', 'Rear delts', 'Core stability']
    },
    createdAt: new Date(),
    expiresAt: new Date(Date.now() + 12 * 60 * 60 * 1000), // expires in 12 hours
    priority: 'high'
  },
  {
    id: 'rec-workout-2',
    type: 'workout',
    title: 'Add active recovery session',
    description: 'Based on your current training cycle, we recommend adding a light active recovery session tomorrow.',
    reasoning: [
      "You're in week 3 of your training cycle, which is typically high volume",
      'Your HRV trend shows a gradual decline over the past 5 days',
      "You've completed 3 high-intensity sessions this week",
      'Weather forecast shows favorable conditions for outdoor activity'
    ],
    confidenceScore: 92,
    workoutRecommendation: {
      workoutType: 'Active Recovery',
      duration: 40,
      intensityAdjustment: -50,
      activities: [
        'Light mobility work',
        'Gentle cycling or walking',
        'Dynamic stretching',
        'Foam rolling'
      ],
      focusAreas: ['Lower body mobility', 'Spine mobility', 'Circulation']
    },
    createdAt: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2 hours ago
    expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000), // expires in 24 hours
    priority: 'medium'
  },
  
  // NUTRITION RECOMMENDATIONS
  {
    id: 'rec-nutrition-1',
    type: 'nutrition',
    title: 'Adjust post-workout nutrition',
    description: 'Based on your current training and recent food logging, we recommend adjusting your post-workout nutrition.',
    reasoning: [
      'Your muscle-building goal requires optimal post-workout nutrition',
      'Recent protein intake has been slightly below target (average 165g vs 180g target)',
      'Today\'s strength session will focus on large muscle groups',
      'Your body weight trend shows slower than expected progression'
    ],
    confidenceScore: 89,
    nutritionRecommendation: {
      calorieAdjustment: 10, // increase by 10%
      macroAdjustments: {
        protein: 15, // increase by 15%
        carbs: 5, // increase by 5%
        fat: -5 // decrease by 5%
      },
      mealTiming: 'Within 45 minutes post-workout',
      foodSuggestions: [
        'Greek yogurt with berries',
        'Whey protein shake with banana',
        'Chicken breast with sweet potato',
        'Tuna with whole grain wrap',
        'Salmon with quinoa',
        'Cottage cheese with pineapple'
      ]
    },
    createdAt: new Date(Date.now() - 1 * 60 * 60 * 1000), // 1 hour ago
    expiresAt: new Date(Date.now() + 6 * 60 * 60 * 1000), // expires in 6 hours
    priority: 'high'
  },
  {
    id: 'rec-nutrition-2',
    type: 'nutrition',
    title: 'Increase hydration today',
    description: 'Your hydration levels have been trending lower than optimal. Today is a good day to focus on rehydration.',
    reasoning: [
      'Your logged water intake has averaged 2.8L vs your 3.5L target over the past 3 days',
      'Today\'s weather is warmer than usual (22°C)',
      'Your urine color tracking indicates possible mild dehydration',
      'You have a strength training session scheduled later today'
    ],
    confidenceScore: 94,
    nutritionRecommendation: {
      hydrationFocus: true,
      hydrationRecommendation: 'Increase water intake to 4L today',
      hydrationSchedule: [
        '500ml upon waking',
        '500ml mid-morning',
        '500ml pre-workout',
        '500ml during workout',
        '500ml post-workout',
        '1L throughout the afternoon',
        '500ml evening'
      ],
      electrolyteFocus: true,
      foodSuggestions: [
        'Coconut water',
        'Watermelon',
        'Cucumber',
        'Celery',
        'Strawberries'
      ]
    },
    createdAt: new Date(Date.now() - 3 * 60 * 60 * 1000), // 3 hours ago
    expiresAt: new Date(Date.now() + 10 * 60 * 60 * 1000), // expires in 10 hours
    priority: 'medium'
  },

  // RECOVERY RECOMMENDATIONS
  {
    id: 'rec-recovery-1',
    type: 'recovery',
    title: 'Enhance sleep quality tonight',
    description: 'Based on your recent training load and recovery metrics, focusing on sleep quality will be particularly beneficial tonight.',
    reasoning: [
      'Your workout intensity has been high for 3 consecutive sessions',
      'Your HRV is showing a downward trend (68ms vs your baseline of 75ms)',
      'Sleep tracking shows an average of 6.5 hours vs your optimal 7.5 hours',
      'Your self-reported stress level is moderate (5/10)'
    ],
    confidenceScore: 91,
    recoveryRecommendation: {
      sleepFocus: true,
      targetSleepDuration: 8.0,
      sleepRoutine: [
        'Avoid screens 1 hour before bed',
        'Set room temperature to 65-68°F (18-20°C)',
        'Complete 10-minute relaxation breathing',
        'Avoid caffeine after 2pm'
      ],
      supplementConsideration: 'Consider 200-300mg magnesium glycinate before bed',
      mobilityWork: [
        'Hip openers',
        'Thoracic extension',
        'Ankle mobility',
        'Shoulder circles'
      ],
      stressReduction: [
        'Box breathing',
        'Progressive muscle relaxation',
        'Mindfulness meditation',
        'Nature sounds'
      ]
    },
    createdAt: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2 hours ago
    expiresAt: new Date(Date.now() + 8 * 60 * 60 * 1000), // expires in 8 hours
    priority: 'high'
  },
  {
    id: 'rec-recovery-2',
    type: 'recovery',
    title: 'Add 15-min mobility work',
    description: 'Adding a targeted mobility session will help address some movement limitations observed in your recent workouts.',
    reasoning: [
      'Video analysis from your squats shows limited ankle dorsiflexion',
      'Your left hip mobility is more restricted than right (15% difference)',
      "You've reported mild lower back stiffness after workouts",
      'Your mobility assessment scores have decreased slightly over the past month'
    ],
    confidenceScore: 88,
    recoveryRecommendation: {
      mobilityWork: [
        'Ankle dorsiflexion with band',
        'Hip 90/90 stretches',
        'Thoracic extensions on foam roller',
        'Cat-cow spinal movements',
        'Shoulder rotations with band'
      ],
      duration: 15,
      timing: 'Pre-workout or on rest days',
      focusAreas: ['Ankles', 'Hips', 'Thoracic spine'],
      progressionStrategy: 'Increase hold times by 5 seconds each week',
      activeRecovery: 'After your mobility work, consider 10 minutes of light walking to increase blood flow and enhance recovery.'
    },
    createdAt: new Date(Date.now() - 5 * 60 * 60 * 1000), // 5 hours ago
    expiresAt: new Date(Date.now() + 48 * 60 * 60 * 1000), // expires in 48 hours
    priority: 'medium'
  }
];

// Mock onboarding questions
export const mockOnboardingQuestions: OnboardingQuestion[] = [
  {
    id: 'q1',
    questionText: 'What is your primary fitness goal?',
    answerType: 'multiple-choice',
    options: ['Lose weight', 'Build muscle', 'Improve fitness', 'Increase strength', 'Train for event', 'General health'],
    required: true,
    helpText: 'This helps us prioritize your training and nutrition recommendations',
    targetField: 'primaryGoal'
  },
  {
    id: 'q2',
    questionText: 'How would you rate your current fitness level?',
    answerType: 'multiple-choice',
    options: ['Beginner', 'Intermediate', 'Advanced', 'Elite'],
    required: true,
    helpText: 'Be honest - this helps us set appropriate starting points',
    targetField: 'fitnessLevel'
  },
  {
    id: 'q3',
    questionText: 'How many years have you been training consistently?',
    answerType: 'number',
    minValue: 0,
    maxValue: 50,
    step: 0.5,
    required: true,
    targetField: 'yearsOfExperience'
  },
  {
    id: 'q4',
    questionText: 'Do you have any injuries or medical conditions we should know about?',
    answerType: 'text',
    required: false,
    helpText: 'This helps us provide safe exercise recommendations',
    targetField: 'medicalInfo'
  },
  {
    id: 'q5',
    questionText: 'On a scale of 1-10, how would you rate your current stress levels?',
    answerType: 'slider',
    minValue: 1,
    maxValue: 10,
    step: 1,
    required: true,
    helpText: 'Stress impacts recovery and training capacity',
    targetField: 'stressLevel'
  },
  {
    id: 'q6',
    questionText: 'How many days per week can you commit to working out?',
    answerType: 'number',
    minValue: 1,
    maxValue: 7,
    step: 1,
    required: true,
    targetField: 'workoutFrequency'
  },
  {
    id: 'q7',
    questionText: 'What types of training do you enjoy most?',
    answerType: 'multiple-choice',
    options: ['Weight training', 'Cardio', 'HIIT', 'Sports', 'Bodyweight', 'Yoga/Pilates', 'Outdoor activities'],
    required: true,
    helpText: 'You can select multiple options',
    targetField: 'trainingPreferences'
  },
  {
    id: 'q8',
    questionText: 'What is your typical sleep duration on weeknights?',
    answerType: 'number',
    minValue: 3,
    maxValue: 12,
    step: 0.5,
    required: true,
    helpText: 'Sleep is critical for recovery and progress',
    targetField: 'sleepDuration'
  },
  {
    id: 'q9',
    questionText: 'Do you have access to a gym?',
    answerType: 'boolean',
    required: true,
    targetField: 'gymAccess'
  },
  {
    id: 'q10',
    questionText: 'What equipment do you have access to?',
    answerType: 'multiple-choice',
    options: ['Full gym', 'Dumbbells', 'Barbells', 'Kettlebells', 'Resistance bands', 'Bodyweight only', 'Cardio machines', 'Cable machines'],
    required: true,
    helpText: 'You can select multiple options',
    targetField: 'equipmentAccess'
  }
];

// Mock onboarding flow
export const mockOnboardingFlow: OnboardingFlow = {
  id: 'flow1',
  name: 'Standard User Onboarding',
  sections: [
    {
      id: 'section1',
      title: 'Basic Information',
      description: 'Let\'s get to know you better',
      questions: ['q1', 'q2', 'q3']
    },
    {
      id: 'section2',
      title: 'Health & Limitations',
      description: 'Help us keep your workouts safe and effective',
      questions: ['q4', 'q5']
    },
    {
      id: 'section3',
      title: 'Training Preferences',
      description: 'Tell us how you like to train',
      questions: ['q6', 'q7']
    },
    {
      id: 'section4',
      title: 'Lifestyle & Recovery',
      description: 'These factors significantly impact your results',
      questions: ['q8']
    },
    {
      id: 'section5',
      title: 'Equipment & Access',
      description: 'Let us know what you have available to work with',
      questions: ['q9', 'q10'],
      conditionalDisplay: {
        dependsOn: 'q9',
        showIfValue: true
      }
    }
  ]
};

// Mock premium personalization
export const mockPremiumPersonalization: PremiumPersonalization = {
  id: 'premium1',
  userId: 'user123',
  activeUntil: new Date(new Date().setMonth(new Date().getMonth() + 6)),
  
  // Features
  aiProgramsRemaining: 3,
  detailedReportsRemaining: 5,
  personalConsultationsRemaining: 1,
  
  // Settings
  predictionModelsEnabled: true,
  comparativeAnalysisEnabled: true,
  geneticFactorsIncluded: false,
  
  // Usage History
  usageHistory: [
    {
      feature: 'AI Program Generation',
      date: new Date(new Date().setDate(new Date().getDate() - 15)),
      result: 'Strength Focus 8-Week Program'
    },
    {
      feature: 'Detailed Analysis Report',
      date: new Date(new Date().setDate(new Date().getDate() - 10)),
      result: 'Quarterly Progress Analysis'
    }
  ]
}; 
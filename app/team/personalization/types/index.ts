// Personalization System Types

// User Fitness Profile
export interface FitnessProfile {
  id: string;
  userId: string;
  createdAt: Date;
  updatedAt: Date;
  
  // Basic Information
  age: number;
  gender: 'male' | 'female' | 'non-binary' | 'prefer-not-to-say';
  height: number; // in cm
  weight: number; // in kg
  
  // Fitness Background
  fitnessLevel: 'beginner' | 'intermediate' | 'advanced' | 'elite';
  fitnessBackground: string[];
  activityLevel: 'sedentary' | 'lightly_active' | 'moderately_active' | 'very_active' | 'extremely_active';
  yearsOfExperience: number;
  
  // Health Information
  medicalConditions: string[];
  injuries: string[];
  limitations: string[];
  
  // Lifestyle Factors
  sleepAverage: number; // in hours
  stressLevel: 'low' | 'moderate' | 'high';
  occupation: string;
  workSchedule: 'standard' | 'shift' | 'flexible' | 'remote';
  commute: number; // in minutes
  
  // Current Fitness Stats
  currentBodyFat?: number;
  muscleGroups: {
    [key: string]: {
      strength: number; // 1-10 scale
      experience: number; // 1-10 scale
      preference: number; // 1-10 scale
      injuries: string[];
    }
  };
  
  // Fitness Testing Results
  fitnessAssessments: FitnessAssessment[];
  
  // Availability
  availableDays: ('monday' | 'tuesday' | 'wednesday' | 'thursday' | 'friday' | 'saturday' | 'sunday')[];
  timeAvailability: {
    [key: string]: TimeSlot[]; // key is day of week
  };
  
  // Equipment Access
  equipmentAccess: string[];
  gymAccess: boolean;
  homeEquipment: string[];
}

// User Goals
export interface UserGoals {
  id: string;
  userId: string;
  createdAt: Date;
  updatedAt: Date;
  
  // Primary Goals
  primaryGoals: Goal[];
  activeGoal: string; // ID of the active goal
  
  // Body Composition Goals
  targetWeight?: number;
  targetBodyFat?: number;
  
  // Performance Goals
  strengthGoals: StrengthGoal[];
  enduranceGoals: EnduranceGoal[];
  mobilityGoals: MobilityGoal[];
  
  // Timeline
  desiredCompletionDate: Date;
  
  // Priorities
  priorities: {
    strength: number; // 1-10 scale
    endurance: number;
    flexibility: number;
    balance: number;
    speed: number;
    power: number;
    muscleGain: number;
    fatLoss: number;
  };
}

// User Preferences
export interface UserPreferences {
  id: string;
  userId: string;
  updatedAt: Date;
  
  // Workout Preferences
  preferredWorkoutDuration: number; // in minutes
  preferredWorkoutFrequency: number; // per week
  preferredExerciseTypes: string[];
  dislikedExerciseTypes: string[];
  preferredMuscleGroups: string[];
  musicPreference?: string;
  outdoorPreference: boolean;
  
  // Rest Preferences
  preferredRestPeriods: number; // in seconds
  
  // Learning Style
  learningStyle: 'visual' | 'auditory' | 'reading' | 'kinesthetic' | 'mixed';
  instructionDetail: 'minimal' | 'moderate' | 'detailed';
  
  // Motivation Style
  motivationType: 'achievement' | 'social' | 'enjoyment' | 'appearance' | 'health' | 'competition';
  motivationFactors: string[];
  demotivationFactors: string[];
  
  // Nutrition Preferences
  dietaryPreferences: string[];
  mealFrequency: number;
  mealPreptTime: number; // minutes willing to spend
  calorieTarget?: number;
  macroPreferences?: {
    protein: number; // percentage
    carbs: number;
    fat: number;
  };
  
  // Notification Preferences
  reminderTiming: 'day_before' | 'same_day' | 'hour_before';
  notificationFrequency: 'low' | 'moderate' | 'high';
  
  // Personalization Level
  dataSharing: 'minimal' | 'standard' | 'full';
  recommendationAggressiveness: 'conservative' | 'moderate' | 'aggressive';
}

// User Behavioral Insights
export interface BehavioralInsights {
  id: string;
  userId: string;
  updatedAt: Date;
  
  // Adherence Patterns
  workoutAdherence: number; // percentage
  nutritionAdherence: number;
  recoveryAdherence: number;
  adherenceTrend: 'increasing' | 'stable' | 'decreasing';
  
  // Timing Patterns
  optimalWorkoutTimes: string[];
  consistentDays: string[];
  inconsistentDays: string[];
  adherenceByDayOfWeek: {
    [key: string]: number; // percentage
  };
  
  // Exercise Behavior
  exerciseCompletionRate: {
    [key: string]: number; // percentage by exercise type
  };
  averageIntensity: number; // 1-10 scale
  intensityPreference: 'lower' | 'moderate' | 'higher';
  restBehavior: 'shorter' | 'recommended' | 'longer';
  
  // Response to Changes
  adaptationToIntensity: 'positive' | 'neutral' | 'negative';
  adaptationToVolume: 'positive' | 'neutral' | 'negative';
  responseToVariety: 'positive' | 'neutral' | 'negative';
  
  // Session Behavior
  averageSessionDuration: number; // in minutes
  sessionCompletionRate: number; // percentage
  exercisesSkipped: string[]; // most commonly skipped
  exercisesModified: string[]; // most commonly modified
  
  // Recovery Patterns
  recoveryNeeds: 'lower' | 'average' | 'higher';
  sleepConsistency: number; // percentage
  stressManagement: number; // 1-10 scale
  
  // Identified Habits
  positiveHabits: string[];
  negativeHabits: string[];
  habitFormationSpeed: 'slow' | 'average' | 'fast';
}

// Recommendation Contexts
export interface RecommendationContext {
  id: string;
  userId: string;
  timestamp: Date;
  
  // Current State
  currentEnergySelf: number; // 1-10 scale
  currentStressSelf: number;
  currentMotivationSelf: number;
  currentSorenessSelf: number;
  sleepLastNight: number; // in hours
  
  // Biometrics (if available)
  heartRateVariability?: number;
  restingHeartRate?: number;
  bodyTemperature?: number;
  
  // Recent Activity
  lastWorkout?: {
    date: Date;
    type: string;
    intensity: number;
    duration: number;
    feedback: string;
  };
  
  // Environmental Factors
  weather?: string;
  temperature?: number;
  schedule: 'light' | 'normal' | 'busy';
  travelStatus: boolean;
  
  // Special Situations
  upcomingEvents?: {
    date: Date;
    type: string;
    importance: number; // 1-10
  }[];
}

// Recommendations
export interface Recommendation {
  id: string;
  userId: string;
  timestamp: Date;
  context: string; // ID of related RecommendationContext
  
  // Recommendation Type
  type: 'workout' | 'nutrition' | 'recovery' | 'goal' | 'habit';
  
  // Content
  title: string;
  description: string;
  reasoning: string[];
  confidenceScore: number;
  
  // Workout Recommendations
  workoutRecommendation?: {
    workoutId?: string;
    modifications?: {
      exerciseId: string;
      originalExercise: string;
      newExercise: string;
      reason: string;
    }[];
    intensityAdjustment?: number; // percentage
    volumeAdjustment?: number;
    focusAreas?: string[];
  };
  
  // Nutrition Recommendations
  nutritionRecommendation?: {
    calorieAdjustment?: number;
    macroAdjustments?: {
      protein?: number;
      carbs?: number;
      fat?: number;
    };
    foodSuggestions?: string[];
    mealTimingSuggestions?: string[];
    hydrationFocus?: boolean;
  };
  
  // Recovery Recommendations
  recoveryRecommendation?: {
    sleepFocus?: boolean;
    mobilityWork?: string[];
    stressReduction?: string[];
    activeRecovery?: string;
  };
  
  // User Response
  userAction: 'accepted' | 'modified' | 'rejected' | 'pending';
  userFeedback?: string;
  effectivenessRating?: number; // 1-10 scale
}

// Goal
export interface Goal {
  id: string;
  type: 'weight' | 'strength' | 'endurance' | 'habit' | 'custom';
  title: string;
  description: string;
  targetValue: number;
  currentValue: number;
  unit: string;
  startDate: Date;
  targetDate: Date;
  status: 'active' | 'completed' | 'abandoned';
  priority: 'low' | 'medium' | 'high';
  relatedMetrics: string[];
  milestones: {
    value: number;
    achieved: boolean;
    date?: Date;
  }[];
}

// Strength Goal
export interface StrengthGoal {
  id: string;
  exercise: string;
  currentMax: number;
  targetMax: number;
  unit: 'kg' | 'lb' | 'reps';
  deadline: Date;
}

// Endurance Goal
export interface EnduranceGoal {
  id: string;
  activityType: string;
  currentCapacity: number;
  targetCapacity: number;
  unit: 'km' | 'miles' | 'minutes' | 'hours';
  deadline: Date;
}

// Mobility Goal
export interface MobilityGoal {
  id: string;
  jointOrMuscle: string;
  currentRange: number;
  targetRange: number;
  unit: 'degrees' | 'cm' | 'level';
  deadline: Date;
}

// Fitness Assessment
export interface FitnessAssessment {
  id: string;
  date: Date;
  
  // Cardio Assessment
  cardio?: {
    vo2Max?: number;
    restingHeartRate?: number;
    timeToExhaustion?: number;
    recoveryRate?: number;
  };
  
  // Strength Assessment
  strength?: {
    [key: string]: number; // exercise name: 1RM
  };
  
  // Endurance Assessment
  endurance?: {
    [key: string]: number; // activity: distance/time
  };
  
  // Flexibility Assessment
  flexibility?: {
    [key: string]: number; // joint/muscle: range of motion
  };
  
  // Body Composition
  bodyComposition?: {
    weight: number;
    bodyFat?: number;
    muscleMass?: number;
    measurements?: {
      [key: string]: number; // body part: measurement
    };
  };
}

// Time Slot
export interface TimeSlot {
  startTime: string; // 24hr format: '08:00'
  endTime: string;
  preferred: boolean;
}

// Premium Personalization
export interface PremiumPersonalization {
  id: string;
  userId: string;
  activeUntil: Date;
  
  // Features
  aiProgramsRemaining: number;
  detailedReportsRemaining: number;
  personalConsultationsRemaining: number;
  
  // Settings
  predictionModelsEnabled: boolean;
  comparativeAnalysisEnabled: boolean;
  geneticFactorsIncluded: boolean;
  
  // Usage History
  usageHistory: {
    feature: string;
    date: Date;
    result: string;
  }[];
}

// System Improvement Metrics
export interface SystemImprovement {
  recommendationId: string;
  recommendationType: string;
  successRate: number;
  userSatisfaction: number;
  adaptationSpeed: number;
  improvementSuggestions: string[];
  abTestVariant?: string;
  modelVersion: string;
}

// Onboarding Question
export interface OnboardingQuestion {
  id: string;
  questionText: string;
  answerType: 'multiple-choice' | 'slider' | 'text' | 'number' | 'boolean';
  options?: string[];
  minValue?: number;
  maxValue?: number;
  step?: number;
  required: boolean;
  helpText?: string;
  targetField: string; // which field in the user profile this populates
}

// Onboarding Flow
export interface OnboardingFlow {
  id: string;
  name: string;
  sections: {
    id: string;
    title: string;
    description: string;
    questions: string[]; // IDs of questions
    conditionalDisplay?: {
      dependsOn: string; // question ID
      showIfValue: any;
    };
  }[];
}

// AI Model Performance
export interface AIModelPerformance {
  modelId: string;
  modelVersion: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  userSatisfaction: number;
  trainingData: {
    size: number;
    lastUpdated: Date;
    sources: string[];
  };
  improvementAreas: string[];
}

// Engagement Campaign
export interface EngagementCampaign {
  id: string;
  name: string;
  targetUserSegment: string;
  startDate: Date;
  endDate?: Date;
  active: boolean;
  
  // Content
  messages: {
    type: 'email' | 'push' | 'in-app';
    title: string;
    body: string;
    trigger: 'time-based' | 'behavior-based';
    triggerDetails: string;
  }[];
  
  // Performance
  metrics: {
    impressions: number;
    interactions: number;
    conversions: number;
    retention: number;
  };
}

// Adaptive Challenge
export interface AdaptiveChallenge {
  id: string;
  userId: string;
  title: string;
  description: string;
  difficulty: number; // 1-10 scale
  type: 'strength' | 'endurance' | 'habit' | 'nutrition' | 'combined';
  startDate: Date;
  endDate: Date;
  
  // Goals
  goals: {
    metric: string;
    target: number;
    unit: string;
    current: number;
  }[];
  
  // Adaptation
  baselinePerformance: number;
  currentDifficulty: number;
  adaptationHistory: {
    date: Date;
    difficulty: number;
    performance: number;
    adjustment: number;
  }[];
  
  // Rewards
  rewardType: 'points' | 'badge' | 'feature' | 'physical';
  rewardDetails: string;
  claimed: boolean;
} 
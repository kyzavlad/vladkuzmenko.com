// Progression type enum
export enum ProgressionType {
  LINEAR = 'linear',
  UNDULATING = 'undulating',
  BLOCK = 'block'
}

// Workout Program Architecture - Core Types

// Program categories
export enum WorkoutCategory {
  STRENGTH = 'Strength Training',
  CARDIO = 'Cardio',
  FLEXIBILITY = 'Flexibility',
  HIIT = 'HIIT',
  YOGA = 'Yoga',
  BALANCE = 'Balance',
  FUNCTIONAL = 'Functional Training',
  SPORT_SPECIFIC = 'Sport Specific',
  REHABILITATION = 'Rehabilitation',
}

// Difficulty levels
export enum DifficultyLevel {
  BEGINNER = 'Beginner',
  INTERMEDIATE = 'Intermediate',
  ADVANCED = 'Advanced',
}

// Duration options
export enum WorkoutDuration {
  QUICK = 'quick',         // 10-20 minutes
  STANDARD = 'standard',   // 30-45 minutes
  EXTENDED = 'extended'    // 60+ minutes
}

// Equipment requirements
export enum EquipmentLevel {
  NONE = 'Bodyweight Only',
  MINIMAL = 'Minimal Equipment',
  STANDARD = 'Standard Equipment',
  FULL = 'Full Gym',
}

// User fitness goals
export enum FitnessGoal {
  STRENGTH = 'Strength',
  MUSCLE_GAIN = 'Muscle Gain',
  FAT_LOSS = 'Fat Loss',
  ENDURANCE = 'Endurance',
  FLEXIBILITY = 'Flexibility',
  ATHLETIC_PERFORMANCE = 'Athletic Performance',
  GENERAL_FITNESS = 'General Fitness',
  REHABILITATION = 'Rehabilitation',
  POSTURE = 'Posture',
  BALANCE = 'Balance',
}

// Target muscle groups
export enum MuscleGroup {
  FULL_BODY = 'Full Body',
  UPPER_BODY = 'Upper Body',
  LOWER_BODY = 'Lower Body',
  CORE = 'Core',
  CHEST = 'Chest',
  BACK = 'Back',
  SHOULDERS = 'Shoulders',
  ARMS = 'Arms',
  LEGS = 'Legs',
  GLUTES = 'Glutes',
}

// Exercise type
export enum ExerciseType {
  STRENGTH = 'strength',
  CARDIO = 'cardio',
  FLEXIBILITY = 'flexibility',
  BALANCE = 'balance',
  PLYOMETRIC = 'plyometric',
  CALISTHENICS = 'calisthenics'
}

// Set types
export enum SetType {
  REGULAR = 'regular',
  SUPERSET = 'superset',
  DROPSET = 'dropset',
  CIRCUIT = 'circuit',
  AMRAP = 'amrap',       // As Many Reps As Possible
  EMOM = 'emom',         // Every Minute On the Minute
  TABATA = 'tabata',     // 20 seconds work, 10 seconds rest
  FOR_TIME = 'for-time'  // Complete as fast as possible
}

// Program target audience
export enum TargetAudience {
  MEN = 'Men',
  WOMEN = 'Women',
  BOTH = 'Men & Women',
  SENIORS = 'Seniors',
  YOUTH = 'Youth',
  PRENATAL = 'Prenatal',
  POSTNATAL = 'Postnatal',
}

// Exercise interface
export interface Exercise {
  id: string;
  name: string;
  type: ExerciseType;
  muscleGroups: MuscleGroup[];
  equipment: EquipmentLevel;
  description: string;
  instructions: string[];
  videoUrl: string;
  thumbnailUrl: string;
  difficultyLevel: DifficultyLevel;
  modifications: {
    easier: string;
    harder: string;
  };
  metrics: {
    calories: number;      // estimated calories per minute
    strengthFocus: number; // 1-10 scale
    cardioFocus: number;   // 1-10 scale
  };
}

// Set interface
export interface ExerciseSet {
  type: SetType;
  exercises: {
    exerciseId: string;
    reps?: number;
    time?: number;        // in seconds
    distance?: number;    // in meters
    restAfter?: number;   // in seconds
    weight?: number;      // in kg or lb (specified in user preferences)
    intensity?: number;   // 1-10 scale
  }[];
  rest: number;          // rest time in seconds
  notes?: string;
}

// Workout interface
export interface Workout {
  id: string;
  name: string;
  description: string;
  category: WorkoutCategory;
  difficultyLevel: DifficultyLevel;
  duration: WorkoutDuration;
  estimatedTimeMinutes: number;
  equipment: EquipmentLevel;
  targetAudience: TargetAudience;
  muscleGroups: MuscleGroup[];
  fitnessGoals: FitnessGoal[];
  warmup: {
    exercises: string[];  // exercise IDs
    duration: number;     // in minutes
  };
  sets: ExerciseSet[];
  cooldown: {
    exercises: string[];  // exercise IDs
    duration: number;     // in minutes
  };
  tips: string[];
  imageUrl: string;
  metrics: {
    caloriesBurned: number;  // estimated total
    strengthScore: number;   // 1-10 scale
    cardioScore: number;     // 1-10 scale
    flexibilityScore: number; // 1-10 scale
  };
  relatedWorkouts: string[]; // related workout IDs
}

// Workout Program interface
export interface WorkoutProgram {
  id: string;
  name: string;
  description: string;
  category: WorkoutCategory;
  difficultyLevel: DifficultyLevel;
  duration: {
    weeks: number;
    daysPerWeek: number;
  };
  equipment: EquipmentLevel;
  targetAudience: TargetAudience;
  fitnessGoals: FitnessGoal[];
  muscleGroups: MuscleGroup[];
  workouts: {
    week: number;
    day: number;
    workoutId: string;
    notes?: string;
  }[];
  progression: {
    type: 'linear' | 'undulating' | 'block';
    deloadFrequency?: number;  // every X weeks
    autoAdjust: boolean;       // adjusts based on user performance
  };
  prerequisites?: {
    level?: DifficultyLevel;
    completedPrograms?: string[];  // program IDs
  };
  creator: {
    name: string;
    credentials?: string;
    imageUrl?: string;
  };
  imageUrl: string;
  featured: boolean;
  premium: boolean;
  tokenCost?: number;
  tags: string[];
}

// User progress tracking
export interface UserWorkoutProgress {
  userId: string;
  workoutId: string;
  programId?: string;
  date: string;
  completed: boolean;
  duration: number;  // actual time taken in minutes
  sets: {
    setIndex: number;
    exercises: {
      exerciseId: string;
      reps?: number;
      weight?: number;
      completed: boolean;
      difficulty: 1 | 2 | 3 | 4 | 5;  // user rating of difficulty
    }[];
  }[];
  caloriesBurned?: number;
  notes?: string;
  rating?: 1 | 2 | 3 | 4 | 5;  // user rating of workout
}

// User program progress
export interface UserProgramProgress {
  userId: string;
  programId: string;
  startDate: string;
  currentWeek: number;
  currentDay: number;
  workoutsCompleted: number;
  totalWorkouts: number;
  lastWorkoutDate?: string;
  adaptations: {
    exerciseId: string;
    adjustment: 'easier' | 'harder';
    reason?: string;
  }[];
  notes: string;
  status: 'active' | 'completed' | 'paused' | 'abandoned';
}

// Achievement interface
export interface Achievement {
  id: string;
  name: string;
  description: string;
  imageUrl: string;
  criteria: {
    type: 'workout_count' | 'program_completion' | 'streak' | 'specific_workout' | 'custom';
    count?: number;
    workoutIds?: string[];
    programIds?: string[];
    daysInARow?: number;
    specificRequirement?: string;
  };
  reward?: {
    tokens?: number;
    unlocks?: string[];
  };
  rarity: 'common' | 'uncommon' | 'rare' | 'epic' | 'legendary';
}

// Challenge interface
export interface Challenge {
  id: string;
  name: string;
  description: string;
  imageUrl: string;
  startDate: string;
  endDate: string;
  criteria: {
    type: 'workout_count' | 'specific_workouts' | 'total_time' | 'total_weight' | 'custom';
    count?: number;
    workoutIds?: string[];
    programIds?: string[];
    minutes?: number;
    weight?: number;
    customRequirement?: string;
  };
  participants: {
    userId: string;
    progress: number;  // percentage of completion
    completed: boolean;
    completionDate?: string;
  }[];
  rewards: {
    tokens?: number;
    achievementId?: string;
    badgeUrl?: string;
  };
  premium: boolean;
  tokenCost?: number;
} 
// User Profile Types
export interface UserProfile {
  id: string;
  name: string;
  email: string;
  level: number;
  experience: number;
  rank: UserRank;
  joinDate: string | Date;
  avatar?: string;
  streak: {
    current: number;
    longest: number;
    lastActive: string | Date;
  };
  coins: number;
  totalWorkoutsCompleted: number;
}

// User Ranks
export type UserRank = 'Beginner' | 'Rookie' | 'Athlete' | 'Advanced' | 'Elite' | 'Master' | 'Champion';

// Experience and Level System
export interface LevelInfo {
  level: number;
  title: string;
  minExperience: number;
  maxExperience: number;
  benefits: string[];
  icon: string;
}

// Achievement Types
export interface Achievement {
  id: string;
  title: string;
  description: string;
  category: AchievementCategory;
  type: AchievementType;
  icon: string;
  requirement: number;
  progress: number;
  completed: boolean;
  completedDate?: string | Date;
  reward: {
    experience: number;
    coins: number;
    specialReward?: SpecialReward;
  };
  secret?: boolean;
}

export type AchievementCategory = 
  | 'workout' 
  | 'nutrition' 
  | 'strength' 
  | 'consistency' 
  | 'challenge' 
  | 'social' 
  | 'milestone' 
  | 'event';

export type AchievementType = 
  | 'cumulative' // Accumulate X (workouts, meals, etc.)
  | 'streak' // Maintain a streak for X days
  | 'threshold' // Reach a specific value
  | 'progression' // Improve by X amount
  | 'completion' // Complete a specific thing
  | 'collection' // Collect X different things
  | 'one_time'; // One time achievement

export interface SpecialReward {
  type: 'content' | 'feature' | 'discount' | 'virtual_session' | 'custom_workout';
  title: string;
  description: string;
  duration?: number; // In days if applicable
  code?: string; // For discounts/unlocks
}

// Body Measurement Types
export interface BodyMeasurements {
  date: string | Date;
  weight?: number;
  height?: number;
  bodyFat?: number;
  chest?: number;
  waist?: number;
  hips?: number;
  shoulders?: number;
  leftArm?: number;
  rightArm?: number;
  leftThigh?: number;
  rightThigh?: number;
  leftCalf?: number;
  rightCalf?: number;
  notes?: string;
}

// Strength Metrics
export interface StrengthMetric {
  exercise: string;
  date: string | Date;
  value: number; // weight in kg/lbs
  reps: number;
  oneRepMax: number; // calculated 1RM
  unit: 'kg' | 'lb';
}

// Performance Metrics
export interface PerformanceMetric {
  activity: string;
  date: string | Date;
  value: number;
  unit: string;
  previousBest: number;
  improvement: number;
}

// Progress Photo
export interface ProgressPhoto {
  id: string;
  date: string | Date;
  url: string;
  type: 'front' | 'back' | 'side' | 'custom';
  notes?: string;
  measurements?: BodyMeasurements;
}

// Goals
export interface Goal {
  id: string;
  title: string;
  description: string;
  category: 'strength' | 'body' | 'performance' | 'habit' | 'nutrition' | 'custom';
  targetValue: number;
  currentValue: number;
  unit: string;
  startDate: string | Date;
  targetDate: string | Date;
  completedDate?: string | Date;
  progress: number; // 0-100
  status: 'not_started' | 'in_progress' | 'completed' | 'overdue';
  checkIns: GoalCheckIn[];
  reminderFrequency?: 'daily' | 'weekly' | 'monthly';
}

export interface GoalCheckIn {
  date: string | Date;
  value: number;
  notes?: string;
}

// Consistency Calendar
export interface ConsistencyRecord {
  date: string | Date;
  workoutCompleted: boolean;
  nutritionTracked: boolean;
  measurementsUpdated: boolean;
  goalProgress: boolean;
  type?: 'rest_day' | 'workout_day';
  notes?: string;
}

// Challenges
export interface Challenge {
  id: string;
  title: string;
  description: string;
  startDate: string | Date;
  endDate: string | Date;
  category: 'personal' | 'group' | 'community' | 'event';
  status: 'upcoming' | 'active' | 'completed' | 'failed';
  tasks: ChallengeTask[];
  progress: number; // 0-100
  participants?: number;
  reward: {
    experience: number;
    coins: number;
    specialReward?: SpecialReward;
  };
}

export interface ChallengeTask {
  id: string;
  title: string;
  description: string;
  completed: boolean;
  completedDate?: string | Date;
}

// Leaderboards
export interface LeaderboardEntry {
  userId: string;
  username: string;
  avatar?: string;
  rank: number;
  score: number;
  change?: number; // Change in rank compared to previous period
}

export interface Leaderboard {
  id: string;
  title: string;
  category: 'workouts' | 'consistency' | 'strength' | 'nutrition' | 'community' | 'challenge';
  period: 'weekly' | 'monthly' | 'all_time';
  entries: LeaderboardEntry[];
}

// Reward Shop Item
export interface RewardShopItem {
  id: string;
  title: string;
  description: string;
  category: 'content' | 'feature' | 'discount' | 'virtual' | 'physical';
  price: number; // in coins
  image: string;
  available: boolean;
  limitedTime?: boolean;
  expiryDate?: string | Date;
  stock?: number;
  requiredLevel?: number;
}

// Progress stats
export interface ProgressStats {
  workoutsThisWeek: number;
  workoutsTotal: number;
  currentStreak: number;
  nutritionAdherence: number; // 0-100
  weightChange: {
    lastMonth: number;
    total: number;
  };
  strengthGains: {
    lastMonth: number; // percentage
    total: number; // percentage
  };
  achievements: {
    total: number;
    unlocked: number;
    recentlyUnlocked: Achievement[];
  };
} 
import { 
  UserProfile, 
  LevelInfo, 
  Achievement, 
  BodyMeasurements, 
  StrengthMetric,
  ProgressPhoto,
  Goal,
  ConsistencyRecord,
  Challenge,
  Leaderboard,
  RewardShopItem,
  ProgressStats
} from '../types';

// Mock user profile
export const userProfile: UserProfile = {
  id: 'user123',
  name: 'Alex Johnson',
  email: 'alex@example.com',
  level: 14,
  experience: 7450,
  rank: 'Advanced',
  joinDate: new Date('2023-02-15'),
  avatar: '/avatars/user-1.png',
  streak: {
    current: 12,
    longest: 28,
    lastActive: new Date()
  },
  coins: 450,
  totalWorkoutsCompleted: 87
};

// Level system
export const levelSystem: LevelInfo[] = [
  {
    level: 1,
    title: 'Beginner',
    minExperience: 0,
    maxExperience: 100,
    benefits: ['Basic workout tracking'],
    icon: '🔰'
  },
  {
    level: 5,
    title: 'Rookie',
    minExperience: 500,
    maxExperience: 1000,
    benefits: ['Custom workouts', 'Basic analytics'],
    icon: '⭐'
  },
  {
    level: 10,
    title: 'Athlete',
    minExperience: 2500,
    maxExperience: 4000,
    benefits: ['Group challenges', 'Advanced analytics'],
    icon: '🏅'
  },
  {
    level: 15,
    title: 'Advanced',
    minExperience: 7500,
    maxExperience: 10000,
    benefits: ['Personalized recommendations', 'Priority support'],
    icon: '💪'
  },
  {
    level: 25,
    title: 'Elite',
    minExperience: 20000,
    maxExperience: 25000,
    benefits: ['Expert workout reviews', 'Beta features access'],
    icon: '⚡'
  },
  {
    level: 40,
    title: 'Master',
    minExperience: 40000,
    maxExperience: 50000,
    benefits: ['One-on-one coaching session', 'VIP content'],
    icon: '🔱'
  },
  {
    level: 50,
    title: 'Champion',
    minExperience: 75000,
    maxExperience: 100000,
    benefits: ['Featured user profile', 'Full platform unlocked'],
    icon: '👑'
  }
];

// Achievements
export const achievements: Achievement[] = [
  {
    id: 'a1',
    title: 'First Workout',
    description: 'Complete your first workout',
    category: 'workout',
    type: 'one_time',
    icon: '🏋️',
    requirement: 1,
    progress: 1,
    completed: true,
    completedDate: new Date('2023-02-16'),
    reward: {
      experience: 50,
      coins: 10
    }
  },
  {
    id: 'a2',
    title: 'Consistency Novice',
    description: 'Complete workouts for 7 days in a row',
    category: 'consistency',
    type: 'streak',
    icon: '📆',
    requirement: 7,
    progress: 12,
    completed: true,
    completedDate: new Date('2023-03-01'),
    reward: {
      experience: 150,
      coins: 25
    }
  },
  {
    id: 'a3',
    title: 'Consistency Master',
    description: 'Complete workouts for 30 days in a row',
    category: 'consistency',
    type: 'streak',
    icon: '🔥',
    requirement: 30,
    progress: 12,
    completed: false,
    reward: {
      experience: 500,
      coins: 100,
      specialReward: {
        type: 'feature',
        title: 'Advanced Progress Analytics',
        description: 'Unlock advanced progress analytics for 30 days',
        duration: 30
      }
    }
  },
  {
    id: 'a4',
    title: 'Strength Milestone: Deadlift',
    description: 'Deadlift 100kg/220lbs',
    category: 'strength',
    type: 'threshold',
    icon: '🏋️',
    requirement: 100,
    progress: 90,
    completed: false,
    reward: {
      experience: 200,
      coins: 30
    }
  },
  {
    id: 'a5',
    title: 'Nutrition Pro',
    description: 'Log your nutrition for 30 days total',
    category: 'nutrition',
    type: 'cumulative',
    icon: '🥗',
    requirement: 30,
    progress: 22,
    completed: false,
    reward: {
      experience: 300,
      coins: 50
    }
  },
  {
    id: 'a6',
    title: 'Group Challenge Champion',
    description: 'Win a group challenge',
    category: 'challenge',
    type: 'completion',
    icon: '🏆',
    requirement: 1,
    progress: 0,
    completed: false,
    reward: {
      experience: 400,
      coins: 75,
      specialReward: {
        type: 'custom_workout',
        title: 'Custom Workout Plan',
        description: 'Get a custom workout plan designed by a professional trainer'
      }
    }
  },
  {
    id: 'a7',
    title: 'Workout Explorer',
    description: 'Try 10 different workout types',
    category: 'workout',
    type: 'collection',
    icon: '🧭',
    requirement: 10,
    progress: 7,
    completed: false,
    reward: {
      experience: 250,
      coins: 40
    }
  },
  {
    id: 'a8',
    title: 'Century Club',
    description: 'Complete 100 workouts',
    category: 'milestone',
    type: 'cumulative',
    icon: '💯',
    requirement: 100,
    progress: 87,
    completed: false,
    reward: {
      experience: 750,
      coins: 150,
      specialReward: {
        type: 'content',
        title: 'Premium Workout Library',
        description: 'Unlock the premium workout library'
      }
    }
  },
  {
    id: 'a9',
    title: 'Hidden Achievement',
    description: 'This will be revealed when completed',
    category: 'milestone',
    type: 'one_time',
    icon: '❓',
    requirement: 1,
    progress: 0,
    completed: false,
    secret: true,
    reward: {
      experience: 100,
      coins: 20
    }
  }
];

// Body Measurements
export const bodyMeasurements: BodyMeasurements[] = [
  {
    date: new Date('2023-02-15'),
    weight: 85.5,
    height: 180.0,
    bodyFat: 18.2,
    chest: 98.0,
    waist: 88.0,
    hips: 100.0,
    shoulders: 120.0,
    leftArm: 35.0,
    rightArm: 35.5,
    leftThigh: 60.0,
    rightThigh: 60.5,
    leftCalf: 38.0,
    rightCalf: 38.0,
    notes: 'Starting measurements'
  },
  {
    date: new Date('2023-03-15'),
    weight: 84.2,
    bodyFat: 17.8,
    chest: 99.0,
    waist: 86.5,
    shoulders: 121.0,
    leftArm: 35.5,
    rightArm: 36.0,
    notes: 'One month progress. Feeling stronger!'
  },
  {
    date: new Date('2023-04-15'),
    weight: 83.1,
    bodyFat: 17.0,
    chest: 100.0,
    waist: 85.0,
    shoulders: 122.0,
    leftArm: 36.0,
    rightArm: 36.5,
    leftThigh: 61.0,
    rightThigh: 61.5,
    notes: 'Two months progress. Diet going well.'
  },
  {
    date: new Date('2023-05-15'),
    weight: 82.3,
    bodyFat: 16.5,
    chest: 100.5,
    waist: 84.0,
    shoulders: 123.0,
    leftArm: 36.5,
    rightArm: 37.0,
    notes: 'Three months progress. Feeling great!'
  },
  {
    date: new Date('2023-06-15'),
    weight: 81.8,
    bodyFat: 16.0,
    chest: 101.0,
    waist: 83.0,
    shoulders: 124.0,
    leftArm: 37.0,
    rightArm: 37.5,
    leftThigh: 62.0,
    rightThigh: 62.5,
    notes: 'Four months progress. Program is working well.'
  }
];

// Strength Metrics
export const strengthMetrics: StrengthMetric[] = [
  // Bench Press
  {
    exercise: 'Bench Press',
    date: new Date('2023-02-20'),
    value: 80,
    reps: 5,
    oneRepMax: 90,
    unit: 'kg'
  },
  {
    exercise: 'Bench Press',
    date: new Date('2023-03-20'),
    value: 85,
    reps: 5,
    oneRepMax: 96,
    unit: 'kg'
  },
  {
    exercise: 'Bench Press',
    date: new Date('2023-04-20'),
    value: 87.5,
    reps: 5,
    oneRepMax: 99,
    unit: 'kg'
  },
  {
    exercise: 'Bench Press',
    date: new Date('2023-05-20'),
    value: 90,
    reps: 5,
    oneRepMax: 102,
    unit: 'kg'
  },
  // Squat
  {
    exercise: 'Squat',
    date: new Date('2023-02-22'),
    value: 100,
    reps: 5,
    oneRepMax: 113,
    unit: 'kg'
  },
  {
    exercise: 'Squat',
    date: new Date('2023-03-22'),
    value: 110,
    reps: 5,
    oneRepMax: 124,
    unit: 'kg'
  },
  {
    exercise: 'Squat',
    date: new Date('2023-04-22'),
    value: 120,
    reps: 5,
    oneRepMax: 136,
    unit: 'kg'
  },
  {
    exercise: 'Squat',
    date: new Date('2023-05-22'),
    value: 125,
    reps: 5,
    oneRepMax: 141,
    unit: 'kg'
  },
  // Deadlift
  {
    exercise: 'Deadlift',
    date: new Date('2023-02-24'),
    value: 120,
    reps: 3,
    oneRepMax: 129,
    unit: 'kg'
  },
  {
    exercise: 'Deadlift',
    date: new Date('2023-03-24'),
    value: 130,
    reps: 3,
    oneRepMax: 140,
    unit: 'kg'
  },
  {
    exercise: 'Deadlift',
    date: new Date('2023-04-24'),
    value: 140,
    reps: 3,
    oneRepMax: 151,
    unit: 'kg'
  },
  {
    exercise: 'Deadlift',
    date: new Date('2023-05-24'),
    value: 145,
    reps: 3,
    oneRepMax: 156,
    unit: 'kg'
  }
];

// Progress photos
export const progressPhotos: ProgressPhoto[] = [
  {
    id: 'p1',
    date: new Date('2023-02-15'),
    url: '/images/progress/front-1.jpg',
    type: 'front',
    notes: 'Starting point'
  },
  {
    id: 'p2',
    date: new Date('2023-02-15'),
    url: '/images/progress/side-1.jpg',
    type: 'side',
    notes: 'Starting point'
  },
  {
    id: 'p3',
    date: new Date('2023-04-15'),
    url: '/images/progress/front-2.jpg',
    type: 'front',
    notes: 'Two months progress'
  },
  {
    id: 'p4',
    date: new Date('2023-04-15'),
    url: '/images/progress/side-2.jpg',
    type: 'side',
    notes: 'Two months progress'
  },
  {
    id: 'p5',
    date: new Date('2023-06-15'),
    url: '/images/progress/front-3.jpg',
    type: 'front',
    notes: 'Four months progress'
  },
  {
    id: 'p6',
    date: new Date('2023-06-15'),
    url: '/images/progress/side-3.jpg',
    type: 'side',
    notes: 'Four months progress'
  }
];

// Goals
export const goals: Goal[] = [
  {
    id: 'g1',
    title: 'Deadlift 150kg',
    description: 'Achieve a deadlift of 150kg (330lbs) for 1 rep',
    category: 'strength',
    targetValue: 150,
    currentValue: 145,
    unit: 'kg',
    startDate: new Date('2023-02-15'),
    targetDate: new Date('2023-07-15'),
    progress: 97, // (145/150) * 100
    status: 'in_progress',
    checkIns: [
      { date: new Date('2023-02-24'), value: 120, notes: 'First attempt' },
      { date: new Date('2023-03-24'), value: 130, notes: 'Getting stronger' },
      { date: new Date('2023-04-24'), value: 140, notes: 'Almost there' },
      { date: new Date('2023-05-24'), value: 145, notes: 'So close!' }
    ]
  },
  {
    id: 'g2',
    title: 'Reach 15% Body Fat',
    description: 'Reduce body fat percentage to 15%',
    category: 'body',
    targetValue: 15,
    currentValue: 16,
    unit: '%',
    startDate: new Date('2023-02-15'),
    targetDate: new Date('2023-08-15'),
    progress: 93, // Progress towards target
    status: 'in_progress',
    checkIns: [
      { date: new Date('2023-02-15'), value: 18.2, notes: 'Starting point' },
      { date: new Date('2023-03-15'), value: 17.8, notes: 'Small progress' },
      { date: new Date('2023-04-15'), value: 17.0, notes: 'Improving diet' },
      { date: new Date('2023-05-15'), value: 16.5, notes: 'Good progress' },
      { date: new Date('2023-06-15'), value: 16.0, notes: 'Almost there' }
    ]
  },
  {
    id: 'g3',
    title: 'Run 5km in 25 minutes',
    description: 'Improve 5km running time to under 25 minutes',
    category: 'performance',
    targetValue: 25,
    currentValue: 27.5,
    unit: 'min',
    startDate: new Date('2023-03-01'),
    targetDate: new Date('2023-09-01'),
    progress: 75, // Progress towards target
    status: 'in_progress',
    checkIns: [
      { date: new Date('2023-03-01'), value: 32, notes: 'Starting point' },
      { date: new Date('2023-04-01'), value: 30, notes: 'Added interval training' },
      { date: new Date('2023-05-01'), value: 29, notes: 'Slowly improving' },
      { date: new Date('2023-06-01'), value: 27.5, notes: 'Getting closer' }
    ]
  },
  {
    id: 'g4',
    title: 'Track Nutrition Daily',
    description: 'Log nutrition every day for 60 days',
    category: 'habit',
    targetValue: 60,
    currentValue: 45,
    unit: 'days',
    startDate: new Date('2023-04-01'),
    targetDate: new Date('2023-06-30'),
    progress: 75, // (45/60) * 100
    status: 'in_progress',
    checkIns: [
      { date: new Date('2023-04-15'), value: 15, notes: 'Good start' },
      { date: new Date('2023-05-01'), value: 30, notes: 'Halfway there' },
      { date: new Date('2023-06-01'), value: 45, notes: 'Maintaining streak' }
    ]
  }
];

// Consistency Calendar - Last 10 days
export const consistencyRecords: ConsistencyRecord[] = Array.from({ length: 30 }, (_, i) => {
  const date = new Date();
  date.setDate(date.getDate() - i);
  
  // Generate some gaps in the data for realism
  const missed = [5, 10, 18, 25].includes(i);
  const restDay = [7, 14, 21, 28].includes(i);
  
  return {
    date: new Date(date),
    workoutCompleted: !missed && !restDay, 
    nutritionTracked: !missed,
    measurementsUpdated: i % 7 === 0, // Weekly measurements
    goalProgress: i % 3 === 0, // Update goals every 3 days
    type: restDay ? 'rest_day' : 'workout_day',
    notes: restDay ? 'Scheduled rest day' : undefined
  } as ConsistencyRecord; // Use type assertion to ensure correct type
}).reverse(); // Recent dates first

// Challenges
export const challenges: Challenge[] = [
  {
    id: 'c1',
    title: '30-Day Consistency Challenge',
    description: 'Complete your scheduled workouts for 30 days straight',
    startDate: new Date('2023-06-01'),
    endDate: new Date('2023-06-30'),
    category: 'personal',
    status: 'active',
    tasks: [
      {
        id: 't1',
        title: 'Week 1 Completion',
        description: 'Complete all workouts in week 1',
        completed: true,
        completedDate: new Date('2023-06-07')
      },
      {
        id: 't2',
        title: 'Week 2 Completion',
        description: 'Complete all workouts in week 2',
        completed: true,
        completedDate: new Date('2023-06-14')
      },
      {
        id: 't3',
        title: 'Week 3 Completion',
        description: 'Complete all workouts in week 3',
        completed: false
      },
      {
        id: 't4',
        title: 'Week 4 Completion',
        description: 'Complete all workouts in week 4',
        completed: false
      }
    ],
    progress: 50, // 2/4 tasks completed
    reward: {
      experience: 500,
      coins: 100,
      specialReward: {
        type: 'content',
        title: 'Premium Workout Series',
        description: 'Unlock the premium HIIT workout series'
      }
    }
  },
  {
    id: 'c2',
    title: 'Summer Shred Challenge',
    description: 'A community challenge to get in shape for summer',
    startDate: new Date('2023-05-01'),
    endDate: new Date('2023-07-31'),
    category: 'community',
    status: 'active',
    tasks: [
      {
        id: 't1',
        title: 'Complete 20 workouts',
        description: 'Finish at least 20 workouts during the challenge period',
        completed: true,
        completedDate: new Date('2023-06-10')
      },
      {
        id: 't2',
        title: 'Log nutrition for 30 days',
        description: 'Track your nutrition for at least 30 days',
        completed: false
      },
      {
        id: 't3',
        title: 'Lose 2% body fat',
        description: 'Reduce your body fat percentage by at least 2%',
        completed: false
      },
      {
        id: 't4',
        title: 'Complete 5 outdoor workouts',
        description: 'Do at least 5 workouts outdoors',
        completed: true,
        completedDate: new Date('2023-06-05')
      }
    ],
    progress: 50, // 2/4 tasks completed
    participants: 1450,
    reward: {
      experience: 750,
      coins: 150,
      specialReward: {
        type: 'feature',
        title: 'Body Composition Analysis',
        description: 'Unlock the body composition analysis feature for 3 months',
        duration: 90
      }
    }
  },
  {
    id: 'c3',
    title: 'Strength Fundamentals',
    description: 'Master the fundamental strength exercises',
    startDate: new Date('2023-03-01'),
    endDate: new Date('2023-05-31'),
    category: 'personal',
    status: 'completed',
    tasks: [
      {
        id: 't1',
        title: 'Perfect Squat Form',
        description: 'Submit a video of your squat form for review',
        completed: true,
        completedDate: new Date('2023-03-15')
      },
      {
        id: 't2',
        title: 'Bench Press Progress',
        description: 'Improve your bench press by 10%',
        completed: true,
        completedDate: new Date('2023-04-10')
      },
      {
        id: 't3',
        title: 'Deadlift Milestone',
        description: 'Deadlift 100kg/220lbs with proper form',
        completed: true,
        completedDate: new Date('2023-05-05')
      }
    ],
    progress: 100, // Completed
    reward: {
      experience: 400,
      coins: 80
    }
  }
];

// Leaderboards
export const leaderboards: Leaderboard[] = [
  {
    id: 'l1',
    title: 'Weekly Workout Count',
    category: 'workouts',
    period: 'weekly',
    entries: [
      { userId: 'user1', username: 'FitnessKing', avatar: '/avatars/user1.png', rank: 1, score: 12, change: 0 },
      { userId: 'user2', username: 'WorkoutQueen', avatar: '/avatars/user2.png', rank: 2, score: 10, change: 1 },
      { userId: 'user3', username: 'GymRat', avatar: '/avatars/user3.png', rank: 3, score: 9, change: -1 },
      { userId: 'user4', username: 'IronPumper', avatar: '/avatars/user4.png', rank: 4, score: 8, change: 2 },
      { userId: 'user123', username: 'Alex Johnson', avatar: '/avatars/user-1.png', rank: 5, score: 7, change: 1 },
      { userId: 'user6', username: 'FitnessFanatic', avatar: '/avatars/user6.png', rank: 6, score: 6, change: -2 },
      { userId: 'user7', username: 'MuscleBuilder', avatar: '/avatars/user7.png', rank: 7, score: 5, change: 0 },
      { userId: 'user8', username: 'CardioKing', avatar: '/avatars/user8.png', rank: 8, score: 4, change: -3 },
      { userId: 'user9', username: 'PowerLifter', avatar: '/avatars/user9.png', rank: 9, score: 3, change: 1 },
      { userId: 'user10', username: 'FlexMaster', avatar: '/avatars/user10.png', rank: 10, score: 2, change: -1 }
    ]
  },
  {
    id: 'l2',
    title: 'Monthly Consistency',
    category: 'consistency',
    period: 'monthly',
    entries: [
      { userId: 'user11', username: 'ConsistencyQueen', avatar: '/avatars/user11.png', rank: 1, score: 98, change: 2 },
      { userId: 'user12', username: 'DailyGrinder', avatar: '/avatars/user12.png', rank: 2, score: 96, change: 0 },
      { userId: 'user13', username: 'NeverMissMonday', avatar: '/avatars/user13.png', rank: 3, score: 95, change: 1 },
      { userId: 'user14', username: 'RoutineRoyalty', avatar: '/avatars/user14.png', rank: 4, score: 93, change: -3 },
      { userId: 'user15', username: 'DisciplineDiva', avatar: '/avatars/user15.png', rank: 5, score: 92, change: 0 },
      { userId: 'user16', username: 'HabitHero', avatar: '/avatars/user16.png', rank: 6, score: 90, change: 3 },
      { userId: 'user123', username: 'Alex Johnson', avatar: '/avatars/user-1.png', rank: 7, score: 88, change: 5 },
      { userId: 'user18', username: 'PersistencePro', avatar: '/avatars/user18.png', rank: 8, score: 85, change: -2 },
      { userId: 'user19', username: 'RegularRhythm', avatar: '/avatars/user19.png', rank: 9, score: 84, change: 1 },
      { userId: 'user20', username: 'SteadyStreak', avatar: '/avatars/user20.png', rank: 10, score: 83, change: -4 }
    ]
  }
];

// Reward Shop Items
export const rewardShopItems: RewardShopItem[] = [
  {
    id: 'rs1',
    title: 'Premium Workout Program',
    description: 'Unlock a premium 8-week workout program',
    category: 'content',
    price: 200,
    image: '/images/rewards/program.jpg',
    available: true
  },
  {
    id: 'rs2',
    title: 'Custom Meal Plan',
    description: 'Get a 4-week custom meal plan designed for your goals',
    category: 'content',
    price: 300,
    image: '/images/rewards/meal-plan.jpg',
    available: true
  },
  {
    id: 'rs3',
    title: 'Advanced Analytics',
    description: 'Unlock advanced analytics features for 30 days',
    category: 'feature',
    price: 150,
    image: '/images/rewards/analytics.jpg',
    available: true,
    limitedTime: false
  },
  {
    id: 'rs4',
    title: 'Personal Trainer Consultation',
    description: '30-minute video call with a certified personal trainer',
    category: 'virtual',
    price: 500,
    image: '/images/rewards/trainer.jpg',
    available: true
  },
  {
    id: 'rs5',
    title: 'Exclusive App Theme',
    description: 'Unlock a special dark gold app theme',
    category: 'feature',
    price: 100,
    image: '/images/rewards/theme.jpg',
    available: true
  },
  {
    id: 'rs6',
    title: 'Premium Profile Badge',
    description: 'Show off your dedication with a special profile badge',
    category: 'feature',
    price: 75,
    image: '/images/rewards/badge.jpg',
    available: true
  },
  {
    id: 'rs7',
    title: 'Branded Water Bottle',
    description: 'High-quality stainless steel water bottle',
    category: 'physical',
    price: 400,
    image: '/images/rewards/bottle.jpg',
    available: true,
    stock: 50
  },
  {
    id: 'rs8',
    title: 'Custom Workout Creator',
    description: 'Create and save unlimited custom workouts',
    category: 'feature',
    price: 250,
    image: '/images/rewards/creator.jpg',
    available: true,
    requiredLevel: 10
  }
];

// User Progress Stats
export const progressStats: ProgressStats = {
  workoutsThisWeek: 4,
  workoutsTotal: 87,
  currentStreak: 12,
  nutritionAdherence: 85,
  weightChange: {
    lastMonth: -1.3, // kg or lbs
    total: -3.7, // kg or lbs
  },
  strengthGains: {
    lastMonth: 5, // percentage
    total: 15 // percentage
  },
  achievements: {
    total: achievements.length,
    unlocked: achievements.filter(a => a.completed).length,
    recentlyUnlocked: achievements.filter(a => a.completed && new Date(a.completedDate!).getTime() > Date.now() - 7 * 24 * 60 * 60 * 1000)
  }
}; 
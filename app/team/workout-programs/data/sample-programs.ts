import { 
  WorkoutCategory, 
  DifficultyLevel, 
  WorkoutDuration, 
  EquipmentLevel,
  FitnessGoal,
  MuscleGroup,
  TargetAudience,
  WorkoutProgram,
  ProgressionType
} from '../types';

// Sample Men's Programs
export const menPrograms: WorkoutProgram[] = [
  {
    id: "men-strength-01",
    name: "Iron Body: Maximum Strength",
    description: "A comprehensive program designed to maximize strength gains through progressive overload and compound movements.",
    longDescription: "This program focuses on building maximal strength using the primary compound lifts: squat, bench press, deadlift, and overhead press. Following a linear progression model, you will gradually increase weight while maintaining perfect form. Ideal for men looking to significantly increase their strength foundation.",
    category: WorkoutCategory.STRENGTH,
    difficultyLevel: DifficultyLevel.INTERMEDIATE,
    equipment: EquipmentLevel.STANDARD,
    targetAudience: [TargetAudience.MEN],
    fitnessGoals: [FitnessGoal.STRENGTH, FitnessGoal.MUSCLE_GAIN],
    muscleGroups: [MuscleGroup.FULL_BODY],
    duration: { weeks: 8, daysPerWeek: 4 },
    progressionType: ProgressionType.LINEAR,
    recommendations: {
      diet: "High protein intake (1.6-2g per kg of bodyweight), caloric surplus of 200-300 calories",
      supplements: "Creatine monohydrate, protein powder, ZMA",
      recovery: "Focus on 7-8 hours of quality sleep, foam rolling, and contrast showers"
    },
    tags: ["strength", "powerlifting", "muscle building", "compound lifts"],
    featured: true,
    premium: false,
    rating: 4.8,
    reviewCount: 342
  },
  {
    id: 'men-hypertrophy-01',
    name: 'Mass Constructor',
    description: 'Designed for maximum muscle growth through high volume training and strategic nutrition planning.',
    longDescription: 'This hypertrophy-focused program utilizes training splits, volume manipulation, and various intensity techniques to stimulate maximum muscle growth. The program incorporates progressive overload principles while focusing on mind-muscle connection and optimal time under tension.',
    category: WorkoutCategory.STRENGTH,
    difficultyLevel: DifficultyLevel.INTERMEDIATE,
    equipment: EquipmentLevel.STANDARD,
    targetAudience: [TargetAudience.MEN],
    fitnessGoals: [FitnessGoal.MUSCLE_GAIN],
    muscleGroups: [MuscleGroup.FULL_BODY],
    duration: { weeks: 12, daysPerWeek: 5 },
    progressionType: ProgressionType.UNDULATING,
    recommendations: {
      diet: 'Caloric surplus of 300-500 calories, 2g of protein per kg of bodyweight, carb cycling around workouts',
      supplements: 'Creatine, whey protein, essential amino acids, pre-workout',
      recovery: 'Active recovery between training days, foam rolling, sports massage every 2 weeks'
    },
    tags: ['hypertrophy', 'muscle building', 'bodybuilding', 'mass gain'],
    featured: false,
    premium: true,
    rating: 4.7,
    reviewCount: 256
  },
  {
    id: 'men-cutting-01',
    name: 'Shred Protocol',
    description: 'A cutting program designed to maintain muscle mass while reducing body fat through intelligent training and diet strategies.',
    longDescription: 'This comprehensive fat loss program combines strength training to preserve muscle mass with strategic cardio and HIIT work. Designed specifically for men looking to reduce body fat while maintaining their hard-earned muscle.',
    category: WorkoutCategory.HIIT,
    difficultyLevel: DifficultyLevel.ADVANCED,
    equipment: EquipmentLevel.STANDARD,
    targetAudience: [TargetAudience.MEN],
    fitnessGoals: [FitnessGoal.FAT_LOSS, FitnessGoal.MUSCLE_GAIN],
    muscleGroups: [MuscleGroup.FULL_BODY],
    duration: { weeks: 8, daysPerWeek: 5 },
    progressionType: ProgressionType.CUSTOMIZED,
    recommendations: {
      diet: 'Moderate caloric deficit (300-500 calories), high protein (2.2g per kg), carb cycling',
      supplements: 'Whey protein, BCAAs, caffeine, L-carnitine',
      recovery: 'Adequate sleep, stress management, selective deload weeks'
    },
    tags: ['fat loss', 'cutting', 'definition', 'shredded'],
    featured: true,
    premium: true,
    rating: 4.6,
    reviewCount: 189
  },
  {
    id: 'men-functional-01',
    name: 'Functional Strong',
    description: 'Build practical strength and mobility with this functional fitness program designed for real-world performance.',
    longDescription: "This program focuses on developing strength that transfers to everyday activities and sports performance. Emphasizing multi-joint movements, core stability, and mobility work, you'll build a body that's not just strong in the gym but capable in all aspects of life.",
    category: WorkoutCategory.FUNCTIONAL,
    difficultyLevel: DifficultyLevel.INTERMEDIATE,
    equipment: EquipmentLevel.MINIMAL,
    targetAudience: [TargetAudience.MEN],
    fitnessGoals: [FitnessGoal.STRENGTH, FitnessGoal.ATHLETIC_PERFORMANCE, FitnessGoal.GENERAL_FITNESS],
    muscleGroups: [MuscleGroup.FULL_BODY],
    duration: { weeks: 6, daysPerWeek: 4 },
    progressionType: ProgressionType.BLOCK,
    recommendations: {
      diet: 'Clean eating approach with focus on whole foods and adequate protein',
      supplements: 'Minimal supplementation - protein and fish oil recommended',
      recovery: 'Focus on mobility work, proper warm-ups and cool-downs'
    },
    tags: ['functional fitness', 'practical strength', 'mobility', 'performance'],
    featured: false,
    premium: false,
    rating: 4.5,
    reviewCount: 178
  },
  {
    id: 'men-bodyweight-01',
    name: 'Bodyweight Mastery: Calisthenics Progression',
    description: 'Master impressive bodyweight skills and build functional strength without equipment.',
    longDescription: 'Progress from basic push-ups and pull-ups to advanced calisthenics movements like muscle-ups, front levers, and handstand push-ups. This progressive program builds extraordinary strength, body control, and impressive physique using just your bodyweight.',
    category: WorkoutCategory.STRENGTH,
    difficultyLevel: DifficultyLevel.BEGINNER,
    equipment: EquipmentLevel.NONE,
    targetAudience: [TargetAudience.MEN],
    fitnessGoals: [FitnessGoal.STRENGTH, FitnessGoal.MUSCLE_GAIN, FitnessGoal.GENERAL_FITNESS],
    muscleGroups: [MuscleGroup.FULL_BODY],
    duration: { weeks: 12, daysPerWeek: 3 },
    progressionType: ProgressionType.CUSTOMIZED,
    recommendations: {
      diet: 'Clean eating with adequate protein and carbohydrates to fuel training',
      supplements: 'Optional protein supplementation',
      recovery: 'Mobility work, adequate rest between training sessions'
    },
    tags: ['calisthenics', 'bodyweight', 'no equipment', 'skills'],
    featured: true,
    premium: false,
    rating: 4.9,
    reviewCount: 412
  }
];

// Sample Women's Programs
export const womenPrograms: WorkoutProgram[] = [
  {
    id: 'women-strength-01',
    name: 'Strong Curves',
    description: 'Build strength and sculpt an aesthetic physique with this women-focused strength training program.',
    longDescription: "Designed specifically for women, this program focuses on building overall strength with special emphasis on lower body and glute development. Using progressive overload principles, you'll build muscle in all the right places while developing functional strength.",
    category: WorkoutCategory.STRENGTH,
    difficultyLevel: DifficultyLevel.BEGINNER,
    equipment: EquipmentLevel.STANDARD,
    targetAudience: [TargetAudience.WOMEN],
    fitnessGoals: [FitnessGoal.STRENGTH, FitnessGoal.MUSCLE_GAIN],
    muscleGroups: [MuscleGroup.FULL_BODY, MuscleGroup.GLUTES, MuscleGroup.LEGS],
    duration: { weeks: 8, daysPerWeek: 3 },
    progressionType: ProgressionType.LINEAR,
    recommendations: {
      diet: 'Focus on adequate protein intake and slight caloric surplus for muscle growth',
      supplements: 'Protein powder, creatine (optional), vitamin D and calcium',
      recovery: 'Emphasis on recovery and proper sleep to maximize results'
    },
    tags: ['women', 'strength', 'glutes', 'toning'],
    featured: true,
    premium: false,
    rating: 4.8,
    reviewCount: 367
  },
  {
    id: 'women-hiit-01',
    name: 'Metabolic Accelerator',
    description: 'High-intensity interval training designed to maximize fat loss while preserving lean muscle.',
    longDescription: 'This HIIT-focused program combines short bursts of intense effort with strategic rest periods to maximize calorie burn and metabolic boost. Sessions are time-efficient but incredibly effective for fat loss while maintaining muscle tone.',
    category: WorkoutCategory.HIIT,
    difficultyLevel: DifficultyLevel.INTERMEDIATE,
    equipment: EquipmentLevel.MINIMAL,
    targetAudience: [TargetAudience.WOMEN],
    fitnessGoals: [FitnessGoal.FAT_LOSS, FitnessGoal.ENDURANCE],
    muscleGroups: [MuscleGroup.FULL_BODY],
    duration: { weeks: 6, daysPerWeek: 4 },
    progressionType: ProgressionType.CUSTOMIZED,
    recommendations: {
      diet: 'Moderate caloric deficit with adequate protein and strategic carb timing',
      supplements: 'BCAAs, caffeine (pre-workout), protein',
      recovery: 'Focus on quality sleep and stress management to control cortisol'
    },
    tags: ['HIIT', 'fat burning', 'quick workouts', 'metabolism'],
    featured: false,
    premium: true,
    rating: 4.7,
    reviewCount: 289
  },
  {
    id: 'women-tone-01',
    name: 'Sculpt & Tone',
    description: 'A full-body toning program focused on creating long, lean muscles and improving body composition.',
    longDescription: "This program focuses on moderate weight, higher rep training to stimulate muscle toning without excessive bulk. Incorporating strategic cardio and functional movement patterns, it's designed to create a lean, athletic physique.",
    category: WorkoutCategory.STRENGTH,
    difficultyLevel: DifficultyLevel.BEGINNER,
    equipment: EquipmentLevel.MINIMAL,
    targetAudience: [TargetAudience.WOMEN],
    fitnessGoals: [FitnessGoal.MUSCLE_GAIN, FitnessGoal.FAT_LOSS, FitnessGoal.GENERAL_FITNESS],
    muscleGroups: [MuscleGroup.FULL_BODY],
    duration: { weeks: 8, daysPerWeek: 4 },
    progressionType: ProgressionType.LINEAR,
    recommendations: {
      diet: 'Balanced nutrition with slight deficit or maintenance calories depending on goals',
      supplements: 'Protein powder, multivitamin, omega-3s',
      recovery: 'Stretching routines, adequate hydration, quality sleep'
    },
    tags: ['toning', 'sculpting', 'lean muscle', 'definition'],
    featured: true,
    premium: false,
    rating: 4.6,
    reviewCount: 312
  },
  {
    id: 'women-yoga-01',
    name: 'Power Yoga Flow',
    description: 'Build strength, flexibility and mindfulness with this progressive yoga program.',
    longDescription: 'This power yoga program builds both strength and flexibility while promoting mindfulness and stress reduction. Progress from basic poses to challenging sequences that build functional strength, improve mobility, and enhance mind-body connection.',
    category: WorkoutCategory.YOGA,
    difficultyLevel: DifficultyLevel.BEGINNER,
    equipment: EquipmentLevel.NONE,
    targetAudience: [TargetAudience.WOMEN],
    fitnessGoals: [FitnessGoal.FLEXIBILITY, FitnessGoal.STRENGTH, FitnessGoal.GENERAL_FITNESS],
    muscleGroups: [MuscleGroup.FULL_BODY, MuscleGroup.CORE],
    duration: { weeks: 6, daysPerWeek: 5 },
    progressionType: ProgressionType.CUSTOMIZED,
    recommendations: {
      diet: 'Plant-forward nutrition with adequate protein, focus on whole foods',
      supplements: 'Optional vitamin B12, D3, and omega-3 supplements',
      recovery: 'Meditation, proper hydration, adequate sleep'
    },
    tags: ['yoga', 'flexibility', 'mindfulness', 'strength'],
    featured: false,
    premium: true,
    rating: 4.9,
    reviewCount: 234
  },
  {
    id: 'women-prenatal-01',
    name: 'Prenatal Fitness Plan',
    description: 'Safe and effective workouts specifically designed for expecting mothers.',
    longDescription: 'This program provides safe, effective workouts for all three trimesters of pregnancy. Focusing on maintaining fitness, preventing common pregnancy discomforts, and preparing the body for labor and recovery, these workouts adjust as your pregnancy progresses.',
    category: WorkoutCategory.FUNCTIONAL,
    difficultyLevel: DifficultyLevel.BEGINNER,
    equipment: EquipmentLevel.MINIMAL,
    targetAudience: [TargetAudience.WOMEN, TargetAudience.PRENATAL],
    fitnessGoals: [FitnessGoal.GENERAL_FITNESS, FitnessGoal.STRENGTH],
    muscleGroups: [MuscleGroup.FULL_BODY, MuscleGroup.CORE],
    duration: { weeks: 12, daysPerWeek: 3 },
    progressionType: ProgressionType.CUSTOMIZED,
    recommendations: {
      diet: 'Follow prenatal nutrition guidelines, focus on nutrient density',
      supplements: 'Only those recommended by healthcare provider',
      recovery: 'Adequate rest, proper hydration, gentle stretching'
    },
    tags: ['prenatal', 'pregnancy', 'safe exercise', 'maternity'],
    featured: false,
    premium: true,
    rating: 4.9,
    reviewCount: 156
  }
];

// Specialty Programs
export const specialtyPrograms: WorkoutProgram[] = [
  {
    id: 'specialty-senior-01',
    name: 'Vitality After 60',
    description: 'Maintain strength, mobility and independence with this program designed for seniors.',
    longDescription: 'This program focuses on maintaining functional strength, joint health, and mobility for adults over 60. With emphasis on fall prevention, everyday movement patterns, and gentle progression, this program helps maintain independence and quality of life.',
    category: WorkoutCategory.FUNCTIONAL,
    difficultyLevel: DifficultyLevel.BEGINNER,
    equipment: EquipmentLevel.MINIMAL,
    targetAudience: [TargetAudience.SENIORS, TargetAudience.BOTH],
    fitnessGoals: [FitnessGoal.GENERAL_FITNESS, FitnessGoal.STRENGTH, FitnessGoal.BALANCE],
    muscleGroups: [MuscleGroup.FULL_BODY],
    duration: { weeks: 8, daysPerWeek: 3 },
    progressionType: ProgressionType.LINEAR,
    recommendations: {
      diet: 'Focus on protein intake, calcium, and overall nutrient density',
      supplements: 'Vitamin D, calcium, fish oil as recommended by physician',
      recovery: 'Adequate rest between sessions, gentle stretching'
    },
    tags: ['seniors', 'functional fitness', 'mobility', 'balance'],
    featured: false,
    premium: false,
    rating: 4.8,
    reviewCount: 124
  },
  {
    id: 'specialty-rehab-01',
    name: 'Lower Back Rehabilitation',
    description: 'Safely recover from lower back pain while building core strength and stability.',
    longDescription: 'Developed in consultation with physical therapists, this program helps those with chronic or acute lower back pain return to pain-free movement. Progressive exercises focus on core stability, proper movement patterns, and gradual strength building.',
    category: WorkoutCategory.REHABILITATION,
    difficultyLevel: DifficultyLevel.BEGINNER,
    equipment: EquipmentLevel.MINIMAL,
    targetAudience: [TargetAudience.BOTH],
    fitnessGoals: [FitnessGoal.REHABILITATION, FitnessGoal.POSTURE],
    muscleGroups: [MuscleGroup.CORE, MuscleGroup.BACK],
    duration: { weeks: 8, daysPerWeek: 4 },
    progressionType: ProgressionType.CUSTOMIZED,
    recommendations: {
      diet: 'Anti-inflammatory foods, adequate protein for tissue repair',
      supplements: 'Omega-3s, turmeric/curcumin, collagen (consult physician)',
      recovery: 'Proper sleep position, gentle stretching, heat/cold therapy as needed'
    },
    tags: ['rehabilitation', 'back pain', 'core strength', 'recovery'],
    featured: false,
    premium: true,
    rating: 4.9,
    reviewCount: 187
  },
  {
    id: 'specialty-athlete-01',
    name: 'Explosive Power: Athletic Performance',
    description: 'Develop sport-specific power, agility and performance with this advanced athletic program.',
    longDescription: 'Designed for competitive athletes, this program focuses on developing explosive power, speed, agility, and sport-specific performance. Using advanced training methods including plyometrics, Olympic lifting variations, and specialized drills.',
    category: WorkoutCategory.SPORT_SPECIFIC,
    difficultyLevel: DifficultyLevel.ADVANCED,
    equipment: EquipmentLevel.STANDARD,
    targetAudience: [TargetAudience.BOTH],
    fitnessGoals: [FitnessGoal.ATHLETIC_PERFORMANCE, FitnessGoal.STRENGTH, FitnessGoal.ENDURANCE],
    muscleGroups: [MuscleGroup.FULL_BODY],
    duration: { weeks: 8, daysPerWeek: 5 },
    progressionType: ProgressionType.BLOCK,
    recommendations: {
      diet: 'Periodized nutrition approach, carb timing around training',
      supplements: 'Creatine, beta-alanine, protein, electrolytes',
      recovery: 'Contrast therapy, foam rolling, adequate sleep, strategic deloads'
    },
    tags: ['athlete', 'sports', 'explosive', 'performance'],
    featured: true,
    premium: true,
    rating: 4.7,
    reviewCount: 142
  },
  {
    id: 'specialty-desk-01',
    name: 'Desk Worker Relief',
    description: 'Combat the negative effects of sitting with targeted mobility and strength exercises.',
    longDescription: 'This program counteracts the detrimental effects of prolonged sitting and computer work. Focusing on posture correction, targeted stretching, and strengthening of commonly weak muscles, this program helps relieve pain and prevent future issues.',
    category: WorkoutCategory.FUNCTIONAL,
    difficultyLevel: DifficultyLevel.BEGINNER,
    equipment: EquipmentLevel.NONE,
    targetAudience: [TargetAudience.BOTH],
    fitnessGoals: [FitnessGoal.POSTURE, FitnessGoal.REHABILITATION, FitnessGoal.GENERAL_FITNESS],
    muscleGroups: [MuscleGroup.BACK, MuscleGroup.SHOULDERS, MuscleGroup.CORE],
    duration: { weeks: 6, daysPerWeek: 5 },
    progressionType: ProgressionType.LINEAR,
    recommendations: {
      diet: 'Anti-inflammatory diet, adequate hydration',
      supplements: 'Optional fish oil, magnesium, vitamin D',
      recovery: 'Frequent movement breaks, ergonomic workspace setup'
    },
    tags: ['desk job', 'posture', 'mobility', 'pain relief'],
    featured: false,
    premium: false,
    rating: 4.8,
    reviewCount: 276
  },
  {
    id: 'specialty-time-constraint-01',
    name: '20-Minute Maximizer',
    description: 'Efficient, time-saving workouts that deliver maximum results in minimal time.',
    longDescription: 'Designed for busy professionals, this program delivers effective workouts in just 20 minutes per session. Using science-based methods like high-intensity intervals, density training, and compound movements to maximize efficiency.',
    category: WorkoutCategory.HIIT,
    difficultyLevel: DifficultyLevel.INTERMEDIATE,
    equipment: EquipmentLevel.MINIMAL,
    targetAudience: [TargetAudience.BOTH],
    fitnessGoals: [FitnessGoal.GENERAL_FITNESS, FitnessGoal.FAT_LOSS, FitnessGoal.STRENGTH],
    muscleGroups: [MuscleGroup.FULL_BODY],
    duration: { weeks: 4, daysPerWeek: 4 },
    progressionType: ProgressionType.CUSTOMIZED,
    recommendations: {
      diet: 'Focus on meal preparation and planning for convenience',
      supplements: 'Optional pre-workout, protein powder for convenience',
      recovery: 'Quality over quantity for sleep, stress management techniques'
    },
    tags: ['quick workouts', 'time-saving', 'efficient', 'busy schedule'],
    featured: true,
    premium: false,
    rating: 4.6,
    reviewCount: 319
  }
];

// All programs combined
export const allPrograms: WorkoutProgram[] = [
  ...menPrograms,
  ...womenPrograms,
  ...specialtyPrograms
]; 
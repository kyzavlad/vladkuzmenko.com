'use client';

import { useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { 
  Search, 
  Filter, 
  ArrowLeft, 
  Dumbbell, 
  ChevronDown, 
  ChevronRight,
  Info,
  Play,
  Heart,
  StarIcon,
  List,
  Grid3x3 
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger 
} from '@/components/ui/dropdown-menu';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

// Mock data for exercise categories
const exerciseCategories = [
  { id: 'all', name: 'All Exercises' },
  { id: 'strength', name: 'Strength' },
  { id: 'cardio', name: 'Cardio' },
  { id: 'flexibility', name: 'Flexibility' },
  { id: 'calisthenics', name: 'Calisthenics' },
  { id: 'plyometrics', name: 'Plyometrics' },
  { id: 'recovery', name: 'Recovery & Mobility' },
];

// Mock data for muscle groups
const muscleGroups = [
  { id: 'chest', name: 'Chest' },
  { id: 'back', name: 'Back' },
  { id: 'shoulders', name: 'Shoulders' },
  { id: 'arms', name: 'Arms' },
  { id: 'core', name: 'Core' },
  { id: 'legs', name: 'Legs' },
  { id: 'glutes', name: 'Glutes' },
  { id: 'full-body', name: 'Full Body' },
];

// Mock data for equipment types
const equipmentTypes = [
  { id: 'none', name: 'No Equipment' },
  { id: 'dumbbells', name: 'Dumbbells' },
  { id: 'barbell', name: 'Barbell' },
  { id: 'kettlebell', name: 'Kettlebell' },
  { id: 'resistance-bands', name: 'Resistance Bands' },
  { id: 'cable-machine', name: 'Cable Machine' },
  { id: 'smith-machine', name: 'Smith Machine' },
  { id: 'bodyweight', name: 'Bodyweight' },
];

// Mock exercises data
const exercises = [
  {
    id: 'ex1',
    name: 'Barbell Squat',
    category: 'strength',
    muscleGroups: ['legs', 'glutes'],
    equipment: ['barbell'],
    difficulty: 'intermediate',
    description: 'A compound exercise that targets multiple lower body muscles with a focus on quads, hamstrings and glutes.',
    instructions: [
      'Stand with feet shoulder-width apart, barbell across upper back',
      'Bend knees and drop hips back and down',
      'Descend until thighs are parallel to ground',
      'Drive through heels to return to standing position'
    ],
    tips: 'Keep your chest up and core engaged throughout the movement.',
    videoUrl: '/exercises/squat.mp4',
    imageUrl: '/exercises/squat.jpg',
    isFavorite: true,
    variations: ['Front Squat', 'Box Squat', 'Goblet Squat']
  },
  {
    id: 'ex2',
    name: 'Bench Press',
    category: 'strength',
    muscleGroups: ['chest', 'shoulders', 'arms'],
    equipment: ['barbell', 'bench'],
    difficulty: 'intermediate',
    description: 'A compound upper body exercise that targets the chest, shoulders, and triceps.',
    instructions: [
      'Lie on bench with feet flat on floor',
      'Grip barbell slightly wider than shoulder width',
      'Lower bar to mid-chest level',
      'Press bar back up to starting position'
    ],
    tips: 'Keep your wrists straight and elbows at a 45-degree angle from your body.',
    videoUrl: '/exercises/bench-press.mp4',
    imageUrl: '/exercises/bench-press.jpg',
    isFavorite: false,
    variations: ['Incline Bench Press', 'Decline Bench Press', 'Close-Grip Bench Press']
  },
  {
    id: 'ex3',
    name: 'Deadlift',
    category: 'strength',
    muscleGroups: ['back', 'legs', 'glutes'],
    equipment: ['barbell'],
    difficulty: 'advanced',
    description: 'A compound full-body exercise that primarily targets the posterior chain.',
    instructions: [
      'Stand with feet hip-width apart, barbell over mid-foot',
      'Bend at hips and knees to grip bar with hands shoulder-width apart',
      'Keeping back flat, drive through heels to stand up straight',
      'Return weight to floor by hinging at hips and bending knees'
    ],
    tips: 'Keep the bar close to your body throughout the movement and maintain a neutral spine.',
    videoUrl: '/exercises/deadlift.mp4',
    imageUrl: '/exercises/deadlift.jpg',
    isFavorite: true,
    variations: ['Romanian Deadlift', 'Sumo Deadlift', 'Trap Bar Deadlift']
  },
  {
    id: 'ex4',
    name: 'Pull-up',
    category: 'strength',
    muscleGroups: ['back', 'arms'],
    equipment: ['bodyweight', 'pull-up-bar'],
    difficulty: 'intermediate',
    description: 'A bodyweight exercise that targets the upper body, especially the lats and biceps.',
    instructions: [
      'Hang from pull-up bar with hands slightly wider than shoulder-width',
      'Pull body up until chin is over the bar',
      'Lower body back to starting position with control',
      'Repeat while maintaining proper form'
    ],
    tips: 'Focus on pulling with your back muscles, not just your arms.',
    videoUrl: '/exercises/pull-up.mp4',
    imageUrl: '/exercises/pull-up.jpg',
    isFavorite: false,
    variations: ['Chin-up', 'Wide-grip Pull-up', 'Commando Pull-up']
  },
  {
    id: 'ex5',
    name: 'Push-up',
    category: 'strength',
    muscleGroups: ['chest', 'shoulders', 'arms', 'core'],
    equipment: ['bodyweight'],
    difficulty: 'beginner',
    description: 'A classic bodyweight exercise that targets the chest, triceps, and shoulders.',
    instructions: [
      'Start in plank position with hands slightly wider than shoulder-width',
      'Lower body until chest nearly touches the floor',
      'Keep elbows at a 45-degree angle from body',
      'Push back up to starting position'
    ],
    tips: 'Keep your core tight and body in a straight line from head to heels.',
    videoUrl: '/exercises/push-up.mp4',
    imageUrl: '/exercises/push-up.jpg',
    isFavorite: false,
    variations: ['Incline Push-up', 'Decline Push-up', 'Diamond Push-up']
  },
  {
    id: 'ex6',
    name: 'Overhead Press',
    category: 'strength',
    muscleGroups: ['shoulders', 'arms'],
    equipment: ['barbell', 'dumbbells'],
    difficulty: 'intermediate',
    description: 'A compound exercise targeting the shoulders, upper chest, and triceps.',
    instructions: [
      'Stand with feet shoulder-width apart',
      'Hold barbell at shoulder height with palms facing forward',
      'Press weight overhead until arms are fully extended',
      'Lower weight back to shoulder level with control'
    ],
    tips: 'Keep your core tight and avoid arching your lower back.',
    videoUrl: '/exercises/overhead-press.mp4',
    imageUrl: '/exercises/overhead-press.jpg',
    isFavorite: true,
    variations: ['Seated Overhead Press', 'Push Press', 'Arnold Press']
  },
  {
    id: 'ex7',
    name: 'Lunge',
    category: 'strength',
    muscleGroups: ['legs', 'glutes'],
    equipment: ['bodyweight', 'dumbbells', 'barbell'],
    difficulty: 'beginner',
    description: 'A unilateral exercise that targets the quadriceps, hamstrings, and glutes.',
    instructions: [
      'Stand with feet hip-width apart',
      'Step forward with one leg and lower body until both knees are at 90-degree angles',
      'Push through front heel to return to standing position',
      'Repeat on opposite leg'
    ],
    tips: 'Keep your torso upright and make sure your front knee stays aligned with your ankle.',
    videoUrl: '/exercises/lunge.mp4',
    imageUrl: '/exercises/lunge.jpg',
    isFavorite: false,
    variations: ['Walking Lunge', 'Reverse Lunge', 'Lateral Lunge']
  },
  {
    id: 'ex8',
    name: 'Russian Twist',
    category: 'strength',
    muscleGroups: ['core'],
    equipment: ['bodyweight', 'medicine-ball', 'weight-plate'],
    difficulty: 'beginner',
    description: 'A rotational exercise that targets the obliques and other core muscles.',
    instructions: [
      'Sit on floor with knees bent and feet elevated',
      'Lean back slightly, keeping back straight',
      'Clasp hands together or hold weight in front of chest',
      'Rotate torso to right, then to left to complete one rep'
    ],
    tips: 'Focus on rotating from your core, not just moving your arms.',
    videoUrl: '/exercises/russian-twist.mp4',
    imageUrl: '/exercises/russian-twist.jpg',
    isFavorite: false,
    variations: ['Weighted Russian Twist', 'Medicine Ball Russian Twist', 'Cable Russian Twist']
  },
  {
    id: 'ex9',
    name: 'Plank',
    category: 'strength',
    muscleGroups: ['core', 'shoulders'],
    equipment: ['bodyweight'],
    difficulty: 'beginner',
    description: 'An isometric core exercise that also engages the shoulders and back.',
    instructions: [
      'Start in push-up position but with weight on forearms',
      'Keep body in straight line from head to heels',
      'Engage core and glutes',
      'Hold position for designated time'
    ],
    tips: "Don't let your hips sag or lift too high. Focus on maintaining a neutral spine.",
    videoUrl: '/exercises/plank.mp4',
    imageUrl: '/exercises/plank.jpg',
    isFavorite: false,
    variations: ['Side Plank', 'Plank with Shoulder Tap', 'Plank Jack']
  },
  {
    id: 'ex10',
    name: 'Romanian Deadlift',
    category: 'strength',
    muscleGroups: ['legs', 'glutes', 'back'],
    equipment: ['barbell', 'dumbbells'],
    difficulty: 'intermediate',
    description: 'A hip-hinge movement that targets the hamstrings, glutes, and lower back.',
    instructions: [
      'Stand with feet hip-width apart, holding weight in front of thighs',
      'Keeping back straight and knees slightly bent, hinge at hips',
      'Lower weight along front of legs until you feel stretch in hamstrings',
      'Drive hips forward to return to standing position'
    ],
    tips: 'Focus on the hip hinge movement and keep the weight close to your body.',
    videoUrl: '/exercises/romanian-deadlift.mp4',
    imageUrl: '/exercises/romanian-deadlift.jpg',
    isFavorite: true,
    variations: ['Single-Leg Romanian Deadlift', 'Dumbbell Romanian Deadlift', 'Kettlebell Romanian Deadlift']
  },
];

export default function ExerciseLibrary() {
  const router = useRouter();
  const searchParams = useSearchParams();
  
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedMuscleGroup, setSelectedMuscleGroup] = useState('');
  const [selectedEquipment, setSelectedEquipment] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [filteredExercises, setFilteredExercises] = useState(exercises);
  
  // Apply filters
  useEffect(() => {
    let results = [...exercises];
    
    // Apply search term filter
    if (searchTerm) {
      results = results.filter(ex => 
        ex.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        ex.description.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    // Apply category filter
    if (selectedCategory && selectedCategory !== 'all') {
      results = results.filter(ex => ex.category === selectedCategory);
    }
    
    // Apply muscle group filter
    if (selectedMuscleGroup) {
      results = results.filter(ex => ex.muscleGroups.includes(selectedMuscleGroup));
    }
    
    // Apply equipment filter
    if (selectedEquipment) {
      results = results.filter(ex => ex.equipment.includes(selectedEquipment));
    }
    
    setFilteredExercises(results);
  }, [searchTerm, selectedCategory, selectedMuscleGroup, selectedEquipment]);
  
  // Toggle favorite status
  const toggleFavorite = (id: string) => {
    setFilteredExercises(prev => 
      prev.map(ex => 
        ex.id === id ? { ...ex, isFavorite: !ex.isFavorite } : ex
      )
    );
  };
  
  return (
    <div className="exercise-library-page">
      {/* Header */}
      <div className="mb-6 flex justify-between items-center">
        <div className="flex items-center">
          <Button
            variant="ghost"
            size="icon"
            className="mr-2"
            asChild
          >
            <Link href="/team/workout-programs">
              <ArrowLeft className="h-5 w-5" />
            </Link>
          </Button>
          <h1 className="text-2xl font-bold text-white">Exercise Library</h1>
        </div>
        
        <div className="flex items-center">
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={() => setViewMode('grid')}
            className={viewMode === 'grid' ? 'text-blue-400' : 'text-gray-400'}
          >
            <Grid3x3 className="h-5 w-5" />
          </Button>
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={() => setViewMode('list')}
            className={viewMode === 'list' ? 'text-blue-400' : 'text-gray-400'}
          >
            <List className="h-5 w-5" />
          </Button>
        </div>
      </div>
      
      {/* Search and Filters */}
      <div className="mb-6 space-y-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <Input 
            type="text" 
            placeholder="Search exercises..." 
            className="pl-10 bg-gray-800 border-gray-700"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        
        <div className="flex flex-wrap gap-2">
          <Tabs 
            defaultValue="all" 
            value={selectedCategory}
            onValueChange={setSelectedCategory}
            className="w-full"
          >
            <TabsList className="bg-gray-800 h-auto flex flex-wrap p-1">
              {exerciseCategories.map(category => (
                <TabsTrigger 
                  key={category.id} 
                  value={category.id}
                  className="flex-grow basis-[calc(25%-0.5rem)] sm:basis-auto py-1 px-3"
                >
                  {category.name}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
          
          <div className="flex flex-wrap gap-2 w-full">
            <Select value={selectedMuscleGroup} onValueChange={setSelectedMuscleGroup}>
              <SelectTrigger className="w-full sm:w-[200px] bg-gray-800 border-gray-700">
                <SelectValue placeholder="Muscle Group" />
              </SelectTrigger>
              <SelectContent className="bg-gray-800 border-gray-700">
                <SelectItem value="">All Muscle Groups</SelectItem>
                {muscleGroups.map(group => (
                  <SelectItem key={group.id} value={group.id}>{group.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            <Select value={selectedEquipment} onValueChange={setSelectedEquipment}>
              <SelectTrigger className="w-full sm:w-[200px] bg-gray-800 border-gray-700">
                <SelectValue placeholder="Equipment" />
              </SelectTrigger>
              <SelectContent className="bg-gray-800 border-gray-700">
                <SelectItem value="">All Equipment</SelectItem>
                {equipmentTypes.map(equipment => (
                  <SelectItem key={equipment.id} value={equipment.id}>{equipment.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            {(searchTerm || selectedMuscleGroup || selectedEquipment) && (
              <Button 
                variant="outline" 
                onClick={() => {
                  setSearchTerm('');
                  setSelectedMuscleGroup('');
                  setSelectedEquipment('');
                }}
                className="ml-auto"
              >
                Clear Filters
              </Button>
            )}
          </div>
        </div>
      </div>
      
      {/* Exercises Display */}
      {filteredExercises.length > 0 ? (
        viewMode === 'grid' ? (
          // Grid View
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {filteredExercises.map(exercise => (
              <Card 
                key={exercise.id} 
                className="bg-gray-800 border-gray-700 h-full flex flex-col"
              >
                <div className="relative aspect-video bg-gray-700 overflow-hidden">
                  {/* This would be an actual image in a real implementation */}
                  <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-gray-700 to-gray-900">
                    <Dumbbell className="h-8 w-8 text-gray-500" />
                  </div>
                  
                  <Button 
                    variant="ghost" 
                    size="icon" 
                    className="absolute top-2 right-2 text-gray-400 hover:text-red-400 bg-gray-800 bg-opacity-50"
                    onClick={() => toggleFavorite(exercise.id)}
                  >
                    <Heart className={`h-4 w-4 ${exercise.isFavorite ? 'text-red-500 fill-red-500' : ''}`} />
                  </Button>
                </div>
                
                <CardContent className="flex-grow p-4">
                  <h3 className="text-lg font-medium text-white mb-1">{exercise.name}</h3>
                  
                  <div className="flex flex-wrap gap-1 mb-3">
                    {exercise.muscleGroups.map((muscle, i) => (
                      <Badge key={i} variant="secondary" className="bg-gray-700 text-xs">
                        {muscleGroups.find(g => g.id === muscle)?.name}
                      </Badge>
                    ))}
                  </div>
                  
                  <p className="text-gray-400 text-sm line-clamp-2 mb-4">
                    {exercise.description}
                  </p>
                  
                  <div className="mt-auto flex justify-between items-center">
                    <Badge variant="outline" className="text-xs capitalize">
                      {exercise.difficulty}
                    </Badge>
                    <Button variant="ghost" className="text-blue-400 p-0 h-auto" asChild>
                      <Link href={`/team/workout-programs/exercises/${exercise.id}`}>
                        Details <ChevronRight className="h-4 w-4 ml-1" />
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
            {filteredExercises.map(exercise => (
              <div 
                key={exercise.id} 
                className="flex items-center p-3 rounded-md bg-gray-800 border border-gray-700"
              >
                <div className="h-12 w-12 rounded bg-gray-700 flex items-center justify-center mr-4">
                  <Dumbbell className="h-6 w-6 text-gray-500" />
                </div>
                
                <div className="flex-grow">
                  <h3 className="text-white font-medium">{exercise.name}</h3>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {exercise.muscleGroups.slice(0, 2).map((muscle, i) => (
                      <Badge key={i} variant="secondary" className="bg-gray-700 text-xs">
                        {muscleGroups.find(g => g.id === muscle)?.name}
                      </Badge>
                    ))}
                    {exercise.muscleGroups.length > 2 && (
                      <Badge variant="secondary" className="bg-gray-700 text-xs">
                        +{exercise.muscleGroups.length - 2} more
                      </Badge>
                    )}
                  </div>
                </div>
                
                <div className="flex items-center gap-2">
                  <Button 
                    variant="ghost" 
                    size="icon" 
                    className="text-gray-400 hover:text-red-400"
                    onClick={() => toggleFavorite(exercise.id)}
                  >
                    <Heart className={`h-4 w-4 ${exercise.isFavorite ? 'text-red-500 fill-red-500' : ''}`} />
                  </Button>
                  
                  <Button variant="ghost" size="icon" className="text-blue-400" asChild>
                    <Link href={`/team/workout-programs/exercises/${exercise.id}`}>
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
            <Filter className="h-6 w-6 text-gray-400" />
          </div>
          <h3 className="text-lg font-medium text-white mb-2">No exercises found</h3>
          <p className="text-gray-400 mb-4">Try adjusting your filters or search term</p>
          <Button 
            variant="outline" 
            onClick={() => {
              setSearchTerm('');
              setSelectedCategory('all');
              setSelectedMuscleGroup('');
              setSelectedEquipment('');
            }}
          >
            Reset Filters
          </Button>
        </div>
      )}
    </div>
  );
} 
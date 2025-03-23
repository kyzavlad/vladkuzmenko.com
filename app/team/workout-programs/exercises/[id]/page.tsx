'use client';

import { useState, useEffect } from 'react';
import { useParams, notFound } from 'next/navigation';
import Link from 'next/link';
import { 
  ArrowLeft, 
  Dumbbell, 
  ChevronRight, 
  Heart, 
  Play, 
  Plus, 
  Share2, 
  CheckCircle,
  BarChart,
  ListChecks,
  Info,
  Clock,
  Award,
  SkipForward
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger 
} from '@/components/ui/dropdown-menu';

// Mock data - in a real app, this would come from an API
const exercises = [
  {
    id: 'ex1',
    name: 'Barbell Squat',
    category: 'strength',
    muscleGroups: [
      { id: 'legs', name: 'Legs' },
      { id: 'glutes', name: 'Glutes' },
    ],
    equipment: [
      { id: 'barbell', name: 'Barbell' }
    ],
    difficulty: 'intermediate',
    description: 'A compound exercise that targets multiple lower body muscles with a focus on quads, hamstrings and glutes.',
    instructions: [
      'Stand with feet shoulder-width apart, barbell across upper back',
      'Bend knees and drop hips back and down',
      'Descend until thighs are parallel to ground',
      'Drive through heels to return to standing position'
    ],
    tips: [
      'Keep your chest up and core engaged throughout the movement',
      'Ensure knees track in line with toes',
      'Avoid rounding your lower back',
      'Use a spotter or safety bars when lifting heavy'
    ],
    videoUrl: '/exercises/squat.mp4',
    imageUrl: '/exercises/squat.jpg',
    isFavorite: true,
    variations: [
      { id: 'front-squat', name: 'Front Squat' },
      { id: 'box-squat', name: 'Box Squat' },
      { id: 'goblet-squat', name: 'Goblet Squat' },
      { id: 'bulgarian-split-squat', name: 'Bulgarian Split Squat' }
    ],
    muscles: {
      primary: ['Quadriceps', 'Glutes'],
      secondary: ['Hamstrings', 'Lower Back', 'Core']
    },
    personalRecords: [
      { weight: '225 lbs', reps: 10, date: '2023-06-15' },
      { weight: '275 lbs', reps: 5, date: '2023-06-25' },
      { weight: '315 lbs', reps: 1, date: '2023-07-05' }
    ],
    recentWorkouts: [
      { date: '2023-07-10', weight: '225 lbs', sets: 4, reps: '10, 8, 8, 6' },
      { date: '2023-07-03', weight: '245 lbs', sets: 4, reps: '8, 6, 6, 5' },
      { date: '2023-06-26', weight: '225 lbs', sets: 3, reps: '10, 8, 8' }
    ]
  },
  {
    id: 'ex2',
    name: 'Bench Press',
    category: 'strength',
    muscleGroups: [
      { id: 'chest', name: 'Chest' },
      { id: 'shoulders', name: 'Shoulders' },
      { id: 'arms', name: 'Arms' }
    ],
    equipment: [
      { id: 'barbell', name: 'Barbell' },
      { id: 'bench', name: 'Bench' }
    ],
    difficulty: 'intermediate',
    description: 'A compound upper body exercise that targets the chest, shoulders, and triceps.',
    instructions: [
      'Lie on bench with feet flat on floor',
      'Grip barbell slightly wider than shoulder width',
      'Lower bar to mid-chest level',
      'Press bar back up to starting position'
    ],
    tips: [
      'Keep your wrists straight and elbows at a 45-degree angle from your body',
      'Drive your feet into the ground for stability',
      'Maintain a slight arch in your lower back',
      'Use a spotter when lifting heavy weights'
    ],
    videoUrl: '/exercises/bench-press.mp4',
    imageUrl: '/exercises/bench-press.jpg',
    isFavorite: false,
    variations: [
      { id: 'incline-bench-press', name: 'Incline Bench Press' },
      { id: 'decline-bench-press', name: 'Decline Bench Press' },
      { id: 'close-grip-bench-press', name: 'Close-Grip Bench Press' },
      { id: 'dumbbell-bench-press', name: 'Dumbbell Bench Press' }
    ],
    muscles: {
      primary: ['Pectorals', 'Triceps'],
      secondary: ['Anterior Deltoids', 'Serratus Anterior']
    },
    personalRecords: [
      { weight: '185 lbs', reps: 10, date: '2023-06-12' },
      { weight: '205 lbs', reps: 5, date: '2023-06-22' },
      { weight: '225 lbs', reps: 2, date: '2023-07-02' }
    ],
    recentWorkouts: [
      { date: '2023-07-12', weight: '185 lbs', sets: 4, reps: '10, 8, 8, 6' },
      { date: '2023-07-05', weight: '195 lbs', sets: 4, reps: '8, 6, 6, 5' },
      { date: '2023-06-28', weight: '185 lbs', sets: 3, reps: '10, 8, 8' }
    ]
  },
  {
    id: 'ex3',
    name: 'Deadlift',
    category: 'strength',
    muscleGroups: [
      { id: 'back', name: 'Back' },
      { id: 'legs', name: 'Legs' },
      { id: 'glutes', name: 'Glutes' }
    ],
    equipment: [
      { id: 'barbell', name: 'Barbell' }
    ],
    difficulty: 'advanced',
    description: 'A compound full-body exercise that primarily targets the posterior chain.',
    instructions: [
      'Stand with feet hip-width apart, barbell over mid-foot',
      'Bend at hips and knees to grip bar with hands shoulder-width apart',
      'Keeping back flat, drive through heels to stand up straight',
      'Return weight to floor by hinging at hips and bending knees'
    ],
    tips: [
      'Keep the bar close to your body throughout the movement',
      'Maintain a neutral spine from start to finish',
      'Engage your lats before initiating the pull',
      'Think about pushing the ground away rather than pulling the weight'
    ],
    videoUrl: '/exercises/deadlift.mp4',
    imageUrl: '/exercises/deadlift.jpg',
    isFavorite: true,
    variations: [
      { id: 'romanian-deadlift', name: 'Romanian Deadlift' },
      { id: 'sumo-deadlift', name: 'Sumo Deadlift' },
      { id: 'trap-bar-deadlift', name: 'Trap Bar Deadlift' },
      { id: 'single-leg-deadlift', name: 'Single-Leg Deadlift' }
    ],
    muscles: {
      primary: ['Hamstrings', 'Glutes', 'Lower Back'],
      secondary: ['Quadriceps', 'Traps', 'Forearms', 'Core']
    },
    personalRecords: [
      { weight: '275 lbs', reps: 8, date: '2023-06-14' },
      { weight: '315 lbs', reps: 5, date: '2023-06-24' },
      { weight: '365 lbs', reps: 1, date: '2023-07-04' }
    ],
    recentWorkouts: [
      { date: '2023-07-11', weight: '275 lbs', sets: 4, reps: '8, 6, 6, 5' },
      { date: '2023-07-04', weight: '315 lbs', sets: 3, reps: '5, 3, 3' },
      { date: '2023-06-27', weight: '275 lbs', sets: 3, reps: '8, 6, 6' }
    ]
  }
];

// Mock workout programs for the dropdown
const workoutPrograms = [
  { id: 'prog1', name: 'Iron Body: Maximum Strength' },
  { id: 'prog2', name: 'Hypertrophy Focus' },
  { id: 'prog3', name: 'Full Body Functional' },
  { id: 'prog4', name: 'Custom Workout' }
];

export default function ExerciseDetail() {
  const params = useParams();
  const [exercise, setExercise] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('instructions');
  const [isFavorite, setIsFavorite] = useState(false);
  
  useEffect(() => {
    // Simulate fetching exercise data
    const foundExercise = exercises.find(ex => ex.id === params.id);
    if (foundExercise) {
      setExercise(foundExercise);
      setIsFavorite(foundExercise.isFavorite);
    }
    setLoading(false);
  }, [params.id]);
  
  // If exercise not found
  if (!loading && !exercise) {
    return notFound();
  }
  
  // Loading state
  if (loading || !exercise) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-pulse flex flex-col items-center">
          <div className="h-10 w-48 bg-gray-700 rounded mb-6"></div>
          <div className="h-72 w-full max-w-3xl bg-gray-800 rounded-lg mb-6"></div>
          <div className="space-y-3 w-full max-w-3xl">
            <div className="h-4 bg-gray-700 rounded w-3/4"></div>
            <div className="h-4 bg-gray-700 rounded w-full"></div>
            <div className="h-4 bg-gray-700 rounded w-5/6"></div>
          </div>
        </div>
      </div>
    );
  }
  
  // Toggle favorite state
  const toggleFavorite = () => {
    setIsFavorite(!isFavorite);
  };
  
  return (
    <div className="exercise-detail-page pb-16">
      {/* Header Navigation */}
      <div className="mb-4 flex items-center">
        <Button
          variant="ghost"
          size="icon"
          className="mr-2"
          asChild
        >
          <Link href="/team/workout-programs/exercises">
            <ArrowLeft className="h-5 w-5" />
          </Link>
        </Button>
        <span className="text-sm text-gray-400">
          <Link href="/team/workout-programs/exercises" className="hover:text-blue-400">Exercises</Link>
          <ChevronRight className="h-4 w-4 inline mx-1" />
          <span>{exercise.name}</span>
        </span>
      </div>
      
      {/* Exercise Header */}
      <div className="mb-6">
        <h1 className="text-2xl md:text-3xl font-bold text-white mb-3">{exercise.name}</h1>
        
        <div className="flex flex-wrap items-center gap-3 mb-4">
          <Badge variant="secondary" className="capitalize">
            {exercise.category}
          </Badge>
          <div className="w-1 h-1 bg-gray-600 rounded-full"></div>
          <Badge variant="outline" className="capitalize">
            {exercise.difficulty}
          </Badge>
          <div className="w-1 h-1 bg-gray-600 rounded-full"></div>
          <div className="flex flex-wrap gap-1">
            {exercise.muscleGroups.map((muscle: any) => (
              <Badge key={muscle.id} className="bg-gray-700">
                {muscle.name}
              </Badge>
            ))}
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <Button className="bg-blue-600 hover:bg-blue-700" asChild>
            <Link href={`/team/workout-programs/exercises/${exercise.id}/demo`}>
              <Play className="h-4 w-4 mr-2" /> Watch Demo
            </Link>
          </Button>
          
          <Button variant="outline" onClick={toggleFavorite}>
            <Heart className={`h-4 w-4 mr-2 ${isFavorite ? 'text-red-500 fill-red-500' : ''}`} />
            {isFavorite ? 'Favorited' : 'Favorite'}
          </Button>
          
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline">
                <Plus className="h-4 w-4 mr-2" /> Add to Workout
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="bg-gray-800 border-gray-700">
              {workoutPrograms.map(program => (
                <DropdownMenuItem 
                  key={program.id}
                  className="cursor-pointer hover:bg-gray-700"
                >
                  {program.name}
                </DropdownMenuItem>
              ))}
              <DropdownMenuItem className="cursor-pointer hover:bg-gray-700">
                Create New Workout
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
      
      {/* Main Content */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Left Column - Exercise Info */}
        <div className="md:col-span-2">
          <Tabs 
            defaultValue="instructions" 
            value={activeTab}
            onValueChange={setActiveTab}
            className="w-full"
          >
            <TabsList className="grid grid-cols-3 mb-6 bg-gray-800">
              <TabsTrigger value="instructions">Instructions</TabsTrigger>
              <TabsTrigger value="history">History</TabsTrigger>
              <TabsTrigger value="variations">Variations</TabsTrigger>
            </TabsList>
            
            {/* Instructions Tab */}
            <TabsContent value="instructions" className="bg-transparent p-0">
              {/* Exercise Image/Video Placeholder */}
              <div className="bg-gray-800 border border-gray-700 rounded-lg aspect-video flex items-center justify-center mb-6">
                <div className="text-center">
                  <Play className="h-12 w-12 text-blue-400 mx-auto mb-2" />
                  <span className="text-gray-400">Video Demo</span>
                </div>
              </div>
              
              {/* Description */}
              <Card className="bg-gray-800 border-gray-700 mb-6">
                <CardHeader>
                  <CardTitle className="text-lg">Description</CardTitle>
                </CardHeader>
                <CardContent className="text-gray-300">
                  <p>{exercise.description}</p>
                </CardContent>
              </Card>
              
              {/* Instructions */}
              <Card className="bg-gray-800 border-gray-700 mb-6">
                <CardHeader>
                  <CardTitle className="text-lg">Instructions</CardTitle>
                </CardHeader>
                <CardContent className="text-gray-300">
                  <ol className="list-decimal pl-5 space-y-2">
                    {exercise.instructions.map((instruction: string, i: number) => (
                      <li key={i}>{instruction}</li>
                    ))}
                  </ol>
                </CardContent>
              </Card>
              
              {/* Tips */}
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-lg">Tips & Advice</CardTitle>
                </CardHeader>
                <CardContent className="text-gray-300">
                  <ul className="list-disc pl-5 space-y-2">
                    {exercise.tips.map((tip: string, i: number) => (
                      <li key={i}>{tip}</li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </TabsContent>
            
            {/* History Tab */}
            <TabsContent value="history" className="bg-transparent p-0">
              {/* Personal Records */}
              <Card className="bg-gray-800 border-gray-700 mb-6">
                <CardHeader>
                  <CardTitle className="text-lg">Personal Records</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {exercise.personalRecords.map((record: any, i: number) => (
                      <div 
                        key={i} 
                        className="flex justify-between items-center p-3 rounded-md bg-gray-700"
                      >
                        <div>
                          <div className="text-white font-medium">{record.weight}</div>
                          <div className="text-xs text-gray-400">{record.date}</div>
                        </div>
                        <div className="flex items-center">
                          <Badge className="bg-blue-600">{record.reps} reps</Badge>
                          {i === 0 && (
                            <Badge className="ml-2 bg-green-600">Current PR</Badge>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  <Button variant="outline" className="w-full mt-4">
                    <Plus className="h-4 w-4 mr-2" /> Add New PR
                  </Button>
                </CardContent>
              </Card>
              
              {/* Recent Workouts */}
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-lg">Recent Workouts</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {exercise.recentWorkouts.map((workout: any, i: number) => (
                      <div 
                        key={i} 
                        className="p-3 rounded-md bg-gray-700"
                      >
                        <div className="flex justify-between items-center mb-2">
                          <div className="text-white font-medium">{workout.date}</div>
                          <Badge variant="outline">
                            {workout.sets} sets
                          </Badge>
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <div className="text-xs text-gray-400">Weight</div>
                            <div className="text-gray-200">{workout.weight}</div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-400">Reps</div>
                            <div className="text-gray-200">{workout.reps}</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  <div className="text-center mt-4">
                    <Button variant="ghost" className="text-blue-400">
                      View Full History
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            {/* Variations Tab */}
            <TabsContent value="variations" className="bg-transparent p-0">
              <Card className="bg-gray-800 border-gray-700 mb-6">
                <CardHeader>
                  <CardTitle className="text-lg">Exercise Variations</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {exercise.variations.map((variation: any) => (
                      <div 
                        key={variation.id}
                        className="p-3 rounded-md bg-gray-700 flex items-center justify-between"
                      >
                        <div className="flex items-center">
                          <div className="h-10 w-10 rounded bg-gray-600 flex items-center justify-center mr-3">
                            <Dumbbell className="h-5 w-5 text-gray-400" />
                          </div>
                          <div className="text-white font-medium">{variation.name}</div>
                        </div>
                        <Button variant="ghost" size="sm" className="text-blue-400">View</Button>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-lg">Alternative Exercises</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div className="p-3 rounded-md bg-gray-700 flex items-center justify-between">
                      <div className="flex items-center">
                        <div className="h-10 w-10 rounded bg-gray-600 flex items-center justify-center mr-3">
                          <Dumbbell className="h-5 w-5 text-gray-400" />
                        </div>
                        <div className="text-white font-medium">Leg Press</div>
                      </div>
                      <Button variant="ghost" size="sm" className="text-blue-400">View</Button>
                    </div>
                    <div className="p-3 rounded-md bg-gray-700 flex items-center justify-between">
                      <div className="flex items-center">
                        <div className="h-10 w-10 rounded bg-gray-600 flex items-center justify-center mr-3">
                          <Dumbbell className="h-5 w-5 text-gray-400" />
                        </div>
                        <div className="text-white font-medium">Hack Squat</div>
                      </div>
                      <Button variant="ghost" size="sm" className="text-blue-400">View</Button>
                    </div>
                    <div className="p-3 rounded-md bg-gray-700 flex items-center justify-between">
                      <div className="flex items-center">
                        <div className="h-10 w-10 rounded bg-gray-600 flex items-center justify-center mr-3">
                          <Dumbbell className="h-5 w-5 text-gray-400" />
                        </div>
                        <div className="text-white font-medium">Smith Machine Squat</div>
                      </div>
                      <Button variant="ghost" size="sm" className="text-blue-400">View</Button>
                    </div>
                    <div className="p-3 rounded-md bg-gray-700 flex items-center justify-between">
                      <div className="flex items-center">
                        <div className="h-10 w-10 rounded bg-gray-600 flex items-center justify-center mr-3">
                          <Dumbbell className="h-5 w-5 text-gray-400" />
                        </div>
                        <div className="text-white font-medium">Lunge Variations</div>
                      </div>
                      <Button variant="ghost" size="sm" className="text-blue-400">View</Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
        
        {/* Right Column - Sidebar */}
        <div>
          {/* Muscles Worked */}
          <Card className="bg-gray-800 border-gray-700 mb-6">
            <CardHeader>
              <CardTitle className="text-lg">Muscles Worked</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="mb-4">
                <h3 className="text-white font-medium mb-2">Primary Muscles</h3>
                <div className="flex flex-wrap gap-2">
                  {exercise.muscles.primary.map((muscle: string, i: number) => (
                    <Badge key={i} className="bg-blue-600">
                      {muscle}
                    </Badge>
                  ))}
                </div>
              </div>
              
              <div>
                <h3 className="text-white font-medium mb-2">Secondary Muscles</h3>
                <div className="flex flex-wrap gap-2">
                  {exercise.muscles.secondary.map((muscle: string, i: number) => (
                    <Badge key={i} variant="outline" className="border-gray-600 text-gray-300">
                      {muscle}
                    </Badge>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Equipment Needed */}
          <Card className="bg-gray-800 border-gray-700 mb-6">
            <CardHeader>
              <CardTitle className="text-lg">Equipment Needed</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {exercise.equipment.map((eq: any) => (
                  <div key={eq.id} className="flex items-center bg-gray-700 p-2 rounded-md">
                    <Dumbbell className="h-4 w-4 text-gray-400 mr-2" />
                    <span className="text-gray-200">{eq.name}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
          
          {/* Quick Stats */}
          <Card className="bg-gray-800 border-gray-700 mb-6">
            <CardHeader>
              <CardTitle className="text-lg">Your Stats</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-700 p-3 rounded-md">
                  <div className="text-xs text-gray-400 mb-1">Personal Best</div>
                  <div className="text-white font-medium">{exercise.personalRecords[0]?.weight}</div>
                </div>
                
                <div className="bg-gray-700 p-3 rounded-md">
                  <div className="text-xs text-gray-400 mb-1">Last Workout</div>
                  <div className="text-white font-medium">{exercise.recentWorkouts[0]?.weight}</div>
                </div>
                
                <div className="bg-gray-700 p-3 rounded-md">
                  <div className="text-xs text-gray-400 mb-1">Times Performed</div>
                  <div className="text-white font-medium">12</div>
                </div>
                
                <div className="bg-gray-700 p-3 rounded-md">
                  <div className="text-xs text-gray-400 mb-1">Last Performed</div>
                  <div className="text-white font-medium">{exercise.recentWorkouts[0]?.date}</div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Suggested Workouts */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-lg">Programs With This Exercise</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center p-3 rounded-md bg-gray-700 hover:bg-gray-600 transition-colors">
                  <div className="h-10 w-10 rounded bg-gray-800 flex items-center justify-center mr-3">
                    <BarChart className="h-5 w-5 text-blue-400" />
                  </div>
                  <div className="flex-grow">
                    <div className="text-white font-medium">Iron Body: Maximum Strength</div>
                    <div className="text-xs text-gray-400">Week 1, Day 1</div>
                  </div>
                  <ChevronRight className="h-5 w-5 text-gray-500" />
                </div>
                
                <div className="flex items-center p-3 rounded-md bg-gray-700 hover:bg-gray-600 transition-colors">
                  <div className="h-10 w-10 rounded bg-gray-800 flex items-center justify-center mr-3">
                    <BarChart className="h-5 w-5 text-blue-400" />
                  </div>
                  <div className="flex-grow">
                    <div className="text-white font-medium">Functional Strong</div>
                    <div className="text-xs text-gray-400">Week 2, Day 3</div>
                  </div>
                  <ChevronRight className="h-5 w-5 text-gray-500" />
                </div>
                
                <Button variant="outline" className="w-full">
                  <Plus className="h-4 w-4 mr-2" /> Create Custom Workout
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
} 
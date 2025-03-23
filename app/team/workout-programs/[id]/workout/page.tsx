'use client';

import { useState, useEffect, useRef } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { 
  Play, 
  Pause, 
  SkipForward, 
  ArrowLeft, 
  CheckCircle, 
  RotateCcw, 
  Timer, 
  X, 
  ChevronUp, 
  ChevronDown, 
  Info,
  Dumbbell,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

// Mock workout data - in a real app this would come from an API or database
const mockExercises = [
  {
    id: 'ex1',
    name: 'Barbell Squat',
    muscleGroups: ['Quadriceps', 'Glutes', 'Lower Back'],
    sets: [
      { setNumber: 1, reps: 12, weight: '135 lbs', type: 'warmup', completed: false },
      { setNumber: 2, reps: 10, weight: '185 lbs', type: 'regular', completed: false },
      { setNumber: 3, reps: 8, weight: '205 lbs', type: 'regular', completed: false },
      { setNumber: 4, reps: 6, weight: '225 lbs', type: 'regular', completed: false },
    ],
    instructions: [
      'Stand with feet shoulder-width apart, barbell across upper back',
      'Bend knees and drop hips back and down',
      'Descend until thighs are parallel to ground',
      'Drive through heels to return to standing position'
    ],
    tips: 'Keep your chest up and core engaged throughout the movement.',
    videoUrl: '/exercises/squat.mp4',
    restBetweenSets: 90 // in seconds
  },
  {
    id: 'ex2',
    name: 'Romanian Deadlift',
    muscleGroups: ['Hamstrings', 'Glutes', 'Lower Back'],
    sets: [
      { setNumber: 1, reps: 12, weight: '135 lbs', type: 'warmup', completed: false },
      { setNumber: 2, reps: 10, weight: '155 lbs', type: 'regular', completed: false },
      { setNumber: 3, reps: 10, weight: '175 lbs', type: 'regular', completed: false },
      { setNumber: 4, reps: 8, weight: '195 lbs', type: 'regular', completed: false },
    ],
    instructions: [
      'Stand hip-width apart, barbell in front of thighs',
      'With slight knee bend, hinge at hips pushing them back',
      'Lower the bar keeping it close to your legs',
      'Stop when you feel a stretch in hamstrings, then return to start'
    ],
    tips: 'Keep your back flat and neck neutral throughout the movement.',
    videoUrl: '/exercises/rdl.mp4',
    restBetweenSets: 90 // in seconds
  },
  {
    id: 'ex3',
    name: 'Walking Lunges',
    muscleGroups: ['Quadriceps', 'Glutes', 'Hamstrings'],
    sets: [
      { setNumber: 1, reps: '12 per leg', weight: 'Bodyweight', type: 'warmup', completed: false },
      { setNumber: 2, reps: '12 per leg', weight: '20 lbs dumbbells', type: 'regular', completed: false },
      { setNumber: 3, reps: '12 per leg', weight: '25 lbs dumbbells', type: 'regular', completed: false },
      { setNumber: 4, reps: '10 per leg', weight: '30 lbs dumbbells', type: 'regular', completed: false },
    ],
    instructions: [
      'Stand with feet hip-width apart, holding dumbbells at sides',
      'Take a step forward with right foot, lowering left knee toward floor',
      'Push through right heel to stand, bringing left foot forward into next lunge',
      'Continue alternating legs, walking forward'
    ],
    tips: 'Keep your torso upright and core engaged.',
    videoUrl: '/exercises/lunges.mp4',
    restBetweenSets: 60 // in seconds
  },
  {
    id: 'ex4',
    name: 'Leg Press',
    muscleGroups: ['Quadriceps', 'Glutes', 'Hamstrings'],
    sets: [
      { setNumber: 1, reps: 15, weight: '180 lbs', type: 'warmup', completed: false },
      { setNumber: 2, reps: 12, weight: '270 lbs', type: 'regular', completed: false },
      { setNumber: 3, reps: 10, weight: '360 lbs', type: 'regular', completed: false },
      { setNumber: 4, reps: 8, weight: '410 lbs', type: 'regular', completed: false },
    ],
    instructions: [
      'Sit in leg press machine with feet shoulder-width apart on platform',
      'Release safety and lower platform until knees are at 90-degree angle',
      'Press through heels to extend legs without locking knees',
      'Control the descent back to starting position'
    ],
    tips: 'Don\'t allow your lower back to round at the bottom of the movement.',
    videoUrl: '/exercises/leg-press.mp4',
    restBetweenSets: 90 // in seconds
  },
  {
    id: 'ex5',
    name: 'Standing Calf Raises',
    muscleGroups: ['Calves'],
    sets: [
      { setNumber: 1, reps: 15, weight: 'Bodyweight', type: 'warmup', completed: false },
      { setNumber: 2, reps: 15, weight: '100 lbs', type: 'regular', completed: false },
      { setNumber: 3, reps: 15, weight: '120 lbs', type: 'regular', completed: false },
      { setNumber: 4, reps: 12, weight: '140 lbs', type: 'regular', completed: false },
    ],
    instructions: [
      'Stand on calf raise machine with balls of feet on platform',
      'Lower heels as far as comfortable to feel a stretch',
      'Raise heels as high as possible, contracting calves at top',
      'Hold the contracted position briefly before lowering'
    ],
    tips: 'Focus on a full range of motion rather than heavy weight.',
    videoUrl: '/exercises/calf-raises.mp4',
    restBetweenSets: 60 // in seconds
  },
];

const mockWorkout = {
  id: 'workout-1',
  name: 'Lower Body Strength',
  description: 'Focus on building quad, hamstring, and glute strength with compound movements',
  exercises: mockExercises,
  estimatedDuration: 60, // in minutes
  difficultyLevel: 'Intermediate',
  notes: 'Take extra time to warm up properly. Foam roll any tight areas before beginning.',
};

export default function WorkoutExecution() {
  const params = useParams();
  const router = useRouter();
  const [workout, setWorkout] = useState(mockWorkout);
  const [currentExerciseIndex, setCurrentExerciseIndex] = useState(0);
  const [currentSetIndex, setCurrentSetIndex] = useState(0);
  const [timerRunning, setTimerRunning] = useState(false);
  const [timerSeconds, setTimerSeconds] = useState(0);
  const [isShowingInstructions, setIsShowingInstructions] = useState(false);
  const [workoutComplete, setWorkoutComplete] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);
  const elapsedTimeRef = useRef<NodeJS.Timeout | null>(null);
  
  // Exercise and Set references for easier access
  const currentExercise = workout.exercises[currentExerciseIndex];
  const currentSet = currentExercise?.sets[currentSetIndex];
  const totalExercises = workout.exercises.length;
  const totalSets = currentExercise?.sets.length || 0;
  
  // Calculate overall workout progress
  const totalSetsInWorkout = workout.exercises.reduce((acc, ex) => acc + ex.sets.length, 0);
  const completedSetsInWorkout = workout.exercises.reduce((acc, ex) => 
    acc + ex.sets.filter(set => set.completed).length, 0);
  const workoutProgress = Math.round((completedSetsInWorkout / totalSetsInWorkout) * 100);
  
  // Start workout elapsed time counter
  useEffect(() => {
    elapsedTimeRef.current = setInterval(() => {
      setElapsedTime(prev => prev + 1);
    }, 1000);
    
    return () => {
      if (elapsedTimeRef.current) {
        clearInterval(elapsedTimeRef.current);
      }
    };
  }, []);
  
  // Handle timer logic
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    
    if (timerRunning) {
      interval = setInterval(() => {
        setTimerSeconds(seconds => {
          if (seconds <= 0) {
            setTimerRunning(false);
            return 0;
          }
          return seconds - 1;
        });
      }, 1000);
    }
    
    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [timerRunning]);
  
  // Format time as MM:SS
  const formatTime = (timeInSeconds: number) => {
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = timeInSeconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };
  
  // Start rest timer
  const startRestTimer = () => {
    setTimerSeconds(currentExercise.restBetweenSets);
    setTimerRunning(true);
  };
  
  // Mark current set as completed and advance to next set or exercise
  const completeSet = () => {
    // Create copy of workout data to update
    const updatedWorkout = { ...workout };
    updatedWorkout.exercises[currentExerciseIndex].sets[currentSetIndex].completed = true;
    setWorkout(updatedWorkout);
    
    if (currentSetIndex + 1 < totalSets) {
      // Move to next set
      setCurrentSetIndex(currentSetIndex + 1);
      startRestTimer();
    } else if (currentExerciseIndex + 1 < totalExercises) {
      // Move to next exercise's first set
      setCurrentExerciseIndex(currentExerciseIndex + 1);
      setCurrentSetIndex(0);
    } else {
      // Workout complete
      setWorkoutComplete(true);
      if (elapsedTimeRef.current) {
        clearInterval(elapsedTimeRef.current);
      }
    }
  };
  
  // Skip current set and move to next
  const skipSet = () => {
    if (currentSetIndex + 1 < totalSets) {
      // Move to next set
      setCurrentSetIndex(currentSetIndex + 1);
    } else if (currentExerciseIndex + 1 < totalExercises) {
      // Move to next exercise's first set
      setCurrentExerciseIndex(currentExerciseIndex + 1);
      setCurrentSetIndex(0);
    }
  };
  
  // Go back to previous set or exercise
  const previousSet = () => {
    if (currentSetIndex > 0) {
      // Go back to previous set
      setCurrentSetIndex(currentSetIndex - 1);
    } else if (currentExerciseIndex > 0) {
      // Go back to previous exercise's last set
      setCurrentExerciseIndex(currentExerciseIndex - 1);
      setCurrentSetIndex(workout.exercises[currentExerciseIndex - 1].sets.length - 1);
    }
  };
  
  // Restart timer for current set
  const resetTimer = () => {
    setTimerSeconds(currentExercise.restBetweenSets);
    setTimerRunning(true);
  };
  
  // Complete workout and navigate back
  const finishWorkout = () => {
    // In a real app, you'd save workout results to a database here
    router.push(`/team/workout-programs/${params.id}`);
  };
  
  return (
    <div className="workout-execution-page pb-24">
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 py-3 px-4 mb-6 sticky top-0 z-10">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Button
              variant="ghost"
              size="icon"
              asChild
              className="mr-2"
            >
              <Link href={`/team/workout-programs/${params.id}`}>
                <ArrowLeft className="h-5 w-5" />
              </Link>
            </Button>
            <div>
              <h1 className="text-lg font-bold text-white">{workout.name}</h1>
              <div className="text-xs text-gray-400">
                Elapsed Time: {formatTime(elapsedTime)}
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="flex items-center">
              <Dumbbell className="h-3 w-3 mr-1" />
              {workout.difficultyLevel}
            </Badge>
          </div>
        </div>
        
        {/* Overall Progress */}
        <div className="mt-3">
          <div className="flex justify-between items-center text-xs text-gray-400 mb-1">
            <span>Overall Progress</span>
            <span>{completedSetsInWorkout} / {totalSetsInWorkout} sets</span>
          </div>
          <Progress value={workoutProgress} className="h-1 bg-gray-700" />
        </div>
      </div>
      
      {!workoutComplete ? (
        <div className="px-4">
          {/* Current Exercise */}
          <div className="mb-6">
            <div className="flex justify-between items-center mb-2">
              <h2 className="text-xl font-bold text-white">
                {currentExercise.name}
              </h2>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsShowingInstructions(!isShowingInstructions)}
              >
                <Info className="h-4 w-4 mr-1" /> 
                {isShowingInstructions ? 'Hide' : 'View'} Instructions
              </Button>
            </div>
            
            <div className="text-sm text-gray-400 mb-2">
              Target: {currentExercise.muscleGroups.join(', ')}
            </div>
            
            {/* Instructions Panel */}
            {isShowingInstructions && (
              <Card className="bg-gray-800 border-gray-700 mb-4">
                <CardContent className="pt-4">
                  <h3 className="text-white font-medium mb-2">Instructions:</h3>
                  <ol className="list-decimal pl-5 mb-3 text-gray-300 space-y-1">
                    {currentExercise.instructions.map((instruction, i) => (
                      <li key={i}>{instruction}</li>
                    ))}
                  </ol>
                  
                  <h3 className="text-white font-medium mb-2">Tips:</h3>
                  <p className="text-gray-300">{currentExercise.tips}</p>
                </CardContent>
              </Card>
            )}
            
            {/* Exercise Progress */}
            <div className="flex justify-between items-center text-xs text-gray-400 mb-1">
              <span>Exercise Progress</span>
              <span>
                Set {currentSetIndex + 1} / {totalSets}
              </span>
            </div>
            <Progress 
              value={((currentSetIndex + (currentSet?.completed ? 1 : 0)) / totalSets) * 100} 
              className="h-1 bg-gray-700" 
            />
          </div>
          
          {/* Current Set */}
          <Card className="bg-gray-800 border-gray-700 mb-6">
            <CardContent className="pt-6">
              <div className="flex justify-between items-center mb-4">
                <div>
                  <Badge 
                    className={
                      currentSet.type === 'warmup' 
                        ? 'bg-amber-600' 
                        : currentSet.type === 'dropset' 
                          ? 'bg-purple-600' 
                          : 'bg-blue-600'
                    }
                  >
                    {currentSet.type === 'warmup' ? 'Warm-up Set' : 
                     currentSet.type === 'dropset' ? 'Drop Set' : 
                     'Working Set'}
                  </Badge>
                  <h3 className="text-xl font-bold text-white mt-2">Set {currentSet.setNumber}</h3>
                </div>
                
                <div className="text-right">
                  <div className="text-lg font-bold text-white">{currentSet.reps}</div>
                  <div className="text-gray-400">Repetitions</div>
                </div>
              </div>
              
              <div className="flex justify-between items-center mb-6">
                <div>
                  <div className="text-gray-400 text-sm">Weight</div>
                  <div className="text-white font-medium">{currentSet.weight}</div>
                </div>
                
                {timerRunning && (
                  <div className="flex items-center">
                    <Timer className="h-4 w-4 text-blue-400 mr-1" />
                    <span className="text-blue-400 font-medium">
                      Rest: {formatTime(timerSeconds)}
                    </span>
                  </div>
                )}
              </div>
              
              <div className="flex flex-col sm:flex-row gap-3">
                <Button 
                  onClick={completeSet} 
                  className="bg-green-600 hover:bg-green-700 flex-grow"
                  disabled={timerRunning && timerSeconds > 0}
                >
                  <CheckCircle className="h-5 w-5 mr-2" /> 
                  Complete Set
                </Button>
                
                {timerRunning ? (
                  <Button 
                    variant="outline" 
                    onClick={() => setTimerRunning(false)}
                  >
                    <Pause className="h-5 w-5 mr-2" /> 
                    Pause Timer
                  </Button>
                ) : (
                  timerSeconds > 0 ? (
                    <Button 
                      variant="outline" 
                      onClick={() => setTimerRunning(true)}
                    >
                      <Play className="h-5 w-5 mr-2" /> 
                      Resume Timer
                    </Button>
                  ) : currentSet.completed ? null : (
                    <Button 
                      variant="outline" 
                      onClick={resetTimer}
                    >
                      <RotateCcw className="h-5 w-5 mr-2" /> 
                      Start Rest Timer
                    </Button>
                  )
                )}
              </div>
            </CardContent>
          </Card>
          
          {/* Navigation Controls */}
          <div className="flex gap-3 mb-6">
            <Button 
              variant="outline" 
              onClick={previousSet} 
              disabled={currentExerciseIndex === 0 && currentSetIndex === 0}
              className="flex-grow"
            >
              <ChevronUp className="h-5 w-5 mr-2" /> 
              Previous
            </Button>
            <Button 
              variant="outline" 
              onClick={skipSet}
              disabled={
                currentExerciseIndex === totalExercises - 1 && 
                currentSetIndex === totalSets - 1
              }
              className="flex-grow"
            >
              <ChevronDown className="h-5 w-5 mr-2" /> 
              Skip
            </Button>
          </div>
          
          {/* Upcoming Exercises */}
          {currentExerciseIndex < totalExercises - 1 && (
            <div>
              <h3 className="text-white font-medium mb-3">Coming Up Next:</h3>
              <div className="space-y-2">
                {workout.exercises.slice(currentExerciseIndex + 1, currentExerciseIndex + 3).map((exercise, i) => (
                  <div 
                    key={exercise.id}
                    className="flex items-center bg-gray-800 p-3 rounded-md border border-gray-700"
                  >
                    <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center mr-3">
                      {currentExerciseIndex + i + 2}
                    </div>
                    <div>
                      <div className="text-white font-medium">{exercise.name}</div>
                      <div className="text-xs text-gray-400">
                        {exercise.sets.length} sets • {exercise.muscleGroups.join(', ')}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        // Workout Complete View
        <div className="px-4 flex flex-col items-center justify-center py-10">
          <div className="h-20 w-20 rounded-full bg-green-600 flex items-center justify-center mb-6">
            <CheckCircle className="h-10 w-10 text-white" />
          </div>
          
          <h2 className="text-2xl font-bold text-white mb-2">Workout Complete!</h2>
          <p className="text-gray-400 text-center mb-6">
            Great job! You've completed all exercises in this workout.
          </p>
          
          <Card className="bg-gray-800 border-gray-700 mb-6 w-full max-w-md">
            <CardContent className="pt-6">
              <h3 className="text-white font-medium mb-3">Workout Summary</h3>
              
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <div className="text-gray-400">Total Time</div>
                  <div className="text-white font-medium">{formatTime(elapsedTime)}</div>
                </div>
                
                <div className="flex justify-between items-center">
                  <div className="text-gray-400">Exercises Completed</div>
                  <div className="text-white font-medium">{totalExercises}</div>
                </div>
                
                <div className="flex justify-between items-center">
                  <div className="text-gray-400">Sets Completed</div>
                  <div className="text-white font-medium">{completedSetsInWorkout}</div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Button onClick={finishWorkout} className="bg-blue-600 hover:bg-blue-700">
            Finish & Return to Program
          </Button>
        </div>
      )}
    </div>
  );
} 
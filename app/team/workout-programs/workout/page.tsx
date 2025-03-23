'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { 
  ArrowLeft, 
  Play,
  Pause,
  RotateCcw,
  Clock,
  ChevronRight,
  ChevronDown,
  CheckCircle,
  XCircle,
  Edit,
  Plus,
  Save,
  Dumbbell,
  MoreVertical,
  TimerReset,
  ChevronUp,
  Share2
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger 
} from '@/components/ui/dropdown-menu';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from '@/components/ui/input';

// Mock workout data
const workoutData = {
  id: 'workout-1',
  name: 'Upper Body Strength - Day 1',
  program: 'Iron Body: Maximum Strength',
  description: 'Focus on compound movements for the upper body with progressive overload.',
  duration: '45-60 minutes',
  difficulty: 'intermediate',
  exercises: [
    {
      id: 'ex1',
      name: 'Barbell Bench Press',
      sets: 4,
      reps: '8-10',
      rest: 90,
      weight: '185 lbs',
      notes: 'Keep shoulders back and down, feet planted firmly',
      isCompleted: true,
      progress: [
        { setNumber: 1, weight: '185', reps: 10, isCompleted: true },
        { setNumber: 2, weight: '185', reps: 9, isCompleted: true },
        { setNumber: 3, weight: '185', reps: 8, isCompleted: true },
        { setNumber: 4, weight: '185', reps: 8, isCompleted: true }
      ]
    },
    {
      id: 'ex2',
      name: 'Barbell Row',
      sets: 4,
      reps: '8-10',
      rest: 90,
      weight: '155 lbs',
      notes: 'Keep back straight, pull to lower chest',
      isCompleted: true,
      progress: [
        { setNumber: 1, weight: '155', reps: 10, isCompleted: true },
        { setNumber: 2, weight: '155', reps: 10, isCompleted: true },
        { setNumber: 3, weight: '155', reps: 9, isCompleted: true },
        { setNumber: 4, weight: '155', reps: 8, isCompleted: true }
      ]
    },
    {
      id: 'ex3',
      name: 'Overhead Press',
      sets: 3,
      reps: '8-10',
      rest: 120,
      weight: '105 lbs',
      notes: 'Brace core, avoid excessive back arch',
      isCompleted: false,
      progress: [
        { setNumber: 1, weight: '105', reps: 10, isCompleted: true },
        { setNumber: 2, weight: '105', reps: 8, isCompleted: false },
        { setNumber: 3, weight: '105', reps: 0, isCompleted: false }
      ]
    },
    {
      id: 'ex4',
      name: 'Pull-Ups',
      sets: 3,
      reps: '8-10',
      rest: 120,
      weight: 'Bodyweight',
      notes: 'Use assistance band if needed',
      isCompleted: false,
      progress: [
        { setNumber: 1, weight: 'BW', reps: 0, isCompleted: false },
        { setNumber: 2, weight: 'BW', reps: 0, isCompleted: false },
        { setNumber: 3, weight: 'BW', reps: 0, isCompleted: false }
      ]
    },
    {
      id: 'ex5',
      name: 'Tricep Extensions',
      sets: 3,
      reps: '10-12',
      rest: 60,
      weight: '50 lbs',
      notes: 'Keep elbows steady and close to body',
      isCompleted: false,
      progress: [
        { setNumber: 1, weight: '50', reps: 0, isCompleted: false },
        { setNumber: 2, weight: '50', reps: 0, isCompleted: false },
        { setNumber: 3, weight: '50', reps: 0, isCompleted: false }
      ]
    },
    {
      id: 'ex6',
      name: 'Bicep Curls',
      sets: 3,
      reps: '10-12',
      rest: 60,
      weight: '40 lbs',
      notes: 'Avoid swinging, control the movement',
      isCompleted: false,
      progress: [
        { setNumber: 1, weight: '40', reps: 0, isCompleted: false },
        { setNumber: 2, weight: '40', reps: 0, isCompleted: false },
        { setNumber: 3, weight: '40', reps: 0, isCompleted: false }
      ]
    }
  ]
};

export default function WorkoutExecution() {
  const [workout, setWorkout] = useState<any>(workoutData);
  const [currentExerciseIndex, setCurrentExerciseIndex] = useState(2); // Start with the incomplete exercise
  const [currentSetIndex, setCurrentSetIndex] = useState(1); // Start with the incomplete set
  const [timerActive, setTimerActive] = useState(false);
  const [restTime, setRestTime] = useState(0);
  const [timerInterval, setTimerInterval] = useState<NodeJS.Timeout | null>(null);
  const [expandedExercises, setExpandedExercises] = useState<Record<string, boolean>>({});
  const [editWeightModal, setEditWeightModal] = useState(false);
  const [editRepsModal, setEditRepsModal] = useState(false);
  const [tempWeight, setTempWeight] = useState('');
  const [tempReps, setTempReps] = useState('');
  
  const currentExercise = workout.exercises[currentExerciseIndex];
  const workoutProgress = calculateWorkoutProgress();
  
  // Calculate overall workout progress
  function calculateWorkoutProgress() {
    const totalSets = workout.exercises.reduce((total: number, ex: any) => total + ex.sets, 0);
    const completedSets = workout.exercises.reduce((total: number, ex: any) => {
      return total + ex.progress.filter((set: any) => set.isCompleted).length;
    }, 0);
    
    return Math.round((completedSets / totalSets) * 100);
  }
  
  // Toggle expanded state for an exercise
  const toggleExpanded = (exerciseId: string) => {
    setExpandedExercises(prev => ({
      ...prev,
      [exerciseId]: !prev[exerciseId]
    }));
  };
  
  // Start rest timer
  const startTimer = () => {
    if (timerInterval) {
      clearInterval(timerInterval);
    }
    
    setRestTime(currentExercise.rest);
    setTimerActive(true);
    
    const interval = setInterval(() => {
      setRestTime(prev => {
        if (prev <= 1) {
          clearInterval(interval);
          setTimerActive(false);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    
    setTimerInterval(interval);
  };
  
  // Reset timer
  const resetTimer = () => {
    if (timerInterval) {
      clearInterval(timerInterval);
    }
    setRestTime(currentExercise.rest);
    setTimerActive(false);
  };
  
  // Pause timer
  const pauseTimer = () => {
    if (timerInterval) {
      clearInterval(timerInterval);
      setTimerInterval(null);
    }
    setTimerActive(false);
  };
  
  // Complete current set
  const completeSet = (reps: number) => {
    const updatedExercises = [...workout.exercises];
    const exercise = updatedExercises[currentExerciseIndex];
    
    // Update the current set
    exercise.progress[currentSetIndex].reps = reps;
    exercise.progress[currentSetIndex].isCompleted = true;
    
    // Check if we need to move to the next set or exercise
    if (currentSetIndex < exercise.sets - 1) {
      // Move to next set
      setCurrentSetIndex(currentSetIndex + 1);
      startTimer(); // Start rest timer
    } else {
      // This exercise is complete
      exercise.isCompleted = true;
      
      // Find the next incomplete exercise
      const nextIncompleteIndex = updatedExercises.findIndex(
        (ex, index) => index > currentExerciseIndex && !ex.isCompleted
      );
      
      if (nextIncompleteIndex !== -1) {
        setCurrentExerciseIndex(nextIncompleteIndex);
        setCurrentSetIndex(0);
      }
      
      resetTimer();
    }
    
    setWorkout({ ...workout, exercises: updatedExercises });
  };
  
  // Save edited weight
  const saveWeight = () => {
    if (!tempWeight) return;
    
    const updatedExercises = [...workout.exercises];
    const exercise = updatedExercises[currentExerciseIndex];
    
    // Update all remaining sets with new weight
    for (let i = currentSetIndex; i < exercise.progress.length; i++) {
      exercise.progress[i].weight = tempWeight;
    }
    
    setWorkout({ ...workout, exercises: updatedExercises });
    setEditWeightModal(false);
  };
  
  // Save edited reps
  const saveReps = () => {
    if (!tempReps) return;
    
    completeSet(parseInt(tempReps));
    setEditRepsModal(false);
  };
  
  // Skip current exercise
  const skipExercise = () => {
    const nextIncompleteIndex = workout.exercises.findIndex(
      (ex: any, index: number) => index > currentExerciseIndex && !ex.isCompleted
    );
    
    if (nextIncompleteIndex !== -1) {
      setCurrentExerciseIndex(nextIncompleteIndex);
      setCurrentSetIndex(0);
    }
    
    resetTimer();
  };
  
  // Effect to cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (timerInterval) {
        clearInterval(timerInterval);
      }
    };
  }, [timerInterval]);
  
  return (
    <div className="workout-execution-page pb-16">
      {/* Header Navigation */}
      <div className="mb-4 flex items-center">
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
        <span className="text-sm text-gray-400">
          <Link href="/team/workout-programs" className="hover:text-blue-400">Programs</Link>
          <ChevronRight className="h-4 w-4 inline mx-1" />
          <span>{workout.program}</span>
          <ChevronRight className="h-4 w-4 inline mx-1" />
          <span>{workout.name}</span>
        </span>
      </div>
      
      {/* Workout Header */}
      <div className="mb-6">
        <h1 className="text-2xl md:text-3xl font-bold text-white mb-3">{workout.name}</h1>
        
        <div className="flex flex-wrap items-center gap-3 mb-4">
          <Badge variant="secondary">{workout.program}</Badge>
          <div className="w-1 h-1 bg-gray-600 rounded-full"></div>
          <Badge variant="outline" className="capitalize">
            {workout.difficulty}
          </Badge>
          <div className="w-1 h-1 bg-gray-600 rounded-full"></div>
          <div className="flex items-center">
            <Clock className="h-4 w-4 mr-1 text-gray-400" />
            <span className="text-gray-400 text-sm">{workout.duration}</span>
          </div>
        </div>
        
        <div className="flex items-center justify-between mb-4">
          <div className="w-full max-w-md">
            <div className="flex justify-between items-center mb-1">
              <span className="text-sm text-gray-400">Workout Progress</span>
              <span className="text-sm text-gray-400">{workoutProgress}%</span>
            </div>
            <Progress value={workoutProgress} className="h-2" />
          </div>
          
          <div className="flex gap-2">
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="outline" className="bg-gray-800">
                  <Edit className="h-4 w-4 mr-2" /> Edit
                </Button>
              </DialogTrigger>
              <DialogContent className="bg-gray-800 border-gray-700">
                <DialogHeader>
                  <DialogTitle>Edit Workout</DialogTitle>
                  <DialogDescription>
                    Make changes to your workout routine.
                  </DialogDescription>
                </DialogHeader>
                <div className="mt-4">
                  <div className="space-y-4">
                    <div>
                      <h3 className="font-medium mb-2">Exercises</h3>
                      <div className="space-y-2">
                        {workout.exercises.map((ex: any, index: number) => (
                          <div key={ex.id} className="flex items-center justify-between bg-gray-700 p-3 rounded-md">
                            <div className="flex items-center">
                              <div className="h-8 w-8 rounded bg-gray-600 flex items-center justify-center mr-3">
                                <span className="text-sm font-medium">{index + 1}</span>
                              </div>
                              <span>{ex.name}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <Button variant="ghost" size="sm">
                                <ChevronUp className="h-4 w-4" />
                              </Button>
                              <Button variant="ghost" size="sm">
                                <ChevronDown className="h-4 w-4" />
                              </Button>
                              <Button variant="ghost" size="sm">
                                <XCircle className="h-4 w-4 text-red-500" />
                              </Button>
                            </div>
                          </div>
                        ))}
                      </div>
                      <Button className="w-full mt-2">
                        <Plus className="h-4 w-4 mr-2" /> Add Exercise
                      </Button>
                    </div>
                  </div>
                  
                  <div className="flex justify-end gap-2 mt-6">
                    <Button variant="ghost">Cancel</Button>
                    <Button>Save Changes</Button>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
            
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="bg-gray-800">
                  <MoreVertical className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="bg-gray-800 border-gray-700">
                <DropdownMenuItem className="cursor-pointer">
                  <Share2 className="h-4 w-4 mr-2" /> Share Workout
                </DropdownMenuItem>
                <DropdownMenuItem className="cursor-pointer">
                  <Save className="h-4 w-4 mr-2" /> Save as Template
                </DropdownMenuItem>
                <DropdownMenuItem className="cursor-pointer text-red-500">
                  <XCircle className="h-4 w-4 mr-2" /> End Workout
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
        
        <p className="text-gray-300">{workout.description}</p>
      </div>
      
      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Current Exercise Column */}
        <div className="lg:col-span-2">
          <Card className="bg-gray-800 border-gray-700 mb-6">
            <CardHeader>
              <CardTitle className="text-lg">Current Exercise</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col md:flex-row">
                {/* Exercise Image Placeholder */}
                <div className="bg-gray-700 border border-gray-600 rounded-lg aspect-square md:aspect-video md:w-1/3 flex items-center justify-center mb-4 md:mb-0 md:mr-4">
                  <div className="text-center">
                    <Dumbbell className="h-12 w-12 text-gray-500 mx-auto mb-2" />
                    <span className="text-gray-400">Exercise Demo</span>
                  </div>
                </div>
                
                <div className="md:w-2/3">
                  <h2 className="text-xl font-bold text-white mb-2">{currentExercise.name}</h2>
                  
                  <div className="flex flex-wrap gap-2 mb-4">
                    <Badge variant="outline">
                      {currentExercise.sets} sets
                    </Badge>
                    <Badge variant="outline">
                      {currentExercise.reps} reps
                    </Badge>
                    <Badge variant="outline">
                      {currentExercise.weight}
                    </Badge>
                    <Badge variant="outline">
                      {currentExercise.rest}s rest
                    </Badge>
                  </div>
                  
                  {currentExercise.notes && (
                    <div className="p-3 bg-gray-700 rounded-md mb-4">
                      <h3 className="text-sm text-gray-400 mb-1">Notes:</h3>
                      <p className="text-gray-200">{currentExercise.notes}</p>
                    </div>
                  )}
                  
                  {/* Current Set Progress */}
                  <div className="mb-4">
                    <div className="flex justify-between items-center mb-2">
                      <h3 className="text-white font-medium">
                        Set {currentSetIndex + 1} of {currentExercise.sets}
                      </h3>
                      
                      <div className="flex items-center gap-2">
                        <Dialog open={editWeightModal} onOpenChange={setEditWeightModal}>
                          <DialogTrigger asChild>
                            <Button variant="outline" size="sm">
                              <Edit className="h-3 w-3 mr-1" /> Weight
                            </Button>
                          </DialogTrigger>
                          <DialogContent className="bg-gray-800 border-gray-700">
                            <DialogHeader>
                              <DialogTitle>Edit Weight</DialogTitle>
                              <DialogDescription>
                                Update the weight for remaining sets.
                              </DialogDescription>
                            </DialogHeader>
                            <div className="mt-4">
                              <Input
                                type="text"
                                placeholder="Enter weight (e.g., 135 lbs)"
                                value={tempWeight}
                                onChange={(e) => setTempWeight(e.target.value)}
                                className="bg-gray-700 border-gray-600 text-white"
                              />
                              <div className="flex justify-end gap-2 mt-4">
                                <Button variant="ghost" onClick={() => setEditWeightModal(false)}>
                                  Cancel
                                </Button>
                                <Button onClick={saveWeight}>Save</Button>
                              </div>
                            </div>
                          </DialogContent>
                        </Dialog>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      <Button 
                        variant="outline" 
                        className="bg-gray-700 hover:bg-gray-600"
                        onClick={() => completeSet(parseInt(currentExercise.reps.split('-')[0]))}
                      >
                        {currentExercise.reps.split('-')[0]} reps
                      </Button>
                      
                      <Dialog open={editRepsModal} onOpenChange={setEditRepsModal}>
                        <DialogTrigger asChild>
                          <Button variant="outline" className="bg-gray-700 hover:bg-gray-600">
                            Custom Reps
                          </Button>
                        </DialogTrigger>
                        <DialogContent className="bg-gray-800 border-gray-700">
                          <DialogHeader>
                            <DialogTitle>Custom Rep Count</DialogTitle>
                            <DialogDescription>
                              Enter the number of reps you completed.
                            </DialogDescription>
                          </DialogHeader>
                          <div className="mt-4">
                            <Input
                              type="number"
                              placeholder="Enter reps"
                              value={tempReps}
                              onChange={(e) => setTempReps(e.target.value)}
                              className="bg-gray-700 border-gray-600 text-white"
                            />
                            <div className="flex justify-end gap-2 mt-4">
                              <Button variant="ghost" onClick={() => setEditRepsModal(false)}>
                                Cancel
                              </Button>
                              <Button onClick={saveReps}>Save</Button>
                            </div>
                          </div>
                        </DialogContent>
                      </Dialog>
                      
                      <Button 
                        variant="outline"
                        className="text-yellow-500 border-yellow-500 hover:bg-yellow-500/20"
                        onClick={skipExercise}
                      >
                        Skip Exercise
                      </Button>
                      
                      <Button 
                        variant="outline"
                        className="text-red-500 border-red-500 hover:bg-red-500/20"
                        onClick={() => completeSet(0)}
                      >
                        Failed Set
                      </Button>
                    </div>
                  </div>
                  
                  {/* Rest Timer */}
                  <div className="bg-gray-700 rounded-md p-4">
                    <div className="flex justify-between items-center mb-2">
                      <h3 className="text-white font-medium">Rest Timer</h3>
                      <span className="text-xl font-bold text-blue-400">
                        {Math.floor(restTime / 60)}:{(restTime % 60).toString().padStart(2, '0')}
                      </span>
                    </div>
                    
                    <div className="flex justify-between gap-2">
                      {timerActive ? (
                        <Button 
                          variant="outline" 
                          className="flex-1"
                          onClick={pauseTimer}
                        >
                          <Pause className="h-4 w-4 mr-2" /> Pause
                        </Button>
                      ) : (
                        <Button 
                          variant="outline" 
                          className="flex-1"
                          onClick={startTimer}
                        >
                          <Play className="h-4 w-4 mr-2" /> Start
                        </Button>
                      )}
                      
                      <Button 
                        variant="outline" 
                        className="flex-1"
                        onClick={resetTimer}
                      >
                        <RotateCcw className="h-4 w-4 mr-2" /> Reset
                      </Button>
                      
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="outline" className="flex-1">
                            <TimerReset className="h-4 w-4 mr-2" /> Presets
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent className="bg-gray-800 border-gray-700">
                          <DropdownMenuItem 
                            onClick={() => setRestTime(30)}
                            className="cursor-pointer"
                          >
                            30 seconds
                          </DropdownMenuItem>
                          <DropdownMenuItem 
                            onClick={() => setRestTime(60)}
                            className="cursor-pointer"
                          >
                            1 minute
                          </DropdownMenuItem>
                          <DropdownMenuItem 
                            onClick={() => setRestTime(90)}
                            className="cursor-pointer"
                          >
                            1.5 minutes
                          </DropdownMenuItem>
                          <DropdownMenuItem 
                            onClick={() => setRestTime(120)}
                            className="cursor-pointer"
                          >
                            2 minutes
                          </DropdownMenuItem>
                          <DropdownMenuItem 
                            onClick={() => setRestTime(180)}
                            className="cursor-pointer"
                          >
                            3 minutes
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Set History for Current Exercise */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-lg">Set History</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="text-left border-b border-gray-700">
                      <th className="pb-2 font-medium text-gray-400">Set</th>
                      <th className="pb-2 font-medium text-gray-400">Weight</th>
                      <th className="pb-2 font-medium text-gray-400">Reps</th>
                      <th className="pb-2 font-medium text-gray-400">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {currentExercise.progress.map((set: any, index: number) => (
                      <tr key={index} className="border-b border-gray-700 last:border-0">
                        <td className="py-3 pr-4">Set {index + 1}</td>
                        <td className="py-3 pr-4">{set.weight}</td>
                        <td className="py-3 pr-4">{set.reps > 0 ? set.reps : '-'}</td>
                        <td className="py-3">
                          {set.isCompleted ? (
                            <Badge className="bg-green-600">Completed</Badge>
                          ) : index === currentSetIndex ? (
                            <Badge className="bg-blue-600">Current</Badge>
                          ) : (
                            <Badge variant="outline">Pending</Badge>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
        
        {/* Workout Overview Column */}
        <div>
          <Card className="bg-gray-800 border-gray-700 mb-6">
            <CardHeader>
              <CardTitle className="text-lg">Workout Overview</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {workout.exercises.map((exercise: any, index: number) => (
                  <div key={exercise.id} className={`rounded-md border ${currentExerciseIndex === index ? 'border-blue-500 bg-blue-900/20' : 'border-gray-700 bg-gray-700'}`}>
                    <div 
                      className="p-3 flex items-center justify-between cursor-pointer"
                      onClick={() => toggleExpanded(exercise.id)}
                    >
                      <div className="flex items-center">
                        <div className={`h-8 w-8 rounded-full flex items-center justify-center mr-3 ${
                          exercise.isCompleted ? 'bg-green-600/20 text-green-500' : 
                          currentExerciseIndex === index ? 'bg-blue-600/20 text-blue-500' : 
                          'bg-gray-600 text-gray-400'
                        }`}>
                          {exercise.isCompleted ? (
                            <CheckCircle className="h-5 w-5" />
                          ) : (
                            <span className="text-sm font-medium">{index + 1}</span>
                          )}
                        </div>
                        <div>
                          <div className={`font-medium ${exercise.isCompleted ? 'text-gray-400 line-through' : 'text-white'}`}>
                            {exercise.name}
                          </div>
                          <div className="text-xs text-gray-400">
                            {exercise.sets} sets • {exercise.reps} reps • {exercise.weight}
                          </div>
                        </div>
                      </div>
                      <div>
                        {expandedExercises[exercise.id] ? (
                          <ChevronUp className="h-5 w-5 text-gray-500" />
                        ) : (
                          <ChevronDown className="h-5 w-5 text-gray-500" />
                        )}
                      </div>
                    </div>
                    
                    {expandedExercises[exercise.id] && (
                      <div className="p-3 pt-0 border-t border-gray-600">
                        <div className="space-y-2">
                          {exercise.progress.map((set: any, setIndex: number) => (
                            <div key={setIndex} className="flex justify-between items-center p-2 rounded bg-gray-800">
                              <div className="text-sm">Set {setIndex + 1}</div>
                              <div className="flex items-center gap-3">
                                <div className="text-sm">{set.weight}</div>
                                <div className="text-sm">{set.reps > 0 ? `${set.reps} reps` : '-'}</div>
                                {set.isCompleted ? (
                                  <CheckCircle className="h-4 w-4 text-green-500" />
                                ) : (
                                  <div className="h-4 w-4 rounded-full border border-gray-600"></div>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                        
                        {!exercise.isCompleted && index !== currentExerciseIndex && (
                          <Button 
                            className="w-full mt-3"
                            onClick={() => {
                              setCurrentExerciseIndex(index);
                              setCurrentSetIndex(exercise.progress.findIndex((set: any) => !set.isCompleted));
                            }}
                          >
                            Jump to Exercise
                          </Button>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
              
              <div className="mt-6 p-4 bg-gray-700 rounded-md">
                <h3 className="text-white font-medium mb-2">Workout Stats</h3>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <div className="text-xs text-gray-400">Total Sets</div>
                    <div className="text-white font-medium">
                      {workout.exercises.reduce((acc: number, ex: any) => acc + ex.sets, 0)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">Completed</div>
                    <div className="text-white font-medium">
                      {workout.exercises.reduce((acc: number, ex: any) => 
                        acc + ex.progress.filter((set: any) => set.isCompleted).length, 0
                      )}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">Total Time</div>
                    <div className="text-white font-medium">45:22</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">Remaining</div>
                    <div className="text-white font-medium">
                      {workout.exercises.reduce((acc: number, ex: any) => 
                        acc + ex.progress.filter((set: any) => !set.isCompleted).length, 0
                      )} sets
                    </div>
                  </div>
                </div>
              </div>
              
              <Button className="w-full mt-4" variant="outline">
                <Share2 className="h-4 w-4 mr-2" /> Share Workout Log
              </Button>
            </CardContent>
          </Card>
          
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-lg">Workout Notes</CardTitle>
            </CardHeader>
            <CardContent>
              <textarea 
                className="w-full h-32 bg-gray-700 border border-gray-600 rounded-md p-3 text-white resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Add notes about your workout experience, soreness, energy levels, etc."
              ></textarea>
              <Button className="w-full mt-3">
                <Save className="h-4 w-4 mr-2" /> Save Notes
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
} 
'use client';

import { useState } from 'react';
import { 
  Plus, 
  Trash2, 
  Copy, 
  Save, 
  Dumbbell, 
  Calendar, 
  Clock, 
  RotateCw, 
  ChevronDown, 
  ChevronUp,
  Target,
  Users,
  Zap
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@/components/ui/tabs';

// Exercise database (mock)
const exerciseDatabase = [
  { id: 'ex1', name: 'Barbell Squat', category: 'Lower Body', equipment: 'Barbell', targetMuscles: ['Quadriceps', 'Glutes', 'Hamstrings'] },
  { id: 'ex2', name: 'Bench Press', category: 'Chest', equipment: 'Barbell', targetMuscles: ['Chest', 'Triceps', 'Shoulders'] },
  { id: 'ex3', name: 'Deadlift', category: 'Full Body', equipment: 'Barbell', targetMuscles: ['Lower Back', 'Glutes', 'Hamstrings'] },
  { id: 'ex4', name: 'Pull-up', category: 'Back', equipment: 'Bodyweight', targetMuscles: ['Lats', 'Biceps', 'Upper Back'] },
  { id: 'ex5', name: 'Dumbbell Shoulder Press', category: 'Shoulders', equipment: 'Dumbbells', targetMuscles: ['Shoulders', 'Triceps'] },
  { id: 'ex6', name: 'Romanian Deadlift', category: 'Lower Body', equipment: 'Barbell', targetMuscles: ['Hamstrings', 'Glutes', 'Lower Back'] },
  { id: 'ex7', name: 'Lat Pulldown', category: 'Back', equipment: 'Cable Machine', targetMuscles: ['Lats', 'Biceps'] },
  { id: 'ex8', name: 'Leg Press', category: 'Lower Body', equipment: 'Machine', targetMuscles: ['Quadriceps', 'Glutes', 'Hamstrings'] },
  { id: 'ex9', name: 'Dumbbell Row', category: 'Back', equipment: 'Dumbbells', targetMuscles: ['Upper Back', 'Lats', 'Biceps'] },
  { id: 'ex10', name: 'Incline Bench Press', category: 'Chest', equipment: 'Barbell', targetMuscles: ['Upper Chest', 'Shoulders', 'Triceps'] },
];

// Template programs (mock)
const templatePrograms = [
  { id: 'tmp1', name: '5x5 Strength Program', difficulty: 'Intermediate', focus: 'Strength', duration: '8 weeks', days: 3 },
  { id: 'tmp2', name: 'PPL Hypertrophy', difficulty: 'Intermediate', focus: 'Muscle Growth', duration: '12 weeks', days: 6 },
  { id: 'tmp3', name: 'Full Body Beginner', difficulty: 'Beginner', focus: 'Strength & Form', duration: '4 weeks', days: 3 },
  { id: 'tmp4', name: 'Upper/Lower Split', difficulty: 'Advanced', focus: 'Strength & Size', duration: '10 weeks', days: 4 },
];

type ExerciseSet = {
  id: string;
  reps: string;
  weight: string;
  rpe?: string;
  restTime?: number;
};

type WorkoutExercise = {
  id: string;
  exerciseId: string;
  name: string;
  sets: ExerciseSet[];
  notes?: string;
  superset?: boolean;
};

type WorkoutDay = {
  id: string;
  name: string;
  description?: string;
  exercises: WorkoutExercise[];
  restDay?: boolean;
};

type WorkoutProgram = {
  id: string;
  name: string;
  description?: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  focus: string;
  duration: string;
  days: WorkoutDay[];
};

export default function WorkoutProgramCreator() {
  const [activeTab, setActiveTab] = useState<string>('builder');
  const [exerciseSearch, setExerciseSearch] = useState<string>('');
  const [selectedDay, setSelectedDay] = useState<number>(0);
  const [expandedExercise, setExpandedExercise] = useState<string | null>(null);
  
  // Initialize with a basic template
  const [program, setProgram] = useState<WorkoutProgram>({
    id: 'program-1',
    name: 'My Workout Program',
    description: 'A customized workout program',
    difficulty: 'Intermediate',
    focus: 'Strength',
    duration: '8 weeks',
    days: [
      {
        id: 'day-1',
        name: 'Day 1 - Full Body',
        exercises: [
          {
            id: 'workout-ex-1',
            exerciseId: 'ex1',
            name: 'Barbell Squat',
            sets: [
              { id: 'set-1-1', reps: '5', weight: '135', restTime: 120 },
              { id: 'set-1-2', reps: '5', weight: '135', restTime: 120 },
              { id: 'set-1-3', reps: '5', weight: '135', restTime: 120 },
            ]
          },
          {
            id: 'workout-ex-2',
            exerciseId: 'ex2',
            name: 'Bench Press',
            sets: [
              { id: 'set-2-1', reps: '5', weight: '95', restTime: 120 },
              { id: 'set-2-2', reps: '5', weight: '95', restTime: 120 },
              { id: 'set-2-3', reps: '5', weight: '95', restTime: 120 },
            ]
          }
        ]
      },
      {
        id: 'day-2',
        name: 'Day 2 - Rest',
        restDay: true,
        exercises: []
      },
      {
        id: 'day-3',
        name: 'Day 3 - Full Body',
        exercises: [
          {
            id: 'workout-ex-3',
            exerciseId: 'ex3',
            name: 'Deadlift',
            sets: [
              { id: 'set-3-1', reps: '5', weight: '185', restTime: 180 },
              { id: 'set-3-2', reps: '5', weight: '185', restTime: 180 },
              { id: 'set-3-3', reps: '5', weight: '185', restTime: 180 },
            ]
          },
          {
            id: 'workout-ex-4',
            exerciseId: 'ex7',
            name: 'Lat Pulldown',
            sets: [
              { id: 'set-4-1', reps: '8', weight: '120', restTime: 90 },
              { id: 'set-4-2', reps: '8', weight: '120', restTime: 90 },
              { id: 'set-4-3', reps: '8', weight: '120', restTime: 90 },
            ]
          }
        ]
      }
    ]
  });
  
  // Filtered exercises for search
  const filteredExercises = exerciseDatabase.filter(ex => 
    ex.name.toLowerCase().includes(exerciseSearch.toLowerCase()) ||
    ex.category.toLowerCase().includes(exerciseSearch.toLowerCase()) ||
    ex.equipment.toLowerCase().includes(exerciseSearch.toLowerCase())
  );
  
  // Add a new exercise to the selected day
  const addExerciseToDay = (exerciseId: string) => {
    const exercise = exerciseDatabase.find(ex => ex.id === exerciseId);
    if (!exercise) return;
    
    const newExercise: WorkoutExercise = {
      id: `workout-ex-${Date.now()}`,
      exerciseId: exercise.id,
      name: exercise.name,
      sets: [
        { id: `set-${Date.now()}-1`, reps: '8', weight: '0', restTime: 60 }
      ]
    };
    
    setProgram(prev => {
      const updatedDays = [...prev.days];
      updatedDays[selectedDay] = {
        ...updatedDays[selectedDay],
        exercises: [...updatedDays[selectedDay].exercises, newExercise]
      };
      return { ...prev, days: updatedDays };
    });
    
    // Auto-expand the newly added exercise
    setExpandedExercise(newExercise.id);
  };
  
  // Add a new set to an exercise
  const addSetToExercise = (exerciseIndex: number) => {
    setProgram(prev => {
      const updatedDays = [...prev.days];
      const exercise = updatedDays[selectedDay].exercises[exerciseIndex];
      const lastSet = exercise.sets[exercise.sets.length - 1];
      
      const newSet: ExerciseSet = {
        id: `set-${Date.now()}`,
        reps: lastSet.reps,
        weight: lastSet.weight,
        restTime: lastSet.restTime
      };
      
      updatedDays[selectedDay].exercises[exerciseIndex].sets.push(newSet);
      return { ...prev, days: updatedDays };
    });
  };
  
  // Remove a set from an exercise
  const removeSet = (exerciseIndex: number, setIndex: number) => {
    setProgram(prev => {
      const updatedDays = [...prev.days];
      const exercise = updatedDays[selectedDay].exercises[exerciseIndex];
      
      // Don't remove if it's the only set
      if (exercise.sets.length <= 1) return prev;
      
      exercise.sets.splice(setIndex, 1);
      return { ...prev, days: updatedDays };
    });
  };
  
  // Remove an exercise from a day
  const removeExercise = (exerciseIndex: number) => {
    setProgram(prev => {
      const updatedDays = [...prev.days];
      updatedDays[selectedDay].exercises.splice(exerciseIndex, 1);
      return { ...prev, days: updatedDays };
    });
  };
  
  // Add a new day to the program
  const addDay = () => {
    const newDay: WorkoutDay = {
      id: `day-${program.days.length + 1}`,
      name: `Day ${program.days.length + 1}`,
      exercises: []
    };
    
    setProgram(prev => ({
      ...prev,
      days: [...prev.days, newDay]
    }));
    
    // Switch to the new day
    setSelectedDay(program.days.length);
  };
  
  // Add a rest day
  const addRestDay = () => {
    const newDay: WorkoutDay = {
      id: `day-${program.days.length + 1}`,
      name: `Day ${program.days.length + 1} - Rest`,
      restDay: true,
      exercises: []
    };
    
    setProgram(prev => ({
      ...prev,
      days: [...prev.days, newDay]
    }));
    
    // Switch to the new day
    setSelectedDay(program.days.length);
  };
  
  // Remove a day
  const removeDay = (dayIndex: number) => {
    if (program.days.length <= 1) return;
    
    setProgram(prev => {
      const updatedDays = prev.days.filter((_, index) => index !== dayIndex);
      return { ...prev, days: updatedDays };
    });
    
    // Adjust selected day if needed
    if (selectedDay >= dayIndex && selectedDay > 0) {
      setSelectedDay(selectedDay - 1);
    }
  };
  
  // Update program metadata
  const updateProgram = (field: keyof WorkoutProgram, value: string) => {
    setProgram(prev => ({
      ...prev,
      [field]: value
    }));
  };
  
  // Update day name
  const updateDayName = (dayIndex: number, name: string) => {
    setProgram(prev => {
      const updatedDays = [...prev.days];
      updatedDays[dayIndex] = {
        ...updatedDays[dayIndex],
        name
      };
      return { ...prev, days: updatedDays };
    });
  };
  
  // Toggle exercise expansion
  const toggleExerciseExpansion = (exerciseId: string) => {
    setExpandedExercise(prev => prev === exerciseId ? null : exerciseId);
  };
  
  // Load a template program
  const loadTemplate = (templateId: string) => {
    // In a real app, this would fetch the template data from an API
    alert(`Template ${templateId} would be loaded`);
  };
  
  // Save the program
  const saveProgram = () => {
    // In a real app, this would save to an API
    console.log("Saving program:", program);
    alert("Program saved successfully!");
  };

  return (
    <div className="workout-program-creator bg-gray-900 rounded-xl p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-white text-2xl font-medium">Workout Program Creator</h2>
        
        <div className="flex gap-2">
          <Button variant="outline" onClick={saveProgram}>
            <Save className="mr-2 h-4 w-4" />
            Save Program
          </Button>
          
          <Button>
            <Zap className="mr-2 h-4 w-4" />
            Generate AI Workout
          </Button>
        </div>
      </div>
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="builder">Program Builder</TabsTrigger>
          <TabsTrigger value="templates">Templates</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>
        
        <TabsContent value="builder" className="space-y-6">
          {/* Program Metadata */}
          <Card className="bg-gray-800 border-gray-700">
            <CardContent className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <label className="text-sm text-gray-400 mb-1 block">Program Name</label>
                    <Input 
                      value={program.name} 
                      onChange={(e) => updateProgram('name', e.target.value)} 
                      className="bg-gray-750 border-gray-600"
                    />
                  </div>
                  
                  <div>
                    <label className="text-sm text-gray-400 mb-1 block">Description</label>
                    <Textarea 
                      value={program.description || ''} 
                      onChange={(e) => updateProgram('description', e.target.value)} 
                      className="bg-gray-750 border-gray-600 h-20"
                    />
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm text-gray-400 mb-1 block">Difficulty</label>
                    <Select 
                      value={program.difficulty}
                      onValueChange={(value) => updateProgram('difficulty', value)}
                    >
                      <SelectTrigger className="bg-gray-750 border-gray-600">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Beginner">Beginner</SelectItem>
                        <SelectItem value="Intermediate">Intermediate</SelectItem>
                        <SelectItem value="Advanced">Advanced</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div>
                    <label className="text-sm text-gray-400 mb-1 block">Primary Focus</label>
                    <Select 
                      value={program.focus}
                      onValueChange={(value) => updateProgram('focus', value)}
                    >
                      <SelectTrigger className="bg-gray-750 border-gray-600">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Strength">Strength</SelectItem>
                        <SelectItem value="Hypertrophy">Hypertrophy</SelectItem>
                        <SelectItem value="Endurance">Endurance</SelectItem>
                        <SelectItem value="Power">Power</SelectItem>
                        <SelectItem value="General Fitness">General Fitness</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div>
                    <label className="text-sm text-gray-400 mb-1 block">Duration</label>
                    <Select 
                      value={program.duration}
                      onValueChange={(value) => updateProgram('duration', value)}
                    >
                      <SelectTrigger className="bg-gray-750 border-gray-600">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="4 weeks">4 weeks</SelectItem>
                        <SelectItem value="6 weeks">6 weeks</SelectItem>
                        <SelectItem value="8 weeks">8 weeks</SelectItem>
                        <SelectItem value="12 weeks">12 weeks</SelectItem>
                        <SelectItem value="16 weeks">16 weeks</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="flex items-end">
                    <div className="flex items-center gap-2 text-white">
                      <Calendar className="h-5 w-5 text-gray-400" />
                      <span>{program.days.length} days per week</span>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Day Selection Tabs */}
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center overflow-x-auto pb-2 mb-2 gap-2">
              {program.days.map((day, index) => (
                <button
                  key={day.id}
                  className={`px-4 py-2 rounded whitespace-nowrap flex items-center gap-2 ${
                    selectedDay === index 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-750 text-gray-300 hover:bg-gray-700'
                  }`}
                  onClick={() => setSelectedDay(index)}
                >
                  {day.restDay ? (
                    <RotateCw className="h-4 w-4" />
                  ) : (
                    <Dumbbell className="h-4 w-4" />
                  )}
                  <span className="text-sm font-medium">{day.name}</span>
                </button>
              ))}
              
              <div className="flex gap-1">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button 
                        size="icon" 
                        variant="outline" 
                        onClick={addDay}
                        className="h-8 w-8 rounded-full"
                      >
                        <Plus className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Add Workout Day</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button 
                        size="icon" 
                        variant="outline" 
                        onClick={addRestDay}
                        className="h-8 w-8 rounded-full"
                      >
                        <RotateCw className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Add Rest Day</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
            </div>
            
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Input 
                  value={program.days[selectedDay]?.name || ''}
                  onChange={(e) => updateDayName(selectedDay, e.target.value)}
                  className="bg-gray-750 border-gray-600"
                  placeholder="Day Name"
                />
                
                {program.days.length > 1 && (
                  <Button 
                    variant="destructive" 
                    size="icon"
                    onClick={() => removeDay(selectedDay)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </div>
            
            {/* Rest Day Content */}
            {program.days[selectedDay]?.restDay ? (
              <div className="bg-gray-750 rounded-lg p-6 text-center">
                <RotateCw className="h-16 w-16 text-blue-500 mx-auto mb-4" />
                <h3 className="text-white text-lg font-medium mb-2">Rest Day</h3>
                <p className="text-gray-400 max-w-md mx-auto mb-4">
                  This is a scheduled rest day. Take time to recover, stretch, or engage in light activity like walking or yoga.
                </p>
                <Button variant="outline" onClick={() => {
                  setProgram(prev => {
                    const updatedDays = [...prev.days];
                    updatedDays[selectedDay] = {
                      ...updatedDays[selectedDay],
                      restDay: false
                    };
                    return { ...prev, days: updatedDays };
                  });
                }}>
                  Convert to Workout Day
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Exercise List */}
                {program.days[selectedDay]?.exercises.map((exercise, exerciseIndex) => (
                  <div 
                    key={exercise.id} 
                    className="bg-gray-750 rounded-lg border border-gray-700"
                  >
                    <div 
                      className="p-4 flex items-center justify-between cursor-pointer"
                      onClick={() => toggleExerciseExpansion(exercise.id)}
                    >
                      <div className="flex items-center gap-3">
                        <Badge className="bg-blue-600">
                          {exerciseIndex + 1}
                        </Badge>
                        <h4 className="text-white font-medium">{exercise.name}</h4>
                        <Badge variant="outline">
                          {exercise.sets.length} × {exercise.sets[0].reps}
                        </Badge>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        <Button 
                          variant="destructive" 
                          size="icon"
                          className="h-8 w-8"
                          onClick={(e) => {
                            e.stopPropagation();
                            removeExercise(exerciseIndex);
                          }}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                        
                        <Button variant="ghost" size="icon" className="h-8 w-8">
                          {expandedExercise === exercise.id ? (
                            <ChevronUp className="h-4 w-4" />
                          ) : (
                            <ChevronDown className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                    </div>
                    
                    {expandedExercise === exercise.id && (
                      <div className="border-t border-gray-700 p-4">
                        <div className="mb-4">
                          <div className="grid grid-cols-6 gap-2 text-gray-400 text-sm mb-2">
                            <div className="col-span-1 text-center">Set</div>
                            <div className="col-span-2 text-center">Reps</div>
                            <div className="col-span-2 text-center">Weight</div>
                            <div className="col-span-1"></div>
                          </div>
                          
                          {exercise.sets.map((set, setIndex) => (
                            <div key={set.id} className="grid grid-cols-6 gap-2 mb-2 items-center">
                              <div className="col-span-1 text-center">
                                <Badge variant="outline">{setIndex + 1}</Badge>
                              </div>
                              <div className="col-span-2">
                                <Input 
                                  value={set.reps} 
                                  onChange={(e) => {
                                    setProgram(prev => {
                                      const updatedDays = [...prev.days];
                                      updatedDays[selectedDay].exercises[exerciseIndex].sets[setIndex].reps = e.target.value;
                                      return { ...prev, days: updatedDays };
                                    });
                                  }}
                                  className="bg-gray-800 border-gray-600 h-8 text-center"
                                />
                              </div>
                              <div className="col-span-2">
                                <Input 
                                  value={set.weight} 
                                  onChange={(e) => {
                                    setProgram(prev => {
                                      const updatedDays = [...prev.days];
                                      updatedDays[selectedDay].exercises[exerciseIndex].sets[setIndex].weight = e.target.value;
                                      return { ...prev, days: updatedDays };
                                    });
                                  }}
                                  className="bg-gray-800 border-gray-600 h-8 text-center"
                                />
                              </div>
                              <div className="col-span-1 flex justify-center">
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-8 w-8"
                                  onClick={() => removeSet(exerciseIndex, setIndex)}
                                  disabled={exercise.sets.length <= 1}
                                >
                                  <Trash2 className="h-3 w-3" />
                                </Button>
                              </div>
                            </div>
                          ))}
                        </div>
                        
                        <div className="flex justify-between">
                          <Button variant="outline" size="sm" onClick={() => addSetToExercise(exerciseIndex)}>
                            <Plus className="mr-1 h-3 w-3" />
                            Add Set
                          </Button>
                          
                          <div>
                            <Input 
                              value={exercise.notes || ''} 
                              onChange={(e) => {
                                setProgram(prev => {
                                  const updatedDays = [...prev.days];
                                  updatedDays[selectedDay].exercises[exerciseIndex].notes = e.target.value;
                                  return { ...prev, days: updatedDays };
                                });
                              }}
                              className="bg-gray-800 border-gray-600"
                              placeholder="Exercise notes (optional)"
                            />
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
                
                {/* Exercise Selection */}
                <Card className="bg-gray-800 border-gray-700">
                  <CardContent className="p-4">
                    <div className="mb-4">
                      <Input 
                        value={exerciseSearch} 
                        onChange={(e) => setExerciseSearch(e.target.value)} 
                        className="bg-gray-750 border-gray-600"
                        placeholder="Search exercises..."
                      />
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 max-h-60 overflow-y-auto pr-2">
                      {filteredExercises.map(exercise => (
                        <div 
                          key={exercise.id} 
                          className="bg-gray-750 p-2 rounded-lg border border-gray-700 cursor-pointer hover:bg-gray-700 transition-colors"
                          onClick={() => addExerciseToDay(exercise.id)}
                        >
                          <div className="flex justify-between items-start">
                            <div>
                              <div className="text-white font-medium text-sm">{exercise.name}</div>
                              <div className="text-gray-400 text-xs">{exercise.equipment}</div>
                            </div>
                            <Badge variant="outline" className="text-xs">{exercise.category}</Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        </TabsContent>
        
        <TabsContent value="templates" className="space-y-6">
          <Card className="bg-gray-800 border-gray-700">
            <CardContent className="p-6">
              <h3 className="text-white font-medium mb-4">Workout Templates</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {templatePrograms.map(template => (
                  <div 
                    key={template.id} 
                    className="bg-gray-750 p-4 rounded-lg border border-gray-700 cursor-pointer hover:bg-gray-700 transition-colors"
                    onClick={() => loadTemplate(template.id)}
                  >
                    <div className="flex justify-between mb-2">
                      <h4 className="text-white font-medium">{template.name}</h4>
                      <Badge variant={
                        template.difficulty === 'Beginner' ? 'outline' :
                        template.difficulty === 'Intermediate' ? 'secondary' : 'default'
                      }>
                        {template.difficulty}
                      </Badge>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-2 text-sm mb-3">
                      <div className="flex items-center gap-1 text-gray-400">
                        <Target className="h-3 w-3" />
                        <span>{template.focus}</span>
                      </div>
                      <div className="flex items-center gap-1 text-gray-400">
                        <Calendar className="h-3 w-3" />
                        <span>{template.duration}</span>
                      </div>
                      <div className="flex items-center gap-1 text-gray-400">
                        <Dumbbell className="h-3 w-3" />
                        <span>{template.days}×/week</span>
                      </div>
                    </div>
                    
                    <Button className="w-full" size="sm">
                      <Copy className="mr-2 h-4 w-4" />
                      Use Template
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
          
          <div className="bg-blue-900/20 border border-blue-800 rounded-lg p-4">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 rounded-full bg-blue-800/50 flex items-center justify-center shrink-0">
                <Zap className="h-5 w-5 text-blue-300" />
              </div>
              
              <div>
                <h3 className="text-white font-medium mb-1">Create AI Workout Program</h3>
                <p className="text-gray-300 text-sm mb-3">
                  Let our AI build a personalized workout program based on your goals, equipment, and fitness level.
                </p>
                
                <Button className="bg-blue-600 hover:bg-blue-700">
                  <Zap className="mr-2 h-4 w-4" />
                  Generate AI Program
                </Button>
              </div>
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="analytics" className="space-y-6">
          <Card className="bg-gray-800 border-gray-700">
            <CardContent className="p-6 text-center">
              <Users className="h-16 w-16 text-gray-700 mx-auto mb-4" />
              <h3 className="text-white text-lg font-medium mb-2">Workout Program Analytics</h3>
              <p className="text-gray-400 mb-4 max-w-md mx-auto">
                Analytics are available after you save and begin using a workout program.
                Track progress, adherence, and results once you start following this program.
              </p>
              <Button onClick={saveProgram}>
                <Save className="mr-2 h-4 w-4" />
                Save Program to Access Analytics
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
} 
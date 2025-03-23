'use client';

import { useEffect, useState } from 'react';
import { useParams, notFound } from 'next/navigation';
import Link from 'next/link';
import { 
  ArrowLeft, 
  Calendar, 
  Clock, 
  Dumbbell, 
  Users, 
  Target, 
  BarChart2, 
  CheckCircle2, 
  Award, 
  ChevronRight, 
  Play, 
  Info, 
  Menu, 
  Star, 
  Share2,
  Heart 
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Accordion, 
  AccordionContent, 
  AccordionItem, 
  AccordionTrigger 
} from '@/components/ui/accordion';
import { Progress } from '@/components/ui/progress';

import { 
  allPrograms, 
  menPrograms, 
  womenPrograms, 
  specialtyPrograms 
} from '../data/sample-programs';

import {
  WorkoutCategory,
  DifficultyLevel,
  EquipmentLevel,
  TargetAudience,
  FitnessGoal,
  MuscleGroup,
  ProgressionType,
  WorkoutProgram
} from '../types';

// Helper function to get category icon (same as in list page)
const getCategoryIcon = (category: WorkoutCategory) => {
  switch (category) {
    case WorkoutCategory.STRENGTH:
      return <Dumbbell className="h-5 w-5" />;
    case WorkoutCategory.CARDIO:
      return <Heart className="h-5 w-5" />;
    case WorkoutCategory.HIIT:
      return <BarChart2 className="h-5 w-5" />;
    case WorkoutCategory.FLEXIBILITY:
    case WorkoutCategory.YOGA:
      return <Users className="h-5 w-5" />;
    case WorkoutCategory.FUNCTIONAL:
      return <Target className="h-5 w-5" />;
    case WorkoutCategory.SPORT_SPECIFIC:
      return <Award className="h-5 w-5" />;
    case WorkoutCategory.REHABILITATION:
      return <Heart className="h-5 w-5" />;
    default:
      return <Dumbbell className="h-5 w-5" />;
  }
};

// Format duration text
const formatDuration = (weeks: number, daysPerWeek: number) => {
  return `${weeks} ${weeks === 1 ? 'week' : 'weeks'} • ${daysPerWeek} ${daysPerWeek === 1 ? 'day' : 'days'}/week`;
};

export default function ProgramDetailPage() {
  const params = useParams();
  const [program, setProgram] = useState<WorkoutProgram | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  
  useEffect(() => {
    // Fetch program by ID
    if (params.id) {
      const foundProgram = allPrograms.find(p => p.id === params.id);
      if (foundProgram) {
        setProgram(foundProgram);
      }
      setLoading(false);
    }
  }, [params.id]);
  
  // If program not found
  if (!loading && !program) {
    return notFound();
  }
  
  if (loading || !program) {
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
  
  // Mock data for schedule tab
  const weeklySchedule = Array.from({ length: program.duration.weeks }, (_, weekIndex) => ({
    weekNumber: weekIndex + 1,
    workouts: Array.from({ length: program.duration.daysPerWeek }, (_, dayIndex) => ({
      id: `${program.id}-w${weekIndex + 1}-d${dayIndex + 1}`,
      name: `Day ${dayIndex + 1} - ${['Push', 'Pull', 'Legs', 'Upper Body', 'Lower Body', 'Core', 'Full Body'][dayIndex % 7]} Workout`,
      duration: Math.floor(Math.random() * 30) + 30, // Random duration between 30-60 minutes
      difficulty: ['Moderate', 'Challenging', 'Recovery'][Math.floor(Math.random() * 3)],
      completed: false
    }))
  }));
  
  return (
    <div className="program-detail-page pb-16">
      {/* Program Header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-4">
          <Button
            variant="ghost"
            size="icon"
            asChild
          >
            <Link href="/team/workout-programs">
              <ArrowLeft className="h-5 w-5" />
            </Link>
          </Button>
          <span className="text-sm text-gray-400">
            <Link href="/team/workout-programs" className="hover:text-blue-400">Programs</Link>
            <ChevronRight className="h-4 w-4 inline mx-1" />
            <span>{program.name}</span>
          </span>
        </div>
        
        {program.premium && (
          <Badge className="mb-3 bg-gradient-to-r from-yellow-400 to-amber-600 text-black font-semibold">
            <Star className="h-3 w-3 mr-1" /> Premium Program
          </Badge>
        )}
        
        <h1 className="text-2xl md:text-3xl font-bold text-white mb-2">{program.name}</h1>
        
        <div className="flex flex-wrap items-center gap-3 mb-4">
          <div className="flex items-center text-gray-400">
            {getCategoryIcon(program.category)}
            <span className="ml-1">{program.category}</span>
          </div>
          <div className="w-1 h-1 bg-gray-600 rounded-full"></div>
          <div className="flex items-center text-gray-400">
            <Clock className="h-4 w-4 mr-1" />
            <span>{formatDuration(program.duration.weeks, program.duration.daysPerWeek)}</span>
          </div>
          <div className="w-1 h-1 bg-gray-600 rounded-full"></div>
          <div className="flex items-center text-gray-400">
            <Dumbbell className="h-4 w-4 mr-1" />
            <span>{program.equipment}</span>
          </div>
          <div className="w-1 h-1 bg-gray-600 rounded-full"></div>
          <div className="flex items-center text-gray-400">
            <BarChart2 className="h-4 w-4 mr-1" />
            <span>{program.difficultyLevel}</span>
          </div>
        </div>
        
        <div className="flex items-center gap-3 mb-6">
          <Button size="lg" className="bg-blue-600 hover:bg-blue-700">
            <Play className="h-4 w-4 mr-2" /> Start Program
          </Button>
          <Button variant="outline" size="icon">
            <Heart className="h-5 w-5" />
          </Button>
          <Button variant="outline" size="icon">
            <Share2 className="h-5 w-5" />
          </Button>
        </div>
      </div>
      
      {/* Program Content */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Main Content */}
        <div className="md:col-span-2">
          <Tabs 
            defaultValue="overview" 
            value={activeTab}
            onValueChange={setActiveTab}
            className="w-full"
          >
            <TabsList className="grid grid-cols-3 mb-6 bg-gray-800">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="schedule">Schedule</TabsTrigger>
              <TabsTrigger value="reviews">Reviews</TabsTrigger>
            </TabsList>
            
            <TabsContent value="overview" className="bg-transparent p-0">
              <Card className="bg-gray-800 border-gray-700 mb-6">
                <CardHeader>
                  <CardTitle className="text-lg">Program Description</CardTitle>
                </CardHeader>
                <CardContent className="text-gray-300">
                  <p className="mb-4">{program.longDescription || program.description}</p>
                  
                  <div className="mt-5 space-y-2">
                    <h3 className="text-white font-medium">Recommended For:</h3>
                    <div className="flex flex-wrap gap-2">
                      {program.targetAudience.map((audience, i) => (
                        <Badge key={i} variant="secondary" className="bg-gray-700">
                          <Users className="h-3 w-3 mr-1" /> {audience}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  
                  <div className="mt-5 space-y-2">
                    <h3 className="text-white font-medium">Fitness Goals:</h3>
                    <div className="flex flex-wrap gap-2">
                      {program.fitnessGoals.map((goal, i) => (
                        <Badge key={i} variant="secondary" className="bg-gray-700">
                          <Target className="h-3 w-3 mr-1" /> {goal}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  
                  <div className="mt-5 space-y-2">
                    <h3 className="text-white font-medium">Target Muscle Groups:</h3>
                    <div className="flex flex-wrap gap-2">
                      {program.muscleGroups.map((muscle, i) => (
                        <Badge key={i} variant="secondary" className="bg-gray-700">
                          <Dumbbell className="h-3 w-3 mr-1" /> {muscle}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-gray-800 border-gray-700 mb-6">
                <CardHeader>
                  <CardTitle className="text-lg">Program Features</CardTitle>
                </CardHeader>
                <CardContent className="text-gray-300">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="flex items-start">
                      <div className="bg-gray-700 p-2 rounded-full mr-3">
                        <BarChart2 className="h-5 w-5 text-blue-400" />
                      </div>
                      <div>
                        <h3 className="text-white font-medium">Progression Type</h3>
                        <p className="text-sm text-gray-400">{program.progressionType}</p>
                      </div>
                    </div>
                    
                    <div className="flex items-start">
                      <div className="bg-gray-700 p-2 rounded-full mr-3">
                        <Calendar className="h-5 w-5 text-blue-400" />
                      </div>
                      <div>
                        <h3 className="text-white font-medium">Program Duration</h3>
                        <p className="text-sm text-gray-400">{formatDuration(program.duration.weeks, program.duration.daysPerWeek)}</p>
                      </div>
                    </div>
                    
                    <div className="flex items-start">
                      <div className="bg-gray-700 p-2 rounded-full mr-3">
                        <Dumbbell className="h-5 w-5 text-blue-400" />
                      </div>
                      <div>
                        <h3 className="text-white font-medium">Equipment Needed</h3>
                        <p className="text-sm text-gray-400">{program.equipment}</p>
                      </div>
                    </div>
                    
                    <div className="flex items-start">
                      <div className="bg-gray-700 p-2 rounded-full mr-3">
                        <BarChart2 className="h-5 w-5 text-blue-400" />
                      </div>
                      <div>
                        <h3 className="text-white font-medium">Difficulty Level</h3>
                        <p className="text-sm text-gray-400">{program.difficultyLevel}</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
              
              {program.recommendations && (
                <Card className="bg-gray-800 border-gray-700">
                  <CardHeader>
                    <CardTitle className="text-lg">Recommendations</CardTitle>
                  </CardHeader>
                  <CardContent className="text-gray-300">
                    <Accordion type="single" collapsible className="w-full">
                      {program.recommendations.diet && (
                        <AccordionItem value="diet" className="border-gray-700">
                          <AccordionTrigger className="text-white">Diet Recommendations</AccordionTrigger>
                          <AccordionContent className="text-gray-300">
                            {program.recommendations.diet}
                          </AccordionContent>
                        </AccordionItem>
                      )}
                      
                      {program.recommendations.supplements && (
                        <AccordionItem value="supplements" className="border-gray-700">
                          <AccordionTrigger className="text-white">Supplement Recommendations</AccordionTrigger>
                          <AccordionContent className="text-gray-300">
                            {program.recommendations.supplements}
                          </AccordionContent>
                        </AccordionItem>
                      )}
                      
                      {program.recommendations.recovery && (
                        <AccordionItem value="recovery" className="border-gray-700">
                          <AccordionTrigger className="text-white">Recovery Recommendations</AccordionTrigger>
                          <AccordionContent className="text-gray-300">
                            {program.recommendations.recovery}
                          </AccordionContent>
                        </AccordionItem>
                      )}
                    </Accordion>
                  </CardContent>
                </Card>
              )}
            </TabsContent>
            
            <TabsContent value="schedule" className="bg-transparent p-0">
              <Card className="bg-gray-800 border-gray-700 mb-6">
                <CardHeader>
                  <CardTitle className="text-lg">Program Schedule</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-400 mb-4">
                    {program.duration.weeks} week program with {program.duration.daysPerWeek} workouts per week.
                  </p>
                  
                  <Accordion type="single" collapsible className="w-full">
                    {weeklySchedule.map((week, i) => (
                      <AccordionItem 
                        key={`week-${i+1}`} 
                        value={`week-${i+1}`}
                        className="border-gray-700"
                      >
                        <AccordionTrigger className="text-white">
                          <div className="flex items-center justify-between w-full pr-4">
                            <div className="flex items-center">
                              <span>Week {week.weekNumber}</span>
                              {i === 0 && (
                                <Badge className="ml-3 bg-blue-600 text-white text-xs">Current Week</Badge>
                              )}
                            </div>
                            <div className="text-sm text-gray-400">
                              {week.workouts.filter(w => w.completed).length} / {week.workouts.length} completed
                            </div>
                          </div>
                        </AccordionTrigger>
                        <AccordionContent>
                          <div className="space-y-3 pt-2">
                            {week.workouts.map((workout, j) => (
                              <div 
                                key={workout.id} 
                                className="flex items-center justify-between bg-gray-700 p-3 rounded-md"
                              >
                                <div className="flex items-center">
                                  <div className="w-6 h-6 rounded-full bg-gray-600 flex items-center justify-center mr-3">
                                    {j + 1}
                                  </div>
                                  <div>
                                    <div className="text-white font-medium">{workout.name}</div>
                                    <div className="text-xs text-gray-400">
                                      {workout.duration} min • {workout.difficulty}
                                    </div>
                                  </div>
                                </div>
                                <Button 
                                  variant={workout.completed ? "default" : "outline"} 
                                  size="sm"
                                  className={workout.completed ? "bg-green-600 hover:bg-green-700" : ""}
                                >
                                  {workout.completed ? (
                                    <>
                                      <CheckCircle2 className="h-4 w-4 mr-1" /> Completed
                                    </>
                                  ) : (
                                    <>
                                      <Play className="h-4 w-4 mr-1" /> Start
                                    </>
                                  )}
                                </Button>
                              </div>
                            ))}
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    ))}
                  </Accordion>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="reviews" className="bg-transparent p-0">
              <Card className="bg-gray-800 border-gray-700 mb-6">
                <CardHeader>
                  <CardTitle className="text-lg">Reviews & Ratings</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-col md:flex-row items-start gap-6 mb-6">
                    <div className="flex flex-col items-center">
                      <div className="text-4xl font-bold text-white mb-1">{program.rating?.toFixed(1) || "N/A"}</div>
                      <div className="flex items-center mb-1">
                        {[1, 2, 3, 4, 5].map((_, i) => (
                          <Star 
                            key={i} 
                            className={`h-4 w-4 ${i < Math.floor(program.rating || 0) ? "text-yellow-400 fill-yellow-400" : "text-gray-500"}`} 
                          />
                        ))}
                      </div>
                      <div className="text-sm text-gray-400">{program.reviewCount || 0} reviews</div>
                    </div>
                    
                    <div className="flex-grow">
                      <div className="space-y-2">
                        {[5, 4, 3, 2, 1].map((rating) => {
                          // Calculate mock percentages for this example
                          let percent = 0;
                          if (program.rating) {
                            if (rating === 5) percent = 70;
                            else if (rating === 4) percent = 20;
                            else if (rating === 3) percent = 7;
                            else if (rating === 2) percent = 2;
                            else percent = 1;
                          }
                          
                          return (
                            <div key={rating} className="flex items-center">
                              <div className="flex items-center w-20">
                                <span className="text-gray-400 mr-1">{rating}</span>
                                <Star className="h-3 w-3 text-yellow-400 fill-yellow-400" />
                              </div>
                              <Progress value={percent} className="h-2 flex-grow bg-gray-700" />
                              <span className="text-gray-400 ml-2 w-12 text-right">{percent}%</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-gray-700 p-4 rounded-md text-center">
                    <p className="text-gray-300 mb-3">
                      Full reviews will be available once the program has been released.
                    </p>
                    <Button variant="outline">
                      <Star className="h-4 w-4 mr-1" /> Write a Review
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
        
        {/* Sidebar */}
        <div>
          <Card className="bg-gray-800 border-gray-700 mb-6 sticky top-4">
            <CardHeader>
              <CardTitle className="text-lg">Program Progress</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="mb-4">
                <div className="flex justify-between mb-2">
                  <span className="text-sm text-gray-400">Overall Progress</span>
                  <span className="text-sm font-medium">0%</span>
                </div>
                <Progress value={0} className="h-2 bg-gray-700" />
              </div>
              
              <div className="space-y-3 mb-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center text-gray-300">
                    <Calendar className="h-4 w-4 mr-2 text-gray-400" />
                    <span>Current Week</span>
                  </div>
                  <Badge variant="outline" className="bg-gray-700 text-white">
                    Week 1
                  </Badge>
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center text-gray-300">
                    <CheckCircle2 className="h-4 w-4 mr-2 text-gray-400" />
                    <span>Workouts Completed</span>
                  </div>
                  <Badge variant="outline" className="bg-gray-700 text-white">
                    0 / {program.duration.weeks * program.duration.daysPerWeek}
                  </Badge>
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center text-gray-300">
                    <Clock className="h-4 w-4 mr-2 text-gray-400" />
                    <span>Time Remaining</span>
                  </div>
                  <Badge variant="outline" className="bg-gray-700 text-white">
                    {program.duration.weeks} weeks
                  </Badge>
                </div>
              </div>
              
              <Button className="w-full bg-blue-600 hover:bg-blue-700">
                <Play className="h-4 w-4 mr-2" /> Start Program
              </Button>
            </CardContent>
          </Card>
          
          {program.premium && (
            <Card className="bg-gradient-to-br from-blue-900 to-purple-900 border-0 mb-6">
              <CardContent className="pt-6">
                <div className="flex items-center justify-center mb-3">
                  <Star className="h-6 w-6 text-yellow-400 fill-yellow-400" />
                </div>
                <h3 className="text-center text-white font-bold text-lg mb-2">Premium Program</h3>
                <p className="text-center text-blue-100 text-sm mb-4">
                  This program is part of our premium offerings with advanced features and support.
                </p>
                <Button variant="secondary" className="w-full bg-white text-blue-900 hover:bg-blue-50">
                  Learn About Premium
                </Button>
              </CardContent>
            </Card>
          )}
          
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-lg">Related Programs</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {allPrograms
                .filter(p => 
                  p.id !== program.id && 
                  (p.category === program.category || 
                   p.fitnessGoals.some(g => program.fitnessGoals.includes(g))))
                .slice(0, 3)
                .map(relatedProgram => (
                  <Link 
                    href={`/team/workout-programs/${relatedProgram.id}`} 
                    key={relatedProgram.id}
                    className="block"
                  >
                    <div className="flex items-center p-3 rounded-md bg-gray-700 hover:bg-gray-600 transition-colors">
                      <div className="h-10 w-10 rounded bg-gray-800 flex items-center justify-center mr-3">
                        {getCategoryIcon(relatedProgram.category)}
                      </div>
                      <div className="flex-grow">
                        <div className="text-white font-medium">{relatedProgram.name}</div>
                        <div className="text-xs text-gray-400">
                          {relatedProgram.difficultyLevel} • {formatDuration(relatedProgram.duration.weeks, relatedProgram.duration.daysPerWeek)}
                        </div>
                      </div>
                      <ChevronRight className="h-5 w-5 text-gray-500" />
                    </div>
                  </Link>
                ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
} 
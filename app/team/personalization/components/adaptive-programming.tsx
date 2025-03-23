'use client';

import { useState } from 'react';
import { 
  Zap, 
  ArrowLeft, 
  Calendar, 
  TrendingUp, 
  BarChart3, 
  Clock, 
  CheckCircle2, 
  XCircle, 
  Sliders, 
  ChevronRight, 
  ChevronLeft,
  LineChart,
  BarChart,
  Dumbbell,
  Heart,
  Activity,
  ArrowUpRight,
  ArrowDownRight,
  Repeat,
  ArrowRight
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

interface AdaptiveProgrammingProps {
  onBack?: () => void;
}

export default function AdaptiveProgramming({ onBack }: AdaptiveProgrammingProps) {
  const [activeTab, setActiveTab] = useState<string>('current');
  const [selectedProgram, setSelectedProgram] = useState<string | null>(null);
  const [timeFrame, setTimeFrame] = useState<string>('week');
  
  // Format date
  const formatDate = (date: Date): string => {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    }).format(date);
  };
  
  // Mock data for active program
  const activeProgram = {
    id: 'program-1',
    name: 'Strength & Hypertrophy Focus',
    phase: 'Progressive Overload - Phase 2',
    currentWeek: 6,
    totalWeeks: 12,
    startDate: new Date('2023-07-15'),
    targetEndDate: new Date('2023-10-07'),
    completionRate: 87,
    lastAdapted: new Date(Date.now() - 48 * 60 * 60 * 1000), // 2 days ago
    nextAssessment: new Date(Date.now() + 3 * 24 * 60 * 60 * 1000), // 3 days from now
    adaptationReason: 'Strength metrics improved; Recovery capacity increased',
    adaptationDetails: 'Upper body volume increased by 15%; Rest periods reduced by 15 seconds',
    createdBy: 'AI Program Generator',
    overallAdherenceRate: 89,
    statProgress: {
      strength: 12,
      endurance: 5,
      mobility: 8,
      bodyComposition: 9
    },
    recentWorkoutScores: [85, 92, 88, 94, 90, 91, 87],
    keyLifts: [
      { name: 'Bench Press', start: 90, current: 100, target: 110, unit: 'kg' },
      { name: 'Squat', start: 120, current: 135, target: 150, unit: 'kg' },
      { name: 'Deadlift', start: 140, current: 155, target: 170, unit: 'kg' }
    ]
  };
  
  // Mock data for recent adaptations
  const recentAdaptations = [
    {
      id: 'adapt-1',
      date: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
      type: 'intensity',
      description: 'Increased load in chest exercises',
      details: 'Based on 8% strength improvement and positive recovery metrics',
      impact: 'positive',
      score: 8.5
    },
    {
      id: 'adapt-2',
      date: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
      type: 'frequency',
      description: 'Modified leg training frequency',
      details: 'Adjusted to 2x weekly based on recovery data and scheduling patterns',
      impact: 'positive',
      score: 7.8
    },
    {
      id: 'adapt-3',
      date: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000),
      type: 'exercise',
      description: 'Substituted 2 exercises',
      details: 'Changed to exercises better suited to your equipment and preferences',
      impact: 'neutral',
      score: 6.5
    },
    {
      id: 'adapt-4',
      date: new Date(Date.now() - 21 * 24 * 60 * 60 * 1000),
      type: 'rest',
      description: 'Modified rest periods between sets',
      details: 'Decreased rest for hypertrophy focus based on heart rate recovery data',
      impact: 'positive',
      score: 8.2
    },
    {
      id: 'adapt-5',
      date: new Date(Date.now() - 28 * 24 * 60 * 60 * 1000),
      type: 'volume',
      description: 'Adjusted weekly training volume',
      details: 'Reduced overall volume by 15% to account for work stress and sleep patterns',
      impact: 'positive',
      score: 9.0
    }
  ];
  
  // Mock data for upcoming workouts
  const upcomingWorkouts = [
    {
      id: 'workout-1',
      name: 'Upper Body Power',
      date: new Date(Date.now() + 1 * 24 * 60 * 60 * 1000),
      estimatedDuration: 65,
      intensity: 'High',
      focusAreas: ['Chest', 'Back', 'Shoulders'],
      recentAdaptations: ['Added incline bench press', 'Increased pull-up volume'],
      readiness: 8.5
    },
    {
      id: 'workout-2',
      name: 'Lower Body Strength',
      date: new Date(Date.now() + 3 * 24 * 60 * 60 * 1000),
      estimatedDuration: 70,
      intensity: 'High',
      focusAreas: ['Quads', 'Hamstrings', 'Glutes'],
      recentAdaptations: ['Reduced squat volume by 10%', 'Added single-leg work'],
      readiness: 7.8
    },
    {
      id: 'workout-3',
      name: 'Upper Body Hypertrophy',
      date: new Date(Date.now() + 5 * 24 * 60 * 60 * 1000),
      estimatedDuration: 60,
      intensity: 'Moderate',
      focusAreas: ['Arms', 'Chest', 'Upper Back'],
      recentAdaptations: ['Increased time under tension', 'Added supersets'],
      readiness: 8.2
    },
    {
      id: 'workout-4',
      name: 'Active Recovery',
      date: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
      estimatedDuration: 45,
      intensity: 'Low',
      focusAreas: ['Full Body', 'Mobility', 'Flexibility'],
      recentAdaptations: ['Personalized for recovery needs', 'Added neural mobility work'],
      readiness: 9.3
    }
  ];
  
  // Mock data for adaptation insights
  const adaptationInsights = [
    {
      id: 'insight-1',
      title: 'Volume Tolerance',
      value: 'Above Average',
      description: 'You respond well to higher training volumes, especially for upper body',
      recommendation: 'Gradually increasing chest and back volume may benefit progress'
    },
    {
      id: 'insight-2',
      title: 'Recovery Pattern',
      value: '48-72 hours',
      description: 'Optimal muscle group training frequency is every 48-72 hours',
      recommendation: 'Your program has been adjusted to allow adequate recovery between muscle groups'
    },
    {
      id: 'insight-3',
      title: 'Intensity Response',
      value: 'High',
      description: 'Your strength gains correlate strongly with higher intensity (85%+ 1RM)',
      recommendation: 'Emphasis on heavy compound movements in your strength days'
    },
    {
      id: 'insight-4',
      title: 'Exercise Adherence',
      value: 'Varied by Type',
      description: 'You complete 92% of compound lifts but only 78% of isolation exercises',
      recommendation: 'Program now prioritizes compound movements with strategic isolation work'
    }
  ];
  
  // Get adaptation icon based on type
  const getAdaptationIcon = (type: string) => {
    switch (type) {
      case 'intensity':
        return <TrendingUp className="h-4 w-4 text-blue-400" />;
      case 'frequency':
        return <Calendar className="h-4 w-4 text-purple-400" />;
      case 'exercise':
        return <Dumbbell className="h-4 w-4 text-amber-400" />;
      case 'rest':
        return <Clock className="h-4 w-4 text-green-400" />;
      case 'volume':
        return <BarChart3 className="h-4 w-4 text-red-400" />;
      default:
        return <Sliders className="h-4 w-4 text-gray-400" />;
    }
  };
  
  // Get intensity badge variant
  const getIntensityVariant = (intensity: string) => {
    switch (intensity) {
      case 'High':
        return 'destructive';
      case 'Moderate':
        return 'default';
      case 'Low':
        return 'secondary';
      default:
        return 'outline';
    }
  };
  
  // Get day name from date
  const getDayName = (date: Date): string => {
    return new Intl.DateTimeFormat('en-US', { weekday: 'long' }).format(date);
  };
  
  return (
    <div className="adaptive-programming">
      {/* Header with back button */}
      <div className="flex justify-between items-start mb-8">
        <div>
          <div className="flex items-center gap-2 mb-2">
            {onBack && (
              <Button 
                variant="ghost" 
                size="icon" 
                onClick={onBack}
                className="h-8 w-8"
              >
                <ArrowLeft className="h-4 w-4" />
              </Button>
            )}
            <h1 className="text-3xl font-bold">Adaptive Programming</h1>
          </div>
          <p className="text-muted-foreground">
            Your workouts and programs that evolve based on your performance, feedback, and recovery
          </p>
        </div>
      </div>
      
      {/* Tabs Navigation */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="mb-8">
        <TabsList className="mb-4">
          <TabsTrigger value="current" className="flex items-center gap-1">
            <Zap className="h-4 w-4" />
            Current Program
          </TabsTrigger>
          <TabsTrigger value="workouts" className="flex items-center gap-1">
            <Calendar className="h-4 w-4" />
            Upcoming Workouts
          </TabsTrigger>
          <TabsTrigger value="adaptations" className="flex items-center gap-1">
            <TrendingUp className="h-4 w-4" />
            Recent Adaptations
          </TabsTrigger>
          <TabsTrigger value="insights" className="flex items-center gap-1">
            <LineChart className="h-4 w-4" />
            Adaptation Insights
          </TabsTrigger>
        </TabsList>
        
        {/* Current Program Tab */}
        <TabsContent value="current" className="space-y-6">
          {/* Program Overview Card */}
          <Card>
            <CardHeader className="pb-3">
              <div className="flex justify-between items-start">
                <div>
                  <CardTitle>{activeProgram.name}</CardTitle>
                  <CardDescription>{activeProgram.phase}</CardDescription>
                </div>
                <Badge>Week {activeProgram.currentWeek} of {activeProgram.totalWeeks}</Badge>
              </div>
            </CardHeader>
            <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <div className="flex justify-between mb-4">
                  <div className="text-sm text-muted-foreground">Program Progress</div>
                  <div className="text-sm font-medium">{activeProgram.completionRate}%</div>
                </div>
                <Progress value={activeProgram.completionRate} className="h-2 mb-4" />
                
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <div className="space-y-1">
                    <div className="text-sm text-muted-foreground">Started</div>
                    <div className="font-medium">{formatDate(activeProgram.startDate)}</div>
                  </div>
                  <div className="space-y-1">
                    <div className="text-sm text-muted-foreground">Target Completion</div>
                    <div className="font-medium">{formatDate(activeProgram.targetEndDate)}</div>
                  </div>
                  <div className="space-y-1">
                    <div className="text-sm text-muted-foreground">Last Adapted</div>
                    <div className="font-medium">{formatDate(activeProgram.lastAdapted)}</div>
                  </div>
                  <div className="space-y-1">
                    <div className="text-sm text-muted-foreground">Next Assessment</div>
                    <div className="font-medium">{formatDate(activeProgram.nextAssessment)}</div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="text-sm font-medium">Recent Adaptation:</div>
                  <div className="bg-muted p-3 rounded-md">
                    <div className="font-medium mb-1">{activeProgram.adaptationReason}</div>
                    <div className="text-sm text-muted-foreground">{activeProgram.adaptationDetails}</div>
                  </div>
                </div>
              </div>
              
              <div>
                <div className="mb-4">
                  <div className="text-sm font-medium mb-2">Key Performance Lifts</div>
                  <div className="space-y-3">
                    {activeProgram.keyLifts.map((lift, index) => (
                      <div key={index} className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span>{lift.name}</span>
                          <span className="font-medium">{lift.current} {lift.unit}</span>
                        </div>
                        <div className="relative">
                          <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                            <div className="h-full bg-muted-foreground rounded-full" style={{ width: `${(lift.start / lift.target) * 100}%` }}></div>
                            <div className="h-full bg-primary rounded-full absolute top-0" style={{ width: `${(lift.current / lift.target) * 100}%` }}></div>
                          </div>
                          <div className="flex justify-between text-xs text-muted-foreground mt-1">
                            <span>Start: {lift.start}</span>
                            <span>Target: {lift.target}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div>
                  <div className="text-sm font-medium mb-2">Progress Metrics</div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-muted p-3 rounded-md">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm">Strength</span>
                        <span className="text-sm font-medium">+{activeProgram.statProgress.strength}%</span>
                      </div>
                      <Progress value={activeProgram.statProgress.strength} max={20} className="h-1" />
                    </div>
                    <div className="bg-muted p-3 rounded-md">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm">Endurance</span>
                        <span className="text-sm font-medium">+{activeProgram.statProgress.endurance}%</span>
                      </div>
                      <Progress value={activeProgram.statProgress.endurance} max={20} className="h-1" />
                    </div>
                    <div className="bg-muted p-3 rounded-md">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm">Mobility</span>
                        <span className="text-sm font-medium">+{activeProgram.statProgress.mobility}%</span>
                      </div>
                      <Progress value={activeProgram.statProgress.mobility} max={20} className="h-1" />
                    </div>
                    <div className="bg-muted p-3 rounded-md">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm">Body Comp.</span>
                        <span className="text-sm font-medium">+{activeProgram.statProgress.bodyComposition}%</span>
                      </div>
                      <Progress value={activeProgram.statProgress.bodyComposition} max={20} className="h-1" />
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
            <CardFooter className="justify-between border-t pt-4">
              <div className="flex items-center">
                <div className="bg-muted-foreground/20 p-2 rounded-full mr-2">
                  <Activity className="h-4 w-4" />
                </div>
                <div className="text-sm">
                  <span className="text-muted-foreground">Recent Performance: </span>
                  <span className="font-medium">Excellent</span>
                </div>
              </div>
              <Button>
                View Program Details
                <ChevronRight className="ml-1 h-4 w-4" />
              </Button>
            </CardFooter>
          </Card>
          
          {/* Program Adaptation Visualizer */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Program Evolution</CardTitle>
              <CardDescription>How your program has evolved based on your progress</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex justify-between mb-4">
                <Select value={timeFrame} onValueChange={setTimeFrame}>
                  <SelectTrigger className="w-[150px]">
                    <SelectValue placeholder="Time frame" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="week">Last Week</SelectItem>
                    <SelectItem value="month">Last Month</SelectItem>
                    <SelectItem value="3month">Last 3 Months</SelectItem>
                    <SelectItem value="program">Entire Program</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              {/* Program Evolution Timeline - Just a mock visualization */}
              <div className="mt-4 space-y-6">
                <div className="relative">
                  <div className="h-1 w-full bg-muted rounded-full"></div>
                  <div className="absolute top-1/2 -translate-y-1/2 left-[30%] w-3 h-3 rounded-full bg-blue-500 border-2 border-white"></div>
                  <div className="absolute top-1/2 -translate-y-1/2 left-[50%] w-3 h-3 rounded-full bg-purple-500 border-2 border-white"></div>
                  <div className="absolute top-1/2 -translate-y-1/2 left-[70%] w-3 h-3 rounded-full bg-amber-500 border-2 border-white"></div>
                  <div className="absolute top-1/2 -translate-y-1/2 left-[90%] w-3 h-3 rounded-full bg-green-500 border-2 border-white"></div>
                </div>
                
                <div className="grid grid-cols-4 gap-4">
                  <div className="bg-blue-500/10 rounded-lg p-3 border border-blue-500/20">
                    <div className="text-sm font-medium text-blue-500 mb-1">Initial Program</div>
                    <div className="text-xs">Baseline assessments used to create customized plan</div>
                  </div>
                  <div className="bg-purple-500/10 rounded-lg p-3 border border-purple-500/20">
                    <div className="text-sm font-medium text-purple-500 mb-1">First Adaptation</div>
                    <div className="text-xs">Volume adjusted based on recovery capacity</div>
                  </div>
                  <div className="bg-amber-500/10 rounded-lg p-3 border border-amber-500/20">
                    <div className="text-sm font-medium text-amber-500 mb-1">Major Progression</div>
                    <div className="text-xs">Strength improvements triggered intensity increase</div>
                  </div>
                  <div className="bg-green-500/10 rounded-lg p-3 border border-green-500/20">
                    <div className="text-sm font-medium text-green-500 mb-1">Current Phase</div>
                    <div className="text-xs">Balanced approach with strategic deload planned</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Recent Performance */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Recent Workout Performance</CardTitle>
              <CardDescription>How well you've scored on recent workouts</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[150px] relative mt-4 mb-6">
                {/* Mock bar chart */}
                <div className="flex h-full items-end justify-between">
                  {activeProgram.recentWorkoutScores.map((score, index) => (
                    <div key={index} className="w-10 mx-auto">
                      <div 
                        className="bg-gradient-to-t from-indigo-600 to-blue-400 rounded-t-sm"
                        style={{ height: `${(score / 100) * 120}px` }}
                      ></div>
                    </div>
                  ))}
                </div>
                {/* X-axis labels */}
                <div className="flex justify-between mt-2">
                  {activeProgram.recentWorkoutScores.map((_, index) => (
                    <div key={index} className="w-10 text-center text-xs text-muted-foreground">
                      {index + 1}
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="flex justify-between items-center">
                <div className="flex items-center">
                  <ArrowUpRight className="h-4 w-4 text-green-500 mr-1" />
                  <span className="text-sm">Trending upward over past 7 workouts</span>
                </div>
                <Button variant="outline" size="sm">View Full History</Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* Upcoming Workouts Tab */}
        <TabsContent value="workouts" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Upcoming Workouts</CardTitle>
              <CardDescription>
                Your next workouts, adapted for optimal progress
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {upcomingWorkouts.map((workout) => (
                  <div 
                    key={workout.id}
                    className="border rounded-lg overflow-hidden"
                  >
                    <div className="flex items-center bg-muted p-3">
                      <div className="bg-background rounded-full p-2 mr-3">
                        <Calendar className="h-4 w-4" />
                      </div>
                      <div className="flex-1">
                        <div className="font-medium">{workout.name}</div>
                        <div className="text-sm text-muted-foreground">
                          {getDayName(workout.date)}, {formatDate(workout.date)}
                        </div>
                      </div>
                      <Badge variant={getIntensityVariant(workout.intensity)}>
                        {workout.intensity} Intensity
                      </Badge>
                    </div>
                    
                    <div className="p-3">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                          <div className="text-sm text-muted-foreground mb-1">Focus Areas</div>
                          <div className="flex flex-wrap gap-1">
                            {workout.focusAreas.map((area, index) => (
                              <Badge key={index} variant="outline">{area}</Badge>
                            ))}
                          </div>
                        </div>
                        
                        <div>
                          <div className="text-sm text-muted-foreground mb-1">Recent Adaptations</div>
                          <ul className="text-sm space-y-1">
                            {workout.recentAdaptations.map((adaptation, index) => (
                              <li key={index} className="flex items-center">
                                <ArrowRight className="h-3 w-3 text-blue-400 mr-1 flex-shrink-0" />
                                <span>{adaptation}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                        
                        <div>
                          <div className="text-sm text-muted-foreground mb-1">Readiness Score</div>
                          <div className="flex items-center">
                            <div className="w-full mr-2">
                              <Progress value={workout.readiness * 10} className="h-2" />
                            </div>
                            <span className="font-medium">{workout.readiness}</span>
                          </div>
                          <div className="text-xs text-muted-foreground mt-1">
                            Based on your recovery metrics and previous performance
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex justify-end mt-3">
                        <Button>
                          View Workout
                          <ChevronRight className="ml-1 h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* Recent Adaptations Tab */}
        <TabsContent value="adaptations" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Recent Program Adaptations</CardTitle>
              <CardDescription>
                How your program has evolved based on your performance and feedback
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {recentAdaptations.map((adaptation) => (
                <div 
                  key={adaptation.id}
                  className="border rounded-lg p-4"
                >
                  <div className="flex justify-between items-start mb-3">
                    <div className="flex items-center">
                      <div className="bg-muted p-2 rounded-full mr-3">
                        {getAdaptationIcon(adaptation.type)}
                      </div>
                      <div>
                        <div className="font-medium">{adaptation.description}</div>
                        <div className="text-sm text-muted-foreground">
                          {formatDate(adaptation.date)}
                        </div>
                      </div>
                    </div>
                    <Badge variant={adaptation.impact === 'positive' ? 'default' : adaptation.impact === 'neutral' ? 'secondary' : 'destructive'}>
                      {adaptation.impact === 'positive' ? (
                        <span className="flex items-center">
                          <ArrowUpRight className="h-3 w-3 mr-1" />
                          Positive Impact
                        </span>
                      ) : adaptation.impact === 'neutral' ? (
                        <span className="flex items-center">
                          <Repeat className="h-3 w-3 mr-1" />
                          Neutral Impact
                        </span>
                      ) : (
                        <span className="flex items-center">
                          <ArrowDownRight className="h-3 w-3 mr-1" />
                          Negative Impact
                        </span>
                      )}
                    </Badge>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <div className="text-sm text-muted-foreground mb-1">Adaptation Reasoning</div>
                      <div className="text-sm">{adaptation.details}</div>
                    </div>
                    
                    <div>
                      <div className="text-sm text-muted-foreground mb-1">Effectiveness Score</div>
                      <div className="flex items-center">
                        <div className="w-full mr-2">
                          <Progress value={adaptation.score * 10} className="h-2" />
                        </div>
                        <span className="font-medium">{adaptation.score.toFixed(1)}</span>
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        Based on performance improvement and your feedback
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </CardContent>
            <CardFooter className="border-t pt-4">
              <Button variant="outline" className="w-full">
                View All Adaptations History
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>
        
        {/* Adaptation Insights Tab */}
        <TabsContent value="insights" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Personalized Adaptation Insights</CardTitle>
              <CardDescription>
                How your body responds to different training variables
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {adaptationInsights.map((insight) => (
                  <div key={insight.id} className="border rounded-lg p-4">
                    <div className="flex justify-between items-start mb-3">
                      <h3 className="font-medium">{insight.title}</h3>
                      <Badge variant="outline">{insight.value}</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-3">
                      {insight.description}
                    </p>
                    <div className="bg-muted rounded-md p-3">
                      <div className="text-xs text-muted-foreground mb-1">Recommendation</div>
                      <div className="text-sm font-medium">{insight.recommendation}</div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
            <CardFooter className="border-t pt-4 flex justify-between">
              <Button variant="outline">
                Advanced Insights
              </Button>
              <Button>
                Apply to My Program
              </Button>
            </CardFooter>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Automated Adaptation System</CardTitle>
              <CardDescription>
                How your program adapts to your training responses
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="bg-muted rounded-lg p-4">
                  <h3 className="font-medium mb-2">Continuous Optimization</h3>
                  <p className="text-sm">
                    Our AI system analyzes 42+ variables from your workouts, recovery metrics, and feedback to 
                    continuously optimize your training program for maximum progress toward your goals.
                  </p>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="border rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="bg-blue-500/20 p-1.5 rounded text-blue-500">
                        <BarChart className="h-4 w-4" />
                      </div>
                      <div className="font-medium">Performance</div>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Tracks strength, endurance, and technical execution metrics
                    </div>
                  </div>
                  
                  <div className="border rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="bg-green-500/20 p-1.5 rounded text-green-500">
                        <Heart className="h-4 w-4" />
                      </div>
                      <div className="font-medium">Recovery</div>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Monitors sleep quality, HRV, soreness, and fatigue levels
                    </div>
                  </div>
                  
                  <div className="border rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="bg-amber-500/20 p-1.5 rounded text-amber-500">
                        <Activity className="h-4 w-4" />
                      </div>
                      <div className="font-medium">External Factors</div>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Accounts for stress, schedule changes, and life events
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
} 
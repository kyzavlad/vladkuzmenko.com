'use client';

import { useState } from 'react';
import Link from 'next/link';
import { 
  ArrowLeft, 
  Calendar, 
  BarChart, 
  Activity, 
  Dumbbell, 
  Trophy,
  Clock,
  Flame,
  ChevronRight,
  Heart,
  List,
  TrendingUp,
  Calendar as CalendarIcon,
  Star
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';

// Mock data
const currentWeek = {
  number: 12,
  completedWorkouts: 3,
  totalWorkouts: 5,
  percentComplete: 60,
  stats: {
    totalVolume: '24,750 lbs',
    averageIntensity: '72%',
    totalTime: '3h 45m',
    caloriesBurned: 1840,
  }
};

const recentWorkouts = [
  {
    id: 'w1',
    date: '2023-07-15',
    name: 'Upper Body Power',
    program: 'Iron Body: Maximum Strength',
    duration: '58m',
    volume: '12,350 lbs',
    performance: 'good',
    completed: true
  },
  {
    id: 'w2',
    date: '2023-07-13',
    name: 'Lower Body Strength',
    program: 'Iron Body: Maximum Strength',
    duration: '62m',
    volume: '16,400 lbs',
    performance: 'great',
    completed: true
  },
  {
    id: 'w3',
    date: '2023-07-10',
    name: 'Push Hypertrophy',
    program: 'Iron Body: Maximum Strength',
    duration: '55m',
    volume: '9,200 lbs',
    performance: 'average',
    completed: true
  },
  {
    id: 'w4',
    date: '2023-07-08',
    name: 'Pull Day',
    program: 'Iron Body: Maximum Strength',
    duration: '50m',
    volume: '8,800 lbs',
    performance: 'good',
    completed: true
  },
];

const upcomingWorkouts = [
  {
    id: 'uw1',
    date: '2023-07-17',
    name: 'Legs & Core',
    program: 'Iron Body: Maximum Strength',
    estimatedDuration: '60m',
    mainExercises: ['Squat', 'Romanian Deadlift', 'Leg Press']
  },
  {
    id: 'uw2',
    date: '2023-07-19',
    name: 'Upper Body Volume',
    program: 'Iron Body: Maximum Strength',
    estimatedDuration: '55m',
    mainExercises: ['Bench Press', 'Rows', 'Shoulder Press']
  }
];

const personalRecords = [
  {
    exercise: 'Back Squat',
    weight: '315 lbs',
    date: '2023-07-01',
    isRecent: true
  },
  {
    exercise: 'Bench Press',
    weight: '225 lbs',
    date: '2023-06-28',
    isRecent: true
  },
  {
    exercise: 'Deadlift',
    weight: '365 lbs',
    date: '2023-06-15',
    isRecent: false
  },
  {
    exercise: 'Overhead Press',
    weight: '145 lbs',
    date: '2023-07-10',
    isRecent: true
  },
  {
    exercise: 'Pull-up',
    weight: 'BW+45 lbs',
    date: '2023-06-22',
    isRecent: false
  }
];

const bodyMeasurements = {
  current: {
    weight: '180 lbs',
    bodyFat: '15%',
    chest: '42 in',
    arms: '15 in',
    waist: '32 in',
    thighs: '24 in'
  },
  changeFromLastMonth: {
    weight: -2,
    bodyFat: -0.5,
    chest: 0.5,
    arms: 0.25,
    waist: -0.75,
    thighs: 0.25
  }
};

const monthlyActivity = {
  completed: 16,
  skipped: 2,
  totalPlanned: 20,
  streak: 8
};

export default function WorkoutTracking() {
  const [activeTab, setActiveTab] = useState('overview');
  
  return (
    <div className="workout-tracking-page">
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
          <h1 className="text-2xl font-bold text-white">Workout Tracking</h1>
        </div>
      </div>
      
      {/* Tabs */}
      <Tabs 
        defaultValue="overview" 
        value={activeTab}
        onValueChange={setActiveTab}
        className="w-full mb-6"
      >
        <TabsList className="grid grid-cols-3 mb-6 bg-gray-800">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
          <TabsTrigger value="progress">Progress</TabsTrigger>
        </TabsList>
        
        {/* Overview Tab */}
        <TabsContent value="overview" className="bg-transparent p-0">
          {/* Current Week Status */}
          <Card className="bg-gray-800 border-gray-700 mb-6">
            <CardHeader>
              <CardTitle className="text-lg">Current Program Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col md:flex-row gap-6 mb-6">
                <div className="flex-grow">
                  <div className="flex justify-between items-center mb-2">
                    <div className="text-sm text-gray-400">Week {currentWeek.number} Progress</div>
                    <div className="text-sm font-medium">{currentWeek.percentComplete}%</div>
                  </div>
                  <Progress value={currentWeek.percentComplete} className="h-2 mb-3 bg-gray-700" />
                  
                  <div className="grid grid-cols-2 gap-3">
                    <div className="flex items-center">
                      <Badge variant="outline" className="mr-2 bg-gray-700 text-white">
                        {currentWeek.completedWorkouts}/{currentWeek.totalWorkouts}
                      </Badge>
                      <span className="text-gray-400 text-sm">Workouts</span>
                    </div>
                    
                    <div className="flex items-center">
                      <Badge variant="outline" className="mr-2 bg-blue-900 text-blue-200">
                        {currentWeek.stats.totalTime}
                      </Badge>
                      <span className="text-gray-400 text-sm">Total Time</span>
                    </div>
                  </div>
                </div>
                
                <div className="w-full md:w-auto">
                  <Button className="bg-blue-600 hover:bg-blue-700 w-full">
                    Resume Program
                  </Button>
                </div>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="flex items-center text-gray-400 text-xs mb-1">
                    <Dumbbell className="h-3 w-3 mr-1" />
                    <span>Volume</span>
                  </div>
                  <div className="text-white font-medium">{currentWeek.stats.totalVolume}</div>
                </div>
                
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="flex items-center text-gray-400 text-xs mb-1">
                    <Activity className="h-3 w-3 mr-1" />
                    <span>Intensity</span>
                  </div>
                  <div className="text-white font-medium">{currentWeek.stats.averageIntensity}</div>
                </div>
                
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="flex items-center text-gray-400 text-xs mb-1">
                    <Clock className="h-3 w-3 mr-1" />
                    <span>Duration</span>
                  </div>
                  <div className="text-white font-medium">{currentWeek.stats.totalTime}</div>
                </div>
                
                <div className="bg-gray-700 rounded-lg p-3">
                  <div className="flex items-center text-gray-400 text-xs mb-1">
                    <Flame className="h-3 w-3 mr-1" />
                    <span>Calories</span>
                  </div>
                  <div className="text-white font-medium">{currentWeek.stats.caloriesBurned}</div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Upcoming Workouts */}
          <Card className="bg-gray-800 border-gray-700 mb-6">
            <CardHeader>
              <CardTitle className="text-lg">Upcoming Workouts</CardTitle>
            </CardHeader>
            <CardContent>
              {upcomingWorkouts.length ? (
                <div className="space-y-3">
                  {upcomingWorkouts.map(workout => (
                    <Link 
                      key={workout.id} 
                      href={`/team/workout-programs/${workout.program}/workout`}
                      className="block"
                    >
                      <div className="flex items-center justify-between p-3 rounded-md bg-gray-700 hover:bg-gray-600 transition-colors">
                        <div className="flex items-center">
                          <div className="bg-gray-800 rounded-lg p-2 mr-3">
                            <Calendar className="h-5 w-5 text-blue-400" />
                          </div>
                          <div>
                            <div className="text-white font-medium">{workout.name}</div>
                            <div className="text-xs text-gray-400">
                              {new Date(workout.date).toLocaleDateString('en-US', { 
                                month: 'short', 
                                day: 'numeric', 
                                weekday: 'short' 
                              })} • {workout.estimatedDuration}
                            </div>
                          </div>
                        </div>
                        <Button variant="ghost" size="sm" className="text-blue-400">
                          Start <ChevronRight className="h-4 w-4 ml-1" />
                        </Button>
                      </div>
                    </Link>
                  ))}
                </div>
              ) : (
                <div className="text-center text-gray-400 py-4">
                  No upcoming workouts scheduled
                </div>
              )}
            </CardContent>
          </Card>
          
          {/* Quick Stats */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            {/* Recent Activity */}
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-lg">This Month's Activity</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="bg-gray-700 rounded-lg p-3 text-center">
                    <div className="text-3xl font-bold text-white">{monthlyActivity.completed}</div>
                    <div className="text-gray-400 text-sm">Workouts Completed</div>
                  </div>
                  
                  <div className="bg-gray-700 rounded-lg p-3 text-center">
                    <div className="text-3xl font-bold text-white">{monthlyActivity.streak}</div>
                    <div className="text-gray-400 text-sm">Day Streak</div>
                  </div>
                </div>
                
                <div className="flex items-center justify-between bg-gray-700 rounded-lg p-3">
                  <div>
                    <div className="text-gray-400 text-xs mb-1">Adherence Rate</div>
                    <div className="text-white font-medium">
                      {Math.round((monthlyActivity.completed / monthlyActivity.totalPlanned) * 100)}%
                    </div>
                  </div>
                  <Progress 
                    value={(monthlyActivity.completed / monthlyActivity.totalPlanned) * 100} 
                    className="h-2 w-24 bg-gray-600" 
                  />
                </div>
              </CardContent>
            </Card>
            
            {/* Personal Records */}
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-lg">Personal Records</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {personalRecords.slice(0, 3).map((record, i) => (
                    <div key={i} className="flex justify-between items-center p-2 rounded-md bg-gray-700">
                      <div>
                        <div className="text-white font-medium">{record.exercise}</div>
                        <div className="text-xs text-gray-400">{record.date}</div>
                      </div>
                      <div className="flex items-center">
                        <div className="text-white font-medium mr-2">{record.weight}</div>
                        {record.isRecent && (
                          <Badge className="bg-green-600">New</Badge>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
                
                <Button variant="ghost" className="w-full mt-3 text-blue-400">
                  View All Records
                </Button>
              </CardContent>
            </Card>
          </div>
          
          {/* Body Measurements */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-lg">Body Measurements</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {Object.entries(bodyMeasurements.current).map(([key, value], i) => {
                  const change = bodyMeasurements.changeFromLastMonth[key as keyof typeof bodyMeasurements.changeFromLastMonth];
                  
                  return (
                    <div key={i} className="bg-gray-700 rounded-lg p-3">
                      <div className="text-gray-400 text-xs capitalize mb-1">{key}</div>
                      <div className="flex items-center">
                        <div className="text-white font-medium mr-2">{value}</div>
                        {change !== 0 && (
                          <Badge 
                            className={
                              (key === 'weight' || key === 'bodyFat' || key === 'waist') 
                                ? (change < 0 ? 'bg-green-600' : 'bg-red-600')
                                : (change > 0 ? 'bg-green-600' : 'bg-red-600')
                            }
                          >
                            {change > 0 ? '+' : ''}{change}
                          </Badge>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
              
              <Button variant="outline" className="w-full mt-4">
                Update Measurements
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* History Tab */}
        <TabsContent value="history" className="bg-transparent p-0">
          <Card className="bg-gray-800 border-gray-700 mb-6">
            <CardHeader>
              <CardTitle className="text-lg">Recent Workouts</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentWorkouts.map(workout => (
                  <div 
                    key={workout.id} 
                    className="p-4 rounded-lg border border-gray-700 bg-gray-800"
                  >
                    <div className="flex justify-between items-start mb-3">
                      <div>
                        <div className="text-white font-medium text-lg">{workout.name}</div>
                        <div className="text-sm text-gray-400">
                          {new Date(workout.date).toLocaleDateString('en-US', { 
                            month: 'short', 
                            day: 'numeric', 
                            year: 'numeric' 
                          })}
                        </div>
                      </div>
                      <Badge 
                        className={
                          workout.performance === 'great' 
                            ? 'bg-green-600' 
                            : workout.performance === 'good' 
                              ? 'bg-blue-600' 
                              : 'bg-amber-600'
                        }
                      >
                        {workout.performance === 'great' 
                          ? 'Great Performance' 
                          : workout.performance === 'good' 
                            ? 'Good Performance' 
                            : 'Average Performance'}
                      </Badge>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3">
                      <div className="flex items-center">
                        <Clock className="h-4 w-4 text-gray-500 mr-1" />
                        <span className="text-gray-200 text-sm">{workout.duration}</span>
                      </div>
                      
                      <div className="flex items-center">
                        <Dumbbell className="h-4 w-4 text-gray-500 mr-1" />
                        <span className="text-gray-200 text-sm">{workout.volume}</span>
                      </div>
                      
                      <div className="flex items-center col-span-2">
                        <List className="h-4 w-4 text-gray-500 mr-1" />
                        <span className="text-gray-200 text-sm">{workout.program}</span>
                      </div>
                    </div>
                    
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm" asChild>
                        <Link href={`/team/workout-programs/tracking/workouts/${workout.id}`}>
                          View Details
                        </Link>
                      </Button>
                      
                      <Button variant="ghost" size="sm">
                        <Star className="h-4 w-4 mr-1" /> Rate
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
              
              <Button variant="outline" className="w-full mt-4">
                View All History
              </Button>
            </CardContent>
          </Card>
          
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-lg">Monthly Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                <div className="bg-gray-700 rounded-lg p-3 text-center">
                  <div className="text-gray-400 text-xs mb-1">Workouts</div>
                  <div className="text-2xl font-bold text-white">18</div>
                </div>
                
                <div className="bg-gray-700 rounded-lg p-3 text-center">
                  <div className="text-gray-400 text-xs mb-1">Total Time</div>
                  <div className="text-2xl font-bold text-white">16h 30m</div>
                </div>
                
                <div className="bg-gray-700 rounded-lg p-3 text-center">
                  <div className="text-gray-400 text-xs mb-1">Volume</div>
                  <div className="text-2xl font-bold text-white">182K lbs</div>
                </div>
                
                <div className="bg-gray-700 rounded-lg p-3 text-center">
                  <div className="text-gray-400 text-xs mb-1">Calories</div>
                  <div className="text-2xl font-bold text-white">12,600</div>
                </div>
              </div>
              
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="text-center text-gray-300 py-10">
                  [Activity Calendar Chart Placeholder]
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* Progress Tab */}
        <TabsContent value="progress" className="bg-transparent p-0">
          <Card className="bg-gray-800 border-gray-700 mb-6">
            <CardHeader>
              <CardTitle className="text-lg">Strength Progress</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-gray-700 p-4 rounded-lg mb-4">
                <div className="text-center text-gray-300 py-10">
                  [Strength Progress Chart Placeholder]
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {personalRecords.map((record, i) => (
                  <div key={i} className="flex justify-between items-center p-3 rounded-md bg-gray-700">
                    <div>
                      <div className="text-white font-medium">{record.exercise}</div>
                      <div className="text-xs text-gray-400">{record.date}</div>
                    </div>
                    <div className="flex items-center">
                      <div className="text-white font-medium mr-2">{record.weight}</div>
                      {record.isRecent && (
                        <Badge className="bg-green-600">New</Badge>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-gray-800 border-gray-700 mb-6">
            <CardHeader>
              <CardTitle className="text-lg">Body Composition</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-4">
                <div className="bg-gray-700 p-4 rounded-lg">
                  <div className="text-white font-medium mb-2">Weight Trend</div>
                  <div className="text-center text-gray-300 py-8">
                    [Weight Chart Placeholder]
                  </div>
                </div>
                
                <div className="bg-gray-700 p-4 rounded-lg">
                  <div className="text-white font-medium mb-2">Body Fat % Trend</div>
                  <div className="text-center text-gray-300 py-8">
                    [Body Fat Chart Placeholder]
                  </div>
                </div>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {Object.entries(bodyMeasurements.current).map(([key, value], i) => {
                  const change = bodyMeasurements.changeFromLastMonth[key as keyof typeof bodyMeasurements.changeFromLastMonth];
                  
                  return (
                    <div key={i} className="flex justify-between items-center p-3 rounded-md bg-gray-700">
                      <div className="text-gray-400 capitalize">{key}</div>
                      <div className="flex items-center">
                        <div className="text-white font-medium mr-2">{value}</div>
                        {change !== 0 && (
                          <Badge 
                            className={
                              (key === 'weight' || key === 'bodyFat' || key === 'waist') 
                                ? (change < 0 ? 'bg-green-600' : 'bg-red-600')
                                : (change > 0 ? 'bg-green-600' : 'bg-red-600')
                            }
                          >
                            {change > 0 ? '+' : ''}{change}
                          </Badge>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-lg">Workout Volume</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-gray-700 p-4 rounded-lg mb-4">
                <div className="text-center text-gray-300 py-10">
                  [Volume Chart Placeholder]
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-700 rounded-lg p-3 text-center">
                  <div className="text-gray-400 text-xs mb-1">This Week</div>
                  <div className="text-2xl font-bold text-white">24,750 lbs</div>
                </div>
                
                <div className="bg-gray-700 rounded-lg p-3 text-center">
                  <div className="text-gray-400 text-xs mb-1">vs Last Week</div>
                  <div className="flex items-center justify-center">
                    <TrendingUp className="h-4 w-4 text-green-500 mr-1" />
                    <span className="text-green-500 font-medium">+8.3%</span>
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
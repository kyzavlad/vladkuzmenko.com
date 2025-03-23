'use client';

import { useState, useEffect } from 'react';
import { 
  Activity, 
  Trophy, 
  Utensils, 
  Heart, 
  Weight, 
  Flame, 
  BarChart3,
  Dumbbell,
  Moon,
  ArrowRight,
  Check,
  Plus,
  Clock,
  Zap,
  CalendarClock,
  Footprints
} from 'lucide-react';
import { format } from 'date-fns';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

interface DailyGoal {
  id: string;
  name: string;
  current: number;
  target: number;
  unit: string;
  category: 'activity' | 'nutrition' | 'sleep' | 'recovery';
  icon: keyof typeof iconMap;
}

interface ActivitySummary {
  date: Date;
  calories: {
    burned: number;
    target: number;
    BMR: number;
  };
  nutrition: {
    consumed: number;
    protein: number;
    carbs: number;
    fat: number;
    water: number;
  };
  activity: {
    steps: number;
    activeMinutes: number;
    standHours: number;
    workouts: {
      completed: boolean;
      name?: string;
      duration?: number;
      time?: string;
    }[];
  };
  sleep: {
    duration: number;
    quality: number;
    debtHours: number;
    bedtime: string;
    wakeup: string;
  };
  recovery: {
    score: number;
    strain: number;
    readiness: 'high' | 'medium' | 'low';
    recommendation: string;
  };
  completedTasks: string[];
  streak: number;
}

interface DailyActivitySummaryProps {
  date?: Date;
  onNavigate?: (direction: 'prev' | 'next') => void;
}

// Icon mapping for different metrics
const iconMap = {
  steps: Footprints,
  calories: Flame,
  activeMinutes: Activity,
  protein: Weight,
  carbs: Utensils,
  fat: Utensils,
  water: Utensils,
  sleep: Moon,
  recovery: Heart,
  workout: Dumbbell
};

export default function DailyActivitySummary({ 
  date = new Date(), 
  onNavigate 
}: DailyActivitySummaryProps) {
  const [activeTab, setActiveTab] = useState<string>('summary');
  
  // Mock data for the daily summary
  const [summary, setSummary] = useState<ActivitySummary>({
    date: new Date(),
    calories: {
      burned: 2350,
      target: 2500,
      BMR: 1700
    },
    nutrition: {
      consumed: 2100,
      protein: 145,
      carbs: 205,
      fat: 70,
      water: 2.5
    },
    activity: {
      steps: 8750,
      activeMinutes: 45,
      standHours: 10,
      workouts: [
        {
          completed: true,
          name: 'Morning Strength Training',
          duration: 45,
          time: '06:30'
        },
        {
          completed: false,
          name: 'Evening Cardio',
          duration: 30,
          time: '18:00'
        }
      ]
    },
    sleep: {
      duration: 7.2,
      quality: 83,
      debtHours: 0.8,
      bedtime: '23:15',
      wakeup: '06:25'
    },
    recovery: {
      score: 85,
      strain: 65,
      readiness: 'high',
      recommendation: 'You're well recovered and ready for a challenging workout today.'
    },
    completedTasks: ['Tracked all meals', 'Completed morning workout', 'Reached protein target'],
    streak: 7
  });
  
  // Generate daily goals array from summary data
  const dailyGoals: DailyGoal[] = [
    {
      id: 'calories',
      name: 'Calories Burned',
      current: summary.calories.burned,
      target: summary.calories.target,
      unit: 'kcal',
      category: 'activity',
      icon: 'calories'
    },
    {
      id: 'steps',
      name: 'Steps',
      current: summary.activity.steps,
      target: 10000,
      unit: 'steps',
      category: 'activity',
      icon: 'steps'
    },
    {
      id: 'active-minutes',
      name: 'Active Minutes',
      current: summary.activity.activeMinutes,
      target: 60,
      unit: 'min',
      category: 'activity',
      icon: 'activeMinutes'
    },
    {
      id: 'protein',
      name: 'Protein',
      current: summary.nutrition.protein,
      target: 150,
      unit: 'g',
      category: 'nutrition',
      icon: 'protein'
    },
    {
      id: 'water',
      name: 'Water',
      current: summary.nutrition.water,
      target: 3,
      unit: 'L',
      category: 'nutrition',
      icon: 'water'
    },
    {
      id: 'sleep',
      name: 'Sleep',
      current: summary.sleep.duration,
      target: 8,
      unit: 'hrs',
      category: 'sleep',
      icon: 'sleep'
    }
  ];
  
  // Calculate calorie balance
  const calorieBalance = summary.calories.burned - summary.nutrition.consumed;
  
  // Calculate progress percentages
  const calculateProgress = (current: number, target: number) => {
    const progress = (current / target) * 100;
    return Math.min(progress, 100); // Cap at 100%
  };

  return (
    <div className="daily-activity-summary">
      {/* Header with date navigation */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center gap-3">
          <CalendarClock className="h-5 w-5 text-blue-500" />
          <h2 className="text-white text-xl font-medium">
            {format(date, 'EEEE, MMMM d')}
          </h2>
          {summary.streak > 0 && (
            <Badge className="bg-orange-900 text-orange-300">
              <Flame className="mr-1 h-3 w-3" /> {summary.streak} Day Streak
            </Badge>
          )}
        </div>
        
        <div className="flex gap-2">
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => onNavigate && onNavigate('prev')}
          >
            Yesterday
          </Button>
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => onNavigate && onNavigate('next')}
            disabled={date.toDateString() === new Date().toDateString()}
          >
            Tomorrow
          </Button>
        </div>
      </div>
      
      {/* Daily Summary Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        {/* Activity & Calories */}
        <Card className="bg-gradient-to-br from-blue-900/40 to-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex justify-between items-start mb-3">
              <h3 className="text-white font-medium flex items-center gap-2">
                <Activity className="h-5 w-5 text-blue-400" />
                Activity
              </h3>
              <Badge variant="outline">{summary.activity.activeMinutes} min</Badge>
            </div>
            
            <div className="grid grid-cols-2 gap-4 mb-3">
              <div>
                <div className="text-gray-400 text-sm mb-1">Steps</div>
                <div className="text-white text-2xl font-light">{summary.activity.steps.toLocaleString()}</div>
                <Progress 
                  value={calculateProgress(summary.activity.steps, 10000)} 
                  className="h-1.5 mt-1" 
                />
              </div>
              
              <div>
                <div className="text-gray-400 text-sm mb-1">Calories</div>
                <div className="text-white text-2xl font-light">{summary.calories.burned.toLocaleString()}</div>
                <Progress 
                  value={calculateProgress(summary.calories.burned, summary.calories.target)} 
                  className="h-1.5 mt-1" 
                />
              </div>
            </div>
            
            <div className="bg-black/20 rounded-lg p-3">
              <div className="text-gray-300 text-sm flex justify-between mb-1">
                <span>Calorie Balance</span>
                <span className={calorieBalance >= 0 ? 'text-green-400' : 'text-red-400'}>
                  {calorieBalance >= 0 ? '+' : ''}{calorieBalance}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs text-gray-400">
                <div>
                  <div>Burned: {summary.calories.burned}</div>
                  <div>BMR: {summary.calories.BMR}</div>
                </div>
                <div>
                  <div>Consumed: {summary.nutrition.consumed}</div>
                  <div>Target: {summary.calories.target}</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {/* Nutrition */}
        <Card className="bg-gradient-to-br from-green-900/40 to-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex justify-between items-start mb-3">
              <h3 className="text-white font-medium flex items-center gap-2">
                <Utensils className="h-5 w-5 text-green-400" />
                Nutrition
              </h3>
              <Badge variant="outline">{summary.nutrition.consumed} kcal</Badge>
            </div>
            
            <div className="grid grid-cols-3 gap-2 mb-3">
              <div>
                <div className="text-gray-400 text-xs mb-1">Protein</div>
                <div className="text-white text-lg font-light">{summary.nutrition.protein}g</div>
                <Progress 
                  value={calculateProgress(summary.nutrition.protein, 150)} 
                  className="h-1.5 mt-1 bg-gray-700" 
                />
              </div>
              
              <div>
                <div className="text-gray-400 text-xs mb-1">Carbs</div>
                <div className="text-white text-lg font-light">{summary.nutrition.carbs}g</div>
                <Progress 
                  value={calculateProgress(summary.nutrition.carbs, 250)} 
                  className="h-1.5 mt-1 bg-gray-700" 
                />
              </div>
              
              <div>
                <div className="text-gray-400 text-xs mb-1">Fat</div>
                <div className="text-white text-lg font-light">{summary.nutrition.fat}g</div>
                <Progress 
                  value={calculateProgress(summary.nutrition.fat, 80)} 
                  className="h-1.5 mt-1 bg-gray-700" 
                />
              </div>
            </div>
            
            <div className="bg-black/20 rounded-lg p-3">
              <div className="flex justify-between items-center">
                <div>
                  <div className="text-gray-300 text-sm">Water</div>
                  <div className="text-white text-lg font-light">{summary.nutrition.water}L</div>
                </div>
                
                <div className="flex items-center gap-1">
                  {[1, 2, 3, 4].map(i => (
                    <div 
                      key={i}
                      className={`w-5 h-8 rounded-sm ${
                        i <= summary.nutrition.water * 2 
                          ? 'bg-blue-500/70' 
                          : 'bg-gray-700'
                      }`}
                    ></div>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {/* Recovery & Sleep */}
        <Card className="bg-gradient-to-br from-purple-900/40 to-gray-800 border-gray-700">
          <CardContent className="p-4">
            <div className="flex justify-between items-start mb-3">
              <h3 className="text-white font-medium flex items-center gap-2">
                <Heart className="h-5 w-5 text-purple-400" />
                Recovery
              </h3>
              <Badge variant={
                summary.recovery.readiness === 'high' ? 'default' :
                summary.recovery.readiness === 'medium' ? 'secondary' : 'outline'
              }>
                {summary.recovery.readiness === 'high' ? 'High Readiness' :
                 summary.recovery.readiness === 'medium' ? 'Moderate' : 'Low Energy'}
              </Badge>
            </div>
            
            <div className="grid grid-cols-2 gap-4 mb-3">
              <div>
                <div className="text-gray-400 text-sm mb-1">Recovery Score</div>
                <div className="text-white text-2xl font-light">{summary.recovery.score}%</div>
                <Progress 
                  value={summary.recovery.score} 
                  className={`h-1.5 mt-1 ${
                    summary.recovery.score > 80 ? 'bg-green-500' :
                    summary.recovery.score > 60 ? 'bg-yellow-500' : 'bg-red-500'
                  }`} 
                />
              </div>
              
              <div>
                <div className="text-gray-400 text-sm mb-1">Sleep</div>
                <div className="text-white text-2xl font-light">{summary.sleep.duration}hrs</div>
                <Progress 
                  value={calculateProgress(summary.sleep.duration, 8)} 
                  className="h-1.5 mt-1" 
                />
              </div>
            </div>
            
            <div className="bg-black/20 rounded-lg p-3">
              <div className="text-gray-300 text-sm mb-2">Sleep Quality</div>
              <div className="flex items-center justify-between mb-1">
                <div className="text-xs text-gray-400">{summary.sleep.bedtime}</div>
                <div className="text-xs text-gray-400">{summary.sleep.wakeup}</div>
              </div>
              <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-blue-600"
                  style={{ width: `${summary.sleep.quality}%` }}
                ></div>
              </div>
              <div className="mt-1 text-xs text-right text-blue-400">
                Quality: {summary.sleep.quality}%
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* Daily Goals Progress */}
      <h3 className="text-white font-medium mb-3 flex items-center gap-2">
        <Trophy className="h-5 w-5 text-yellow-500" />
        Daily Goals
      </h3>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 mb-6">
        {dailyGoals.map(goal => (
          <div 
            key={goal.id}
            className="bg-gray-800 border border-gray-700 rounded-lg p-3 flex items-center gap-3"
          >
            <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
              goal.category === 'activity' ? 'bg-blue-900/30 text-blue-500' :
              goal.category === 'nutrition' ? 'bg-green-900/30 text-green-500' :
              goal.category === 'sleep' ? 'bg-purple-900/30 text-purple-500' :
              'bg-orange-900/30 text-orange-500'
            }`}>
              {(() => {
                const IconComponent = iconMap[goal.icon];
                return <IconComponent className="h-5 w-5" />;
              })()}
            </div>
            
            <div className="flex-grow">
              <div className="flex justify-between items-baseline">
                <div className="text-white font-medium text-sm">{goal.name}</div>
                <div className="text-gray-400 text-xs">
                  {goal.current} / {goal.target} {goal.unit}
                </div>
              </div>
              
              <Progress 
                value={calculateProgress(goal.current, goal.target)} 
                className="h-1.5 mt-1" 
              />
            </div>
            
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center ${
                    calculateProgress(goal.current, goal.target) >= 100
                      ? 'bg-green-900/30 text-green-500'
                      : 'bg-gray-700 text-gray-500'
                  }`}>
                    {calculateProgress(goal.current, goal.target) >= 100 ? (
                      <Check className="h-3 w-3" />
                    ) : (
                      <Plus className="h-3 w-3" />
                    )}
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  {calculateProgress(goal.current, goal.target) >= 100 
                    ? 'Goal completed!' 
                    : `${Math.round(calculateProgress(goal.current, goal.target))}% completed`}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        ))}
      </div>
      
      {/* Workouts & Tasks */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Scheduled Workouts */}
        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <h3 className="text-white font-medium mb-3 flex items-center gap-2">
              <Dumbbell className="h-5 w-5 text-orange-500" />
              Today's Workouts
            </h3>
            
            {summary.activity.workouts.length > 0 ? (
              <div className="space-y-3">
                {summary.activity.workouts.map((workout, index) => (
                  <div 
                    key={index}
                    className={`p-3 rounded-lg border ${
                      workout.completed 
                        ? 'bg-green-900/10 border-green-800/30' 
                        : 'bg-gray-750 border-gray-700'
                    }`}
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="text-white font-medium">{workout.name}</div>
                        <div className="flex items-center gap-2 text-gray-400 text-sm">
                          {workout.time && (
                            <div className="flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              <span>{workout.time}</span>
                            </div>
                          )}
                          {workout.duration && (
                            <div className="flex items-center gap-1">
                              <Activity className="h-3 w-3" />
                              <span>{workout.duration} min</span>
                            </div>
                          )}
                        </div>
                      </div>
                      
                      {workout.completed ? (
                        <Badge className="bg-green-900 text-green-300">Completed</Badge>
                      ) : (
                        <Button size="sm">
                          <Zap className="mr-1 h-3 w-3" />
                          Start
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-6 text-gray-500">
                <Dumbbell className="h-10 w-10 mx-auto mb-2 opacity-30" />
                <p>No workouts scheduled for today</p>
              </div>
            )}
          </CardContent>
        </Card>
        
        {/* Completed Tasks */}
        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-4">
            <h3 className="text-white font-medium mb-3 flex items-center gap-2">
              <Check className="h-5 w-5 text-green-500" />
              Completed Goals
            </h3>
            
            {summary.completedTasks.length > 0 ? (
              <div className="space-y-2">
                {summary.completedTasks.map((task, index) => (
                  <div key={index} className="flex items-center gap-2 text-gray-300">
                    <div className="w-5 h-5 rounded-full bg-green-900/30 flex items-center justify-center text-green-500">
                      <Check className="h-3 w-3" />
                    </div>
                    <span>{task}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-6 text-gray-500">
                <Check className="h-10 w-10 mx-auto mb-2 opacity-30" />
                <p>No completed tasks yet today</p>
              </div>
            )}
            
            <div className="mt-4 pt-3 border-t border-gray-700">
              <div className="flex justify-between items-center">
                <div className="text-gray-400 text-sm">Recovery recommendation:</div>
                <Badge variant="outline" className="text-blue-400">Coach Tip</Badge>
              </div>
              <p className="text-gray-300 text-sm mt-1">
                {summary.recovery.recommendation}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
} 
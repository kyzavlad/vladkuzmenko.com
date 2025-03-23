'use client';

import { useState } from 'react';
import { 
  LineChart, 
  BarChart3, 
  Calendar, 
  Clock, 
  ArrowUpRight, 
  ArrowDownRight, 
  ArrowRight, 
  BarChart, 
  PieChart, 
  Activity, 
  Settings, 
  Filter, 
  ChevronRight,
  ArrowLeft,
  FileText,
  TrendingUp,
  Repeat,
  Heart,
  Zap,
  Sun,
  Moon,
  CheckCircle,
  XCircle,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

interface BehavioralInsightsProps {
  onBack?: () => void;
}

export default function BehavioralInsights({ onBack }: BehavioralInsightsProps) {
  const [activeTab, setActiveTab] = useState<string>('adherence');
  const [timeframe, setTimeframe] = useState<string>('month');
  
  // Mock behavioral insights
  const adherenceInsights = [
    {
      id: 'adh-1',
      title: 'Workout Consistency',
      value: '85%',
      trend: 'up',
      description: 'You are most consistent on Tuesday and Saturday mornings',
      action: 'Schedule important workouts on these days for best results'
    },
    {
      id: 'adh-2',
      title: 'Completion Rate',
      value: '92%',
      trend: 'up',
      description: 'You complete more workouts that are under 60 minutes',
      action: 'Your program has been optimized for 45-60 minute sessions'
    },
    {
      id: 'adh-3',
      title: 'Nutrition Tracking',
      value: '72%',
      trend: 'down',
      description: 'Weekend eating patterns deviate from your plan',
      action: 'Try meal prepping on Fridays for weekend adherence'
    },
    {
      id: 'adh-4',
      title: 'Check-in Frequency',
      value: '4.8 days/week',
      trend: 'neutral',
      description: 'You consistently log data on weekdays but less on weekends',
      action: 'Set weekend reminders to maintain tracking consistency'
    }
  ];

  const patternInsights = [
    {
      id: 'pat-1',
      title: 'Optimal Workout Time',
      value: '6:30-8:00 AM',
      trend: 'neutral',
      description: 'Morning workouts show 23% better performance than evening',
      action: 'Schedule strength training in the morning when possible'
    },
    {
      id: 'pat-2',
      title: 'Rest Day Pattern',
      value: 'Wednesdays, Sundays',
      trend: 'neutral',
      description: 'Your current rest day schedule aligns with recovery needs',
      action: 'Maintain this pattern for optimal recovery'
    },
    {
      id: 'pat-3',
      title: 'Sleep Impact',
      value: 'High Correlation',
      trend: 'neutral',
      description: 'Each additional hour of sleep improves next-day performance by 7%',
      action: 'Prioritize 7+ hours sleep before key training days'
    },
    {
      id: 'pat-4',
      title: 'Hydration Pattern',
      value: '75% of Target',
      trend: 'down',
      description: 'Lower hydration on busy workdays affects energy levels',
      action: 'Set hourly water reminders on Mondays and Thursdays'
    }
  ];

  const responseInsights = [
    {
      id: 'res-1',
      title: 'Volume Response',
      value: 'Above Average',
      trend: 'up',
      description: 'You respond well to higher training volumes for upper body',
      action: 'Your upper body program includes more volume for faster progress'
    },
    {
      id: 'res-2',
      title: 'Recovery Rate',
      value: '48-72 hours',
      trend: 'neutral',
      description: 'Optimal muscle group training frequency is every 48-72 hours',
      action: 'Your split has been optimized for this recovery window'
    },
    {
      id: 'res-3',
      title: 'Intensity Threshold',
      value: '85% 1RM',
      trend: 'up',
      description: 'Strength gains are highest when training above 85% 1RM',
      action: 'Your strength days now include more work in the 85-90% range'
    },
    {
      id: 'res-4',
      title: 'Exercise Preference',
      value: 'Compound Movements',
      trend: 'neutral',
      description: 'You complete 92% of compound lifts but only 78% of isolation exercises',
      action: 'Program now prioritizes compounds with strategic isolation work'
    }
  ];

  const habitInsights = [
    {
      id: 'hab-1',
      title: 'Positive Habit',
      value: 'Morning Protein',
      trend: 'up',
      description: 'You consistently consume protein within 30 min of waking',
      action: 'This habit supports your muscle building goals - keep it up!'
    },
    {
      id: 'hab-2',
      title: 'Positive Habit',
      value: 'Consistent Sleep Schedule',
      trend: 'up',
      description: 'Your sleep timing is consistent on weekdays',
      action: 'This supports recovery and hormonal balance'
    },
    {
      id: 'hab-3',
      title: 'Habit to Improve',
      value: 'Evening Screen Time',
      trend: 'down',
      description: 'Screen time after 9pm correlates with poorer sleep quality',
      action: 'Try blue light blocking or reading instead of screens at night'
    },
    {
      id: 'hab-4',
      title: 'Habit to Improve',
      value: 'Stress Management',
      trend: 'down',
      description: 'Work stress spikes on Mondays affect recovery for 24-48 hours',
      action: 'Added breathwork exercises to your Monday routine'
    }
  ];

  // Mock weekly pattern data for visualization
  const weeklyPatternData = {
    workoutTiming: [
      { day: 'Mon', morning: 85, afternoon: 72, evening: 68 },
      { day: 'Tue', morning: 89, afternoon: 75, evening: 70 },
      { day: 'Wed', morning: 0, afternoon: 0, evening: 0 }, // Rest day
      { day: 'Thu', morning: 0, afternoon: 78, evening: 71 },
      { day: 'Fri', morning: 83, afternoon: 76, evening: 66 },
      { day: 'Sat', morning: 87, afternoon: 74, evening: 65 },
      { day: 'Sun', morning: 0, afternoon: 0, evening: 0 } // Rest day
    ],
    nutritionAdherence: [
      { day: 'Mon', value: 88 },
      { day: 'Tue', value: 92 },
      { day: 'Wed', value: 85 },
      { day: 'Thu', value: 90 },
      { day: 'Fri', value: 84 },
      { day: 'Sat', value: 68 },
      { day: 'Sun', value: 62 }
    ]
  };

  // Function to get color based on trend
  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'up':
        return 'text-green-500';
      case 'down':
        return 'text-red-500';
      default:
        return 'text-blue-500';
    }
  };

  // Function to get trend icon based on trend direction
  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up':
        return <ArrowUpRight className="h-4 w-4 text-green-500" />;
      case 'down':
        return <ArrowDownRight className="h-4 w-4 text-red-500" />;
      default:
        return <Repeat className="h-4 w-4 text-blue-500" />;
    }
  };

  // Get current insights based on active tab
  const getCurrentInsights = () => {
    switch (activeTab) {
      case 'adherence':
        return adherenceInsights;
      case 'patterns':
        return patternInsights;
      case 'response':
        return responseInsights;
      case 'habits':
        return habitInsights;
      default:
        return adherenceInsights;
    }
  };

  return (
    <div className="behavioral-insights">
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
            <h1 className="text-3xl font-bold">Behavioral Insights</h1>
          </div>
          <p className="text-muted-foreground">
            Understanding your patterns to optimize your fitness journey
          </p>
        </div>
        <div className="flex gap-2">
          <Select value={timeframe} onValueChange={setTimeframe}>
            <SelectTrigger className="w-[150px]">
              <SelectValue placeholder="Select timeframe" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="week">Last Week</SelectItem>
              <SelectItem value="month">Last Month</SelectItem>
              <SelectItem value="3month">Last 3 Months</SelectItem>
              <SelectItem value="year">Last Year</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" size="icon">
            <Filter className="h-4 w-4" />
          </Button>
        </div>
      </div>
      
      {/* Insights Summary Card */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-lg">Behavioral Intelligence Summary</CardTitle>
          <CardDescription>
            Key insights from analyzing your behavioral patterns
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-muted rounded-lg p-4">
              <div className="flex justify-between items-start mb-3">
                <div className="bg-green-500/20 text-green-500 p-2 rounded-full">
                  <Calendar className="h-5 w-5" />
                </div>
                <Badge variant="outline">Adherence</Badge>
              </div>
              <h3 className="font-medium mb-1">Strong Workout Consistency</h3>
              <p className="text-sm text-muted-foreground">
                You maintain 85% adherence to your scheduled workouts, with best compliance on Tuesday and Saturday mornings.
              </p>
            </div>
            
            <div className="bg-muted rounded-lg p-4">
              <div className="flex justify-between items-start mb-3">
                <div className="bg-blue-500/20 text-blue-500 p-2 rounded-full">
                  <Clock className="h-5 w-5" />
                </div>
                <Badge variant="outline">Pattern</Badge>
              </div>
              <h3 className="font-medium mb-1">Optimal Training Window</h3>
              <p className="text-sm text-muted-foreground">
                Your performance is 23% better during morning workouts (6:30-8:00 AM) compared to evening sessions.
              </p>
            </div>
            
            <div className="bg-muted rounded-lg p-4">
              <div className="flex justify-between items-start mb-3">
                <div className="bg-amber-500/20 text-amber-500 p-2 rounded-full">
                  <Activity className="h-5 w-5" />
                </div>
                <Badge variant="outline">Response</Badge>
              </div>
              <h3 className="font-medium mb-1">Volume Responsiveness</h3>
              <p className="text-sm text-muted-foreground">
                You respond better to higher training volumes on upper body exercises compared to lower body.
              </p>
            </div>
            
            <div className="bg-muted rounded-lg p-4">
              <div className="flex justify-between items-start mb-3">
                <div className="bg-purple-500/20 text-purple-500 p-2 rounded-full">
                  <Zap className="h-5 w-5" />
                </div>
                <Badge variant="outline">Habit</Badge>
              </div>
              <h3 className="font-medium mb-1">Nutrition Pattern</h3>
              <p className="text-sm text-muted-foreground">
                Your nutrition adherence drops by 22% on weekends, affecting Monday energy and recovery.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Behavioral Insights Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="mb-8">
        <TabsList className="mb-4">
          <TabsTrigger value="adherence" className="flex items-center gap-1">
            <Calendar className="h-4 w-4" />
            Adherence
          </TabsTrigger>
          <TabsTrigger value="patterns" className="flex items-center gap-1">
            <BarChart3 className="h-4 w-4" />
            Patterns
          </TabsTrigger>
          <TabsTrigger value="response" className="flex items-center gap-1">
            <LineChart className="h-4 w-4" />
            Responses
          </TabsTrigger>
          <TabsTrigger value="habits" className="flex items-center gap-1">
            <Activity className="h-4 w-4" />
            Habits
          </TabsTrigger>
        </TabsList>
        
        {/* Tab Content */}
        <TabsContent value={activeTab} className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>{activeTab === 'adherence' ? 'Adherence Insights' : 
                         activeTab === 'patterns' ? 'Pattern Insights' : 
                         activeTab === 'response' ? 'Response Insights' : 'Habit Insights'}</CardTitle>
              <CardDescription>
                {activeTab === 'adherence' ? 'How consistently you follow your fitness program' : 
                activeTab === 'patterns' ? 'Recurring patterns in your fitness activities' : 
                activeTab === 'response' ? 'How your body responds to different training variables' : 
                'Positive and negative habits affecting your progress'}
              </CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {getCurrentInsights().map((insight) => (
                <div 
                  key={insight.id}
                  className="border rounded-lg p-4"
                >
                  <div className="flex justify-between mb-3">
                    <h3 className="font-medium">{insight.title}</h3>
                    <div className="flex items-center gap-1">
                      {getTrendIcon(insight.trend)}
                      <Badge variant={insight.trend === 'up' ? 'default' : 
                                     insight.trend === 'down' ? 'destructive' : 'secondary'}>
                        {insight.value}
                      </Badge>
                    </div>
                  </div>
                  <p className="text-sm mb-3">{insight.description}</p>
                  <div className="bg-muted rounded-md p-3">
                    <div className="text-xs text-muted-foreground mb-1">Recommended Action</div>
                    <div className="text-sm font-medium">{insight.action}</div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
          
          {/* Visualization Card - Different for each tab */}
          {activeTab === 'adherence' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Weekly Adherence Pattern</CardTitle>
                <CardDescription>How your adherence varies throughout the week</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[200px] relative mt-4 mb-6">
                  {/* Mock adherence chart - just a visualization representation */}
                  <div className="flex h-full items-end justify-between">
                    {weeklyPatternData.nutritionAdherence.map((day, index) => (
                      <div key={index} className="w-10 mx-auto relative">
                        <div 
                          className={`rounded-t-sm w-8 ${day.value > 80 ? 'bg-green-500' : day.value > 70 ? 'bg-amber-500' : 'bg-red-500'}`}
                          style={{ height: `${day.value * 1.8}px` }}
                        ></div>
                      </div>
                    ))}
                  </div>
                  {/* X-axis labels */}
                  <div className="flex justify-between mt-2">
                    {weeklyPatternData.nutritionAdherence.map((day, index) => (
                      <div key={index} className="w-10 text-center text-xs text-muted-foreground">
                        {day.day}
                      </div>
                    ))}
                  </div>
                </div>
                <div className="flex justify-between">
                  <div className="text-sm">
                    <span className="text-muted-foreground">Average Adherence: </span>
                    <span className="font-medium">81.3%</span>
                  </div>
                  <Button variant="outline" size="sm">
                    View Detailed Analysis
                    <ChevronRight className="ml-1 h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
          
          {activeTab === 'patterns' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Performance by Time of Day</CardTitle>
                <CardDescription>How your workout performance varies by time of day</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex gap-3 mb-4">
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                    <span className="text-xs">Morning</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                    <span className="text-xs">Afternoon</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-amber-500 rounded-full"></div>
                    <span className="text-xs">Evening</span>
                  </div>
                </div>
                
                <div className="h-[200px] relative mt-4 mb-6">
                  {/* Mock time pattern chart - just a visualization representation */}
                  <div className="flex h-full items-end justify-between">
                    {weeklyPatternData.workoutTiming.map((day, index) => (
                      <div key={index} className="w-12 mx-auto relative flex items-end justify-center gap-1">
                        {day.morning > 0 && (
                          <div 
                            className="rounded-t-sm w-3 bg-blue-500"
                            style={{ height: `${day.morning * 1.8}px` }}
                          ></div>
                        )}
                        {day.afternoon > 0 && (
                          <div 
                            className="rounded-t-sm w-3 bg-purple-500"
                            style={{ height: `${day.afternoon * 1.8}px` }}
                          ></div>
                        )}
                        {day.evening > 0 && (
                          <div 
                            className="rounded-t-sm w-3 bg-amber-500"
                            style={{ height: `${day.evening * 1.8}px` }}
                          ></div>
                        )}
                      </div>
                    ))}
                  </div>
                  {/* X-axis labels */}
                  <div className="flex justify-between mt-2">
                    {weeklyPatternData.workoutTiming.map((day, index) => (
                      <div key={index} className="w-12 text-center text-xs text-muted-foreground">
                        {day.day}
                      </div>
                    ))}
                  </div>
                </div>
                <div className="flex justify-between">
                  <div className="flex items-center">
                    <Sun className="h-4 w-4 text-blue-500 mr-1" />
                    <span className="text-sm font-medium">Morning workouts consistently perform best</span>
                  </div>
                  <Button variant="outline" size="sm">
                    View Time Analysis
                    <ChevronRight className="ml-1 h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
          
          {activeTab === 'response' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Training Response Analysis</CardTitle>
                <CardDescription>How different variables affect your training results</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="border rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <TrendingUp className="h-4 w-4 text-blue-500" />
                      <span className="font-medium text-sm">Strength Correlation</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="bg-blue-500/10 rounded p-2">
                        <div className="text-muted-foreground">Intensity Impact</div>
                        <div className="font-medium mt-1">+24%</div>
                      </div>
                      <div className="bg-blue-500/10 rounded p-2">
                        <div className="text-muted-foreground">Volume Impact</div>
                        <div className="font-medium mt-1">+12%</div>
                      </div>
                      <div className="bg-blue-500/10 rounded p-2">
                        <div className="text-muted-foreground">Rest Impact</div>
                        <div className="font-medium mt-1">+8%</div>
                      </div>
                      <div className="bg-blue-500/10 rounded p-2">
                        <div className="text-muted-foreground">Sleep Impact</div>
                        <div className="font-medium mt-1">+18%</div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="border rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <Heart className="h-4 w-4 text-red-500" />
                      <span className="font-medium text-sm">Recovery Correlation</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="bg-red-500/10 rounded p-2">
                        <div className="text-muted-foreground">Sleep Impact</div>
                        <div className="font-medium mt-1">+35%</div>
                      </div>
                      <div className="bg-red-500/10 rounded p-2">
                        <div className="text-muted-foreground">Nutrition Impact</div>
                        <div className="font-medium mt-1">+28%</div>
                      </div>
                      <div className="bg-red-500/10 rounded p-2">
                        <div className="text-muted-foreground">Stress Impact</div>
                        <div className="font-medium mt-1">-22%</div>
                      </div>
                      <div className="bg-red-500/10 rounded p-2">
                        <div className="text-muted-foreground">Active Recovery</div>
                        <div className="font-medium mt-1">+15%</div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="border rounded-lg p-4">
                  <h3 className="font-medium mb-2 text-sm">Key Response Insight</h3>
                  <p className="text-sm text-muted-foreground">
                    Your data shows that for every 1% increase in training intensity (above 80% 1RM), 
                    your strength gains increase by 2.4%, but recovery needs increase by 3.2%. Your 
                    optimal training intensity appears to be 85-87% for maximal progress without 
                    excessive recovery demand.
                  </p>
                  <div className="mt-3 flex justify-end">
                    <Button size="sm">
                      Apply This Insight
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
          
          {activeTab === 'habits' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Habit Formation Analysis</CardTitle>
                <CardDescription>Progress on forming positive fitness habits</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="border rounded-lg p-3">
                    <div className="flex justify-between items-center mb-3">
                      <div className="flex items-center">
                        <div className="bg-green-500/20 p-2 rounded-full mr-2">
                          <CheckCircle className="h-4 w-4 text-green-500" />
                        </div>
                        <span className="font-medium">Morning Protein Intake</span>
                      </div>
                      <Badge variant="outline">94% Consistent</Badge>
                    </div>
                    <div className="mb-2">
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-muted-foreground">Formation Progress</span>
                        <span>Almost Automatic</span>
                      </div>
                      <Progress value={94} className="h-2" />
                    </div>
                    <div className="text-xs text-muted-foreground">
                      This habit is well-established (66+ days) and likely to persist
                    </div>
                  </div>
                  
                  <div className="border rounded-lg p-3">
                    <div className="flex justify-between items-center mb-3">
                      <div className="flex items-center">
                        <div className="bg-amber-500/20 p-2 rounded-full mr-2">
                          <Activity className="h-4 w-4 text-amber-500" />
                        </div>
                        <span className="font-medium">Post-Workout Recovery Routine</span>
                      </div>
                      <Badge variant="outline">58% Consistent</Badge>
                    </div>
                    <div className="mb-2">
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-muted-foreground">Formation Progress</span>
                        <span>Building</span>
                      </div>
                      <Progress value={58} className="h-2" />
                    </div>
                    <div className="text-xs text-muted-foreground">
                      This habit is still forming (24 days) and needs reinforcement
                    </div>
                  </div>
                  
                  <div className="border rounded-lg p-3">
                    <div className="flex justify-between items-center mb-3">
                      <div className="flex items-center">
                        <div className="bg-red-500/20 p-2 rounded-full mr-2">
                          <XCircle className="h-4 w-4 text-red-500" />
                        </div>
                        <span className="font-medium">Evening Screen Limitation</span>
                      </div>
                      <Badge variant="outline">32% Consistent</Badge>
                    </div>
                    <div className="mb-2">
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-muted-foreground">Formation Progress</span>
                        <span>Early Stage</span>
                      </div>
                      <Progress value={32} className="h-2" />
                    </div>
                    <div className="text-xs text-muted-foreground">
                      This habit needs additional support strategies to develop
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
      
      {/* Action Plan Card */}
      <Card>
        <CardHeader>
          <CardTitle>Behavioral Optimization Plan</CardTitle>
          <CardDescription>
            Recommended actions based on your behavioral insights
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Actions list */}
            <div className="border rounded-lg overflow-hidden">
              <div className="bg-muted p-3">
                <h3 className="font-medium">High-Impact Actions</h3>
              </div>
              <div className="p-4 space-y-3">
                <div className="flex items-start gap-3">
                  <div className="bg-blue-500/20 text-blue-500 p-1.5 rounded-full mt-0.5">
                    <Calendar className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="font-medium mb-1">Schedule key workouts for Tuesday and Saturday mornings</div>
                    <p className="text-sm text-muted-foreground">
                      Your adherence is 22% higher during these time slots.
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3">
                  <div className="bg-amber-500/20 text-amber-500 p-1.5 rounded-full mt-0.5">
                    <FileText className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="font-medium mb-1">Implement weekend meal preparation</div>
                    <p className="text-sm text-muted-foreground">
                      Preparing meals on Friday for the weekend can improve nutrition consistency by 35%.
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3">
                  <div className="bg-purple-500/20 text-purple-500 p-1.5 rounded-full mt-0.5">
                    <Moon className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="font-medium mb-1">Test blue light blocking glasses after 8pm</div>
                    <p className="text-sm text-muted-foreground">
                      May improve sleep quality by 18% based on similar user profiles.
                    </p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="flex justify-end">
              <Button>
                Apply All Recommendations
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 
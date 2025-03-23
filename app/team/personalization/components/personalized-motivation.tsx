'use client';

import { useState } from 'react';
import { 
  ArrowLeft, 
  BarChart, 
  Bell,
  Calendar,
  BarChart3,
  CheckCircle,
  ChevronRight, 
  Clock,
  Edit, 
  HeartHandshake, 
  Info, 
  MessageSquare, 
  Quote,
  Settings, 
  Sparkles,
  Star,
  Target,
  ThumbsDown, 
  ThumbsUp, 
  Trophy,
  User,
  Users,
  Volume2, 
  XCircle,
  Zap
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Progress } from '@/components/ui/progress';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

interface PersonalizedMotivationProps {
  onBack?: () => void;
}

export default function PersonalizedMotivation({ onBack }: PersonalizedMotivationProps) {
  const [activeTab, setActiveTab] = useState<string>('current');
  const [motivationStyle, setMotivationStyle] = useState<string>('encouraging');
  const [motivationPreferences, setMotivationPreferences] = useState({
    quotes: true,
    reminders: true,
    challenges: true,
    achievements: true,
    community: false,
  });
  
  // Mock motivation preferences data
  const userMotivationProfile = {
    primaryMotivationType: 'Achievement',
    secondaryMotivationType: 'Consistency',
    responsiveness: {
      positive: 85,
      challenging: 72,
      community: 45,
      dataVisual: 78
    },
    preferredTimes: ['Early morning', 'Evening'],
    motivationalTriggers: ['Progress visibility', 'Streak maintenance', 'Milestone completion'],
    demotivationalTriggers: ['Missed targets', 'Plateau periods', 'Comparison to others'],
    messageTone: 'Encouraging with moderate challenge',
  };
  
  // Mock active motivation data
  const activeMotivationStrategies = [
    {
      id: 'mot-1',
      type: 'Achievement',
      title: 'Personal Best Proximity',
      description: 'You are just 5kg away from your bench press personal best',
      triggerType: 'Pre-workout',
      effectiveness: 87,
      lastTriggered: '2 days ago'
    },
    {
      id: 'mot-2',
      type: 'Consistency',
      title: 'Streak Protection',
      description: "You're on a 12-day workout streak. Keep it going today!",
      triggerType: 'Morning reminder',
      effectiveness: 92,
      lastTriggered: 'Today'
    },
    {
      id: 'mot-3',
      type: 'Progress',
      title: 'Monthly Progress Highlight',
      description: "You've increased your workout volume by 15% this month",
      triggerType: 'Weekly summary',
      effectiveness: 83,
      lastTriggered: '3 days ago'
    },
    {
      id: 'mot-4',
      type: 'Accountability',
      title: 'Training Partner Check-in',
      description: 'James completed his workout today. Your turn!',
      triggerType: 'Social update',
      effectiveness: 75,
      lastTriggered: 'Today'
    }
  ];
  
  // Mock motivation quotes tailored to user
  const motivationalQuotes = [
    {
      id: 'quote-1',
      text: "The only bad workout is the one that didn't happen. Your consistency over the past month has been exceptional - keep building on it.",
      context: "Based on your consistency pattern",
      effectiveness: 88
    },
    {
      id: 'quote-2',
      text: "Every time you hit a personal best, you redefine what's possible. You're only 5kg away from your next breakthrough.",
      context: "Based on your proximity to personal bests",
      effectiveness: 92
    },
    {
      id: 'quote-3',
      text: "Your body achieves what your mind believes. Your performance increases when you visualize success before starting.",
      context: "Based on your pre-workout mental routine",
      effectiveness: 85
    },
    {
      id: 'quote-4',
      text: "Progress isn't always linear, but your consistency makes success inevitable. Your adherence rate puts you in the top 10% of users.",
      context: "Based on your adherence statistics",
      effectiveness: 90
    }
  ];
  
  // Mock upcoming motivation schedule
  const upcomingMotivation = [
    {
      id: 'upcoming-1',
      time: 'Tomorrow, 6:15 AM',
      type: 'Pre-workout reminder',
      message: 'Leg day - you are close to your squat personal best!',
      trigger: 'Calendar-based'
    },
    {
      id: 'upcoming-2',
      time: 'Wednesday, 7:30 PM',
      type: 'Progress update',
      message: 'Weekly progress summary with visual progress graph',
      trigger: 'Weekly schedule'
    },
    {
      id: 'upcoming-3',
      time: 'Friday, 5:45 AM',
      type: 'Challenge alert',
      message: 'Final day to complete the 4-workout weekly challenge',
      trigger: 'Challenge tracking'
    },
    {
      id: 'upcoming-4',
      time: 'Sunday, 4:00 PM',
      type: 'Reflection prompt',
      message: 'Weekly reflection and next weeks goal setting reminder',
      trigger: 'End of week'
    }
  ];
  
  // Function to get motivation type icon
  const getMotivationTypeIcon = (type: string) => {
    switch (type) {
      case 'Achievement':
        return <Trophy className="h-4 w-4" />;
      case 'Consistency':
        return <Calendar className="h-4 w-4" />;
      case 'Progress':
        return <BarChart3 className="h-4 w-4" />;
      case 'Accountability':
        return <User className="h-4 w-4" />;
      default:
        return <Star className="h-4 w-4" />;
    }
  };
  
  // Function to format timestamp to relative time
  const formatRelativeTime = (time: string) => {
    // This is just a mock function - in a real app, we'd calculate the actual relative time
    return time;
  };

  return (
    <div className="personalized-motivation">
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
            <h1 className="text-3xl font-bold">Personalized Motivation</h1>
          </div>
          <p className="text-muted-foreground">
            Customized motivation based on your preferences and behavioral data
          </p>
        </div>
        <Button variant="outline" className="flex items-center gap-2">
          <Edit className="h-4 w-4" />
          Customize Preferences
        </Button>
      </div>
      
      {/* Motivation Profile Card */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-lg">Your Motivation Profile</CardTitle>
          <CardDescription>
            How our system is personalized to motivate you effectively
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <h3 className="text-sm font-medium mb-3">Motivation Type Analysis</h3>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between items-center mb-1 text-sm">
                    <div className="flex items-center gap-1">
                      <Trophy className="h-3.5 w-3.5 text-amber-500" />
                      <span>Achievement-Driven</span>
                    </div>
                    <span className="text-xs">Primary</span>
                  </div>
                  <Progress value={85} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between items-center mb-1 text-sm">
                    <div className="flex items-center gap-1">
                      <Calendar className="h-3.5 w-3.5 text-blue-500" />
                      <span>Consistency-Focused</span>
                    </div>
                    <span className="text-xs">Secondary</span>
                  </div>
                  <Progress value={72} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between items-center mb-1 text-sm">
                    <div className="flex items-center gap-1">
                      <BarChart3 className="h-3.5 w-3.5 text-green-500" />
                      <span>Progress-Oriented</span>
                    </div>
                    <span className="text-xs">Moderate</span>
                  </div>
                  <Progress value={63} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between items-center mb-1 text-sm">
                    <div className="flex items-center gap-1">
                      <User className="h-3.5 w-3.5 text-purple-500" />
                      <span>Community-Motivated</span>
                    </div>
                    <span className="text-xs">Low</span>
                  </div>
                  <Progress value={35} className="h-2" />
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="text-sm font-medium mb-3">Response Effectiveness</h3>
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className="bg-green-500/20 text-green-500 p-2 rounded-full">
                    <Target className="h-4 w-4" />
                  </div>
                  <div className="flex-1">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-sm">Goal Proximity</span>
                      <span className="text-xs">92% effective</span>
                    </div>
                    <Progress value={92} className="h-1.5" />
                  </div>
                </div>
                
                <div className="flex items-center gap-3">
                  <div className="bg-blue-500/20 text-blue-500 p-2 rounded-full">
                    <Clock className="h-4 w-4" />
                  </div>
                  <div className="flex-1">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-sm">Streak Protection</span>
                      <span className="text-xs">88% effective</span>
                    </div>
                    <Progress value={88} className="h-1.5" />
                  </div>
                </div>
                
                <div className="flex items-center gap-3">
                  <div className="bg-amber-500/20 text-amber-500 p-2 rounded-full">
                    <BarChart className="h-4 w-4" />
                  </div>
                  <div className="flex-1">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-sm">Progress Visualization</span>
                      <span className="text-xs">85% effective</span>
                    </div>
                    <Progress value={85} className="h-1.5" />
                  </div>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="text-sm font-medium mb-3">Optimal Motivation Windows</h3>
              <div className="border rounded-lg p-3 mb-4">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="h-4 w-4 text-amber-500" />
                  <span className="text-sm font-medium">Peak Motivation Times</span>
                </div>
                <div className="space-y-1.5 mb-2">
                  <div className="flex items-center justify-between text-sm">
                    <span>Early Morning (5-7 AM)</span>
                    <Badge variant="outline">Primary</Badge>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span>Evening (7-9 PM)</span>
                    <Badge variant="outline">Secondary</Badge>
                  </div>
                </div>
                <div className="text-xs text-muted-foreground">
                  Motivational content is prioritized during these windows
                </div>
              </div>
              
              <div className="text-sm flex items-start gap-2">
                <Info className="h-4 w-4 text-blue-500 mt-0.5" />
                <p className="text-muted-foreground">
                  Your motivation profile is updated weekly based on your responses to different types of motivational content.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Motivation Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="mb-8">
        <TabsList className="mb-4">
          <TabsTrigger value="current" className="flex items-center gap-1">
            <Zap className="h-4 w-4" />
            Active Strategies
          </TabsTrigger>
          <TabsTrigger value="quotes" className="flex items-center gap-1">
            <Quote className="h-4 w-4" />
            Personalized Quotes
          </TabsTrigger>
          <TabsTrigger value="upcoming" className="flex items-center gap-1">
            <Calendar className="h-4 w-4" />
            Upcoming Motivation
          </TabsTrigger>
          <TabsTrigger value="settings" className="flex items-center gap-1">
            <Settings className="h-4 w-4" />
            Motivation Settings
          </TabsTrigger>
        </TabsList>
        
        {/* Active Strategies Tab */}
        <TabsContent value="current">
          <Card>
            <CardHeader>
              <CardTitle>Active Motivation Strategies</CardTitle>
              <CardDescription>
                Currently active strategies personalized to your motivation profile
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {activeMotivationStrategies.map(strategy => (
                  <div key={strategy.id} className="border rounded-lg p-4">
                    <div className="flex justify-between items-start mb-3">
                      <div className="flex items-center gap-2">
                        <div className={`p-2 rounded-full 
                          ${strategy.type === 'Achievement' ? 'bg-amber-500/20 text-amber-500' : 
                          strategy.type === 'Consistency' ? 'bg-blue-500/20 text-blue-500' :
                          strategy.type === 'Progress' ? 'bg-green-500/20 text-green-500' :
                          'bg-purple-500/20 text-purple-500'}`}
                        >
                          {getMotivationTypeIcon(strategy.type)}
                        </div>
                        <div>
                          <h3 className="font-medium">{strategy.title}</h3>
                          <div className="text-xs text-muted-foreground">
                            {strategy.triggerType} • Last triggered: {strategy.lastTriggered}
                          </div>
                        </div>
                      </div>
                      <Badge variant="outline">{strategy.effectiveness}% effective</Badge>
                    </div>
                    <p className="text-sm mb-3">{strategy.description}</p>
                    <div className="flex justify-between items-center text-sm">
                      <div className="flex gap-2">
                        <Button variant="outline" size="sm" className="h-8">
                          <ThumbsUp className="h-3.5 w-3.5 mr-1" />
                          Helpful
                        </Button>
                        <Button variant="outline" size="sm" className="h-8">
                          <ThumbsDown className="h-3.5 w-3.5 mr-1" />
                          Not Helpful
                        </Button>
                      </div>
                      <Button variant="ghost" size="sm" className="h-8">
                        Adjust <ChevronRight className="h-3.5 w-3.5 ml-1" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* Personalized Quotes Tab */}
        <TabsContent value="quotes">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-start">
                <div>
                  <CardTitle>Personalized Motivational Quotes</CardTitle>
                  <CardDescription>
                    Quotes tailored to your profile and current fitness journey
                  </CardDescription>
                </div>
                <Select>
                  <SelectTrigger className="w-[150px]">
                    <SelectValue placeholder="Style: Encouraging" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="encouraging">Encouraging</SelectItem>
                    <SelectItem value="challenging">Challenging</SelectItem>
                    <SelectItem value="supportive">Supportive</SelectItem>
                    <SelectItem value="insightful">Insightful</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {motivationalQuotes.map(quote => (
                  <div key={quote.id} className="border rounded-lg p-4">
                    <div className="flex justify-between mb-4">
                      <div className="bg-blue-500/20 text-blue-500 p-2 rounded-full">
                        <Quote className="h-5 w-5" />
                      </div>
                      <Badge variant="outline">{quote.effectiveness}% resonance</Badge>
                    </div>
                    <blockquote className="text-lg font-medium italic mb-3">
                      "{quote.text}"
                    </blockquote>
                    <div className="text-sm text-muted-foreground mb-2">
                      {quote.context}
                    </div>
                    <div className="flex justify-between items-center">
                      <div className="flex gap-2">
                        <Button variant="ghost" size="sm" className="h-8">
                          <Volume2 className="h-3.5 w-3.5 mr-1" />
                          Speak
                        </Button>
                        <Button variant="ghost" size="sm" className="h-8">
                          <Sparkles className="h-3.5 w-3.5 mr-1" />
                          Add to Favorites
                        </Button>
                      </div>
                      <div className="flex gap-1">
                        <Button variant="outline" size="icon" className="h-8 w-8">
                          <ThumbsUp className="h-3.5 w-3.5" />
                        </Button>
                        <Button variant="outline" size="icon" className="h-8 w-8">
                          <ThumbsDown className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
                
                <Button variant="outline" className="w-full">
                  Generate More Quotes
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* Upcoming Motivation Tab */}
        <TabsContent value="upcoming">
          <Card>
            <CardHeader>
              <CardTitle>Upcoming Motivation Schedule</CardTitle>
              <CardDescription>
                Planned motivational content based on your calendar and preferences
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {upcomingMotivation.map((item, index) => (
                  <div key={item.id} className="flex gap-4 items-start">
                    <div className="bg-muted text-center p-2 rounded-lg min-w-[60px]">
                      <div className="text-xs text-muted-foreground">
                        {item.time.split(',')[0]}
                      </div>
                      <div className="font-medium">
                        {item.time.split(',')[1]}
                      </div>
                    </div>
                    
                    <div className="flex-1 border rounded-lg p-3">
                      <div className="flex justify-between items-start mb-2">
                        <Badge variant="outline">{item.type}</Badge>
                        <Button variant="ghost" size="icon" className="h-7 w-7">
                          <Edit className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                      <p className="text-sm mb-1.5">{item.message}</p>
                      <div className="text-xs text-muted-foreground">
                        Trigger: {item.trigger}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
            <CardFooter>
              <Button variant="outline" className="w-full">
                View Full Motivation Calendar
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>
        
        {/* Motivation Settings Tab */}
        <TabsContent value="settings">
          <Card>
            <CardHeader>
              <CardTitle>Motivation Preferences</CardTitle>
              <CardDescription>
                Customize how and when you receive motivational content
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div>
                  <h3 className="text-sm font-medium mb-3">Motivation Style</h3>
                  <div className="grid grid-cols-2 gap-3 mb-4">
                    <div 
                      className={`border rounded-lg p-3 cursor-pointer ${motivationStyle === 'encouraging' ? 'border-primary bg-primary/5' : ''}`}
                      onClick={() => setMotivationStyle('encouraging')}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <HeartHandshake className="h-4 w-4 text-pink-500" />
                        <span className="font-medium">Encouraging</span>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Supportive, positive language that focuses on what you can achieve
                      </p>
                    </div>
                    
                    <div 
                      className={`border rounded-lg p-3 cursor-pointer ${motivationStyle === 'challenging' ? 'border-primary bg-primary/5' : ''}`}
                      onClick={() => setMotivationStyle('challenging')}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <Zap className="h-4 w-4 text-amber-500" />
                        <span className="font-medium">Challenging</span>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Direct, challenging language that pushes you to exceed your limits
                      </p>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-sm font-medium mb-3">Content Preferences</h3>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Quote className="h-4 w-4 text-blue-500" />
                        <span>Motivational Quotes</span>
                      </div>
                      <Switch 
                        checked={motivationPreferences.quotes} 
                        onCheckedChange={(checked) => setMotivationPreferences({...motivationPreferences, quotes: checked})}
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Bell className="h-4 w-4 text-amber-500" />
                        <span>Smart Reminders</span>
                      </div>
                      <Switch 
                        checked={motivationPreferences.reminders} 
                        onCheckedChange={(checked) => setMotivationPreferences({...motivationPreferences, reminders: checked})}
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Target className="h-4 w-4 text-green-500" />
                        <span>Weekly Challenges</span>
                      </div>
                      <Switch 
                        checked={motivationPreferences.challenges} 
                        onCheckedChange={(checked) => setMotivationPreferences({...motivationPreferences, challenges: checked})}
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Trophy className="h-4 w-4 text-yellow-500" />
                        <span>Achievement Celebrations</span>
                      </div>
                      <Switch 
                        checked={motivationPreferences.achievements} 
                        onCheckedChange={(checked) => setMotivationPreferences({...motivationPreferences, achievements: checked})}
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Users className="h-4 w-4 text-purple-500" />
                        <span>Community Motivation</span>
                      </div>
                      <Switch 
                        checked={motivationPreferences.community} 
                        onCheckedChange={(checked) => setMotivationPreferences({...motivationPreferences, community: checked})}
                      />
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-sm font-medium mb-3">Frequency Preferences</h3>
                  <div className="space-y-6">
                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="text-sm">Daily Motivation Frequency</span>
                        <span className="text-sm font-medium">3 times daily</span>
                      </div>
                      <Slider defaultValue={[3]} max={5} step={1} />
                      <div className="flex justify-between text-xs text-muted-foreground mt-1">
                        <span>1</span>
                        <span>2</span>
                        <span>3</span>
                        <span>4</span>
                        <span>5</span>
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="text-sm">Notification Intensity</span>
                        <span className="text-sm font-medium">Moderate</span>
                      </div>
                      <Slider defaultValue={[2]} max={3} step={1} />
                      <div className="flex justify-between text-xs text-muted-foreground mt-1">
                        <span>Subtle</span>
                        <span>Moderate</span>
                        <span>Prominent</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline">
                Reset to Defaults
              </Button>
              <Button>
                Save Preferences
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>
      </Tabs>
      
      {/* Motivation Effectiveness Card */}
      <Card>
        <CardHeader>
          <CardTitle>Motivation Effectiveness</CardTitle>
          <CardDescription>
            Impact of personalized motivation on your fitness goals
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-sm font-medium mb-3">Behavior Impact Analysis</h3>
              <div className="border rounded-lg p-4">
                <div className="flex justify-between text-sm mb-3">
                  <span>Workout Adherence Improvement</span>
                  <span className="font-medium">+18%</span>
                </div>
                <Progress value={18} max={25} className="h-2 mb-4" />
                
                <div className="flex justify-between text-sm mb-3">
                  <span>Workout Intensity Improvement</span>
                  <span className="font-medium">+12%</span>
                </div>
                <Progress value={12} max={25} className="h-2 mb-4" />
                
                <div className="flex justify-between text-sm mb-3">
                  <span>Consistency Improvement</span>
                  <span className="font-medium">+23%</span>
                </div>
                <Progress value={23} max={25} className="h-2 mb-4" />
                
                <div className="text-xs text-muted-foreground">
                  Improvements measured since personalized motivation was activated 6 weeks ago
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="text-sm font-medium mb-3">Response Over Time</h3>
              <div className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-medium">Motivation Response Trend</h4>
                  <Badge variant="outline">Improving</Badge>
                </div>
                
                {/* Mock chart */}
                <div className="h-[140px] relative">
                  <div className="absolute inset-0 flex items-end">
                    <div className="w-full flex items-end justify-between">
                      <div className="w-4 bg-blue-500 rounded-t" style={{ height: '40%' }}></div>
                      <div className="w-4 bg-blue-500 rounded-t" style={{ height: '50%' }}></div>
                      <div className="w-4 bg-blue-500 rounded-t" style={{ height: '45%' }}></div>
                      <div className="w-4 bg-blue-500 rounded-t" style={{ height: '60%' }}></div>
                      <div className="w-4 bg-blue-500 rounded-t" style={{ height: '70%' }}></div>
                      <div className="w-4 bg-blue-500 rounded-t" style={{ height: '65%' }}></div>
                      <div className="w-4 bg-blue-500 rounded-t" style={{ height: '75%' }}></div>
                      <div className="w-4 bg-blue-500 rounded-t" style={{ height: '85%' }}></div>
                    </div>
                  </div>
                </div>
                
                <div className="text-xs text-muted-foreground mt-2">
                  System effectiveness improves as it learns your unique motivation patterns
                </div>
              </div>
            </div>
          </div>
        </CardContent>
        <CardFooter>
          <Button variant="outline" className="w-full">
            View Detailed Motivation Analytics
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
} 
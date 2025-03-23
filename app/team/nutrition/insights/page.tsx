'use client';

import { useState } from 'react';
import Link from 'next/link';
import { 
  ArrowLeft, 
  BarChart3,
  AlertTriangle,
  Info,
  Award,
  Filter,
  CheckCircle2,
  GraduationCap,
  ChevronRight,
  Clock,
  Lightbulb,
  Utensils,
  Flame
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '@/components/ui/select';

// Import our types
import { NutritionInsight } from '../types';

// Mock insights data
const mockInsights: NutritionInsight[] = [
  {
    id: 'insight-1',
    type: 'alert',
    title: 'Low Protein Intake',
    description: 'You\'ve consistently consumed less than 80% of your protein target over the past week. This may impact your muscle recovery and growth.',
    actionable: true,
    action: {
      label: 'Protein-Rich Recipes',
      link: '/team/nutrition/recipes?filter=high-protein'
    },
    createdAt: new Date(Date.now() - 86400000), // 1 day ago
    priority: 'high',
    dismissed: false
  },
  {
    id: 'insight-2',
    type: 'suggestion',
    title: 'Missing Micronutrients',
    description: 'Your diet appears to be low in vitamin D and omega-3 fatty acids. Consider adding fatty fish, eggs, or supplements to your diet.',
    actionable: true,
    action: {
      label: 'Learn More',
      link: '/team/nutrition/education/micronutrients'
    },
    createdAt: new Date(Date.now() - 172800000), // 2 days ago
    priority: 'medium',
    dismissed: false
  },
  {
    id: 'insight-3',
    type: 'tip',
    title: 'Meal Timing Optimization',
    description: 'Based on your workout schedule, consider having a protein-rich meal within 30 minutes after your strength training sessions.',
    actionable: false,
    createdAt: new Date(Date.now() - 259200000), // 3 days ago
    priority: 'medium',
    dismissed: false
  },
  {
    id: 'insight-4',
    type: 'achievement',
    title: 'Water Intake Goal Reached',
    description: 'Congratulations! You\'ve hit your water intake goal every day for the past week. Great job staying hydrated!',
    actionable: false,
    createdAt: new Date(Date.now() - 345600000), // 4 days ago
    priority: 'low',
    dismissed: false
  },
  {
    id: 'insight-5',
    type: 'alert',
    title: 'Excessive Sugar Consumption',
    description: 'Your sugar intake has exceeded your daily target by 50% or more for 5 of the last 7 days. This may impact your fitness goals.',
    actionable: true,
    action: {
      label: 'Sugar Reduction Tips',
      link: '/team/nutrition/education/sugar'
    },
    createdAt: new Date(Date.now() - 432000000), // 5 days ago
    priority: 'high',
    dismissed: false
  },
  {
    id: 'insight-6',
    type: 'suggestion',
    title: 'Increase Vegetable Diversity',
    description: 'Try incorporating a wider variety of vegetables into your diet. Your current meals show limited vegetable diversity.',
    actionable: true,
    action: {
      label: 'Vegetable-Rich Recipes',
      link: '/team/nutrition/recipes?filter=vegetables'
    },
    createdAt: new Date(Date.now() - 518400000), // 6 days ago
    priority: 'medium',
    dismissed: false
  },
  {
    id: 'insight-7',
    type: 'tip',
    title: 'Hydration and Workout Performance',
    description: 'Your water intake drops on workout days. Remember to drink water before, during, and after exercise for optimal performance.',
    actionable: false,
    createdAt: new Date(Date.now() - 604800000), // 7 days ago
    priority: 'medium',
    dismissed: false
  }
];

// Educational content
const educationalContent = [
  {
    id: 'edu-1',
    title: 'Macronutrients 101',
    description: 'Understanding proteins, carbs, and fats - the building blocks of nutrition',
    duration: '5 min read',
    category: 'Basics',
    link: '#'
  },
  {
    id: 'edu-2',
    title: 'Pre and Post Workout Nutrition',
    description: 'Optimizing your meals around training for maximum results',
    duration: '8 min read',
    category: 'Workout Nutrition',
    link: '#'
  },
  {
    id: 'edu-3',
    title: 'Reading Food Labels',
    description: 'How to decode nutrition facts and ingredient lists',
    duration: '6 min read',
    category: 'Shopping',
    link: '#'
  }
];

export default function NutritionInsights() {
  const [activeTab, setActiveTab] = useState('insights');
  const [filter, setFilter] = useState('all');
  const [timeRange, setTimeRange] = useState('7-days');
  
  // Filter insights based on current filter
  const getFilteredInsights = () => {
    let filtered = [...mockInsights];
    
    // Apply type filter
    if (filter !== 'all') {
      filtered = filtered.filter(insight => insight.type === filter);
    }
    
    // Apply time range filter
    // In a real app, this would calculate based on actual dates
    
    return filtered;
  };
  
  const filteredInsights = getFilteredInsights();
  
  // Format date for display
  const formatDate = (date: Date | string) => {
    const d = new Date(date);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - d.getTime());
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) {
      return 'Today';
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else {
      return d.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric' 
      });
    }
  };
  
  return (
    <div className="nutrition-insights p-4 max-w-5xl mx-auto">
      <div className="flex items-center mb-6">
        <Button variant="ghost" size="sm" className="mr-2" asChild>
          <Link href="/team/nutrition">
            <ArrowLeft className="h-5 w-5" />
          </Link>
        </Button>
        <div>
          <h1 className="text-3xl font-bold text-white mb-1">Nutrition Insights</h1>
          <p className="text-gray-400">Personalized guidance and nutrition education</p>
        </div>
      </div>
      
      <Tabs defaultValue="insights" className="mb-8" onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-3 mb-6">
          <TabsTrigger value="insights">Insights</TabsTrigger>
          <TabsTrigger value="education">Education</TabsTrigger>
          <TabsTrigger value="trends">Trends</TabsTrigger>
        </TabsList>
        
        <TabsContent value="insights" className="space-y-6">
          {/* Filters */}
          <div className="flex flex-wrap gap-4 mb-4">
            <div className="flex items-center gap-2">
              <Filter className="h-4 w-4 text-gray-400" />
              <span className="text-gray-300 text-sm">Filter:</span>
              <Select value={filter} onValueChange={setFilter}>
                <SelectTrigger className="min-w-[150px] h-9 text-sm bg-gray-800 border-gray-700">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-gray-800 border-gray-700">
                  <SelectItem value="all">All Insights</SelectItem>
                  <SelectItem value="alert">Alerts</SelectItem>
                  <SelectItem value="suggestion">Suggestions</SelectItem>
                  <SelectItem value="tip">Tips</SelectItem>
                  <SelectItem value="achievement">Achievements</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-gray-400" />
              <span className="text-gray-300 text-sm">Time Range:</span>
              <Select value={timeRange} onValueChange={setTimeRange}>
                <SelectTrigger className="min-w-[150px] h-9 text-sm bg-gray-800 border-gray-700">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-gray-800 border-gray-700">
                  <SelectItem value="7-days">Last 7 Days</SelectItem>
                  <SelectItem value="14-days">Last 14 Days</SelectItem>
                  <SelectItem value="30-days">Last 30 Days</SelectItem>
                  <SelectItem value="all-time">All Time</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          
          {/* Insights List */}
          <div className="space-y-4">
            {filteredInsights.length > 0 ? (
              filteredInsights.map(insight => (
                <Card 
                  key={insight.id} 
                  className={`bg-gray-800 border-gray-700 ${
                    insight.priority === 'high' ? 'border-l-4 border-l-red-500' :
                    insight.priority === 'medium' ? 'border-l-4 border-l-amber-500' :
                    'border-l-4 border-l-blue-500'
                  }`}
                >
                  <CardContent className="p-5">
                    <div className="flex items-start">
                      <div className={`w-10 h-10 rounded-full flex items-center justify-center mr-4 flex-shrink-0 ${
                        insight.type === 'alert' ? 'bg-red-900' :
                        insight.type === 'suggestion' ? 'bg-amber-900' :
                        insight.type === 'tip' ? 'bg-blue-900' :
                        'bg-green-900'
                      }`}>
                        {insight.type === 'alert' && <AlertTriangle className="h-5 w-5 text-red-400" />}
                        {insight.type === 'suggestion' && <Lightbulb className="h-5 w-5 text-amber-400" />}
                        {insight.type === 'tip' && <Info className="h-5 w-5 text-blue-400" />}
                        {insight.type === 'achievement' && <Award className="h-5 w-5 text-green-400" />}
                      </div>
                      
                      <div className="flex-grow">
                        <div className="flex justify-between items-start">
                          <div>
                            <div className="flex items-center gap-2">
                              <h3 className="text-white font-medium">{insight.title}</h3>
                              <Badge 
                                variant="outline" 
                                className={
                                  insight.type === 'alert' ? 'text-red-400 border-red-700' :
                                  insight.type === 'suggestion' ? 'text-amber-400 border-amber-700' :
                                  insight.type === 'tip' ? 'text-blue-400 border-blue-700' :
                                  'text-green-400 border-green-700'
                                }
                              >
                                {insight.type.charAt(0).toUpperCase() + insight.type.slice(1)}
                              </Badge>
                            </div>
                            <p className="text-gray-400 text-sm mt-1">{insight.description}</p>
                          </div>
                          <span className="text-xs text-gray-500">{formatDate(insight.createdAt)}</span>
                        </div>
                        
                        {insight.actionable && insight.action && (
                          <div className="mt-4">
                            <Button size="sm" variant="outline" asChild>
                              <Link href={insight.action.link}>
                                {insight.action.label}
                                <ChevronRight className="ml-1 h-4 w-4" />
                              </Link>
                            </Button>
                          </div>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            ) : (
              <Card className="bg-gray-800 border-gray-700">
                <CardContent className="p-8 text-center">
                  <CheckCircle2 className="h-12 w-12 text-gray-500 mx-auto mb-4" />
                  <h3 className="text-white text-lg font-medium mb-2">No Insights Found</h3>
                  <p className="text-gray-400 mb-6">
                    There are no insights matching your current filters.
                  </p>
                  <Button variant="outline" onClick={() => setFilter('all')}>
                    Reset Filters
                  </Button>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>
        
        <TabsContent value="education" className="space-y-6">
          {/* Featured Article */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-white">Featured Article</CardTitle>
              <CardDescription className="text-gray-400">
                Essential nutrition knowledge to support your goals
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="bg-gray-750 rounded-lg overflow-hidden">
                <div className="h-48 bg-gray-700 flex items-center justify-center">
                  <Utensils className="h-12 w-12 text-gray-500" />
                </div>
                <div className="p-4">
                  <Badge className="mb-2">Nutrition Fundamentals</Badge>
                  <h3 className="text-white text-xl font-medium mb-2">Nutrition for Muscle Growth</h3>
                  <p className="text-gray-400 mb-4">Learn the science-backed strategies to optimize your diet for building lean muscle mass.</p>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-500 text-sm">10 min read</span>
                    <Button asChild>
                      <Link href="#">Read Article</Link>
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Articles List */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-white">Nutrition Education</CardTitle>
              <CardDescription className="text-gray-400">
                Build your knowledge about nutrition
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {educationalContent.map(article => (
                  <div key={article.id} className="flex items-start p-3 bg-gray-750 rounded-md">
                    <div className="w-12 h-12 rounded bg-gray-700 flex items-center justify-center mr-3 flex-shrink-0">
                      <GraduationCap className="h-6 w-6 text-gray-500" />
                    </div>
                    <div className="flex-grow">
                      <div className="flex flex-wrap items-center gap-2 mb-1">
                        <h4 className="text-white font-medium">{article.title}</h4>
                        <Badge variant="outline" className="text-xs">{article.category}</Badge>
                      </div>
                      <p className="text-gray-400 text-sm mb-1">{article.description}</p>
                      <span className="text-gray-500 text-xs">{article.duration}</span>
                    </div>
                    <Button variant="ghost" size="sm" asChild>
                      <Link href={article.link}>
                        <ChevronRight className="h-5 w-5" />
                      </Link>
                    </Button>
                  </div>
                ))}
                
                <Button variant="outline" className="w-full" asChild>
                  <Link href="#">
                    Browse All Articles
                  </Link>
                </Button>
              </div>
            </CardContent>
          </Card>
          
          {/* Topics */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-white">Nutrition Topics</CardTitle>
              <CardDescription className="text-gray-400">
                Browse by areas of interest
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                <Button variant="outline" className="h-auto py-4 flex flex-col" asChild>
                  <Link href="#">
                    <Flame className="h-8 w-8 mb-2" />
                    <span>Weight Loss</span>
                  </Link>
                </Button>
                
                <Button variant="outline" className="h-auto py-4 flex flex-col" asChild>
                  <Link href="#">
                    <Utensils className="h-8 w-8 mb-2" />
                    <span>Meal Prep</span>
                  </Link>
                </Button>
                
                <Button variant="outline" className="h-auto py-4 flex flex-col" asChild>
                  <Link href="#">
                    <BarChart3 className="h-8 w-8 mb-2" />
                    <span>Macros</span>
                  </Link>
                </Button>
                
                <Button variant="outline" className="h-auto py-4 flex flex-col" asChild>
                  <Link href="#">
                    <Award className="h-8 w-8 mb-2" />
                    <span>Performance</span>
                  </Link>
                </Button>
                
                <Button variant="outline" className="h-auto py-4 flex flex-col" asChild>
                  <Link href="#">
                    <AlertTriangle className="h-8 w-8 mb-2" />
                    <span>Food Allergies</span>
                  </Link>
                </Button>
                
                <Button variant="outline" className="h-auto py-4 flex flex-col" asChild>
                  <Link href="#">
                    <Info className="h-8 w-8 mb-2" />
                    <span>Supplements</span>
                  </Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="trends" className="space-y-6">
          {/* Trends Coming Soon */}
          <Card className="bg-gray-800 border-gray-700">
            <CardContent className="p-12 text-center">
              <BarChart3 className="h-16 w-16 text-gray-500 mx-auto mb-6" />
              <h2 className="text-white text-2xl font-bold mb-4">Detailed Trends Coming Soon</h2>
              <p className="text-gray-400 text-lg mb-8 max-w-md mx-auto">
                We're building a comprehensive trends analysis feature to help you visualize your nutrition patterns over time.
              </p>
              <Button asChild>
                <Link href="/team/nutrition">
                  Return to Dashboard
                </Link>
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
} 
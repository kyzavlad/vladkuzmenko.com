"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { 
  Activity, 
  Dumbbell, 
  Apple, 
  Camera, 
  Trophy, 
  Clock, 
  Calendar,
  Heart,
  Flame,
  User,
  Utensils,
  Timer,
  Plus,
  ArrowRight,
  DollarSign
} from 'lucide-react';
import { LineChart } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';

export default function TeamDashboard() {
  const [loading, setLoading] = useState(true);
  const [caloriesProgress, setCaloriesProgress] = useState(0);
  const [workoutProgress, setWorkoutProgress] = useState(0);
  const [waterProgress, setWaterProgress] = useState(0);

  useEffect(() => {
    // Simulate loading data
    const timer = setTimeout(() => {
      setLoading(false);
      
      // Animate progress bars
      const progressTimer = setTimeout(() => {
        setCaloriesProgress(72);
        setWorkoutProgress(45);
        setWaterProgress(60);
      }, 300);
      
      return () => clearTimeout(progressTimer);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900 text-white">
        <div className="flex flex-col items-center gap-4">
          <Activity className="w-12 h-12 text-blue-400 animate-pulse" />
          <div className="text-xl">Loading fitness dashboard...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="container mx-auto flex justify-between items-center">
          <Link href="/" className="text-xl font-bold text-blue-400">
            Fitness AI Platform
          </Link>
          <nav>
            <ul className="flex space-x-6">
              <li><Link href="/platform" className="text-gray-300 hover:text-blue-400 transition">Video Platform</Link></li>
              <li><Link href="/team" className="text-white hover:text-blue-400 transition">Fitness</Link></li>
              <li><Link href="/team/nutrition" className="text-gray-300 hover:text-blue-400 transition">Nutrition</Link></li>
              <li><Link href="/team/workouts" className="text-gray-300 hover:text-blue-400 transition">Workouts</Link></li>
              <li><Link href="/team/progress" className="text-gray-300 hover:text-blue-400 transition">Progress</Link></li>
            </ul>
          </nav>
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-300">60 tokens remaining</span>
            <Button variant="outline" className="border-blue-500 text-blue-400">Account</Button>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main className="container mx-auto p-6">
        {/* Top Metrics Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-8">
          <Card className="bg-gray-800 border-gray-700 text-white">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center text-blue-400">
                <Flame className="mr-2 h-5 w-5" />
                Daily Calories
              </CardTitle>
              <CardDescription className="text-gray-400">
                1,450 / 2,000 consumed
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Progress value={caloriesProgress} className="h-2 mb-2" />
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>Breakfast: 450</span>
                <span>Lunch: 650</span>
                <span>Dinner: 350</span>
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-gray-800 border-gray-700 text-white">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center text-blue-400">
                <Dumbbell className="mr-2 h-5 w-5" />
                Workout Progress
              </CardTitle>
              <CardDescription className="text-gray-400">
                2 of 4 weekly workouts completed
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Progress value={workoutProgress} className="h-2 mb-2" />
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>Mon: Completed</span>
                <span>Wed: Completed</span>
                <span>Fri: Up Next</span>
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-gray-800 border-gray-700 text-white">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center text-blue-400">
                <Heart className="mr-2 h-5 w-5" />
                Health Stats
              </CardTitle>
              <CardDescription className="text-gray-400">
                Daily water intake: 1.5L / 2.5L
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Progress value={waterProgress} className="h-2 mb-2" />
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>Steps: 7,235</span>
                <span>Sleep: 7.5 hrs</span>
                <span>BPM: 68 avg</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Tabs Section */}
        <Tabs defaultValue="overview" className="mb-8">
          <TabsList className="grid grid-cols-5 bg-gray-800 text-gray-300">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="nutrition">Nutrition</TabsTrigger>
            <TabsTrigger value="workouts">Workouts</TabsTrigger>
            <TabsTrigger value="progress">Progress</TabsTrigger>
            <TabsTrigger value="challenges">Challenges</TabsTrigger>
          </TabsList>
          
          <TabsContent value="overview" className="mt-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <Card className="bg-gray-800 border-gray-700 text-white md:col-span-2 hover:border-blue-500 transition-all duration-300">
                <CardHeader>
                  <CardTitle className="flex items-center text-blue-400">
                    <Camera className="mr-2 h-5 w-5" />
                    Calorie Analysis Camera
                  </CardTitle>
                  <CardDescription className="text-gray-400">
                    Instantly analyze the calories and nutrition of your food
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="aspect-video bg-gray-700 rounded-lg flex flex-col items-center justify-center mb-4">
                    <Camera className="h-12 w-12 text-gray-500 mb-2" />
                    <p className="text-gray-400 text-sm">Use 5 tokens to analyze a meal with your camera</p>
                  </div>
                  <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white">
                    <Camera className="mr-2 h-4 w-4" /> Analyze Food (5 tokens)
                  </Button>
                </CardContent>
              </Card>
              
              <Card className="bg-gray-800 border-gray-700 text-white hover:border-blue-500 transition-all duration-300">
                <CardHeader>
                  <CardTitle className="flex items-center text-blue-400">
                    <LineChart className="mr-2 h-5 w-5" />
                    Weekly Overview
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Workout Consistency</span>
                    <span className="text-white font-medium">65%</span>
                  </div>
                  <Progress value={65} className="h-1.5" />
                  
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Nutrition Goal Adherence</span>
                    <span className="text-white font-medium">78%</span>
                  </div>
                  <Progress value={78} className="h-1.5" />
                  
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Weight Target Progress</span>
                    <span className="text-white font-medium">45%</span>
                  </div>
                  <Progress value={45} className="h-1.5" />
                  
                  <Button variant="outline" className="w-full mt-4 border-gray-700 text-gray-300 hover:text-white">
                    View Detailed Analysis
                  </Button>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          
          <TabsContent value="nutrition" className="mt-4">
            <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
              <h3 className="text-xl font-semibold mb-4">Nutrition Center</h3>
              <p className="text-gray-400 mb-4">This section will show detailed nutrition tracking and meal planning functionality.</p>
              <Button>Go to Nutrition Center</Button>
            </div>
          </TabsContent>
          
          <TabsContent value="workouts" className="mt-4">
            <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
              <h3 className="text-xl font-semibold mb-4">Workout Programs</h3>
              <p className="text-gray-400 mb-4">This section will contain workout programs and exercise libraries.</p>
              <Button>Go to Workout Center</Button>
            </div>
          </TabsContent>
          
          <TabsContent value="progress" className="mt-4">
            <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
              <h3 className="text-xl font-semibold mb-4">Progress Tracking</h3>
              <p className="text-gray-400 mb-4">Track your fitness journey and body measurements here.</p>
              <Button>View Progress Center</Button>
            </div>
          </TabsContent>
          
          <TabsContent value="challenges" className="mt-4">
            <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
              <h3 className="text-xl font-semibold mb-4">Fitness Challenges</h3>
              <p className="text-gray-400 mb-4">Join community challenges and compete with friends.</p>
              <Button>Explore Challenges</Button>
            </div>
          </TabsContent>
        </Tabs>

        {/* Feature Cards Grid */}
        <h2 className="text-2xl font-bold mb-4">Fitness Programs</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card className="bg-gray-800 border-gray-700 text-white hover:border-blue-500 transition-all duration-300">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center text-blue-400">
                <Dumbbell className="mr-2 h-5 w-5" />
                Strength Training
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-300 mb-4">Build muscle and increase strength with guided programs.</p>
              <div className="flex items-center mb-4">
                <div className="w-2 h-2 rounded-full bg-green-500 mr-2"></div>
                <span className="text-xs text-gray-400">5 Programs Available</span>
              </div>
              <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white">
                Browse Programs
              </Button>
            </CardContent>
          </Card>
          
          <Card className="bg-gray-800 border-gray-700 text-white hover:border-blue-500 transition-all duration-300">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center text-blue-400">
                <Activity className="mr-2 h-5 w-5" />
                Cardio Programs
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-300 mb-4">Improve endurance and burn fat with effective cardio routines.</p>
              <div className="flex items-center mb-4">
                <div className="w-2 h-2 rounded-full bg-green-500 mr-2"></div>
                <span className="text-xs text-gray-400">7 Programs Available</span>
              </div>
              <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white">
                Browse Programs
              </Button>
            </CardContent>
          </Card>
          
          <Card className="bg-gray-800 border-gray-700 text-white hover:border-blue-500 transition-all duration-300">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center text-blue-400">
                <Apple className="mr-2 h-5 w-5" />
                Nutrition Plans
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-300 mb-4">Meal plans and recipes tailored to your fitness goals.</p>
              <div className="flex items-center mb-4">
                <div className="w-2 h-2 rounded-full bg-green-500 mr-2"></div>
                <span className="text-xs text-gray-400">3 Plans Available</span>
              </div>
              <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white">
                Browse Plans
              </Button>
            </CardContent>
          </Card>
          
          <Card className="bg-gray-800 border-gray-700 text-white hover:border-blue-500 transition-all duration-300">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center text-blue-400">
                <Trophy className="mr-2 h-5 w-5" />
                Challenges
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-300 mb-4">Compete in community challenges to stay motivated.</p>
              <div className="flex items-center mb-4">
                <div className="w-2 h-2 rounded-full bg-yellow-500 mr-2"></div>
                <span className="text-xs text-gray-400">2 Active Challenges</span>
              </div>
              <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white">
                Join Challenges
              </Button>
            </CardContent>
          </Card>
        </div>
        
        {/* Subscription Plans */}
        <h2 className="text-2xl font-bold mb-4">Membership Plans</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card className="bg-gray-800 border-gray-700 text-white hover:border-blue-500 transition-all duration-300">
            <CardHeader className="pb-2">
              <div className="px-3 py-1 bg-gray-700 text-gray-300 text-xs rounded-full w-fit mb-2">FREE</div>
              <CardTitle>Basic Plan</CardTitle>
              <div className="text-2xl font-bold mt-2 mb-1">$0<span className="text-sm font-normal text-gray-400">/month</span></div>
            </CardHeader>
            <CardContent className="space-y-4">
              <ul className="space-y-2">
                <li className="flex items-start">
                  <div className="rounded-full bg-blue-400 p-1 mr-2 mt-0.5">
                    <svg width="8" height="8" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M20 6L9 17L4 12" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <span className="text-gray-300 text-sm">60 free tokens</span>
                </li>
                <li className="flex items-start">
                  <div className="rounded-full bg-blue-400 p-1 mr-2 mt-0.5">
                    <svg width="8" height="8" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M20 6L9 17L4 12" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <span className="text-gray-300 text-sm">3 workout programs</span>
                </li>
                <li className="flex items-start">
                  <div className="rounded-full bg-blue-400 p-1 mr-2 mt-0.5">
                    <svg width="8" height="8" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M20 6L9 17L4 12" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <span className="text-gray-300 text-sm">Basic nutrition tracking</span>
                </li>
              </ul>
              <Button disabled className="w-full bg-gray-700 text-gray-300">
                Current Plan
              </Button>
            </CardContent>
          </Card>
          
          <Card className="bg-gray-800 border-gray-700 text-white hover:border-blue-500 transition-all duration-300">
            <CardHeader className="pb-2">
              <div className="px-3 py-1 bg-blue-900 text-blue-300 text-xs rounded-full w-fit mb-2">POPULAR</div>
              <CardTitle>Fitness Pro</CardTitle>
              <div className="text-2xl font-bold mt-2 mb-1">$9.99<span className="text-sm font-normal text-gray-400">/month</span></div>
            </CardHeader>
            <CardContent className="space-y-4">
              <ul className="space-y-2">
                <li className="flex items-start">
                  <div className="rounded-full bg-blue-400 p-1 mr-2 mt-0.5">
                    <svg width="8" height="8" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M20 6L9 17L4 12" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <span className="text-gray-300 text-sm">200 tokens monthly</span>
                </li>
                <li className="flex items-start">
                  <div className="rounded-full bg-blue-400 p-1 mr-2 mt-0.5">
                    <svg width="8" height="8" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M20 6L9 17L4 12" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <span className="text-gray-300 text-sm">Unlimited food analysis</span>
                </li>
                <li className="flex items-start">
                  <div className="rounded-full bg-blue-400 p-1 mr-2 mt-0.5">
                    <svg width="8" height="8" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M20 6L9 17L4 12" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <span className="text-gray-300 text-sm">All premium workouts</span>
                </li>
              </ul>
              <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white">
                Upgrade Now
              </Button>
            </CardContent>
          </Card>
          
          <Card className="bg-gray-800 border-gray-700 text-white hover:border-blue-500 transition-all duration-300">
            <CardHeader className="pb-2">
              <div className="px-3 py-1 bg-purple-900 text-purple-300 text-xs rounded-full w-fit mb-2">FEATURED</div>
              <CardTitle>Personal Coach</CardTitle>
              <div className="text-2xl font-bold mt-2 mb-1">$19.99<span className="text-sm font-normal text-gray-400">/month</span></div>
            </CardHeader>
            <CardContent className="space-y-4">
              <ul className="space-y-2">
                <li className="flex items-start">
                  <div className="rounded-full bg-blue-400 p-1 mr-2 mt-0.5">
                    <svg width="8" height="8" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M20 6L9 17L4 12" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <span className="text-gray-300 text-sm">500 tokens monthly</span>
                </li>
                <li className="flex items-start">
                  <div className="rounded-full bg-blue-400 p-1 mr-2 mt-0.5">
                    <svg width="8" height="8" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M20 6L9 17L4 12" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <span className="text-gray-300 text-sm">AI workout recommendations</span>
                </li>
                <li className="flex items-start">
                  <div className="rounded-full bg-blue-400 p-1 mr-2 mt-0.5">
                    <svg width="8" height="8" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M20 6L9 17L4 12" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <span className="text-gray-300 text-sm">Weekly progress analysis</span>
                </li>
              </ul>
              <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white">
                Upgrade Now
              </Button>
            </CardContent>
          </Card>
          
          <Card className="bg-gray-800 border-gray-700 text-white hover:border-blue-500 transition-all duration-300">
            <CardHeader className="pb-2">
              <div className="px-3 py-1 bg-gradient-to-r from-blue-600 to-purple-600 text-white text-xs rounded-full w-fit mb-2">ELITE</div>
              <CardTitle>Elite Package</CardTitle>
              <div className="text-2xl font-bold mt-2 mb-1">$39.99<span className="text-sm font-normal text-gray-400">/month</span></div>
            </CardHeader>
            <CardContent className="space-y-4">
              <ul className="space-y-2">
                <li className="flex items-start">
                  <div className="rounded-full bg-blue-400 p-1 mr-2 mt-0.5">
                    <svg width="8" height="8" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M20 6L9 17L4 12" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <span className="text-gray-300 text-sm">Unlimited tokens</span>
                </li>
                <li className="flex items-start">
                  <div className="rounded-full bg-blue-400 p-1 mr-2 mt-0.5">
                    <svg width="8" height="8" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M20 6L9 17L4 12" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <span className="text-gray-300 text-sm">Personalized nutrition plans</span>
                </li>
                <li className="flex items-start">
                  <div className="rounded-full bg-blue-400 p-1 mr-2 mt-0.5">
                    <svg width="8" height="8" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M20 6L9 17L4 12" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <span className="text-gray-300 text-sm">1-on-1 AI coaching</span>
                </li>
              </ul>
              <Button className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white">
                Upgrade Now
              </Button>
            </CardContent>
          </Card>
        </div>
        
        {/* User Profile Quick Setup */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-xl font-bold">Complete Your Fitness Profile</h3>
              <p className="text-gray-400">Set your goals and preferences to get personalized recommendations</p>
            </div>
            <div className="bg-blue-900 text-blue-300 px-3 py-1 rounded-full text-sm">60% Complete</div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div className="flex items-center p-3 bg-gray-700 rounded-lg">
              <div className="bg-gray-600 p-2 rounded-full mr-3">
                <User className="h-5 w-5 text-gray-300" />
              </div>
              <div>
                <div className="text-sm text-gray-300">Basic Info</div>
                <div className="text-xs text-green-400">Completed</div>
              </div>
            </div>
            
            <div className="flex items-center p-3 bg-gray-700 rounded-lg">
              <div className="bg-gray-600 p-2 rounded-full mr-3">
                <Flame className="h-5 w-5 text-gray-300" />
              </div>
              <div>
                <div className="text-sm text-gray-300">Fitness Goals</div>
                <div className="text-xs text-green-400">Completed</div>
              </div>
            </div>
            
            <div className="flex items-center p-3 bg-gray-700 rounded-lg">
              <div className="bg-gray-600 p-2 rounded-full mr-3">
                <Utensils className="h-5 w-5 text-gray-300" />
              </div>
              <div>
                <div className="text-sm text-gray-300">Dietary Preferences</div>
                <div className="text-xs text-yellow-400">In Progress</div>
              </div>
            </div>
          </div>
          
          <Button className="bg-blue-600 hover:bg-blue-700 text-white">
            Complete Your Profile <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </main>
    </div>
  );
}
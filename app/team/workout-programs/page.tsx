'use client';

import { useState } from 'react';
import Link from 'next/link';
import { 
  Search, 
  Filter, 
  ChevronRight, 
  Dumbbell, 
  BarChart,
  Clock,
  Calendar,
  CalendarDays,
  Users,
  Trophy,
  Star,
  Heart,
  PlayCircle,
  PlusCircle,
  Settings,
  List,
  Grid,
  Flame
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger 
} from '@/components/ui/dropdown-menu';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';

// Import sample workout programs data
import { menPrograms, womenPrograms, specialtyPrograms } from './data/sample-programs';

export default function WorkoutPrograms() {
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [filterCategory, setFilterCategory] = useState<string | null>(null);
  const [filterDifficulty, setFilterDifficulty] = useState<string | null>(null);
  const [filterDuration, setFilterDuration] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('all');
  
  // Combine all programs
  const allPrograms = [...menPrograms, ...womenPrograms, ...specialtyPrograms];
  
  // Filter programs based on search query and filters
  const filteredPrograms = allPrograms
    .filter(program => {
      // Filter by tab
      if (activeTab === 'men') return program.targetAudience === 'men';
      if (activeTab === 'women') return program.targetAudience === 'women';
      if (activeTab === 'specialty') return program.targetAudience === 'all';
      if (activeTab === 'my-programs') return program.enrolled;
      if (activeTab === 'favorites') return program.favorite;
      return true; // 'all' tab
    })
    .filter(program => {
      // Filter by search query
      if (!searchQuery) return true;
      return program.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
             program.description.toLowerCase().includes(searchQuery.toLowerCase());
    })
    .filter(program => {
      // Filter by category
      if (!filterCategory) return true;
      return program.category === filterCategory;
    })
    .filter(program => {
      // Filter by difficulty
      if (!filterDifficulty) return true;
      return program.level === filterDifficulty;
    })
    .filter(program => {
      // Filter by duration
      if (!filterDuration) return true;
      return program.duration === filterDuration;
    });
  
  // Mock data for current program
  const currentProgram = {
    id: 'program-1',
    name: 'Iron Body: Maximum Strength',
    progress: 35,
    currentDay: 'Day 3: Upper Body Strength',
    nextWorkout: 'Tomorrow',
    stats: {
      daysCompleted: 7,
      totalDays: 20,
      workoutsCompleted: 7,
      totalWorkouts: 20,
      consistency: 85
    }
  };
  
  // Reset all filters
  const resetFilters = () => {
    setSearchQuery('');
    setFilterCategory(null);
    setFilterDifficulty(null);
    setFilterDuration(null);
  };
  
  return (
    <div className="workout-programs-page pb-16">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-2">Workout Programs</h1>
        <p className="text-gray-400">Explore and track workout programs tailored to your fitness goals.</p>
      </div>
      
      {/* Quick Action Buttons */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <Button 
          className="bg-gradient-to-br from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 h-auto py-6 flex flex-col items-center" 
          size="lg"
          asChild
        >
          <Link href="/team/workout-programs/tracking">
            <BarChart className="h-8 w-8 mb-2" />
            <span className="text-sm">Progress Tracking</span>
          </Link>
        </Button>
        
        <Button 
          className="bg-gradient-to-br from-emerald-600 to-emerald-700 hover:from-emerald-500 hover:to-emerald-600 h-auto py-6 flex flex-col items-center" 
          size="lg"
          asChild
        >
          <Link href="/team/workout-programs/exercises">
            <Dumbbell className="h-8 w-8 mb-2" />
            <span className="text-sm">Exercise Library</span>
          </Link>
        </Button>
        
        <Button 
          className="bg-gradient-to-br from-amber-600 to-amber-700 hover:from-amber-500 hover:to-amber-600 h-auto py-6 flex flex-col items-center" 
          size="lg"
          asChild
        >
          <Link href="/team/workout-programs/calendar">
            <CalendarDays className="h-8 w-8 mb-2" />
            <span className="text-sm">Workout Calendar</span>
          </Link>
        </Button>
        
        <Button 
          className="bg-gradient-to-br from-purple-600 to-purple-700 hover:from-purple-500 hover:to-purple-600 h-auto py-6 flex flex-col items-center" 
          size="lg"
          asChild
        >
          <Link href="/team/workout-programs/community">
            <Users className="h-8 w-8 mb-2" />
            <span className="text-sm">Community</span>
          </Link>
        </Button>
      </div>
      
      {/* Current Program Card */}
      <Card className="bg-gray-800 border-gray-700 mb-8">
        <CardHeader>
          <div className="flex justify-between items-start">
            <CardTitle className="text-lg">Current Program</CardTitle>
            <Link href="/team/workout-programs/tracking">
              <Button variant="link" className="text-blue-400 p-0 h-auto">
                View Details
              </Button>
            </Link>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col md:flex-row md:items-center">
            <div className="md:w-2/3 mb-4 md:mb-0 md:pr-6">
              <h3 className="text-xl font-bold text-white mb-2">
                {currentProgram.name}
              </h3>
              
              <div className="flex flex-col space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-400">Program Progress</span>
                    <span className="text-gray-400">{currentProgram.progress}%</span>
                  </div>
                  <Progress value={currentProgram.progress} className="h-2" />
                </div>
                
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-gray-700 p-3 rounded-md">
                    <div className="text-xs text-gray-400">Days Completed</div>
                    <div className="text-white font-medium">{currentProgram.stats.daysCompleted}/{currentProgram.stats.totalDays}</div>
                  </div>
                  
                  <div className="bg-gray-700 p-3 rounded-md">
                    <div className="text-xs text-gray-400">Consistency</div>
                    <div className="text-white font-medium">{currentProgram.stats.consistency}%</div>
                  </div>
                  
                  <div className="bg-gray-700 p-3 rounded-md">
                    <div className="text-xs text-gray-400">Next Workout</div>
                    <div className="text-white font-medium">{currentProgram.nextWorkout}</div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="md:w-1/3 bg-gray-700 p-4 rounded-md">
              <h3 className="text-white font-medium mb-3 flex items-center">
                <Calendar className="h-4 w-4 mr-2 text-blue-400" /> Upcoming Workout
              </h3>
              
              <div className="text-lg font-medium text-white mb-2">
                {currentProgram.currentDay}
              </div>
              
              <div className="flex items-center gap-2 mb-4">
                <Badge variant="outline">
                  <Clock className="h-3 w-3 mr-1" /> 50-60 min
                </Badge>
                <Badge variant="outline">
                  <Flame className="h-3 w-3 mr-1" /> 400 kcal
                </Badge>
              </div>
              
              <div className="flex gap-2">
                <Button asChild className="flex-1">
                  <Link href="/team/workout-programs/workout">
                    <PlayCircle className="h-4 w-4 mr-2" /> Start
                  </Link>
                </Button>
                <Button variant="outline" className="flex-1">
                  <Calendar className="h-4 w-4 mr-2" /> Schedule
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Program Browser */}
      <div>
        <div className="flex flex-col md:flex-row justify-between md:items-center mb-4 gap-4">
          <h2 className="text-xl font-bold text-white">Browse Programs</h2>
          
          <div className="flex flex-col md:flex-row gap-4">
            <div className="relative">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500" />
              <Input
                type="text"
                placeholder="Search programs..."
                className="pl-9 bg-gray-800 border-gray-700 text-white"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            
            <div className="flex gap-2">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" className="bg-gray-800">
                    <Filter className="h-4 w-4 mr-2" /> Filter
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="bg-gray-800 border-gray-700">
                  <DropdownMenuLabel>Category</DropdownMenuLabel>
                  <DropdownMenuItem 
                    className={`cursor-pointer ${filterCategory === 'strength' ? 'bg-blue-600' : ''}`}
                    onClick={() => setFilterCategory(filterCategory === 'strength' ? null : 'strength')}
                  >
                    Strength
                  </DropdownMenuItem>
                  <DropdownMenuItem 
                    className={`cursor-pointer ${filterCategory === 'hypertrophy' ? 'bg-blue-600' : ''}`}
                    onClick={() => setFilterCategory(filterCategory === 'hypertrophy' ? null : 'hypertrophy')}
                  >
                    Hypertrophy
                  </DropdownMenuItem>
                  <DropdownMenuItem 
                    className={`cursor-pointer ${filterCategory === 'endurance' ? 'bg-blue-600' : ''}`}
                    onClick={() => setFilterCategory(filterCategory === 'endurance' ? null : 'endurance')}
                  >
                    Endurance
                  </DropdownMenuItem>
                  <DropdownMenuItem 
                    className={`cursor-pointer ${filterCategory === 'functional' ? 'bg-blue-600' : ''}`}
                    onClick={() => setFilterCategory(filterCategory === 'functional' ? null : 'functional')}
                  >
                    Functional
                  </DropdownMenuItem>
                  
                  <DropdownMenuSeparator />
                  <DropdownMenuLabel>Difficulty</DropdownMenuLabel>
                  <DropdownMenuItem 
                    className={`cursor-pointer ${filterDifficulty === 'beginner' ? 'bg-blue-600' : ''}`}
                    onClick={() => setFilterDifficulty(filterDifficulty === 'beginner' ? null : 'beginner')}
                  >
                    Beginner
                  </DropdownMenuItem>
                  <DropdownMenuItem 
                    className={`cursor-pointer ${filterDifficulty === 'intermediate' ? 'bg-blue-600' : ''}`}
                    onClick={() => setFilterDifficulty(filterDifficulty === 'intermediate' ? null : 'intermediate')}
                  >
                    Intermediate
                  </DropdownMenuItem>
                  <DropdownMenuItem 
                    className={`cursor-pointer ${filterDifficulty === 'advanced' ? 'bg-blue-600' : ''}`}
                    onClick={() => setFilterDifficulty(filterDifficulty === 'advanced' ? null : 'advanced')}
                  >
                    Advanced
                  </DropdownMenuItem>
                  
                  <DropdownMenuSeparator />
                  <DropdownMenuLabel>Duration</DropdownMenuLabel>
                  <DropdownMenuItem 
                    className={`cursor-pointer ${filterDuration === '4-weeks' ? 'bg-blue-600' : ''}`}
                    onClick={() => setFilterDuration(filterDuration === '4-weeks' ? null : '4-weeks')}
                  >
                    4 Weeks
                  </DropdownMenuItem>
                  <DropdownMenuItem 
                    className={`cursor-pointer ${filterDuration === '8-weeks' ? 'bg-blue-600' : ''}`}
                    onClick={() => setFilterDuration(filterDuration === '8-weeks' ? null : '8-weeks')}
                  >
                    8 Weeks
                  </DropdownMenuItem>
                  <DropdownMenuItem 
                    className={`cursor-pointer ${filterDuration === '12-weeks' ? 'bg-blue-600' : ''}`}
                    onClick={() => setFilterDuration(filterDuration === '12-weeks' ? null : '12-weeks')}
                  >
                    12 Weeks
                  </DropdownMenuItem>
                  
                  <DropdownMenuSeparator />
                  <DropdownMenuItem 
                    className="cursor-pointer text-blue-400"
                    onClick={resetFilters}
                  >
                    Reset Filters
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
              
              <div className="flex border border-gray-700 rounded-md overflow-hidden">
                <Button 
                  variant="ghost" 
                  size="icon"
                  className={`rounded-none ${viewMode === 'grid' ? 'bg-gray-700' : 'bg-gray-800'}`}
                  onClick={() => setViewMode('grid')}
                >
                  <Grid className="h-4 w-4" />
                </Button>
                <Button 
                  variant="ghost" 
                  size="icon"
                  className={`rounded-none ${viewMode === 'list' ? 'bg-gray-700' : 'bg-gray-800'}`}
                  onClick={() => setViewMode('list')}
                >
                  <List className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </div>
        
        <Tabs 
          defaultValue="all" 
          value={activeTab}
          onValueChange={setActiveTab}
          className="w-full"
        >
          <TabsList className="grid grid-cols-3 md:grid-cols-6 mb-6 bg-gray-800">
            <TabsTrigger value="all">All</TabsTrigger>
            <TabsTrigger value="men">Men</TabsTrigger>
            <TabsTrigger value="women">Women</TabsTrigger>
            <TabsTrigger value="specialty">Specialty</TabsTrigger>
            <TabsTrigger value="my-programs">My Programs</TabsTrigger>
            <TabsTrigger value="favorites">Favorites</TabsTrigger>
          </TabsList>
          
          <TabsContent value={activeTab} className="bg-transparent p-0">
            {/* Display message when no programs match filters */}
            {filteredPrograms.length === 0 && (
              <div className="text-center py-10">
                <div className="mb-4">
                  <Search className="h-12 w-12 mx-auto text-gray-500" />
                </div>
                <h3 className="text-xl font-medium text-white mb-2">No programs found</h3>
                <p className="text-gray-400 mb-4">Try adjusting your search or filters</p>
                <Button onClick={resetFilters}>Reset Filters</Button>
              </div>
            )}
            
            {/* Grid View */}
            {viewMode === 'grid' && filteredPrograms.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredPrograms.map(program => (
                  <Card key={program.id} className="bg-gray-800 border-gray-700 overflow-hidden hover:border-blue-500 transition-all">
                    <div className="h-40 bg-gray-700 relative">
                      {/* Program Image Placeholder */}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <Dumbbell className="h-12 w-12 text-gray-600" />
                      </div>
                      
                      {/* Program Stats Overlay */}
                      <div className="absolute bottom-0 left-0 right-0 bg-black/60 p-3 flex justify-between items-center">
                        <div className="flex gap-2">
                          <Badge variant="outline" className="bg-black/40 text-xs">
                            <Clock className="h-3 w-3 mr-1" /> {program.duration}
                          </Badge>
                          <Badge variant="outline" className="bg-black/40 text-xs capitalize">
                            {program.level}
                          </Badge>
                        </div>
                        
                        <Button 
                          variant="ghost" 
                          size="icon" 
                          className="h-7 w-7 rounded-full hover:bg-gray-700/50"
                          onClick={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            // Toggle favorite logic would go here
                          }}
                        >
                          <Heart className={`h-4 w-4 ${program.favorite ? 'text-red-500 fill-red-500' : 'text-gray-400'}`} />
                        </Button>
                      </div>
                    </div>
                    
                    <CardContent className="pt-4">
                      <div className="mb-3">
                        <Badge className="capitalize bg-blue-600 mb-2">
                          {program.category}
                        </Badge>
                        <h3 className="text-lg font-bold text-white mb-1">
                          {program.name}
                        </h3>
                        <p className="text-sm text-gray-400 line-clamp-2">
                          {program.description}
                        </p>
                      </div>
                      
                      <div className="flex flex-wrap gap-1 mb-3">
                        {program.muscleGroups.map(muscle => (
                          <Badge key={muscle} variant="outline" className="text-xs">
                            {muscle}
                          </Badge>
                        ))}
                      </div>
                      
                      <div className="flex items-center justify-between text-sm text-gray-400 mb-1">
                        <div>{program.workoutsPerWeek} workouts / week</div>
                        <div>{program.tokensRequired > 0 ? `${program.tokensRequired} tokens` : 'Free'}</div>
                      </div>
                      
                      {program.progression && (
                        <div className="mb-3">
                          <div className="text-xs text-gray-400 mb-1">Progression Type</div>
                          <div className="text-sm">{program.progression}</div>
                        </div>
                      )}
                    </CardContent>
                    
                    <CardFooter className="border-t border-gray-700 pt-3 pb-3">
                      <Button 
                        asChild 
                        className={program.enrolled ? 'bg-green-600 hover:bg-green-700' : ''}
                        variant={program.enrolled ? 'default' : 'outline'}
                      >
                        <Link href={program.enrolled ? '/team/workout-programs/workout' : `/team/workout-programs/${program.id}`}>
                          {program.enrolled ? (
                            <>
                              <PlayCircle className="h-4 w-4 mr-2" /> Continue
                            </>
                          ) : (
                            <>
                              <PlusCircle className="h-4 w-4 mr-2" /> Enroll
                            </>
                          )}
                        </Link>
                      </Button>
                    </CardFooter>
                  </Card>
                ))}
              </div>
            )}
            
            {/* List View */}
            {viewMode === 'list' && filteredPrograms.length > 0 && (
              <div className="space-y-4">
                {filteredPrograms.map(program => (
                  <div 
                    key={program.id} 
                    className="bg-gray-800 border border-gray-700 rounded-lg overflow-hidden hover:border-blue-500 transition-all"
                  >
                    <div className="p-4 flex flex-col md:flex-row md:items-center">
                      {/* Program Icon */}
                      <div className="h-20 w-20 bg-gray-700 rounded-lg flex items-center justify-center mb-4 md:mb-0 md:mr-4">
                        <Dumbbell className="h-8 w-8 text-gray-600" />
                      </div>
                      
                      {/* Program Details */}
                      <div className="md:flex-grow md:mr-4">
                        <div className="flex flex-wrap items-center gap-2 mb-1">
                          <Badge className="capitalize bg-blue-600">
                            {program.category}
                          </Badge>
                          <Badge variant="outline" className="capitalize">
                            {program.level}
                          </Badge>
                          {program.favorite && (
                            <Heart className="h-4 w-4 text-red-500 fill-red-500" />
                          )}
                        </div>
                        
                        <h3 className="text-lg font-bold text-white mb-1">
                          {program.name}
                        </h3>
                        
                        <p className="text-sm text-gray-400 mb-2 line-clamp-1">
                          {program.description}
                        </p>
                        
                        <div className="flex flex-wrap gap-3 text-sm">
                          <div className="flex items-center text-gray-400">
                            <Clock className="h-4 w-4 mr-1" /> {program.duration}
                          </div>
                          <div className="flex items-center text-gray-400">
                            <Calendar className="h-4 w-4 mr-1" /> {program.workoutsPerWeek} workouts/week
                          </div>
                          {program.tokensRequired > 0 ? (
                            <div className="flex items-center text-blue-400">
                              <Star className="h-4 w-4 mr-1" /> {program.tokensRequired} tokens
                            </div>
                          ) : (
                            <div className="flex items-center text-green-400">
                              <Badge variant="outline" className="text-xs border-green-500 text-green-400">Free</Badge>
                            </div>
                          )}
                        </div>
                      </div>
                      
                      {/* Action Button */}
                      <div className="mt-4 md:mt-0">
                        <Button 
                          asChild 
                          className={program.enrolled ? 'bg-green-600 hover:bg-green-700' : ''}
                          variant={program.enrolled ? 'default' : 'outline'}
                        >
                          <Link href={program.enrolled ? '/team/workout-programs/workout' : `/team/workout-programs/${program.id}`}>
                            {program.enrolled ? (
                              <>
                                <PlayCircle className="h-4 w-4 mr-2" /> Continue
                              </>
                            ) : (
                              <>
                                <PlusCircle className="h-4 w-4 mr-2" /> Enroll
                              </>
                            )}
                          </Link>
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
} 
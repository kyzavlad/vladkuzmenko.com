'use client';

import { useState } from 'react';
import { Trophy, Star, Award, Lock, TrendingUp, Gift, Info, Search } from 'lucide-react';
import { Achievement, AchievementCategory } from '../types';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { format } from 'date-fns';

interface AchievementsGalleryProps {
  achievements: Achievement[];
  totalCoins: number;
  totalExperience: number;
}

export default function AchievementsGallery({ 
  achievements, 
  totalCoins, 
  totalExperience 
}: AchievementsGalleryProps) {
  const [activeCategory, setActiveCategory] = useState<AchievementCategory | 'all'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedAchievement, setSelectedAchievement] = useState<Achievement | null>(null);
  
  // Filter achievements by category and search query
  const filteredAchievements = achievements.filter(achievement => {
    const matchesCategory = activeCategory === 'all' || achievement.category === activeCategory;
    const matchesSearch = 
      achievement.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      achievement.description.toLowerCase().includes(searchQuery.toLowerCase());
    
    return matchesCategory && matchesSearch;
  });
  
  // Group achievements by completion status
  const unlockedAchievements = filteredAchievements.filter(a => a.completed);
  const inProgressAchievements = filteredAchievements.filter(a => !a.completed && !a.secret);
  const secretAchievements = filteredAchievements.filter(a => !a.completed && a.secret);
  
  // Calculate totals
  const totalUnlocked = achievements.filter(a => a.completed).length;
  const totalAchievements = achievements.length;
  const unlockedPercentage = Math.round((totalUnlocked / totalAchievements) * 100);
  
  // Calculate rewards from unlocked achievements
  const earnedCoins = achievements
    .filter(a => a.completed)
    .reduce((total, a) => total + a.reward.coins, 0);
  
  const earnedXP = achievements
    .filter(a => a.completed)
    .reduce((total, a) => total + a.reward.experience, 0);
  
  // Categories with counts
  const categories: { id: AchievementCategory | 'all'; label: string; icon: JSX.Element; count: number }[] = [
    { 
      id: 'all', 
      label: 'All', 
      icon: <Trophy className="h-4 w-4" />, 
      count: achievements.length 
    },
    { 
      id: 'workout', 
      label: 'Workout', 
      icon: <TrendingUp className="h-4 w-4" />, 
      count: achievements.filter(a => a.category === 'workout').length 
    },
    { 
      id: 'strength', 
      label: 'Strength', 
      icon: <Award className="h-4 w-4" />, 
      count: achievements.filter(a => a.category === 'strength').length 
    },
    { 
      id: 'consistency', 
      label: 'Consistency', 
      icon: <Star className="h-4 w-4" />, 
      count: achievements.filter(a => a.category === 'consistency').length 
    },
    { 
      id: 'nutrition', 
      label: 'Nutrition', 
      icon: <Trophy className="h-4 w-4" />, 
      count: achievements.filter(a => a.category === 'nutrition').length 
    },
    { 
      id: 'milestone', 
      label: 'Milestones', 
      icon: <Trophy className="h-4 w-4" />, 
      count: achievements.filter(a => a.category === 'milestone').length 
    }
  ];
  
  // Rendering helper for individual achievement card
  const renderAchievementCard = (achievement: Achievement) => {
    const progressPercentage = Math.min(
      100, 
      Math.round((achievement.progress / achievement.requirement) * 100)
    );
    
    return (
      <div 
        key={achievement.id} 
        className={`bg-gray-750 rounded-lg overflow-hidden cursor-pointer transition-all duration-200 hover:bg-gray-700 ${
          achievement.completed ? 'border border-yellow-600/30' : 'border border-gray-700'
        }`}
        onClick={() => setSelectedAchievement(achievement)}
      >
        <div className="p-4">
          <div className="flex items-start gap-3">
            <div className={`w-12 h-12 rounded-lg flex items-center justify-center text-2xl ${
              achievement.completed 
                ? 'bg-yellow-500/20 text-yellow-500' 
                : 'bg-gray-700 text-gray-500'
            }`}>
              {achievement.icon}
            </div>
            <div className="flex-grow">
              <div className="flex items-center justify-between">
                <h4 className="text-white font-medium">{achievement.title}</h4>
                {achievement.completed && (
                  <Badge className="bg-yellow-600 text-xs">Unlocked</Badge>
                )}
              </div>
              <p className="text-gray-400 text-sm mt-1">{achievement.description}</p>
            </div>
          </div>
          
          {!achievement.completed && !achievement.secret && (
            <div className="mt-3">
              <div className="flex justify-between items-center text-xs mb-1">
                <span className="text-gray-400">Progress</span>
                <span className="text-white">{achievement.progress}/{achievement.requirement}</span>
              </div>
              <Progress value={progressPercentage} className="h-2" />
            </div>
          )}
          
          {achievement.secret && !achievement.completed && (
            <div className="mt-3 flex items-center text-sm text-gray-400">
              <Lock className="h-3 w-3 mr-1" />
              <span>Secret achievement - keep exploring!</span>
            </div>
          )}
          
          <div className="mt-3 flex items-center text-xs text-gray-500">
            <Gift className="h-3 w-3 mr-1" />
            <span>Rewards: {achievement.reward.experience} XP, {achievement.reward.coins} coins{
              achievement.reward.specialReward ? ` + ${achievement.reward.specialReward.title}` : ''
            }</span>
          </div>
        </div>
      </div>
    );
  };
  
  return (
    <div className="achievements-gallery">
      {/* Achievements Summary */}
      <Card className="bg-gray-800 border-gray-700 mb-6">
        <CardContent className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Progress Overview */}
            <div className="flex items-center space-x-4">
              <div className="relative">
                <svg className="w-20 h-20" viewBox="0 0 100 100">
                  <circle 
                    className="text-gray-700 stroke-current" 
                    strokeWidth="10" 
                    fill="transparent" 
                    r="40" 
                    cx="50" 
                    cy="50" 
                  />
                  <circle 
                    className="text-yellow-500 stroke-current" 
                    strokeWidth="10" 
                    fill="transparent" 
                    r="40" 
                    cx="50" 
                    cy="50" 
                    strokeDasharray={`${2 * Math.PI * 40}`}
                    strokeDashoffset={`${2 * Math.PI * 40 * (1 - unlockedPercentage / 100)}`}
                    strokeLinecap="round"
                    transform="rotate(-90 50 50)"
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center text-white">
                  {unlockedPercentage}%
                </div>
              </div>
              <div>
                <h3 className="text-white text-lg font-medium">Your Progress</h3>
                <p className="text-gray-400">
                  {totalUnlocked} of {totalAchievements} achievements unlocked
                </p>
              </div>
            </div>
            
            {/* Rewards Earned */}
            <div className="bg-gray-750 rounded-lg p-4">
              <h3 className="text-white font-medium mb-2 flex items-center">
                <Gift className="h-4 w-4 mr-2 text-green-500" />
                Rewards Earned
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <div className="text-gray-400 text-xs">Experience</div>
                  <div className="text-white text-lg font-medium">
                    {earnedXP} <span className="text-xs text-gray-500">XP</span>
                  </div>
                </div>
                <div>
                  <div className="text-gray-400 text-xs">Coins</div>
                  <div className="text-white text-lg font-medium">
                    {earnedCoins} <span className="text-xs text-gray-500">coins</span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Next Milestone */}
            <div className="bg-gray-750 rounded-lg p-4">
              <h3 className="text-white font-medium mb-2 flex items-center">
                <Trophy className="h-4 w-4 mr-2 text-yellow-500" />
                Next Milestone
              </h3>
              
              {inProgressAchievements.length > 0 ? (
                <div>
                  <div className="text-white text-sm">
                    {inProgressAchievements[0].title}
                  </div>
                  <div className="text-gray-400 text-xs mt-1">
                    {inProgressAchievements[0].progress}/{inProgressAchievements[0].requirement} progress
                  </div>
                  <Progress 
                    value={(inProgressAchievements[0].progress / inProgressAchievements[0].requirement) * 100} 
                    className="h-2 mt-2" 
                  />
                </div>
              ) : (
                <div className="text-gray-400 text-sm">
                  All current achievements completed!
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Search and Filters */}
      <div className="flex flex-col md:flex-row gap-4 mb-6">
        <div className="relative flex-grow">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-500" />
          <Input
            placeholder="Search achievements"
            className="bg-gray-800 border-gray-700 pl-9"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        
        <div className="flex overflow-x-auto pb-2 md:pb-0 gap-2">
          {categories.map(category => (
            <Button
              key={category.id}
              variant={activeCategory === category.id ? "default" : "outline"}
              size="sm"
              onClick={() => setActiveCategory(category.id)}
              className="whitespace-nowrap"
            >
              {category.icon}
              <span className="ml-2">{category.label}</span>
              <Badge className="ml-2 text-xs" variant={activeCategory === category.id ? "default" : "outline"}>
                {category.count}
              </Badge>
            </Button>
          ))}
        </div>
      </div>
      
      {/* Achievements Tabs */}
      <Tabs defaultValue="unlocked" className="w-full">
        <TabsList className="grid grid-cols-3 mb-6">
          <TabsTrigger value="unlocked">
            Unlocked <Badge className="ml-2">{unlockedAchievements.length}</Badge>
          </TabsTrigger>
          <TabsTrigger value="in-progress">
            In Progress <Badge className="ml-2">{inProgressAchievements.length}</Badge>
          </TabsTrigger>
          <TabsTrigger value="locked">
            Locked <Badge className="ml-2">{secretAchievements.length}</Badge>
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="unlocked" className="space-y-4">
          {unlockedAchievements.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {unlockedAchievements.map(renderAchievementCard)}
            </div>
          ) : (
            <div className="bg-gray-750 p-12 rounded-lg text-center">
              <Trophy className="h-12 w-12 mx-auto text-gray-600 mb-4" />
              <h3 className="text-white text-lg font-medium mb-2">No achievements unlocked yet</h3>
              <p className="text-gray-400 max-w-md mx-auto mb-4">
                Complete your first workout and start building your fitness journey to unlock achievements
              </p>
              <Button>Start a Workout</Button>
            </div>
          )}
        </TabsContent>
        
        <TabsContent value="in-progress" className="space-y-4">
          {inProgressAchievements.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {inProgressAchievements.map(renderAchievementCard)}
            </div>
          ) : (
            <div className="bg-gray-750 p-12 rounded-lg text-center">
              <Award className="h-12 w-12 mx-auto text-gray-600 mb-4" />
              <h3 className="text-white text-lg font-medium mb-2">All caught up!</h3>
              <p className="text-gray-400 max-w-md mx-auto">
                You've completed all available achievements. Check back soon for new challenges!
              </p>
            </div>
          )}
        </TabsContent>
        
        <TabsContent value="locked" className="space-y-4">
          {secretAchievements.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {secretAchievements.map(renderAchievementCard)}
            </div>
          ) : (
            <div className="bg-gray-750 p-12 rounded-lg text-center">
              <Lock className="h-12 w-12 mx-auto text-gray-600 mb-4" />
              <h3 className="text-white text-lg font-medium mb-2">No locked achievements</h3>
              <p className="text-gray-400 max-w-md mx-auto">
                All achievements are available to view. Keep exploring to discover more!
              </p>
            </div>
          )}
        </TabsContent>
      </Tabs>
      
      {/* Achievement Detail Dialog */}
      {selectedAchievement && (
        <Dialog open={!!selectedAchievement} onOpenChange={(open) => !open && setSelectedAchievement(null)}>
          <DialogContent className="bg-gray-800 text-white border-gray-700">
            <DialogHeader>
              <DialogTitle className="flex items-center">
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-xl mr-2 ${
                  selectedAchievement.completed 
                    ? 'bg-yellow-500/20 text-yellow-500' 
                    : 'bg-gray-700 text-gray-500'
                }`}>
                  {selectedAchievement.icon}
                </div>
                {selectedAchievement.title}
              </DialogTitle>
              <DialogDescription className="text-gray-400">
                {selectedAchievement.description}
              </DialogDescription>
            </DialogHeader>
            
            <div className="space-y-4">
              {selectedAchievement.completed ? (
                <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
                  <div className="flex items-center text-yellow-500 mb-2">
                    <Trophy className="h-5 w-5 mr-2" />
                    <span className="font-medium">Achievement Unlocked!</span>
                  </div>
                  <div className="text-gray-300 text-sm">
                    You unlocked this achievement on {
                      selectedAchievement.completedDate && 
                      format(new Date(selectedAchievement.completedDate), 'MMMM d, yyyy')
                    }
                  </div>
                </div>
              ) : (
                <div className="bg-gray-750 rounded-lg p-4">
                  <div className="flex items-center mb-2">
                    <Info className="h-5 w-5 mr-2 text-blue-500" />
                    <span className="text-white font-medium">Progress Details</span>
                  </div>
                  
                  {!selectedAchievement.secret ? (
                    <>
                      <div className="flex justify-between items-center text-sm mb-2">
                        <span className="text-gray-400">Progress</span>
                        <span className="text-white">
                          {selectedAchievement.progress} / {selectedAchievement.requirement}
                          {selectedAchievement.type === 'cumulative' && ' completed'}
                        </span>
                      </div>
                      <Progress 
                        value={(selectedAchievement.progress / selectedAchievement.requirement) * 100} 
                        className="h-2 mb-3" 
                      />
                      <div className="text-gray-400 text-sm">
                        {100 - Math.round((selectedAchievement.progress / selectedAchievement.requirement) * 100)}% remaining to unlock this achievement
                      </div>
                    </>
                  ) : (
                    <div className="text-gray-400 text-sm">
                      This is a secret achievement. Keep exploring to discover how to unlock it!
                    </div>
                  )}
                </div>
              )}
              
              <div className="bg-gray-750 rounded-lg p-4">
                <div className="flex items-center mb-3">
                  <Gift className="h-5 w-5 mr-2 text-green-500" />
                  <span className="text-white font-medium">Rewards</span>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-700 p-3 rounded-lg">
                    <div className="text-gray-400 text-xs">Experience</div>
                    <div className="text-white text-lg font-medium flex items-center">
                      {selectedAchievement.reward.experience}
                      <Star className="h-4 w-4 ml-1 text-yellow-500" />
                    </div>
                  </div>
                  
                  <div className="bg-gray-700 p-3 rounded-lg">
                    <div className="text-gray-400 text-xs">Coins</div>
                    <div className="text-white text-lg font-medium">
                      {selectedAchievement.reward.coins}
                    </div>
                  </div>
                </div>
                
                {selectedAchievement.reward.specialReward && (
                  <div className="mt-3 bg-blue-900/30 border border-blue-800 p-3 rounded-lg">
                    <div className="text-blue-400 font-medium">
                      {selectedAchievement.reward.specialReward.title}
                    </div>
                    <div className="text-gray-300 text-sm mt-1">
                      {selectedAchievement.reward.specialReward.description}
                    </div>
                    {selectedAchievement.reward.specialReward.duration && (
                      <div className="text-gray-400 text-xs mt-1">
                        Valid for {selectedAchievement.reward.specialReward.duration} days after unlocking
                      </div>
                    )}
                  </div>
                )}
              </div>
              
              <div className="bg-gray-750 rounded-lg p-4">
                <div className="flex items-center mb-2">
                  <Info className="h-5 w-5 mr-2 text-gray-400" />
                  <span className="text-white font-medium">Achievement Details</span>
                </div>
                
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Category</span>
                    <span className="text-white capitalize">{selectedAchievement.category}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Type</span>
                    <span className="text-white capitalize">{selectedAchievement.type.replace('_', ' ')}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">ID</span>
                    <span className="text-gray-300 text-xs">{selectedAchievement.id}</span>
                  </div>
                </div>
              </div>
            </div>
            
            <DialogFooter>
              <Button variant="outline" onClick={() => setSelectedAchievement(null)}>Close</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
} 
'use client';

import { useState, useEffect } from 'react';
import { 
  Brain, 
  CheckCircle, 
  XCircle, 
  Sparkles, 
  Layers, 
  Dumbbell, 
  Clock, 
  ChevronRight, 
  Info, 
  MessageSquare, 
  ThumbsUp, 
  ThumbsDown, 
  RotateCcw,
  Utensils,
  Zap,
  Heart
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Progress } from '@/components/ui/progress';
import { mockRecommendations, mockRecommendationContext } from '../utils/sample-data';

interface RecommendationEngineProps {
  userId?: string;
  onAccept?: (recommendationId: string) => void;
  onReject?: (recommendationId: string) => void;
  onModify?: (recommendationId: string, modifications: any) => void;
}

export default function RecommendationEngine({ 
  userId = 'user123', 
  onAccept, 
  onReject, 
  onModify 
}: RecommendationEngineProps) {
  const [activeTab, setActiveTab] = useState<string>('workout');
  const [activeRecommendation, setActiveRecommendation] = useState<string | null>(null);
  const [contextVisible, setContextVisible] = useState<boolean>(false);
  const [userFeedback, setUserFeedback] = useState<Record<string, { action: string; rating?: number; reason?: string }>>({});
  
  // Get recommendations for the current tab
  const filteredRecommendations = mockRecommendations.filter(
    recommendation => recommendation.type === activeTab
  );
  
  // Accept recommendation
  const handleAccept = (recommendationId: string) => {
    setUserFeedback(prev => ({
      ...prev,
      [recommendationId]: { action: 'accepted' }
    }));
    
    if (onAccept) {
      onAccept(recommendationId);
    }
  };
  
  // Reject recommendation
  const handleReject = (recommendationId: string) => {
    setUserFeedback(prev => ({
      ...prev,
      [recommendationId]: { action: 'rejected' }
    }));
    
    if (onReject) {
      onReject(recommendationId);
    }
  };
  
  // Rate recommendation quality
  const handleRate = (recommendationId: string, rating: number) => {
    setUserFeedback(prev => ({
      ...prev,
      [recommendationId]: { 
        ...prev[recommendationId],
        rating 
      }
    }));
  };
  
  // Format date
  const formatDate = (date: Date): string => {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: 'numeric'
    }).format(date);
  };
  
  // Get icon for recommendation type
  const getRecommendationIcon = (type: string) => {
    switch (type) {
      case 'workout':
        return <Dumbbell className="h-5 w-5 text-blue-400" />;
      case 'nutrition':
        return <Utensils className="h-5 w-5 text-green-400" />;
      case 'recovery':
        return <Heart className="h-5 w-5 text-yellow-400" />;
      case 'goal':
        return <Sparkles className="h-5 w-5 text-purple-400" />;
      case 'habit':
        return <Zap className="h-5 w-5 text-orange-400" />;
      default:
        return <Info className="h-5 w-5 text-gray-400" />;
    }
  };
  
  // Render appropriate content based on recommendation type
  const renderRecommendationContent = (recommendation: typeof mockRecommendations[0]) => {
    switch (recommendation.type) {
      case 'workout':
        return (
          <div>
            {recommendation.workoutRecommendation && (
              <div className="space-y-4">
                {recommendation.workoutRecommendation.modifications && recommendation.workoutRecommendation.modifications.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium text-white mb-2">Exercise Modifications:</h4>
                    <div className="space-y-3">
                      {recommendation.workoutRecommendation.modifications.map((mod, index) => (
                        <div key={index} className="bg-gray-750 rounded-lg p-3">
                          <div className="flex justify-between items-start">
                            <div>
                              <div className="text-white font-medium">{mod.originalExercise}</div>
                              <div className="text-gray-400 text-sm">Changed to: {mod.newExercise}</div>
                            </div>
                            <Badge variant="outline" className="bg-blue-900/20 text-blue-400">
                              Modified
                            </Badge>
                          </div>
                          <div className="text-gray-400 text-sm mt-2">
                            Reason: {mod.reason}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {recommendation.workoutRecommendation.intensityAdjustment && (
                  <div className="bg-gray-750 rounded-lg p-3">
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="text-white font-medium">Intensity Adjustment</div>
                        <div className="text-gray-400 text-sm">
                          {recommendation.workoutRecommendation.intensityAdjustment > 0 ? 'Increased' : 'Decreased'} by {Math.abs(recommendation.workoutRecommendation.intensityAdjustment)}%
                        </div>
                      </div>
                      <Badge variant={recommendation.workoutRecommendation.intensityAdjustment > 0 ? 'default' : 'destructive'}>
                        {recommendation.workoutRecommendation.intensityAdjustment > 0 ? 'Harder' : 'Easier'}
                      </Badge>
                    </div>
                  </div>
                )}
                
                {recommendation.workoutRecommendation.focusAreas && (
                  <div>
                    <h4 className="text-sm font-medium text-white mb-2">Focus Areas:</h4>
                    <div className="flex flex-wrap gap-2">
                      {recommendation.workoutRecommendation.focusAreas.map((area, index) => (
                        <Badge key={index} variant="outline" className="bg-blue-900/20">
                          {area}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        );
        
      case 'nutrition':
        return (
          <div>
            {recommendation.nutritionRecommendation && (
              <div className="space-y-4">
                {(recommendation.nutritionRecommendation.calorieAdjustment !== undefined || 
                  recommendation.nutritionRecommendation.macroAdjustments) && (
                  <div className="bg-gray-750 rounded-lg p-3">
                    <h4 className="text-sm font-medium text-white mb-2">Nutrition Adjustments:</h4>
                    
                    {recommendation.nutritionRecommendation.calorieAdjustment !== undefined && (
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-gray-400">Calorie Adjustment:</span>
                        <Badge variant={recommendation.nutritionRecommendation.calorieAdjustment >= 0 ? 'default' : 'destructive'}>
                          {recommendation.nutritionRecommendation.calorieAdjustment > 0 ? '+' : ''}
                          {recommendation.nutritionRecommendation.calorieAdjustment}%
                        </Badge>
                      </div>
                    )}
                    
                    {recommendation.nutritionRecommendation.macroAdjustments && (
                      <div className="space-y-1">
                        {recommendation.nutritionRecommendation.macroAdjustments.protein !== undefined && (
                          <div className="flex justify-between items-center">
                            <span className="text-gray-400">Protein:</span>
                            <Badge variant={recommendation.nutritionRecommendation.macroAdjustments.protein >= 0 ? 'default' : 'destructive'}>
                              {recommendation.nutritionRecommendation.macroAdjustments.protein > 0 ? '+' : ''}
                              {recommendation.nutritionRecommendation.macroAdjustments.protein}%
                            </Badge>
                          </div>
                        )}
                        
                        {recommendation.nutritionRecommendation.macroAdjustments.carbs !== undefined && (
                          <div className="flex justify-between items-center">
                            <span className="text-gray-400">Carbs:</span>
                            <Badge variant={recommendation.nutritionRecommendation.macroAdjustments.carbs >= 0 ? 'default' : 'destructive'}>
                              {recommendation.nutritionRecommendation.macroAdjustments.carbs > 0 ? '+' : ''}
                              {recommendation.nutritionRecommendation.macroAdjustments.carbs}%
                            </Badge>
                          </div>
                        )}
                        
                        {recommendation.nutritionRecommendation.macroAdjustments.fat !== undefined && (
                          <div className="flex justify-between items-center">
                            <span className="text-gray-400">Fat:</span>
                            <Badge variant={recommendation.nutritionRecommendation.macroAdjustments.fat >= 0 ? 'default' : 'destructive'}>
                              {recommendation.nutritionRecommendation.macroAdjustments.fat > 0 ? '+' : ''}
                              {recommendation.nutritionRecommendation.macroAdjustments.fat}%
                            </Badge>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
                
                {recommendation.nutritionRecommendation.foodSuggestions && (
                  <div>
                    <h4 className="text-sm font-medium text-white mb-2">Suggested Foods:</h4>
                    <div className="grid grid-cols-2 gap-2">
                      {recommendation.nutritionRecommendation.foodSuggestions.map((food, index) => (
                        <div key={index} className="bg-gray-750 rounded-lg p-2 text-gray-300 text-sm flex items-center">
                          <div className="w-2 h-2 rounded-full bg-green-500 mr-2"></div>
                          {food}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {recommendation.nutritionRecommendation.hydrationFocus && (
                  <div className="bg-blue-900/20 text-blue-300 p-3 rounded-lg text-sm flex items-center">
                    <Info className="h-4 w-4 mr-2" />
                    <span>Focus on hydration today</span>
                  </div>
                )}
              </div>
            )}
          </div>
        );
        
      case 'recovery':
        return (
          <div>
            {recommendation.recoveryRecommendation && (
              <div className="space-y-4">
                {recommendation.recoveryRecommendation.sleepFocus && (
                  <div className="bg-purple-900/20 text-purple-300 p-3 rounded-lg text-sm flex items-center">
                    <Clock className="h-4 w-4 mr-2" />
                    <span>Prioritize sleep quality tonight</span>
                  </div>
                )}
                
                {recommendation.recoveryRecommendation.mobilityWork && (
                  <div>
                    <h4 className="text-sm font-medium text-white mb-2">Mobility Focus Areas:</h4>
                    <div className="grid grid-cols-2 gap-2">
                      {recommendation.recoveryRecommendation.mobilityWork.map((area, index) => (
                        <div key={index} className="bg-gray-750 rounded-lg p-2 text-gray-300 text-sm flex items-center">
                          <div className="w-2 h-2 rounded-full bg-yellow-500 mr-2"></div>
                          {area}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {recommendation.recoveryRecommendation.stressReduction && (
                  <div>
                    <h4 className="text-sm font-medium text-white mb-2">Stress Reduction:</h4>
                    <div className="grid grid-cols-2 gap-2">
                      {recommendation.recoveryRecommendation.stressReduction.map((technique, index) => (
                        <div key={index} className="bg-gray-750 rounded-lg p-2 text-gray-300 text-sm flex items-center">
                          <div className="w-2 h-2 rounded-full bg-blue-500 mr-2"></div>
                          {technique}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {recommendation.recoveryRecommendation.activeRecovery && (
                  <div className="bg-gray-750 rounded-lg p-3">
                    <h4 className="text-sm font-medium text-white mb-1">Active Recovery:</h4>
                    <div className="text-gray-300 text-sm">
                      {recommendation.recoveryRecommendation.activeRecovery}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        );
        
      default:
        return <div className="text-gray-400">No detailed recommendation content available</div>;
    }
  };
  
  return (
    <div className="recommendation-engine">
      <div className="flex justify-between items-start mb-6">
        <div>
          <h2 className="text-white text-xl font-medium flex items-center">
            <Brain className="h-5 w-5 text-indigo-400 mr-2" />
            AI Recommendation Engine
          </h2>
          <p className="text-gray-400">
            Personalized suggestions based on your current state, goals, and behavioral patterns
          </p>
        </div>
        
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => setContextVisible(!contextVisible)}
              >
                <Info className="mr-2 h-4 w-4" />
                Context Data
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>View the data being used to generate recommendations</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
      
      {/* Context Data Display */}
      {contextVisible && (
        <Card className="mb-6 bg-gray-800 border-gray-700">
          <CardHeader className="pb-3">
            <CardTitle className="text-white text-base">Recommendation Context</CardTitle>
            <CardDescription>
              Data used to generate your personalized recommendations
            </CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <h3 className="text-white font-medium mb-2">Current State</h3>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Energy Level:</span>
                  <Badge>{mockRecommendationContext.currentEnergySelf}/10</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Stress Level:</span>
                  <Badge>{mockRecommendationContext.currentStressSelf}/10</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Motivation:</span>
                  <Badge>{mockRecommendationContext.currentMotivationSelf}/10</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Soreness:</span>
                  <Badge>{mockRecommendationContext.currentSorenessSelf}/10</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Sleep Last Night:</span>
                  <Badge>{mockRecommendationContext.sleepLastNight} hrs</Badge>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="text-white font-medium mb-2">Biometrics & Activity</h3>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Heart Rate Variability:</span>
                  <Badge>{mockRecommendationContext.heartRateVariability} ms</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Resting Heart Rate:</span>
                  <Badge>{mockRecommendationContext.restingHeartRate} bpm</Badge>
                </div>
                {mockRecommendationContext.lastWorkout && (
                  <>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Last Workout Type:</span>
                      <Badge>{mockRecommendationContext.lastWorkout.type}</Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Last Workout Intensity:</span>
                      <Badge>{mockRecommendationContext.lastWorkout.intensity}/10</Badge>
                    </div>
                  </>
                )}
              </div>
            </div>
            
            <div>
              <h3 className="text-white font-medium mb-2">Environmental Factors</h3>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Weather:</span>
                  <Badge>{mockRecommendationContext.weather}</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Temperature:</span>
                  <Badge>{mockRecommendationContext.temperature}°C</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Schedule:</span>
                  <Badge className="capitalize">{mockRecommendationContext.schedule}</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Travel Status:</span>
                  <Badge>{mockRecommendationContext.travelStatus ? 'Traveling' : 'At Home'}</Badge>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
      
      {/* Recommendation Tabs */}
      <Tabs 
        defaultValue="workout" 
        value={activeTab}
        onValueChange={setActiveTab}
        className="mb-6"
      >
        <TabsList className="mb-4">
          <TabsTrigger value="workout" className="flex items-center gap-1">
            <Dumbbell className="h-4 w-4" />
            Workout
          </TabsTrigger>
          <TabsTrigger value="nutrition" className="flex items-center gap-1">
            <Utensils className="h-4 w-4" />
            Nutrition
          </TabsTrigger>
          <TabsTrigger value="recovery" className="flex items-center gap-1">
            <Heart className="h-4 w-4" />
            Recovery
          </TabsTrigger>
        </TabsList>
        
        {['workout', 'nutrition', 'recovery'].map(tab => (
          <TabsContent key={tab} value={tab} className="mt-0">
            {filteredRecommendations.length > 0 ? (
              <div className="space-y-4">
                {filteredRecommendations.map(recommendation => {
                  const isActive = activeRecommendation === recommendation.id;
                  const feedback = userFeedback[recommendation.id];
                  
                  return (
                    <Card 
                      key={recommendation.id} 
                      className={`border transition-colors ${
                        feedback?.action === 'accepted' 
                          ? 'bg-green-900/10 border-green-800/50' 
                          : feedback?.action === 'rejected'
                            ? 'bg-red-900/10 border-red-800/50'
                            : isActive 
                              ? 'bg-indigo-900/10 border-indigo-800/50' 
                              : 'bg-gray-800 border-gray-700'
                      }`}
                    >
                      <CardHeader className="pb-3">
                        <div className="flex justify-between items-start">
                          <div className="flex items-center gap-2">
                            {getRecommendationIcon(recommendation.type)}
                            <CardTitle className="text-white text-base">{recommendation.title}</CardTitle>
                          </div>
                          <div className="flex items-center gap-2">
                            <Badge variant={
                              recommendation.confidenceScore > 85 ? 'default' :
                              recommendation.confidenceScore > 70 ? 'secondary' : 'outline'
                            }>
                              {recommendation.confidenceScore}% Confidence
                            </Badge>
                            {feedback?.action === 'accepted' && (
                              <CheckCircle className="h-5 w-5 text-green-500" />
                            )}
                            {feedback?.action === 'rejected' && (
                              <XCircle className="h-5 w-5 text-red-500" />
                            )}
                          </div>
                        </div>
                        <CardDescription className="mt-1">
                          {recommendation.description}
                        </CardDescription>
                      </CardHeader>
                      
                      <CardContent>
                        {/* Show detail when active */}
                        {isActive && (
                          <div>
                            <div className="mb-4">
                              <h4 className="text-sm font-medium text-white mb-2">Why we're recommending this:</h4>
                              <ul className="text-sm text-gray-400 space-y-1 ml-4 list-disc">
                                {recommendation.reasoning.map((reason, index) => (
                                  <li key={index}>{reason}</li>
                                ))}
                              </ul>
                            </div>
                            
                            {renderRecommendationContent(recommendation)}
                            
                            {/* Feedback system */}
                            {!feedback?.action && (
                              <div className="flex justify-end gap-2 mt-4">
                                <Button 
                                  variant="outline" 
                                  size="sm"
                                  onClick={() => handleReject(recommendation.id)}
                                >
                                  <XCircle className="mr-2 h-4 w-4" />
                                  Not Now
                                </Button>
                                <Button 
                                  size="sm"
                                  onClick={() => handleAccept(recommendation.id)}
                                >
                                  <CheckCircle className="mr-2 h-4 w-4" />
                                  Apply Recommendation
                                </Button>
                              </div>
                            )}
                            
                            {feedback?.action && (
                              <div className="mt-4 pt-4 border-t border-gray-700">
                                <div className="text-gray-400 text-sm mb-2">
                                  How helpful was this recommendation?
                                </div>
                                <div className="flex gap-2">
                                  {[1, 2, 3, 4, 5].map(rating => (
                                    <Button
                                      key={rating}
                                      variant={feedback.rating === rating ? 'default' : 'outline'}
                                      size="sm"
                                      className="px-3"
                                      onClick={() => handleRate(recommendation.id, rating)}
                                    >
                                      {rating}
                                    </Button>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                        
                        {/* Toggle button when not expanded */}
                        {!isActive && !feedback?.action && (
                          <div className="flex justify-end">
                            <Button 
                              variant="ghost" 
                              size="sm"
                              onClick={() => setActiveRecommendation(recommendation.id)}
                            >
                              View Details
                              <ChevronRight className="ml-1 h-4 w-4" />
                            </Button>
                          </div>
                        )}
                      </CardContent>
                      
                      {feedback?.action && !isActive && (
                        <CardFooter className="border-t border-gray-700 pt-3 flex justify-between">
                          <div className="text-sm text-gray-400">
                            {feedback.action === 'accepted' ? 'Recommendation applied' : 'Recommendation skipped'}
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setActiveRecommendation(recommendation.id)}
                          >
                            Details
                            <ChevronRight className="ml-1 h-4 w-4" />
                          </Button>
                        </CardFooter>
                      )}
                    </Card>
                  );
                })}
              </div>
            ) : (
              <Card className="bg-gray-800 border-gray-700">
                <CardContent className="pt-6 pb-6 text-center">
                  <div className="mx-auto w-12 h-12 rounded-full bg-gray-700 flex items-center justify-center mb-3">
                    <Layers className="h-6 w-6 text-gray-500" />
                  </div>
                  <h3 className="text-white font-medium mb-2">No {activeTab} recommendations</h3>
                  <p className="text-gray-400 text-sm">
                    The AI hasn't generated any {activeTab} recommendations for you at this time.
                    Check back later or update your profile for more personalized recommendations.
                  </p>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        ))}
      </Tabs>
      
      {/* System improvements feedback */}
      <Card className="mt-6 bg-gray-800 border-gray-700">
        <CardHeader className="pb-3">
          <CardTitle className="text-white text-base">Recommendation Quality</CardTitle>
          <CardDescription>
            Help us improve your personalized recommendations
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-start gap-4">
            <div className="bg-indigo-900/20 text-indigo-400 p-2 rounded-full">
              <Brain className="h-5 w-5" />
            </div>
            <div className="flex-1">
              <h3 className="text-white font-medium text-sm">Self-Improving AI System</h3>
              <p className="text-gray-400 text-sm mt-1">
                Each time you interact with a recommendation, our system learns more about what works for you.
                Your feedback directly improves the quality and relevance of future suggestions.
              </p>
              
              <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="bg-gray-750 p-3 rounded-lg">
                  <div className="text-white text-sm font-medium mb-1">Accuracy</div>
                  <Progress value={85} className="h-1 mb-1" />
                  <div className="text-xs text-gray-500">Based on your feedback</div>
                </div>
                
                <div className="bg-gray-750 p-3 rounded-lg">
                  <div className="text-white text-sm font-medium mb-1">Relevance</div>
                  <Progress value={78} className="h-1 mb-1" />
                  <div className="text-xs text-gray-500">Based on your choices</div>
                </div>
                
                <div className="bg-gray-750 p-3 rounded-lg">
                  <div className="text-white text-sm font-medium mb-1">Adaptation</div>
                  <Progress value={92} className="h-1 mb-1" />
                  <div className="text-xs text-gray-500">Learning from outcomes</div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 
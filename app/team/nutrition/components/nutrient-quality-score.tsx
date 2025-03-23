'use client';

import { Award, ChevronRight, Info } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { DailyNutritionLog } from '../types';

interface NutrientQualityScoreProps {
  nutritionData: DailyNutritionLog;
}

export default function NutrientQualityScore({ nutritionData }: NutrientQualityScoreProps) {
  // Get quality category based on score
  const getQualityCategory = (score: number) => {
    if (score >= 90) return { label: 'Excellent', color: 'bg-green-600', textColor: 'text-green-400' };
    if (score >= 80) return { label: 'Very Good', color: 'bg-green-600', textColor: 'text-green-400' };
    if (score >= 70) return { label: 'Good', color: 'bg-green-600', textColor: 'text-green-400' };
    if (score >= 60) return { label: 'Fair', color: 'bg-yellow-600', textColor: 'text-yellow-400' };
    if (score >= 50) return { label: 'Needs Improvement', color: 'bg-orange-600', textColor: 'text-orange-400' };
    return { label: 'Poor', color: 'bg-red-600', textColor: 'text-red-400' };
  };
  
  const quality = getQualityCategory(nutritionData.nutrientQualityScore || 0);
  
  // Mock nutrient breakdown data - in a real app this would come from the nutritionData
  const nutrientBreakdown = [
    { name: 'Protein Variety', score: 85, maxScore: 100 },
    { name: 'Vegetable Diversity', score: 90, maxScore: 100 },
    { name: 'Healthy Fats', score: 75, maxScore: 100 },
    { name: 'Micronutrients', score: 80, maxScore: 100 },
    { name: 'Added Sugar', score: 95, maxScore: 100 },
    { name: 'Processed Foods', score: 90, maxScore: 100 },
  ];
  
  // Mock feedback and recommendations
  const recommendations = [
    'Good protein variety from plant and animal sources',
    'Excellent vegetable intake with diverse colors',
    'Consider adding more omega-3 sources like fatty fish or flaxseeds',
    'You could increase vitamin D and magnesium intake'
  ];
  
  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardHeader className="pb-2">
        <div className="flex items-center">
          <Award className="h-5 w-5 text-green-400 mr-2" />
          <CardTitle className="text-white">Nutrient Quality Score</CardTitle>
        </div>
        <CardDescription className="text-gray-400">
          Analysis of your nutrient intake quality
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-center mb-6">
          <div className="relative w-24 h-24 mr-4">
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className={`${quality.textColor} text-2xl font-bold`}>{nutritionData.nutrientQualityScore}</div>
                <div className="text-gray-400 text-xs">/ 100</div>
              </div>
            </div>
            <svg className="w-full h-full" viewBox="0 0 100 100">
              <circle 
                className="text-gray-700 stroke-current" 
                strokeWidth="10" 
                fill="transparent" 
                r="40" 
                cx="50" 
                cy="50" 
              />
              <circle 
                className={`${quality.textColor} stroke-current`}
                strokeWidth="10" 
                fill="transparent" 
                r="40" 
                cx="50" 
                cy="50" 
                strokeDasharray={`${2 * Math.PI * 40}`}
                strokeDashoffset={`${2 * Math.PI * 40 * (1 - (nutritionData.nutrientQualityScore || 0) / 100)}`}
                strokeLinecap="round"
                transform="rotate(-90 50 50)"
              />
            </svg>
          </div>
          
          <div>
            <Badge className={quality.color}>{quality.label}</Badge>
            <p className="text-gray-400 text-sm mt-2">
              Your diet today provides good nutritional value with balanced macros and diverse food groups.
            </p>
          </div>
        </div>
        
        {/* Nutrient Breakdown */}
        <div className="space-y-3 mb-6">
          <h4 className="text-white font-medium">Breakdown by Category</h4>
          
          {nutrientBreakdown.map((nutrient) => (
            <div key={nutrient.name} className="space-y-1">
              <div className="flex justify-between items-center text-xs">
                <span className="text-gray-400">{nutrient.name}</span>
                <span className="text-white">{nutrient.score}/{nutrient.maxScore}</span>
              </div>
              <Progress 
                value={(nutrient.score / nutrient.maxScore) * 100} 
                className="h-1.5 bg-gray-700" 
                indicatorClassName={
                  nutrient.score >= 80 ? "bg-green-500" : 
                  nutrient.score >= 60 ? "bg-yellow-500" : 
                  "bg-red-500"
                }
              />
            </div>
          ))}
        </div>
        
        {/* Recommendations */}
        <div className="bg-gray-700 rounded-md p-3">
          <div className="flex items-center mb-2">
            <Info className="h-4 w-4 text-blue-400 mr-2" />
            <h4 className="text-white text-sm font-medium">Recommendations</h4>
          </div>
          
          <ul className="space-y-2 text-xs text-gray-300">
            {recommendations.map((recommendation, index) => (
              <li key={index} className="flex items-start">
                <span className="text-green-400 mr-2">✓</span>
                <span>{recommendation}</span>
              </li>
            ))}
          </ul>
          
          <Button variant="ghost" size="sm" className="text-blue-400 w-full mt-3 justify-between">
            <span>View detailed analysis</span>
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
} 
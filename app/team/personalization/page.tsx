'use client';

import { 
  Brain, 
  LightbulbIcon, 
  Settings, 
  LineChart, 
  Calendar, 
  Zap, 
  ChevronRight, 
  Plus,
  Sparkles,
  MessageSquare,
  Users,
  PersonStanding,
  BarChart,
  Activity,
  Star
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';

import SmartOnboarding from './components/smart-onboarding';
import RecommendationEngine from './components/recommendation-engine';
import AdaptiveProgramming from './components/adaptive-programming';
import BehavioralInsights from './components/behavioral-insights';
import PersonalizedMotivation from './components/personalized-motivation';

export default function PersonalizationPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">Personalization Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <SmartOnboarding />
        <RecommendationEngine />
        <AdaptiveProgramming />
        <BehavioralInsights />
        <PersonalizedMotivation />
      </div>
    </div>
  );
} 
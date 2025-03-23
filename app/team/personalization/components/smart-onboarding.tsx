'use client';

import { useState } from 'react';
import { 
  User, 
  Dumbbell, 
  Calendar, 
  AlertTriangle, 
  Clock, 
  Heart, 
  Goal, 
  ArrowRight, 
  Check, 
  ArrowLeft, 
  Save
} from 'lucide-react';
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';
import * as z from 'zod';
import { format } from 'date-fns';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Checkbox } from '@/components/ui/checkbox';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { mockOnboardingFlow, mockOnboardingQuestions } from '../utils/sample-data';

interface SmartOnboardingProps {
  onComplete: () => void;
}

interface Question {
  id: string;
  questionText: string;
  answerType: string;
  options?: string[];
  minValue?: number;
  maxValue?: number;
  step?: number;
  required: boolean;
  helpText?: string;
  targetField: string;
}

interface Section {
  id: string;
  title: string;
  description: string;
  questions: string[];
  conditionalDisplay?: {
    dependsOn: string;
    showIfValue: any;
  };
}

export default function SmartOnboarding({ onComplete }: SmartOnboardingProps) {
  const [activeSection, setActiveSection] = useState<number>(0);
  const [answers, setAnswers] = useState<Record<string, any>>({});
  const [sectionStates, setSectionStates] = useState<Record<string, 'completed' | 'active' | 'pending'>>({
    section1: 'active',
    section2: 'pending',
    section3: 'pending',
    section4: 'pending',
    section5: 'pending'
  });
  
  // Extract sections and questions from mock data
  const sections: Section[] = mockOnboardingFlow.sections;
  const questions: Record<string, Question> = {};
  mockOnboardingQuestions.forEach(q => {
    questions[q.id] = q;
  });
  
  // Track progress
  const totalQuestions = sections.reduce((acc, section) => acc + section.questions.length, 0);
  const answeredQuestions = Object.keys(answers).length;
  const progress = Math.round((answeredQuestions / totalQuestions) * 100);
  
  // Handle section navigation
  const goToNextSection = () => {
    // Validate current section
    const currentSection = sections[activeSection];
    const requiredQuestions = currentSection.questions.filter(qId => questions[qId].required);
    const allRequiredAnswered = requiredQuestions.every(qId => answers[qId] !== undefined);
    
    if (!allRequiredAnswered) {
      // Show error (in a real app, you would highlight the unanswered questions)
      console.error('Please answer all required questions');
      return;
    }
    
    // Mark current section as completed
    setSectionStates(prev => ({
      ...prev,
      [currentSection.id]: 'completed',
      ...(activeSection + 1 < sections.length ? { [sections[activeSection + 1].id]: 'active' } : {})
    }));
    
    // Move to next section or complete
    if (activeSection + 1 < sections.length) {
      setActiveSection(activeSection + 1);
    } else {
      // In a real app, you would submit the data here
      onComplete();
    }
  };
  
  const goToPrevSection = () => {
    if (activeSection > 0) {
      const prevSection = sections[activeSection - 1];
      
      // Mark current section as pending, previous as active
      setSectionStates(prev => ({
        ...prev,
        [sections[activeSection].id]: 'pending',
        [prevSection.id]: 'active'
      }));
      
      setActiveSection(activeSection - 1);
    }
  };
  
  // Handle answer updates
  const updateAnswer = (questionId: string, value: any) => {
    setAnswers(prev => ({
      ...prev,
      [questionId]: value
    }));
  };
  
  // Should we show this section based on conditional logic?
  const shouldShowSection = (section: Section): boolean => {
    if (!section.conditionalDisplay) return true;
    
    const { dependsOn, showIfValue } = section.conditionalDisplay;
    return answers[dependsOn] === showIfValue;
  };
  
  // Render a question based on its type
  const renderQuestion = (questionId: string) => {
    const question = questions[questionId];
    if (!question) return null;
    
    const commonProps = {
      id: question.id,
      className: 'mb-6',
      key: question.id
    };
    
    switch (question.answerType) {
      case 'multiple-choice':
        return (
          <div {...commonProps}>
            <div className="mb-2">
              <Label htmlFor={question.id}>{question.questionText}</Label>
              {question.helpText && (
                <p className="text-gray-400 text-sm">{question.helpText}</p>
              )}
            </div>
            
            <RadioGroup
              value={answers[question.id] || ''}
              onValueChange={(value) => updateAnswer(question.id, value)}
            >
              <div className="space-y-2">
                {question.options?.map((option) => (
                  <div key={option} className="flex items-center space-x-2">
                    <RadioGroupItem value={option} id={`${question.id}-${option}`} />
                    <Label htmlFor={`${question.id}-${option}`}>{option}</Label>
                  </div>
                ))}
              </div>
            </RadioGroup>
          </div>
        );
        
      case 'text':
        return (
          <div {...commonProps}>
            <div className="mb-2">
              <Label htmlFor={question.id}>{question.questionText}</Label>
              {question.helpText && (
                <p className="text-gray-400 text-sm">{question.helpText}</p>
              )}
            </div>
            
            <Textarea
              id={question.id}
              value={answers[question.id] || ''}
              onChange={(e) => updateAnswer(question.id, e.target.value)}
              className="min-h-[100px]"
              placeholder="Your answer..."
            />
          </div>
        );
        
      case 'number':
        return (
          <div {...commonProps}>
            <div className="mb-2">
              <Label htmlFor={question.id}>{question.questionText}</Label>
              {question.helpText && (
                <p className="text-gray-400 text-sm">{question.helpText}</p>
              )}
            </div>
            
            <Input
              id={question.id}
              type="number"
              value={answers[question.id] || ''}
              onChange={(e) => updateAnswer(question.id, parseFloat(e.target.value))}
              min={question.minValue}
              max={question.maxValue}
              step={question.step || 1}
            />
          </div>
        );
        
      case 'slider':
        return (
          <div {...commonProps}>
            <div className="mb-2">
              <Label htmlFor={question.id}>{question.questionText}</Label>
              {question.helpText && (
                <p className="text-gray-400 text-sm">{question.helpText}</p>
              )}
            </div>
            
            <div className="space-y-4">
              <Slider
                id={question.id}
                defaultValue={[answers[question.id] || question.minValue || 1]}
                max={question.maxValue || 10}
                min={question.minValue || 1}
                step={question.step || 1}
                onValueChange={(value) => updateAnswer(question.id, value[0])}
              />
              
              <div className="flex justify-between text-xs text-gray-500">
                <span>Low ({question.minValue || 1})</span>
                <span>High ({question.maxValue || 10})</span>
              </div>
              
              <div className="text-center font-medium">
                Selected: {answers[question.id] || question.minValue || 1}
              </div>
            </div>
          </div>
        );
        
      case 'boolean':
        return (
          <div {...commonProps}>
            <div className="mb-2">
              <Label htmlFor={question.id}>{question.questionText}</Label>
              {question.helpText && (
                <p className="text-gray-400 text-sm">{question.helpText}</p>
              )}
            </div>
            
            <RadioGroup
              value={answers[question.id]?.toString() || ''}
              onValueChange={(value) => updateAnswer(question.id, value === 'true')}
            >
              <div className="flex space-x-4">
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="true" id={`${question.id}-true`} />
                  <Label htmlFor={`${question.id}-true`}>Yes</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="false" id={`${question.id}-false`} />
                  <Label htmlFor={`${question.id}-false`}>No</Label>
                </div>
              </div>
            </RadioGroup>
          </div>
        );
        
      default:
        return (
          <div {...commonProps}>
            <p className="text-red-500">Unknown question type: {question.answerType}</p>
          </div>
        );
    }
  };
  
  // Get the current section
  const currentSection = sections[activeSection];
  
  // Only show questions for this section
  const sectionQuestions = currentSection.questions.filter(qId => questions[qId]);
  
  return (
    <div className="smart-onboarding">
      {/* Progress bar and navigation */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-2">
          <h2 className="text-white font-bold text-xl">Personalized Profile Setup</h2>
          <Button variant="outline" size="sm" onClick={onComplete}>
            Skip for Now
          </Button>
        </div>
        
        <div className="mb-4">
          <Progress value={progress} className="h-2" />
          <div className="flex justify-between mt-1 text-sm text-gray-400">
            <span>Step {activeSection + 1} of {sections.length}</span>
            <span>{progress}% Complete</span>
          </div>
        </div>
        
        <div className="flex gap-2 overflow-x-auto pb-2">
          {sections.map((section, index) => {
            // Skip sections that shouldn't be shown
            if (!shouldShowSection(section)) return null;
            
            const state = sectionStates[section.id];
            
            return (
              <Badge
                key={section.id}
                variant={
                  state === 'completed' ? 'default' : 
                  state === 'active' ? 'secondary' : 'outline'
                }
                className="cursor-pointer whitespace-nowrap"
                onClick={() => {
                  // Only allow navigation to completed sections or the active one
                  if (state === 'completed' || state === 'active') {
                    setActiveSection(index);
                  }
                }}
              >
                {state === 'completed' && <Check className="mr-1 h-3 w-3" />}
                {section.title}
              </Badge>
            );
          })}
        </div>
      </div>
      
      {/* Current section */}
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader>
          <CardTitle className="text-white">{currentSection.title}</CardTitle>
          <CardDescription>{currentSection.description}</CardDescription>
        </CardHeader>
        
        <CardContent>
          <div className="space-y-4">
            {sectionQuestions.map(questionId => renderQuestion(questionId))}
          </div>
        </CardContent>
        
        <CardFooter className="border-t border-gray-700 pt-4 flex justify-between">
          <Button 
            variant="outline"
            onClick={goToPrevSection}
            disabled={activeSection === 0}
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Previous
          </Button>
          
          <Button onClick={goToNextSection}>
            {activeSection === sections.length - 1 ? (
              <>
                Complete Setup
                <Check className="ml-2 h-4 w-4" />
              </>
            ) : (
              <>
                Next
                <ArrowRight className="ml-2 h-4 w-4" />
              </>
            )}
          </Button>
        </CardFooter>
      </Card>
      
      {/* Intelligence indicators */}
      <div className="mt-6 bg-gray-800 border border-gray-700 rounded-lg p-4">
        <h3 className="text-white font-medium mb-2">Intelligent Onboarding</h3>
        <p className="text-gray-400 text-sm">
          Our system is analyzing your answers to create a personalized experience:
        </p>
        
        <div className="mt-3 space-y-2">
          <div className="flex items-center gap-2 text-sm text-gray-300">
            <Badge variant="outline" className="bg-blue-900/20">Active</Badge>
            <span>Adjusting remaining questions based on your profile</span>
          </div>
          
          <div className="flex items-center gap-2 text-sm text-gray-300">
            <Badge variant="outline" className="bg-green-900/20">Active</Badge>
            <span>Calculating your ideal training parameters</span>
          </div>
          
          <div className="flex items-center gap-2 text-sm text-gray-300">
            <Badge variant="outline" className="bg-purple-900/20">Active</Badge>
            <span>Analyzing behavioral indicators for motivation strategy</span>
          </div>
        </div>
      </div>
    </div>
  );
} 
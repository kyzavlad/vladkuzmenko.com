'use client';

import { useState } from 'react';
import { 
  Dumbbell, 
  TrendingUp, 
  BarChart, 
  Trophy, 
  Plus,
  Search,
  Filter,
  ArrowUp,
  ArrowRight,
  ChevronDown,
  Flag
} from 'lucide-react';
import { format } from 'date-fns';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { StrengthMetric } from '../types';

interface StrengthProgressTrackerProps {
  strengthData: StrengthMetric[];
  onAddRecord?: () => void;
}

type TimeRange = '1m' | '3m' | '6m' | '1y' | 'all';
type SortOrder = 'exercise' | 'recent' | 'progress';

export default function StrengthProgressTracker({ 
  strengthData, 
  onAddRecord 
}: StrengthProgressTrackerProps) {
  const [timeRange, setTimeRange] = useState<TimeRange>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortOrder, setSortOrder] = useState<SortOrder>('exercise');
  const [unit, setUnit] = useState<'kg' | 'lb'>('kg');
  
  // Filter and organize strength data
  const filteredData = strengthData.filter(metric => 
    metric.exercise.toLowerCase().includes(searchQuery.toLowerCase()) &&
    (unit === 'kg' || metric.unit === unit)
  );
  
  // Group data by exercise
  const exerciseMap: Record<string, StrengthMetric[]> = {};
  
  filteredData.forEach(metric => {
    if (!exerciseMap[metric.exercise]) {
      exerciseMap[metric.exercise] = [];
    }
    exerciseMap[metric.exercise].push(metric);
  });
  
  // For each exercise, sort by date (newest first)
  Object.keys(exerciseMap).forEach(exercise => {
    exerciseMap[exercise].sort((a, b) => 
      new Date(b.date).getTime() - new Date(a.date).getTime()
    );
  });
  
  // Sort the exercise list based on selected sort order
  const sortedExercises = Object.keys(exerciseMap).sort((a, b) => {
    if (sortOrder === 'exercise') {
      return a.localeCompare(b);
    } else if (sortOrder === 'recent') {
      const dateA = new Date(exerciseMap[a][0].date).getTime();
      const dateB = new Date(exerciseMap[b][0].date).getTime();
      return dateB - dateA;
    } else { // 'progress'
      const progressA = calculateProgress(exerciseMap[a]);
      const progressB = calculateProgress(exerciseMap[b]);
      return progressB - progressA;
    }
  });
  
  // Calculate progress percentage for an exercise
  function calculateProgress(metrics: StrengthMetric[]): number {
    if (metrics.length < 2) return 0;
    
    const newest = metrics[0];
    const oldest = metrics[metrics.length - 1];
    
    return ((newest.oneRepMax - oldest.oneRepMax) / oldest.oneRepMax) * 100;
  }
  
  // Find personal records
  const personalRecords: Record<string, StrengthMetric> = {};
  
  Object.keys(exerciseMap).forEach(exercise => {
    personalRecords[exercise] = exerciseMap[exercise].reduce((max, current) => {
      return current.oneRepMax > max.oneRepMax ? current : max;
    }, exerciseMap[exercise][0]);
  });
  
  // Format date for display
  const formatDate = (date: string | Date) => {
    return format(new Date(date), 'MMM d, yyyy');
  };
  
  // Convert between kg and lb if needed
  const convertWeight = (weight: number, from: 'kg' | 'lb', to: 'kg' | 'lb') => {
    if (from === to) return weight;
    return from === 'kg' ? weight * 2.20462 : weight / 2.20462;
  };
  
  // Display weight based on selected unit
  const displayWeight = (metric: StrengthMetric) => {
    const weight = unit === metric.unit ? 
      metric.value : 
      convertWeight(metric.value, metric.unit, unit);
    
    return `${weight.toFixed(1)} ${unit}`;
  };
  
  // Display 1RM based on selected unit
  const display1RM = (metric: StrengthMetric) => {
    const oneRM = unit === metric.unit ? 
      metric.oneRepMax : 
      convertWeight(metric.oneRepMax, metric.unit, unit);
    
    return `${oneRM.toFixed(1)} ${unit}`;
  };
  
  return (
    <div className="strength-progress-tracker">
      {/* Controls */}
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 mb-6">
        <div className="flex flex-col sm:flex-row gap-4 w-full lg:w-auto">
          <div className="relative flex-grow">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-500" />
            <Input
              placeholder="Search exercises"
              className="bg-gray-800 border-gray-700 pl-9"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          
          <Select value={timeRange} onValueChange={(value) => setTimeRange(value as TimeRange)}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Time Range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1m">Last Month</SelectItem>
              <SelectItem value="3m">Last 3 Months</SelectItem>
              <SelectItem value="6m">Last 6 Months</SelectItem>
              <SelectItem value="1y">Last Year</SelectItem>
              <SelectItem value="all">All Time</SelectItem>
            </SelectContent>
          </Select>
          
          <Select value={sortOrder} onValueChange={(value) => setSortOrder(value as SortOrder)}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Sort By" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="exercise">Exercise Name</SelectItem>
              <SelectItem value="recent">Most Recent</SelectItem>
              <SelectItem value="progress">Most Progress</SelectItem>
            </SelectContent>
          </Select>
        </div>
        
        <div className="flex gap-2 w-full lg:w-auto justify-between lg:justify-start">
          <div className="flex rounded-md overflow-hidden">
            <Button 
              size="sm" 
              variant={unit === 'kg' ? "default" : "outline"}
              className="rounded-r-none"
              onClick={() => setUnit('kg')}
            >
              kg
            </Button>
            <Button 
              size="sm" 
              variant={unit === 'lb' ? "default" : "outline"}
              className="rounded-l-none"
              onClick={() => setUnit('lb')}
            >
              lb
            </Button>
          </div>
          
          <Button size="sm" onClick={onAddRecord}>
            <Plus className="h-4 w-4 mr-2" />
            Add Record
          </Button>
        </div>
      </div>
      
      {/* Personal Records Card */}
      <Card className="bg-gray-800 border-gray-700 mb-6">
        <CardContent className="p-6">
          <div className="flex items-center mb-4">
            <Trophy className="h-5 w-5 mr-2 text-yellow-500" />
            <h3 className="text-white font-medium">Personal Records</h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.keys(personalRecords).slice(0, 6).map(exercise => (
              <div key={exercise} className="bg-gray-750 p-3 rounded-lg">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="text-white font-medium">{exercise}</h4>
                  <Badge className="bg-yellow-600">PR</Badge>
                </div>
                
                <div className="text-2xl text-white font-light">
                  {display1RM(personalRecords[exercise])} <span className="text-sm text-gray-400">1RM</span>
                </div>
                
                <div className="text-gray-400 text-xs mt-1">
                  {displayWeight(personalRecords[exercise])} × {personalRecords[exercise].reps} reps • {formatDate(personalRecords[exercise].date)}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
      
      {/* Exercise Progress List */}
      {sortedExercises.length > 0 ? (
        <Accordion type="multiple" defaultValue={[sortedExercises[0]]} className="space-y-4">
          {sortedExercises.map(exercise => {
            const exerciseData = exerciseMap[exercise];
            const latestRecord = exerciseData[0];
            const progress = calculateProgress(exerciseData);
            const isPR = personalRecords[exercise].date === latestRecord.date;
            
            return (
              <AccordionItem 
                key={exercise} 
                value={exercise}
                className="bg-gray-800 border border-gray-700 rounded-lg overflow-hidden"
              >
                <AccordionTrigger className="px-6 py-4 hover:bg-gray-750">
                  <div className="flex-1 flex items-center">
                    <div className="w-10 h-10 rounded-full bg-blue-500/20 flex items-center justify-center mr-4">
                      <Dumbbell className="h-5 w-5 text-blue-500" />
                    </div>
                    <div className="flex-grow">
                      <h3 className="text-white font-medium text-left">{exercise}</h3>
                      <div className="text-gray-400 text-sm">{exerciseData.length} records • Last: {formatDate(latestRecord.date)}</div>
                    </div>
                    <div className="flex flex-col items-end mr-4">
                      <div className="text-white font-medium">{display1RM(latestRecord)}</div>
                      <div className="text-gray-400 text-sm">1RM</div>
                    </div>
                    {progress !== 0 && (
                      <Badge className={progress > 0 ? 'bg-green-600' : 'bg-red-600'}>
                        {progress > 0 ? '+' : ''}{progress.toFixed(1)}%
                      </Badge>
                    )}
                  </div>
                </AccordionTrigger>
                <AccordionContent>
                  <div className="px-6 py-4 border-t border-gray-700">
                    {/* Progress Chart */}
                    <div className="mb-6">
                      <h4 className="text-white font-medium mb-3">Progress Chart</h4>
                      <div className="h-40 bg-gray-750 rounded-lg p-4 flex items-end">
                        {exerciseData.slice().reverse().map((record, index) => {
                          // Calculate bar height as percentage of max value
                          const maxOneRM = Math.max(...exerciseData.map(d => d.oneRepMax));
                          const heightPercentage = (record.oneRepMax / maxOneRM) * 100;
                          
                          return (
                            <div key={index} className="flex-1 flex flex-col items-center">
                              <div 
                                className={`w-6 rounded-t ${
                                  record.date === personalRecords[exercise].date
                                    ? 'bg-yellow-500'
                                    : 'bg-blue-500'
                                }`}
                                style={{ height: `${heightPercentage}%` }}
                              ></div>
                              <div className="text-gray-400 text-xs mt-2 transform -rotate-45 origin-top-left w-16 truncate">
                                {format(new Date(record.date), 'MMM d')}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                    
                    {/* Exercise Records Table */}
                    <h4 className="text-white font-medium mb-3">History</h4>
                    <div className="overflow-x-auto">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Date</TableHead>
                            <TableHead>Weight</TableHead>
                            <TableHead>Reps</TableHead>
                            <TableHead>1RM</TableHead>
                            <TableHead>Change</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {exerciseData.map((record, index) => {
                            const prevRecord = index < exerciseData.length - 1 ? exerciseData[index + 1] : null;
                            const change = prevRecord 
                              ? ((record.oneRepMax - prevRecord.oneRepMax) / prevRecord.oneRepMax) * 100 
                              : 0;
                            
                            return (
                              <TableRow key={index}>
                                <TableCell className="font-medium">
                                  <div className="flex items-center">
                                    {formatDate(record.date)}
                                    {record.date === personalRecords[exercise].date && (
                                      <Trophy className="h-4 w-4 ml-2 text-yellow-500" />
                                    )}
                                  </div>
                                </TableCell>
                                <TableCell>{displayWeight(record)}</TableCell>
                                <TableCell>{record.reps}</TableCell>
                                <TableCell className="font-medium">{display1RM(record)}</TableCell>
                                <TableCell>
                                  {index < exerciseData.length - 1 && (
                                    <div className={
                                      change > 0 
                                        ? 'text-green-500 flex items-center' 
                                        : change < 0 
                                        ? 'text-red-500 flex items-center' 
                                        : 'text-gray-400 flex items-center'
                                    }>
                                      {change > 0 && <ArrowUp className="h-3 w-3 mr-1" />}
                                      {change < 0 && <ArrowUp className="h-3 w-3 mr-1 transform rotate-180" />}
                                      {change === 0 && <ArrowRight className="h-3 w-3 mr-1" />}
                                      {change !== 0 ? `${Math.abs(change).toFixed(1)}%` : 'No change'}
                                    </div>
                                  )}
                                </TableCell>
                              </TableRow>
                            );
                          })}
                        </TableBody>
                      </Table>
                    </div>
                    
                    {/* Goal Setting */}
                    <div className="mt-6 bg-gray-750 rounded-lg p-4">
                      <div className="flex items-center mb-3">
                        <Flag className="h-4 w-4 mr-2 text-blue-500" />
                        <h4 className="text-white font-medium">Goal Setting</h4>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <div className="text-gray-400 text-sm mb-1">Current 1RM</div>
                          <div className="text-white text-lg font-medium">
                            {display1RM(latestRecord)}
                          </div>
                        </div>
                        
                        <div>
                          <div className="text-gray-400 text-sm mb-1">Next Milestone</div>
                          <div className="text-white text-lg font-medium">
                            {/* Calculate a realistic next goal (e.g., 5% increase) */}
                            {display1RM({
                              ...latestRecord,
                              oneRepMax: latestRecord.oneRepMax * 1.05
                            })}
                          </div>
                        </div>
                      </div>
                      
                      <div className="mt-3">
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-gray-400">Progress toward next milestone</span>
                          <span className="text-white">0%</span>
                        </div>
                        <Progress value={0} className="h-2" />
                      </div>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            );
          })}
        </Accordion>
      ) : (
        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-6 text-center">
            <Dumbbell className="h-16 w-16 mx-auto mb-4 text-gray-700" />
            <h3 className="text-white text-lg font-medium mb-2">No Strength Data Yet</h3>
            <p className="text-gray-400 max-w-md mx-auto mb-6">
              Start tracking your strength progress by adding your first strength record. This will help you visualize improvements over time.
            </p>
            <Button onClick={onAddRecord}>
              <Plus className="mr-2 h-4 w-4" />
              Add Your First Strength Record
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
} 
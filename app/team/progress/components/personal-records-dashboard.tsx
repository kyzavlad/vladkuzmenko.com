'use client';

import { useState } from 'react';
import { 
  Dumbbell, 
  Award, 
  Plus,
  Edit,
  Trash2,
  Calendar,
  TrendingUp,
  BarChart3,
  ChevronRight,
  Search,
  ArrowUpRight,
  Filter,
  Clock,
  Weight,
  Heart,
  Target,
  Share2
} from 'lucide-react';
import { format, parseISO } from 'date-fns';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@/components/ui/tabs';
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from '@/components/ui/hover-card';

interface PersonalRecord {
  id: string;
  exercise: string;
  category: string;
  value: number;
  unit: 'kg' | 'lb' | 'min' | 'sec' | 'reps';
  date: string;
  notes?: string;
  previousRecord?: number;
  improvement?: number;
  improvedBy?: number;
}

interface PersonalRecordsDashboardProps {
  userId?: string;
}

export default function PersonalRecordsDashboard({ userId }: PersonalRecordsDashboardProps) {
  const [activeTab, setActiveTab] = useState<string>('strength');
  const [recordCategory, setRecordCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [currentView, setCurrentView] = useState<'grid' | 'table'>('grid');
  const [showAddDialog, setShowAddDialog] = useState<boolean>(false);
  const [newRecord, setNewRecord] = useState<Partial<PersonalRecord>>({
    exercise: '',
    category: 'strength',
    value: 0,
    unit: 'kg',
    date: format(new Date(), 'yyyy-MM-dd')
  });
  
  // Mock personal records data
  const [personalRecords, setPersonalRecords] = useState<PersonalRecord[]>([
    { 
      id: 'pr1', 
      exercise: 'Bench Press', 
      category: 'strength', 
      value: 100, 
      unit: 'kg', 
      date: '2023-07-15',
      previousRecord: 95,
      improvement: 5,
      improvedBy: 5.26
    },
    { 
      id: 'pr2', 
      exercise: 'Squat', 
      category: 'strength', 
      value: 150, 
      unit: 'kg', 
      date: '2023-07-10',
      previousRecord: 140,
      improvement: 10,
      improvedBy: 7.14
    },
    { 
      id: 'pr3', 
      exercise: 'Deadlift', 
      category: 'strength', 
      value: 180, 
      unit: 'kg', 
      date: '2023-07-05',
      previousRecord: 170,
      improvement: 10,
      improvedBy: 5.88
    },
    { 
      id: 'pr4', 
      exercise: '5K Run', 
      category: 'endurance', 
      value: 22, 
      unit: 'min', 
      date: '2023-07-20',
      previousRecord: 24,
      improvement: 2,
      improvedBy: 8.33
    },
    { 
      id: 'pr5', 
      exercise: 'Pull-ups', 
      category: 'strength', 
      value: 15, 
      unit: 'reps', 
      date: '2023-07-12',
      previousRecord: 12,
      improvement: 3,
      improvedBy: 25
    },
    { 
      id: 'pr6', 
      exercise: '400m Sprint', 
      category: 'speed', 
      value: 62, 
      unit: 'sec', 
      date: '2023-07-18',
      previousRecord: 65,
      improvement: 3,
      improvedBy: 4.61
    },
    { 
      id: 'pr7', 
      exercise: 'Plank', 
      category: 'endurance', 
      value: 180, 
      unit: 'sec', 
      date: '2023-07-14',
      previousRecord: 150,
      improvement: 30,
      improvedBy: 20
    },
    { 
      id: 'pr8', 
      exercise: 'Shoulder Press', 
      category: 'strength', 
      value: 65, 
      unit: 'kg', 
      date: '2023-07-08',
      previousRecord: 60,
      improvement: 5,
      improvedBy: 8.33
    }
  ]);
  
  // Filter records based on current tab, category, and search query
  const filteredRecords = personalRecords.filter(record => {
    // Filter by tab
    if (activeTab !== 'all' && record.category !== activeTab) return false;
    
    // Filter by category
    if (recordCategory !== 'all' && record.exercise !== recordCategory) return false;
    
    // Filter by search
    if (searchQuery && !record.exercise.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    
    return true;
  });
  
  // Get unique exercise names for filter dropdown
  const uniqueExercises = [...new Set(personalRecords.map(record => record.exercise))];
  
  // Add a new record
  const addRecord = () => {
    if (!newRecord.exercise || !newRecord.value) return;
    
    const record: PersonalRecord = {
      id: `pr${Date.now()}`,
      exercise: newRecord.exercise,
      category: newRecord.category as string,
      value: newRecord.value,
      unit: newRecord.unit as 'kg' | 'lb' | 'min' | 'sec' | 'reps',
      date: newRecord.date as string,
      notes: newRecord.notes
    };
    
    // Check for previous record to calculate improvement
    const previousRecord = personalRecords.find(
      pr => pr.exercise === record.exercise && pr.unit === record.unit
    );
    
    if (previousRecord) {
      record.previousRecord = previousRecord.value;
      
      // Calculate improvement and percentage
      if (record.unit === 'min' || record.unit === 'sec') {
        // For time-based records, lower is better
        record.improvement = previousRecord.value - record.value;
        record.improvedBy = (record.improvement / previousRecord.value) * 100;
      } else {
        // For other records, higher is better
        record.improvement = record.value - previousRecord.value;
        record.improvedBy = (record.improvement / previousRecord.value) * 100;
      }
    }
    
    setPersonalRecords(prev => [record, ...prev]);
    
    // Reset form
    setNewRecord({
      exercise: '',
      category: 'strength',
      value: 0,
      unit: 'kg',
      date: format(new Date(), 'yyyy-MM-dd')
    });
    
    setShowAddDialog(false);
  };
  
  // Delete a record
  const deleteRecord = (id: string) => {
    setPersonalRecords(prev => prev.filter(record => record.id !== id));
  };
  
  // Helper to format value with unit
  const formatValueWithUnit = (value: number, unit: string) => {
    if (unit === 'kg' || unit === 'lb') return `${value} ${unit}`;
    if (unit === 'reps') return `${value} reps`;
    
    // Format time
    if (unit === 'min') {
      const minutes = Math.floor(value);
      const seconds = Math.round((value - minutes) * 60);
      return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }
    
    if (unit === 'sec') {
      return `${value} sec`;
    }
    
    return `${value} ${unit}`;
  };
  
  // Get record icon based on category
  const getRecordIcon = (category: string) => {
    switch (category) {
      case 'strength':
        return <Dumbbell className="h-5 w-5 text-blue-500" />;
      case 'endurance':
        return <Heart className="h-5 w-5 text-red-500" />;
      case 'speed':
        return <Clock className="h-5 w-5 text-yellow-500" />;
      default:
        return <Target className="h-5 w-5 text-purple-500" />;
    }
  };
  
  // Get CSS class for record card based on category
  const getRecordCardClass = (category: string) => {
    switch (category) {
      case 'strength':
        return 'border-blue-800/30 bg-blue-900/10';
      case 'endurance':
        return 'border-red-800/30 bg-red-900/10';
      case 'speed':
        return 'border-yellow-800/30 bg-yellow-900/10';
      default:
        return 'border-purple-800/30 bg-purple-900/10';
    }
  };
  
  // Find the latest strength PRs for the highlights section
  const topStrengthPRs = personalRecords
    .filter(record => record.category === 'strength')
    .slice(0, 3);
  
  // Calculate total PRs broken down by category
  const prCounts = personalRecords.reduce((acc, record) => {
    if (!acc[record.category]) {
      acc[record.category] = 0;
    }
    acc[record.category]++;
    return acc;
  }, {} as Record<string, number>);
  
  return (
    <div className="personal-records-dashboard">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-white text-xl font-medium">Personal Records</h2>
          <p className="text-gray-400">Track your fitness milestones and achievements</p>
        </div>
        
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => setCurrentView(currentView === 'grid' ? 'table' : 'grid')}>
            {currentView === 'grid' ? <BarChart3 className="h-4 w-4" /> : <Filter className="h-4 w-4" />}
          </Button>
          
          <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="mr-2 h-4 w-4" />
                Add New PR
              </Button>
            </DialogTrigger>
            <DialogContent className="bg-gray-800 text-white">
              <DialogHeader>
                <DialogTitle>Add New Personal Record</DialogTitle>
                <DialogDescription className="text-gray-400">
                  Record your latest fitness achievement
                </DialogDescription>
              </DialogHeader>
              
              <div className="grid gap-4 py-4">
                <div className="grid gap-2">
                  <label className="text-sm text-gray-400">Exercise</label>
                  <Input 
                    value={newRecord.exercise}
                    onChange={(e) => setNewRecord({...newRecord, exercise: e.target.value})}
                    placeholder="e.g. Bench Press, 5K Run"
                    className="bg-gray-750 border-gray-600"
                  />
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="grid gap-2">
                    <label className="text-sm text-gray-400">Category</label>
                    <Select
                      value={newRecord.category}
                      onValueChange={(value) => setNewRecord({...newRecord, category: value})}
                    >
                      <SelectTrigger className="bg-gray-750 border-gray-600">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="strength">Strength</SelectItem>
                        <SelectItem value="endurance">Endurance</SelectItem>
                        <SelectItem value="speed">Speed</SelectItem>
                        <SelectItem value="other">Other</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="grid gap-2">
                    <label className="text-sm text-gray-400">Date</label>
                    <Input 
                      type="date"
                      value={newRecord.date}
                      onChange={(e) => setNewRecord({...newRecord, date: e.target.value})}
                      className="bg-gray-750 border-gray-600"
                    />
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="grid gap-2">
                    <label className="text-sm text-gray-400">Value</label>
                    <Input 
                      type="number"
                      value={newRecord.value?.toString() || ''}
                      onChange={(e) => setNewRecord({...newRecord, value: parseFloat(e.target.value) || 0})}
                      className="bg-gray-750 border-gray-600"
                    />
                  </div>
                  
                  <div className="grid gap-2">
                    <label className="text-sm text-gray-400">Unit</label>
                    <Select
                      value={newRecord.unit}
                      onValueChange={(value) => setNewRecord({...newRecord, unit: value as any})}
                    >
                      <SelectTrigger className="bg-gray-750 border-gray-600">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="kg">Kilograms (kg)</SelectItem>
                        <SelectItem value="lb">Pounds (lb)</SelectItem>
                        <SelectItem value="reps">Repetitions</SelectItem>
                        <SelectItem value="min">Minutes</SelectItem>
                        <SelectItem value="sec">Seconds</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                
                <div className="grid gap-2">
                  <label className="text-sm text-gray-400">Notes (Optional)</label>
                  <Input 
                    value={newRecord.notes || ''}
                    onChange={(e) => setNewRecord({...newRecord, notes: e.target.value})}
                    placeholder="Any details about this record"
                    className="bg-gray-750 border-gray-600"
                  />
                </div>
              </div>
              
              <DialogFooter>
                <Button variant="outline" onClick={() => setShowAddDialog(false)}>Cancel</Button>
                <Button onClick={addRecord}>Save Record</Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>
      
      {/* PR Highlights */}
      <Card className="bg-gray-800 border-gray-700 mb-6">
        <CardContent className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
            <div className="md:col-span-4">
              <h3 className="text-white font-medium mb-3 flex items-center gap-2">
                <Award className="h-5 w-5 text-yellow-500" />
                PR Highlights
              </h3>
              
              <div className="space-y-3">
                {topStrengthPRs.map(record => (
                  <div key={record.id} className="bg-gray-750 rounded-lg p-3 flex items-center gap-3">
                    <div className="bg-blue-900/30 rounded-full p-2">
                      <Dumbbell className="h-5 w-5 text-blue-500" />
                    </div>
                    
                    <div>
                      <div className="text-white font-medium">{record.exercise}</div>
                      <div className="text-gray-400 text-sm">{formatValueWithUnit(record.value, record.unit)}</div>
                    </div>
                    
                    {record.improvedBy && (
                      <Badge className="ml-auto bg-green-900 text-green-300">
                        +{record.improvedBy.toFixed(1)}%
                      </Badge>
                    )}
                  </div>
                ))}
              </div>
            </div>
            
            <div className="md:col-span-4">
              <h3 className="text-white font-medium mb-3 flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-blue-500" />
                PR Statistics
              </h3>
              
              <div className="bg-gray-750 rounded-lg p-4">
                <div className="mb-3 text-white">
                  <div className="text-3xl font-light">{personalRecords.length}</div>
                  <div className="text-sm text-gray-400">Total Records Tracked</div>
                </div>
                
                <div className="space-y-2">
                  {Object.entries(prCounts).map(([category, count]) => (
                    <div key={category} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {getRecordIcon(category)}
                        <span className="text-gray-300 capitalize">{category}</span>
                      </div>
                      <Badge variant="outline">{count}</Badge>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="md:col-span-4">
              <h3 className="text-white font-medium mb-3 flex items-center gap-2">
                <Calendar className="h-5 w-5 text-purple-500" />
                Recent Progress
              </h3>
              
              <div className="bg-gray-750 rounded-lg p-4">
                <div className="space-y-3">
                  {personalRecords.slice(0, 3).map(record => (
                    <div key={record.id} className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <div className="w-8 text-gray-400">{format(parseISO(record.date), 'MMM d')}</div>
                        <div className="text-white">{record.exercise}</div>
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="text-gray-300">{formatValueWithUnit(record.value, record.unit)}</span>
                        {record.improvement && record.improvement > 0 && (
                          <ArrowUpRight className="h-3 w-3 text-green-500" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
                
                <div className="mt-4 pt-3 border-t border-gray-700 text-center">
                  <Button variant="ghost" size="sm" className="text-gray-400 hover:text-white">
                    View All Progress
                    <ChevronRight className="ml-1 h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* PR Filters and Records */}
      <div className="space-y-6">
        <div className="flex flex-wrap gap-3 justify-between items-center">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full md:w-auto">
            <TabsList>
              <TabsTrigger value="all">All PRs</TabsTrigger>
              <TabsTrigger value="strength">Strength</TabsTrigger>
              <TabsTrigger value="endurance">Endurance</TabsTrigger>
              <TabsTrigger value="speed">Speed</TabsTrigger>
            </TabsList>
          </Tabs>
          
          <div className="flex gap-3 w-full md:w-auto">
            <Select value={recordCategory} onValueChange={setRecordCategory}>
              <SelectTrigger className="bg-gray-800 border-gray-700 w-full md:w-[200px]">
                <SelectValue placeholder="Filter by exercise" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Exercises</SelectItem>
                {uniqueExercises.map(exercise => (
                  <SelectItem key={exercise} value={exercise}>
                    {exercise}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            <div className="relative w-full md:w-auto">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
              <Input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search records..."
                className="bg-gray-800 border-gray-700 pl-9 w-full md:w-[220px]"
              />
            </div>
          </div>
        </div>
        
        {filteredRecords.length > 0 ? (
          currentView === 'grid' ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {filteredRecords.map(record => (
                <div
                  key={record.id}
                  className={`border rounded-lg p-4 ${getRecordCardClass(record.category)}`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <div className="flex items-center gap-2">
                      {getRecordIcon(record.category)}
                      <h4 className="text-white font-medium">{record.exercise}</h4>
                    </div>
                    
                    <div className="flex">
                      <Button variant="ghost" size="icon" className="h-8 w-8 text-gray-400">
                        <Edit className="h-4 w-4" />
                      </Button>
                      <Button 
                        variant="ghost" 
                        size="icon" 
                        className="h-8 w-8 text-gray-400"
                        onClick={() => deleteRecord(record.id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-1 text-gray-400 text-sm mb-3">
                    <Calendar className="h-4 w-4" />
                    <span>{format(parseISO(record.date), 'MMMM d, yyyy')}</span>
                  </div>
                  
                  <div className="flex items-baseline gap-2 mb-2">
                    <div className="text-2xl font-light text-white">
                      {formatValueWithUnit(record.value, record.unit)}
                    </div>
                    
                    {record.improvement && (
                      <HoverCard>
                        <HoverCardTrigger asChild>
                          <Badge className={record.improvement > 0 ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}>
                            {record.improvement > 0 ? '+' : ''}
                            {record.unit === 'min' || record.unit === 'sec' 
                              ? record.improvement.toFixed(1) 
                              : formatValueWithUnit(record.improvement, record.unit)}
                          </Badge>
                        </HoverCardTrigger>
                        <HoverCardContent className="bg-gray-800 border-gray-700 text-white w-64">
                          <div className="flex justify-between mb-2">
                            <span className="text-gray-400">Previous Record:</span>
                            <span>{formatValueWithUnit(record.previousRecord!, record.unit)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Improvement:</span>
                            <span className={record.improvement > 0 ? 'text-green-400' : 'text-red-400'}>
                              {record.improvement > 0 ? '+' : ''}
                              {record.improvedBy!.toFixed(1)}%
                            </span>
                          </div>
                        </HoverCardContent>
                      </HoverCard>
                    )}
                  </div>
                  
                  {record.notes && (
                    <div className="text-gray-400 text-sm italic mt-2">
                      "{record.notes}"
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead className="bg-gray-800">
                  <tr>
                    <th className="text-left p-3 text-gray-400 font-medium">Exercise</th>
                    <th className="text-left p-3 text-gray-400 font-medium">Value</th>
                    <th className="text-left p-3 text-gray-400 font-medium">Category</th>
                    <th className="text-left p-3 text-gray-400 font-medium">Date</th>
                    <th className="text-left p-3 text-gray-400 font-medium">Improvement</th>
                    <th className="text-right p-3 text-gray-400 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-700">
                  {filteredRecords.map(record => (
                    <tr key={record.id} className="border-gray-700 hover:bg-gray-750">
                      <td className="p-3 text-white">{record.exercise}</td>
                      <td className="p-3 text-white">{formatValueWithUnit(record.value, record.unit)}</td>
                      <td className="p-3">
                        <div className="flex items-center gap-2">
                          {getRecordIcon(record.category)}
                          <span className="text-gray-300 capitalize">{record.category}</span>
                        </div>
                      </td>
                      <td className="p-3 text-gray-400">{format(parseISO(record.date), 'MMM d, yyyy')}</td>
                      <td className="p-3">
                        {record.improvement ? (
                          <div className="flex items-center gap-2">
                            <Badge className={record.improvement > 0 ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}>
                              {record.improvement > 0 ? '+' : ''}
                              {record.improvedBy!.toFixed(1)}%
                            </Badge>
                            <span className="text-gray-400 text-sm">
                              {record.improvement > 0 ? 'from' : 'vs.'} {formatValueWithUnit(record.previousRecord!, record.unit)}
                            </span>
                          </div>
                        ) : (
                          <span className="text-gray-500">–</span>
                        )}
                      </td>
                      <td className="p-3 text-right">
                        <div className="flex justify-end gap-1">
                          <Button variant="ghost" size="icon" className="h-8 w-8">
                            <Share2 className="h-4 w-4" />
                          </Button>
                          <Button variant="ghost" size="icon" className="h-8 w-8">
                            <Edit className="h-4 w-4" />
                          </Button>
                          <Button 
                            variant="ghost" 
                            size="icon" 
                            className="h-8 w-8 text-red-500"
                            onClick={() => deleteRecord(record.id)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        ) : (
          <Card className="bg-gray-800 border-gray-700">
            <CardContent className="p-6 text-center">
              <Filter className="h-16 w-16 text-gray-700 mx-auto mb-4" />
              <h3 className="text-white text-lg font-medium mb-2">No records found</h3>
              <p className="text-gray-400 mb-4">
                No personal records match your current filters.
                Try adjusting your filters or add a new PR.
              </p>
              <div className="flex justify-center gap-3">
                <Button variant="outline" onClick={() => {
                  setActiveTab('all');
                  setRecordCategory('all');
                  setSearchQuery('');
                }}>
                  Clear Filters
                </Button>
                <Button onClick={() => setShowAddDialog(true)}>
                  <Plus className="mr-2 h-4 w-4" />
                  Add New PR
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
} 
'use client';

import { useState, useRef } from 'react';
import { 
  ChevronLeft, 
  ChevronRight, 
  Trophy, 
  Calendar, 
  LineChart, 
  Camera, 
  Dumbbell, 
  Award, 
  Star, 
  Share2, 
  Download,
  Milestone,
  Heart,
  FileText,
  Play
} from 'lucide-react';
import { format, subMonths, differenceInMonths, parseISO, isValid } from 'date-fns';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Achievement, 
  BodyMeasurements, 
  StrengthMetric, 
  ProgressPhoto 
} from '../types';

interface TimelineEvent {
  id: string;
  date: string | Date;
  type: 'achievement' | 'milestone' | 'measurement' | 'strength-record' | 'progress-photo' | 'workout';
  title: string;
  description: string;
  mediaUrl?: string;
  data?: any; // Specific data for the event type
  highlighted?: boolean;
}

interface FitnessJourneyTimelineProps {
  achievements: Achievement[];
  measurements: BodyMeasurements[];
  strengthData: StrengthMetric[];
  progressPhotos: ProgressPhoto[];
  joinDate: string | Date;
  onShareStory?: () => void;
  onCreateHighlight?: () => void;
}

export default function FitnessJourneyTimeline({
  achievements,
  measurements,
  strengthData,
  progressPhotos,
  joinDate,
  onShareStory,
  onCreateHighlight
}: FitnessJourneyTimelineProps) {
  const [currentMonth, setCurrentMonth] = useState<Date>(new Date());
  const [viewMode, setViewMode] = useState<'month' | 'all'>('month');
  const [selectedEvent, setSelectedEvent] = useState<TimelineEvent | null>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  
  // Format date for display
  const formatTimelineDate = (date: string | Date) => {
    const parsedDate = typeof date === 'string' ? parseISO(date) : date;
    return isValid(parsedDate) ? format(parsedDate, 'MMM d, yyyy') : 'Invalid date';
  };
  
  // Navigate through timeline
  const navigatePrevious = () => {
    setCurrentMonth(prev => subMonths(prev, 1));
  };
  
  const navigateNext = () => {
    const nextMonth = subMonths(currentMonth, -1);
    if (nextMonth <= new Date()) {
      setCurrentMonth(nextMonth);
    }
  };
  
  // Generate milestone dates (simulated)
  const getMilestones = (): TimelineEvent[] => {
    const startDate = new Date(joinDate);
    const now = new Date();
    const monthsActive = differenceInMonths(now, startDate);
    
    const milestones: TimelineEvent[] = [];
    
    // Join date milestone
    milestones.push({
      id: 'join-milestone',
      date: startDate,
      type: 'milestone',
      title: 'Started Fitness Journey',
      description: 'You began your fitness journey with us!',
      highlighted: true
    });
    
    // Add milestones for 1 month, 3 months, 6 months, etc.
    [1, 3, 6, 12, 24].forEach(month => {
      if (monthsActive >= month) {
        const milestoneDate = new Date(startDate);
        milestoneDate.setMonth(startDate.getMonth() + month);
        
        milestones.push({
          id: `milestone-${month}`,
          date: milestoneDate,
          type: 'milestone',
          title: `${month} Month${month > 1 ? 's' : ''} Anniversary`,
          description: `You've been on your fitness journey for ${month} month${month > 1 ? 's' : ''}!`,
          highlighted: month % 6 === 0 // Highlight 6-month milestones
        });
      }
    });
    
    return milestones;
  };
  
  // Convert all data into timeline events
  const generateTimelineEvents = (): TimelineEvent[] => {
    let events: TimelineEvent[] = [];
    
    // Add achievements
    events = events.concat(
      achievements.filter(a => a.completed).map(a => ({
        id: `achievement-${a.id}`,
        date: a.completedDate || new Date(),
        type: 'achievement' as const,
        title: a.title,
        description: a.description,
        data: a,
        highlighted: a.category === 'milestone'
      }))
    );
    
    // Add body measurements
    events = events.concat(
      measurements.map((m, index) => {
        // Only highlight first measurement and significant changes
        const isFirst = index === measurements.length - 1;
        const prevMeasurement = index < measurements.length - 1 ? measurements[index + 1] : null;
        
        // Calculate change in weight or body fat if available
        let changeDescription = '';
        if (prevMeasurement && m.weight && prevMeasurement.weight) {
          const weightChange = m.weight - prevMeasurement.weight;
          changeDescription = `Weight change: ${weightChange > 0 ? '+' : ''}${weightChange.toFixed(1)} kg`;
        } else if (prevMeasurement && m.bodyFat && prevMeasurement.bodyFat) {
          const fatChange = m.bodyFat - prevMeasurement.bodyFat;
          changeDescription = `Body fat change: ${fatChange > 0 ? '+' : ''}${fatChange.toFixed(1)}%`;
        }
        
        const description = isFirst 
          ? 'Started tracking your body measurements' 
          : `Updated your body measurements. ${changeDescription}`;
          
        // Determine if this is a significant change worth highlighting
        const isSignificantChange = prevMeasurement && (
          (m.weight && prevMeasurement.weight && Math.abs(m.weight - prevMeasurement.weight) >= 5) ||
          (m.bodyFat && prevMeasurement.bodyFat && Math.abs(m.bodyFat - prevMeasurement.bodyFat) >= 3)
        );
        
        return {
          id: `measurement-${index}`,
          date: m.date,
          type: 'measurement' as const,
          title: isFirst ? 'First Measurement' : 'Body Measurement Update',
          description,
          data: m,
          highlighted: isFirst || isSignificantChange
        };
      })
    );
    
    // Group strength data by exercise
    const exerciseMap: Record<string, StrengthMetric[]> = {};
    
    strengthData.forEach(metric => {
      if (!exerciseMap[metric.exercise]) {
        exerciseMap[metric.exercise] = [];
      }
      exerciseMap[metric.exercise].push(metric);
    });
    
    // For each exercise, find personal records
    Object.keys(exerciseMap).forEach(exercise => {
      // Sort by date (newest first)
      exerciseMap[exercise].sort((a, b) => 
        new Date(b.date).getTime() - new Date(a.date).getTime()
      );
      
      // Find PRs (assume the highest oneRepMax for each exercise is a PR)
      let highestOneRM = 0;
      
      exerciseMap[exercise].forEach((record, index) => {
        if (record.oneRepMax > highestOneRM) {
          highestOneRM = record.oneRepMax;
          
          events.push({
            id: `strength-${exercise}-${index}`,
            date: record.date,
            type: 'strength-record',
            title: `New PR: ${exercise}`,
            description: `You set a new personal record for ${exercise} with ${record.oneRepMax.toFixed(1)} ${record.unit} (calculated 1RM)`,
            data: record,
            highlighted: true
          });
        }
      });
    });
    
    // Add progress photos
    events = events.concat(
      progressPhotos.map((photo, index) => ({
        id: `photo-${photo.id}`,
        date: photo.date,
        type: 'progress-photo' as const,
        title: 'Progress Photo',
        description: photo.notes || `You took a progress photo (${photo.type} view)`,
        mediaUrl: photo.url,
        data: photo,
        highlighted: index === 0 || index === progressPhotos.length - 1 // Highlight first and most recent photos
      }))
    );
    
    // Add milestones
    events = events.concat(getMilestones());
    
    // Sort all events by date (newest first for now, we'll reverse for display)
    events.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
    
    return events;
  };
  
  // Get all timeline events
  const allEvents = generateTimelineEvents();
  
  // Filter events for current month if in month view
  const visibleEvents = viewMode === 'month'
    ? allEvents.filter(event => {
        const eventDate = new Date(event.date);
        return eventDate.getMonth() === currentMonth.getMonth() &&
               eventDate.getFullYear() === currentMonth.getFullYear();
      })
    : allEvents;
  
  // Group events by date for display
  const groupEventsByDate = (events: TimelineEvent[]) => {
    const grouped: Record<string, TimelineEvent[]> = {};
    
    events.forEach(event => {
      const dateKey = format(new Date(event.date), 'yyyy-MM-dd');
      if (!grouped[dateKey]) {
        grouped[dateKey] = [];
      }
      grouped[dateKey].push(event);
    });
    
    return grouped;
  };
  
  const groupedEvents = groupEventsByDate(visibleEvents);
  const dateKeys = Object.keys(groupedEvents).sort().reverse(); // Chronological order (newest last)
  
  // Get icon for event type
  const getEventIcon = (type: TimelineEvent['type']) => {
    switch (type) {
      case 'achievement':
        return <Trophy className="h-5 w-5 text-yellow-500" />;
      case 'milestone':
        return <Star className="h-5 w-5 text-purple-500" />;
      case 'measurement':
        return <LineChart className="h-5 w-5 text-blue-500" />;
      case 'strength-record':
        return <Dumbbell className="h-5 w-5 text-green-500" />;
      case 'progress-photo':
        return <Camera className="h-5 w-5 text-pink-500" />;
      case 'workout':
        return <Heart className="h-5 w-5 text-red-500" />;
      default:
        return <Calendar className="h-5 w-5 text-gray-500" />;
    }
  };
  
  // Helper function to get event class based on type
  const getEventClass = (type: TimelineEvent['type']) => {
    switch (type) {
      case 'achievement':
        return 'border-yellow-600/30 bg-yellow-900/20';
      case 'milestone':
        return 'border-purple-600/30 bg-purple-900/20';
      case 'measurement':
        return 'border-blue-600/30 bg-blue-900/20';
      case 'strength-record':
        return 'border-green-600/30 bg-green-900/20';
      case 'progress-photo':
        return 'border-pink-600/30 bg-pink-900/20';
      case 'workout':
        return 'border-red-600/30 bg-red-900/20';
      default:
        return 'border-gray-700 bg-gray-800';
    }
  };
  
  // Determine if there are events for this month
  const hasEventsForCurrentMonth = dateKeys.length > 0;
  
  return (
    <div className="fitness-journey-timeline">
      {/* Header with navigation */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-white text-xl font-medium">Your Fitness Journey</h2>
          <p className="text-gray-400">Tracking your progress since {formatTimelineDate(joinDate)}</p>
        </div>
        
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={onShareStory}>
            <Share2 className="mr-2 h-4 w-4" />
            Share Story
          </Button>
          
          <Button variant="outline" size="sm" onClick={onCreateHighlight}>
            <Play className="mr-2 h-4 w-4" />
            Create Highlight Reel
          </Button>
        </div>
      </div>
      
      {/* Timeline navigation */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex gap-2">
          <Button
            variant={viewMode === 'month' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('month')}
          >
            Month View
          </Button>
          <Button
            variant={viewMode === 'all' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('all')}
          >
            All Time
          </Button>
        </div>
        
        {viewMode === 'month' && (
          <div className="flex items-center gap-3">
            <Button size="icon" variant="outline" onClick={navigatePrevious}>
              <ChevronLeft className="h-4 w-4" />
            </Button>
            
            <h3 className="text-white font-medium w-32 text-center">
              {format(currentMonth, 'MMMM yyyy')}
            </h3>
            
            <Button 
              size="icon" 
              variant="outline" 
              onClick={navigateNext}
              disabled={format(currentMonth, 'yyyy-MM') === format(new Date(), 'yyyy-MM')}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        )}
      </div>
      
      {/* Timeline content */}
      <div className="timeline-container" ref={timelineRef}>
        {hasEventsForCurrentMonth ? (
          <div className="relative">
            {/* Vertical timeline line */}
            <div className="absolute left-[21px] top-8 bottom-4 w-[2px] bg-gray-700 z-0"></div>
            
            {/* Timeline events */}
            <div className="space-y-6">
              {dateKeys.map(dateKey => (
                <div key={dateKey} className="mb-8">
                  {/* Date header */}
                  <div className="text-white font-medium mb-3">
                    {formatTimelineDate(dateKey)}
                  </div>
                  
                  {/* Events for this date */}
                  <div className="space-y-3">
                    {groupedEvents[dateKey].map(event => (
                      <div 
                        key={event.id} 
                        className={`flex gap-4 p-4 rounded-lg border ${getEventClass(event.type)} ${
                          event.highlighted ? 'border-l-4' : ''
                        } cursor-pointer hover:bg-gray-750 transition-colors`}
                        onClick={() => setSelectedEvent(event)}
                      >
                        {/* Timeline bullet and line */}
                        <div className="mt-1 relative">
                          <div className="w-10 h-10 rounded-full bg-gray-800 flex items-center justify-center z-10 relative border border-gray-700">
                            {getEventIcon(event.type)}
                          </div>
                        </div>
                        
                        <div className="flex-grow">
                          <div className="flex justify-between items-start">
                            <div>
                              <h4 className="text-white font-medium">{event.title}</h4>
                              <p className="text-gray-400 text-sm">{event.description}</p>
                            </div>
                            
                            <Badge variant="outline">
                              {event.type.replace('-', ' ')}
                            </Badge>
                          </div>
                          
                          {/* Media preview for photos */}
                          {event.type === 'progress-photo' && event.mediaUrl && (
                            <div className="mt-3 max-w-xs">
                              <div className="aspect-square w-24 rounded overflow-hidden">
                                <img src={event.mediaUrl} alt="Progress" className="w-full h-full object-cover" />
                              </div>
                            </div>
                          )}
                          
                          {/* Achievement details */}
                          {event.type === 'achievement' && event.data && (
                            <div className="mt-3 flex items-center text-sm text-yellow-500">
                              <Award className="h-4 w-4 mr-1" />
                              <span>+{event.data.reward.experience} XP, +{event.data.reward.coins} coins</span>
                            </div>
                          )}
                          
                          {/* Strength record details */}
                          {event.type === 'strength-record' && event.data && (
                            <div className="mt-3 text-sm text-green-500">
                              {event.data.value} {event.data.unit} × {event.data.reps} reps = {event.data.oneRepMax.toFixed(1)} {event.data.unit} (1RM)
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <Card className="bg-gray-800 border-gray-700">
            <CardContent className="p-6 text-center">
              <Calendar className="h-16 w-16 text-gray-700 mx-auto mb-4" />
              <h3 className="text-white text-lg font-medium mb-2">No events for {format(currentMonth, 'MMMM yyyy')}</h3>
              <p className="text-gray-400 mb-4">
                There are no recorded fitness events or milestones during this month.
                Keep up your fitness journey to create more memories!
              </p>
              <Button onClick={() => setViewMode('all')}>
                View Full Timeline
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
      
      {/* Overall journey stats card */}
      <Card className="bg-gray-800 border-gray-700 mt-6">
        <CardContent className="p-6">
          <h3 className="text-white font-medium mb-4">Your Journey Statistics</h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-750 p-4 rounded-lg">
              <div className="text-gray-400 text-sm mb-1">Journey Length</div>
              <div className="text-white text-xl font-light">
                {differenceInMonths(new Date(), new Date(joinDate))} months
              </div>
            </div>
            
            <div className="bg-gray-750 p-4 rounded-lg">
              <div className="text-gray-400 text-sm mb-1">Achievements</div>
              <div className="text-white text-xl font-light">
                {achievements.filter(a => a.completed).length} earned
              </div>
            </div>
            
            <div className="bg-gray-750 p-4 rounded-lg">
              <div className="text-gray-400 text-sm mb-1">Progress Photos</div>
              <div className="text-white text-xl font-light">
                {progressPhotos.length} photos
              </div>
            </div>
            
            <div className="bg-gray-750 p-4 rounded-lg">
              <div className="text-gray-400 text-sm mb-1">Personal Records</div>
              <div className="text-white text-xl font-light">
                {allEvents.filter(e => e.type === 'strength-record').length} PRs
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Video generation prompt */}
      <div className="mt-6 bg-blue-900/20 border border-blue-800 rounded-lg p-4">
        <div className="flex items-start gap-4">
          <div className="w-10 h-10 rounded-full bg-blue-800/50 flex items-center justify-center shrink-0">
            <FileText className="h-5 w-5 text-blue-300" />
          </div>
          
          <div>
            <h3 className="text-white font-medium mb-1">Create Your Journey Story</h3>
            <p className="text-gray-300 text-sm mb-3">
              Turn your fitness journey into a shareable video showcasing your progress,
              achievements, and transformation.
            </p>
            
            <div className="flex flex-wrap gap-2">
              <Button onClick={onCreateHighlight} className="bg-blue-600 hover:bg-blue-700">
                <Play className="mr-2 h-4 w-4" />
                Generate Video
              </Button>
              
              <Button variant="outline">
                <Download className="mr-2 h-4 w-4" />
                Download Timeline
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 
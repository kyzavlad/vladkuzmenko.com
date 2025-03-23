'use client';

import { useState, useRef, useEffect } from 'react';
import { Camera, Calendar, Filter, ChevronLeft, ChevronRight, Plus, Trash, Download, Share } from 'lucide-react';
import { format } from 'date-fns';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { ProgressPhoto } from '../types';

interface ProgressPhotoComparisonProps {
  photos: ProgressPhoto[];
  onAddPhoto?: () => void;
  onDeletePhoto?: (photoId: string) => void;
}

type PhotoView = 'front' | 'back' | 'side' | 'custom';
type ComparisonMode = 'slider' | 'side-by-side' | 'overlay';

export default function ProgressPhotoComparison({ 
  photos,
  onAddPhoto, 
  onDeletePhoto 
}: ProgressPhotoComparisonProps) {
  const [photoView, setPhotoView] = useState<PhotoView>('front');
  const [comparisonMode, setComparisonMode] = useState<ComparisonMode>('slider');
  const [sliderPosition, setSliderPosition] = useState(50);
  const [beforeDate, setBeforeDate] = useState<string | null>(null);
  const [afterDate, setAfterDate] = useState<string | null>(null);
  const sliderContainerRef = useRef<HTMLDivElement>(null);
  
  // Filter photos by view type
  const filteredPhotos = photos.filter(photo => photo.type === photoView);
  
  // Sort photos by date
  const sortedPhotos = [...filteredPhotos].sort((a, b) => {
    return new Date(a.date).getTime() - new Date(b.date).getTime();
  });
  
  // Get unique dates for the dropdown
  const uniqueDates = Array.from(new Set(
    sortedPhotos.map(photo => format(new Date(photo.date), 'yyyy-MM-dd'))
  ));
  
  // Set default before/after dates if not already set and photos are available
  useEffect(() => {
    if (uniqueDates.length >= 2 && (!beforeDate || !afterDate)) {
      setBeforeDate(uniqueDates[0]);
      setAfterDate(uniqueDates[uniqueDates.length - 1]);
    } else if (uniqueDates.length === 1 && (!beforeDate || !afterDate)) {
      setBeforeDate(uniqueDates[0]);
      setAfterDate(uniqueDates[0]);
    }
  }, [uniqueDates, beforeDate, afterDate]);
  
  // Get the before and after photos based on selected dates
  const beforePhoto = sortedPhotos.find(
    photo => format(new Date(photo.date), 'yyyy-MM-dd') === beforeDate
  );
  
  const afterPhoto = sortedPhotos.find(
    photo => format(new Date(photo.date), 'yyyy-MM-dd') === afterDate
  );
  
  // Handle mouse move for the slider
  const handleMouseMove = (e: React.MouseEvent) => {
    if (comparisonMode !== 'slider' || !sliderContainerRef.current) return;
    
    const rect = sliderContainerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const percentage = (x / rect.width) * 100;
    
    setSliderPosition(percentage);
  };
  
  // Format date for display
  const formatPhotoDate = (date: string | Date) => {
    return format(new Date(date), 'MMM d, yyyy');
  };
  
  // Handle date selection in the dropdowns
  const handleBeforeDateChange = (value: string) => {
    setBeforeDate(value);
    
    // If after date is earlier than before date, update it
    if (afterDate && new Date(value) > new Date(afterDate)) {
      setAfterDate(value);
    }
  };
  
  const handleAfterDateChange = (value: string) => {
    setAfterDate(value);
    
    // If before date is later than after date, update it
    if (beforeDate && new Date(value) < new Date(beforeDate)) {
      setBeforeDate(value);
    }
  };
  
  return (
    <div className="progress-photo-comparison">
      {/* Controls */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
        <Tabs 
          defaultValue={photoView} 
          onValueChange={(value) => setPhotoView(value as PhotoView)}
          className="w-full md:w-auto"
        >
          <TabsList className="grid grid-cols-4 w-full">
            <TabsTrigger value="front">Front</TabsTrigger>
            <TabsTrigger value="back">Back</TabsTrigger>
            <TabsTrigger value="side">Side</TabsTrigger>
            <TabsTrigger value="custom">Custom</TabsTrigger>
          </TabsList>
        </Tabs>
        
        <div className="flex flex-wrap gap-2">
          <Button size="sm" variant="outline" className="whitespace-nowrap">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button size="sm" onClick={onAddPhoto} className="whitespace-nowrap">
            <Camera className="h-4 w-4 mr-2" />
            Add Photo
          </Button>
        </div>
      </div>
      
      {/* Main Content */}
      {sortedPhotos.length > 0 ? (
        <div className="space-y-6">
          {/* Comparison Mode Selection */}
          <div className="flex justify-between items-center">
            <Tabs 
              defaultValue={comparisonMode} 
              onValueChange={(value) => setComparisonMode(value as ComparisonMode)}
              className="w-full md:w-auto"
            >
              <TabsList className="grid grid-cols-3 w-full md:w-auto">
                <TabsTrigger value="slider">Slider</TabsTrigger>
                <TabsTrigger value="side-by-side">Side by Side</TabsTrigger>
                <TabsTrigger value="overlay">Overlay</TabsTrigger>
              </TabsList>
            </Tabs>
            
            {comparisonMode === 'slider' && (
              <div className="hidden md:block w-1/3">
                <Slider
                  value={[sliderPosition]}
                  min={0}
                  max={100}
                  step={1}
                  onValueChange={(value) => setSliderPosition(value[0])}
                />
              </div>
            )}
          </div>
          
          {/* Date Selection */}
          <Card className="bg-gray-800 border-gray-700">
            <CardContent className="p-4 flex flex-col md:flex-row justify-between gap-4">
              <div className="space-y-2 flex-grow">
                <label className="text-white text-sm">Before:</label>
                <Select value={beforeDate || ''} onValueChange={handleBeforeDateChange}>
                  <SelectTrigger className="w-full bg-gray-750">
                    <SelectValue placeholder="Select date" />
                  </SelectTrigger>
                  <SelectContent>
                    {uniqueDates.map(date => (
                      <SelectItem key={`before-${date}`} value={date}>
                        {format(new Date(date), 'MMM d, yyyy')}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2 flex-grow">
                <label className="text-white text-sm">After:</label>
                <Select value={afterDate || ''} onValueChange={handleAfterDateChange}>
                  <SelectTrigger className="w-full bg-gray-750">
                    <SelectValue placeholder="Select date" />
                  </SelectTrigger>
                  <SelectContent>
                    {uniqueDates.map(date => (
                      <SelectItem key={`after-${date}`} value={date}>
                        {format(new Date(date), 'MMM d, yyyy')}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>
          
          {/* Photo Comparison Area */}
          <div className="relative bg-gray-900 rounded-lg overflow-hidden">
            {comparisonMode === 'slider' && beforePhoto && afterPhoto && (
              <div 
                ref={sliderContainerRef}
                className="relative w-full aspect-square md:aspect-[4/3] cursor-ew-resize"
                onMouseMove={handleMouseMove}
                onTouchMove={(e) => {
                  if (!sliderContainerRef.current) return;
                  const touch = e.touches[0];
                  const rect = sliderContainerRef.current.getBoundingClientRect();
                  const x = Math.max(0, Math.min(touch.clientX - rect.left, rect.width));
                  setSliderPosition((x / rect.width) * 100);
                }}
              >
                {/* Before Image (full width) */}
                <div className="absolute inset-0">
                  <img 
                    src={beforePhoto.url} 
                    alt={`Before - ${formatPhotoDate(beforePhoto.date)}`} 
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute bottom-4 left-4 bg-black/60 text-white px-2 py-1 rounded text-sm">
                    {formatPhotoDate(beforePhoto.date)}
                  </div>
                </div>
                
                {/* After Image (clipped by slider) */}
                <div 
                  className="absolute inset-0 overflow-hidden"
                  style={{ width: `${sliderPosition}%` }}
                >
                  <img 
                    src={afterPhoto.url} 
                    alt={`After - ${formatPhotoDate(afterPhoto.date)}`} 
                    className="absolute inset-0 w-full h-full object-cover"
                    style={{ width: `${100 / (sliderPosition / 100)}%` }}
                  />
                  <div className="absolute bottom-4 left-4 bg-black/60 text-white px-2 py-1 rounded text-sm">
                    {formatPhotoDate(afterPhoto.date)}
                  </div>
                </div>
                
                {/* Slider Handle */}
                <div 
                  className="absolute top-0 bottom-0 w-1 bg-white cursor-ew-resize"
                  style={{ left: `${sliderPosition}%` }}
                >
                  <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-6 h-6 bg-white rounded-full flex items-center justify-center shadow-lg">
                    <div className="flex flex-col items-center">
                      <ChevronLeft className="h-3 w-3 text-gray-700" />
                      <ChevronRight className="h-3 w-3 text-gray-700" />
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {comparisonMode === 'side-by-side' && (
              <div className="grid grid-cols-2 gap-2 p-2">
                {beforePhoto ? (
                  <div className="relative aspect-square md:aspect-[4/3]">
                    <img 
                      src={beforePhoto.url} 
                      alt={`Before - ${formatPhotoDate(beforePhoto.date)}`} 
                      className="w-full h-full object-cover rounded"
                    />
                    <div className="absolute bottom-2 left-2 bg-black/60 text-white px-2 py-1 rounded text-sm">
                      {formatPhotoDate(beforePhoto.date)}
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center aspect-square md:aspect-[4/3] bg-gray-800 rounded">
                    <div className="text-center">
                      <Calendar className="h-8 w-8 mx-auto mb-2 text-gray-600" />
                      <p className="text-gray-400 text-sm">No before photo</p>
                    </div>
                  </div>
                )}
                
                {afterPhoto ? (
                  <div className="relative aspect-square md:aspect-[4/3]">
                    <img 
                      src={afterPhoto.url} 
                      alt={`After - ${formatPhotoDate(afterPhoto.date)}`} 
                      className="w-full h-full object-cover rounded"
                    />
                    <div className="absolute bottom-2 left-2 bg-black/60 text-white px-2 py-1 rounded text-sm">
                      {formatPhotoDate(afterPhoto.date)}
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center aspect-square md:aspect-[4/3] bg-gray-800 rounded">
                    <div className="text-center">
                      <Calendar className="h-8 w-8 mx-auto mb-2 text-gray-600" />
                      <p className="text-gray-400 text-sm">No after photo</p>
                    </div>
                  </div>
                )}
              </div>
            )}
            
            {comparisonMode === 'overlay' && beforePhoto && afterPhoto && (
              <div className="relative aspect-square md:aspect-[4/3]">
                {/* Before Image */}
                <img 
                  src={beforePhoto.url} 
                  alt={`Before - ${formatPhotoDate(beforePhoto.date)}`} 
                  className="absolute inset-0 w-full h-full object-cover"
                />
                
                {/* After Image with Opacity */}
                <img 
                  src={afterPhoto.url} 
                  alt={`After - ${formatPhotoDate(afterPhoto.date)}`} 
                  className="absolute inset-0 w-full h-full object-cover opacity-50"
                />
                
                {/* Opacity Slider */}
                <div className="absolute bottom-4 left-4 right-4 bg-black/60 p-2 rounded">
                  <Slider
                    value={[50]}
                    min={0}
                    max={100}
                    step={1}
                    onValueChange={(value) => {
                      // In a real app, this would control the opacity of the overlay
                      console.log('Opacity:', value[0]);
                    }}
                  />
                </div>
                
                {/* Labels */}
                <div className="absolute top-4 left-4 bg-black/60 text-white px-2 py-1 rounded text-sm">
                  Before: {formatPhotoDate(beforePhoto.date)}
                </div>
                <div className="absolute top-4 right-4 bg-black/60 text-white px-2 py-1 rounded text-sm">
                  After: {formatPhotoDate(afterPhoto.date)}
                </div>
              </div>
            )}
          </div>
          
          {/* Photo Timeline */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <h3 className="text-white font-medium mb-4">Photo Timeline</h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {sortedPhotos.map((photo) => (
                <div key={photo.id} className="relative group">
                  <div className="aspect-square overflow-hidden rounded-lg bg-gray-900">
                    <img 
                      src={photo.url} 
                      alt={`Photo from ${formatPhotoDate(photo.date)}`}
                      className="w-full h-full object-cover group-hover:opacity-90 transition-opacity"
                    />
                  </div>
                  <div className="absolute bottom-2 left-2 bg-black/60 text-white px-2 py-0.5 rounded text-xs">
                    {formatPhotoDate(photo.date)}
                  </div>
                  
                  {/* Actions */}
                  <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-7 w-7 bg-black/60 text-white hover:bg-black/80">
                          <Filter className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent className="w-40 bg-gray-800 border-gray-700">
                        <DropdownMenuItem 
                          className="flex items-center cursor-pointer"
                          onClick={() => handleBeforeDateChange(format(new Date(photo.date), 'yyyy-MM-dd'))}
                        >
                          <ChevronLeft className="h-4 w-4 mr-2" />
                          <span>Set as Before</span>
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          className="flex items-center cursor-pointer"
                          onClick={() => handleAfterDateChange(format(new Date(photo.date), 'yyyy-MM-dd'))}
                        >
                          <ChevronRight className="h-4 w-4 mr-2" />
                          <span>Set as After</span>
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          className="flex items-center cursor-pointer text-red-400"
                          onClick={() => onDeletePhoto && onDeletePhoto(photo.id)}
                        >
                          <Trash className="h-4 w-4 mr-2" />
                          <span>Delete</span>
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>
              ))}
              
              {/* Add Photo Button */}
              <div 
                className="aspect-square rounded-lg border-2 border-dashed border-gray-700 flex items-center justify-center cursor-pointer hover:border-gray-500 transition-colors"
                onClick={onAddPhoto}
              >
                <div className="text-center">
                  <Plus className="h-8 w-8 mx-auto mb-2 text-gray-600" />
                  <p className="text-gray-400 text-sm">Add Photo</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-6 text-center">
            <Camera className="h-16 w-16 mx-auto mb-4 text-gray-700" />
            <h3 className="text-white text-lg font-medium mb-2">No Progress Photos Yet</h3>
            <p className="text-gray-400 max-w-md mx-auto mb-6">
              Start tracking your physical transformation by taking progress photos. 
              They're a powerful way to visualize changes that may not be reflected on the scale.
            </p>
            <Button onClick={onAddPhoto}>
              <Camera className="mr-2 h-4 w-4" />
              Take Your First Progress Photo
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
} 
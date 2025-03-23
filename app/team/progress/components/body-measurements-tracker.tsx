'use client';

import { useState } from 'react';
import { 
  LineChart, 
  BarChart, 
  ArrowUp, 
  ArrowDown, 
  Calendar, 
  Plus, 
  ChevronLeft, 
  ChevronRight,
  Filter,
  Camera,
  Download
} from 'lucide-react';
import { format, parseISO, isValid, subMonths, isSameMonth } from 'date-fns';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { BodyMeasurements } from '../types';

interface BodyMeasurementsTrackerProps {
  measurements: BodyMeasurements[];
  onAddMeasurement?: () => void;
}

type MeasurementPeriod = '1m' | '3m' | '6m' | '1y' | 'all';
type MeasurementView = 'weight' | 'body-fat' | 'circumference';
type CircumferenceRegion = 'chest' | 'waist' | 'hips' | 'arms' | 'thighs' | 'calves';

export default function BodyMeasurementsTracker({ 
  measurements, 
  onAddMeasurement 
}: BodyMeasurementsTrackerProps) {
  const [period, setPeriod] = useState<MeasurementPeriod>('3m');
  const [view, setView] = useState<MeasurementView>('weight');
  const [circumferenceRegion, setCircumferenceRegion] = useState<CircumferenceRegion>('waist');
  const [selectedDate, setSelectedDate] = useState<Date | null>(
    measurements.length > 0 
      ? new Date(measurements[measurements.length - 1].date) 
      : null
  );
  
  // Filter measurements based on selected period
  const filteredMeasurements = (() => {
    if (period === 'all') return [...measurements].sort((a, b) => {
      return new Date(a.date).getTime() - new Date(b.date).getTime();
    });
    
    const now = new Date();
    const monthsAgo = {
      '1m': subMonths(now, 1),
      '3m': subMonths(now, 3),
      '6m': subMonths(now, 6),
      '1y': subMonths(now, 12)
    }[period];
    
    return [...measurements]
      .filter(m => new Date(m.date) >= monthsAgo)
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  })();
  
  // Get the latest and first measurements for comparison
  const latestMeasurement = measurements.length > 0 
    ? measurements.reduce((latest, current) => {
        const latestDate = new Date(latest.date).getTime();
        const currentDate = new Date(current.date).getTime();
        return currentDate > latestDate ? current : latest;
      }, measurements[0])
    : null;
    
  const firstMeasurement = filteredMeasurements.length > 0 
    ? filteredMeasurements[0] 
    : null;
  
  // Get the selected measurement based on date
  const selectedMeasurement = selectedDate 
    ? measurements.find(m => {
        const measurementDate = new Date(m.date);
        return isSameMonth(measurementDate, selectedDate) && 
               measurementDate.getDate() === selectedDate.getDate();
      }) || latestMeasurement
    : latestMeasurement;
  
  // Functions to handle date navigation
  const handlePreviousDate = () => {
    if (!selectedDate || !measurements.length) return;
    
    const currentIndex = measurements.findIndex(m => {
      const measurementDate = new Date(m.date);
      return isSameMonth(measurementDate, selectedDate) && 
             measurementDate.getDate() === selectedDate.getDate();
    });
    
    if (currentIndex > 0) {
      setSelectedDate(new Date(measurements[currentIndex - 1].date));
    }
  };
  
  const handleNextDate = () => {
    if (!selectedDate || !measurements.length) return;
    
    const currentIndex = measurements.findIndex(m => {
      const measurementDate = new Date(m.date);
      return isSameMonth(measurementDate, selectedDate) && 
             measurementDate.getDate() === selectedDate.getDate();
    });
    
    if (currentIndex < measurements.length - 1) {
      setSelectedDate(new Date(measurements[currentIndex + 1].date));
    }
  };
  
  // Calculate changes between measurements
  const calculateChange = (current: number | undefined, previous: number | undefined) => {
    if (current === undefined || previous === undefined) return null;
    return current - previous;
  };
  
  // Get title and units based on view
  const viewDetails = {
    'weight': { title: 'Weight', unit: 'kg', property: 'weight' },
    'body-fat': { title: 'Body Fat', unit: '%', property: 'bodyFat' },
    'circumference': { 
      title: 'Measurements', 
      unit: 'cm',
      property: circumferenceRegion === 'arms' 
        ? 'leftArm' 
        : circumferenceRegion === 'thighs'
        ? 'leftThigh'
        : circumferenceRegion === 'calves'
        ? 'leftCalf'
        : circumferenceRegion
    }
  };
  
  // Get region specific title
  const getRegionTitle = () => {
    switch (circumferenceRegion) {
      case 'chest': return 'Chest';
      case 'waist': return 'Waist';
      case 'hips': return 'Hips';
      case 'arms': return 'Arms';
      case 'thighs': return 'Thighs';
      case 'calves': return 'Calves';
      default: return 'Circumference';
    }
  };
  
  // Format date for display
  const formatMeasurementDate = (date: string | Date) => {
    const parsedDate = typeof date === 'string' ? parseISO(date) : date;
    return isValid(parsedDate) ? format(parsedDate, 'MMM d, yyyy') : 'Invalid date';
  };
  
  return (
    <div className="body-measurements-tracker">
      {/* Controls */}
      <div className="flex flex-wrap justify-between items-center gap-4 mb-6">
        <div className="flex gap-2 overflow-x-auto pb-2 md:pb-0">
          <Button
            variant={period === '1m' ? "default" : "outline"}
            size="sm"
            onClick={() => setPeriod('1m')}
          >
            1 Month
          </Button>
          <Button
            variant={period === '3m' ? "default" : "outline"}
            size="sm"
            onClick={() => setPeriod('3m')}
          >
            3 Months
          </Button>
          <Button
            variant={period === '6m' ? "default" : "outline"}
            size="sm"
            onClick={() => setPeriod('6m')}
          >
            6 Months
          </Button>
          <Button
            variant={period === '1y' ? "default" : "outline"}
            size="sm"
            onClick={() => setPeriod('1y')}
          >
            1 Year
          </Button>
          <Button
            variant={period === 'all' ? "default" : "outline"}
            size="sm"
            onClick={() => setPeriod('all')}
          >
            All Time
          </Button>
        </div>
        
        <div className="flex gap-2">
          <Button size="sm" variant="outline">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button size="sm" onClick={onAddMeasurement}>
            <Plus className="h-4 w-4 mr-2" />
            Add Measurement
          </Button>
        </div>
      </div>
      
      {/* Main Content with tabs */}
      <Tabs defaultValue={view} onValueChange={(value) => setView(value as MeasurementView)} className="w-full">
        <TabsList className="grid grid-cols-3 mb-6">
          <TabsTrigger value="weight">
            <BarChart className="h-4 w-4 mr-2" />
            Weight
          </TabsTrigger>
          <TabsTrigger value="body-fat">
            <LineChart className="h-4 w-4 mr-2" />
            Body Fat %
          </TabsTrigger>
          <TabsTrigger value="circumference">
            <Filter className="h-4 w-4 mr-2" />
            Circumference
          </TabsTrigger>
        </TabsList>
        
        {/* The chart area would be identical for each tab but with different data */}
        {/* In a real implementation, this would use a charting library like Chart.js or Recharts */}
        <div className="mb-6 bg-gray-800 border border-gray-700 rounded-lg p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-white font-medium">
              {view === 'circumference' 
                ? `${getRegionTitle()} Measurements` 
                : `${viewDetails[view].title} History`}
            </h3>
            
            {view === 'circumference' && (
              <Select 
                value={circumferenceRegion} 
                onValueChange={(value) => setCircumferenceRegion(value as CircumferenceRegion)}
              >
                <SelectTrigger className="w-[160px] h-8 text-sm">
                  <SelectValue placeholder="Select Region" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="chest">Chest</SelectItem>
                  <SelectItem value="waist">Waist</SelectItem>
                  <SelectItem value="hips">Hips</SelectItem>
                  <SelectItem value="arms">Arms</SelectItem>
                  <SelectItem value="thighs">Thighs</SelectItem>
                  <SelectItem value="calves">Calves</SelectItem>
                </SelectContent>
              </Select>
            )}
          </div>
          
          {filteredMeasurements.length > 0 ? (
            <div className="h-72 flex items-end justify-around border-b border-gray-700 mb-2">
              {/* Placeholder for chart - would use a real chart library in production */}
              {filteredMeasurements.map((measurement, index) => {
                const value = measurement[viewDetails[view].property as keyof typeof measurement] as number | undefined;
                if (value === undefined) return null;
                
                // For demo purposes, calculate a bar height
                // In reality, would scale based on min/max values in dataset
                const heightPercentage = Math.min(100, Math.max(10, (value / (view === 'weight' ? 100 : view === 'body-fat' ? 30 : 100)) * 100));
                
                return (
                  <div key={index} className="flex flex-col items-center group cursor-pointer"
                    onClick={() => setSelectedDate(new Date(measurement.date))}>
                    <div 
                      className={`w-10 ${
                        selectedMeasurement && new Date(selectedMeasurement.date).getTime() === new Date(measurement.date).getTime()
                          ? 'bg-blue-500' 
                          : 'bg-blue-800 group-hover:bg-blue-700'
                      } rounded-t-sm transition-all`}
                      style={{ height: `${heightPercentage}%` }}
                    ></div>
                    <div className="text-xs text-gray-400 mt-2 whitespace-nowrap">{format(new Date(measurement.date), 'MMM d')}</div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="h-72 flex items-center justify-center border-b border-gray-700 mb-2 text-gray-500">
              No measurement data available for the selected period
            </div>
          )}
          
          <div className="flex justify-between text-xs text-gray-500">
            <div>
              {filteredMeasurements.length > 0
                ? formatMeasurementDate(filteredMeasurements[0].date)
                : 'No data'}
            </div>
            <div>
              {filteredMeasurements.length > 0
                ? formatMeasurementDate(filteredMeasurements[filteredMeasurements.length - 1].date)
                : ''}
            </div>
          </div>
        </div>
        
        {/* Current vs. Initial Comparison Card */}
        <Card className="bg-gray-800 border-gray-700 mb-6">
          <CardContent className="p-6">
            <div className="flex justify-between items-center mb-5">
              <h3 className="text-white font-medium">Measurement Details</h3>
              <div className="flex gap-2">
                <Button size="icon" variant="outline" onClick={handlePreviousDate} 
                  disabled={!selectedDate || measurements.indexOf(selectedMeasurement!) <= 0}>
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <Button size="icon" variant="outline" onClick={handleNextDate}
                  disabled={!selectedDate || measurements.indexOf(selectedMeasurement!) >= measurements.length - 1}>
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </div>
            
            {selectedMeasurement ? (
              <>
                <div className="text-gray-400 text-sm mb-4 flex items-center">
                  <Calendar className="h-4 w-4 mr-2" />
                  Showing measurements from {formatMeasurementDate(selectedMeasurement.date)}
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {selectedMeasurement.weight !== undefined && (
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <h4 className="text-white font-medium">Weight</h4>
                        {firstMeasurement && firstMeasurement.weight !== undefined && (
                          <Badge className={
                            selectedMeasurement.weight < firstMeasurement.weight
                              ? 'bg-green-600'
                              : selectedMeasurement.weight > firstMeasurement.weight
                              ? 'bg-red-600'
                              : 'bg-gray-600'
                          }>
                            {calculateChange(selectedMeasurement.weight, firstMeasurement.weight)?.toFixed(1)} kg
                          </Badge>
                        )}
                      </div>
                      <div className="text-3xl text-white font-light">{selectedMeasurement.weight} kg</div>
                      {firstMeasurement && firstMeasurement.weight !== undefined && (
                        <div className="text-xs text-gray-400">
                          Started at {firstMeasurement.weight} kg on {formatMeasurementDate(firstMeasurement.date)}
                        </div>
                      )}
                    </div>
                  )}
                  
                  {selectedMeasurement.bodyFat !== undefined && (
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <h4 className="text-white font-medium">Body Fat</h4>
                        {firstMeasurement && firstMeasurement.bodyFat !== undefined && (
                          <Badge className={
                            selectedMeasurement.bodyFat < firstMeasurement.bodyFat
                              ? 'bg-green-600'
                              : selectedMeasurement.bodyFat > firstMeasurement.bodyFat
                              ? 'bg-red-600'
                              : 'bg-gray-600'
                          }>
                            {calculateChange(selectedMeasurement.bodyFat, firstMeasurement.bodyFat)?.toFixed(1)}%
                          </Badge>
                        )}
                      </div>
                      <div className="text-3xl text-white font-light">{selectedMeasurement.bodyFat}%</div>
                      {firstMeasurement && firstMeasurement.bodyFat !== undefined && (
                        <div className="text-xs text-gray-400">
                          Started at {firstMeasurement.bodyFat}% on {formatMeasurementDate(firstMeasurement.date)}
                        </div>
                      )}
                    </div>
                  )}
                  
                  {selectedMeasurement.chest !== undefined && (
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <h4 className="text-white font-medium">Chest</h4>
                        {firstMeasurement && firstMeasurement.chest !== undefined && (
                          <Badge className={
                            selectedMeasurement.chest > firstMeasurement.chest
                              ? 'bg-green-600'
                              : selectedMeasurement.chest < firstMeasurement.chest
                              ? 'bg-red-600'
                              : 'bg-gray-600'
                          }>
                            {calculateChange(selectedMeasurement.chest, firstMeasurement.chest)?.toFixed(1)} cm
                          </Badge>
                        )}
                      </div>
                      <div className="text-3xl text-white font-light">{selectedMeasurement.chest} cm</div>
                      {firstMeasurement && firstMeasurement.chest !== undefined && (
                        <div className="text-xs text-gray-400">
                          Started at {firstMeasurement.chest} cm on {formatMeasurementDate(firstMeasurement.date)}
                        </div>
                      )}
                    </div>
                  )}
                  
                  {selectedMeasurement.waist !== undefined && (
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <h4 className="text-white font-medium">Waist</h4>
                        {firstMeasurement && firstMeasurement.waist !== undefined && (
                          <Badge className={
                            selectedMeasurement.waist < firstMeasurement.waist
                              ? 'bg-green-600'
                              : selectedMeasurement.waist > firstMeasurement.waist
                              ? 'bg-red-600'
                              : 'bg-gray-600'
                          }>
                            {calculateChange(selectedMeasurement.waist, firstMeasurement.waist)?.toFixed(1)} cm
                          </Badge>
                        )}
                      </div>
                      <div className="text-3xl text-white font-light">{selectedMeasurement.waist} cm</div>
                      {firstMeasurement && firstMeasurement.waist !== undefined && (
                        <div className="text-xs text-gray-400">
                          Started at {firstMeasurement.waist} cm on {formatMeasurementDate(firstMeasurement.date)}
                        </div>
                      )}
                    </div>
                  )}
                  
                  {selectedMeasurement.hips !== undefined && (
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <h4 className="text-white font-medium">Hips</h4>
                        {firstMeasurement && firstMeasurement.hips !== undefined && (
                          <Badge className={
                            selectedMeasurement.hips < firstMeasurement.hips
                              ? 'bg-green-600'
                              : selectedMeasurement.hips > firstMeasurement.hips
                              ? 'bg-red-600'
                              : 'bg-gray-600'
                          }>
                            {calculateChange(selectedMeasurement.hips, firstMeasurement.hips)?.toFixed(1)} cm
                          </Badge>
                        )}
                      </div>
                      <div className="text-3xl text-white font-light">{selectedMeasurement.hips} cm</div>
                      {firstMeasurement && firstMeasurement.hips !== undefined && (
                        <div className="text-xs text-gray-400">
                          Started at {firstMeasurement.hips} cm on {formatMeasurementDate(firstMeasurement.date)}
                        </div>
                      )}
                    </div>
                  )}
                  
                  {/* Arms (showing the average) */}
                  {(selectedMeasurement.leftArm !== undefined || selectedMeasurement.rightArm !== undefined) && (
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <h4 className="text-white font-medium">Arms (avg)</h4>
                        {firstMeasurement && 
                         (firstMeasurement.leftArm !== undefined || firstMeasurement.rightArm !== undefined) && (
                          <Badge className="bg-green-600">
                            {(() => {
                              const currentAvg = (
                                (selectedMeasurement.leftArm || 0) + 
                                (selectedMeasurement.rightArm || selectedMeasurement.leftArm || 0)
                              ) / (selectedMeasurement.leftArm && selectedMeasurement.rightArm ? 2 : 1);
                              
                              const firstAvg = (
                                (firstMeasurement.leftArm || 0) + 
                                (firstMeasurement.rightArm || firstMeasurement.leftArm || 0)
                              ) / (firstMeasurement.leftArm && firstMeasurement.rightArm ? 2 : 1);
                              
                              return (currentAvg - firstAvg).toFixed(1);
                            })()} cm
                          </Badge>
                        )}
                      </div>
                      <div className="text-3xl text-white font-light">
                        {(() => {
                          if (selectedMeasurement.leftArm && selectedMeasurement.rightArm) {
                            return ((selectedMeasurement.leftArm + selectedMeasurement.rightArm) / 2).toFixed(1);
                          }
                          return selectedMeasurement.leftArm || selectedMeasurement.rightArm || '-';
                        })()} cm
                      </div>
                      <div className="flex justify-between text-sm">
                        <div className="text-gray-400">L: {selectedMeasurement.leftArm || '-'} cm</div>
                        <div className="text-gray-400">R: {selectedMeasurement.rightArm || '-'} cm</div>
                      </div>
                    </div>
                  )}
                  
                  {/* Thighs (showing the average) */}
                  {(selectedMeasurement.leftThigh !== undefined || selectedMeasurement.rightThigh !== undefined) && (
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <h4 className="text-white font-medium">Thighs (avg)</h4>
                        {firstMeasurement && 
                         (firstMeasurement.leftThigh !== undefined || firstMeasurement.rightThigh !== undefined) && (
                          <Badge className="bg-green-600">
                            {(() => {
                              const currentAvg = (
                                (selectedMeasurement.leftThigh || 0) + 
                                (selectedMeasurement.rightThigh || selectedMeasurement.leftThigh || 0)
                              ) / (selectedMeasurement.leftThigh && selectedMeasurement.rightThigh ? 2 : 1);
                              
                              const firstAvg = (
                                (firstMeasurement.leftThigh || 0) + 
                                (firstMeasurement.rightThigh || firstMeasurement.leftThigh || 0)
                              ) / (firstMeasurement.leftThigh && firstMeasurement.rightThigh ? 2 : 1);
                              
                              return (currentAvg - firstAvg).toFixed(1);
                            })()} cm
                          </Badge>
                        )}
                      </div>
                      <div className="text-3xl text-white font-light">
                        {(() => {
                          if (selectedMeasurement.leftThigh && selectedMeasurement.rightThigh) {
                            return ((selectedMeasurement.leftThigh + selectedMeasurement.rightThigh) / 2).toFixed(1);
                          }
                          return selectedMeasurement.leftThigh || selectedMeasurement.rightThigh || '-';
                        })()} cm
                      </div>
                      <div className="flex justify-between text-sm">
                        <div className="text-gray-400">L: {selectedMeasurement.leftThigh || '-'} cm</div>
                        <div className="text-gray-400">R: {selectedMeasurement.rightThigh || '-'} cm</div>
                      </div>
                    </div>
                  )}
                </div>
                
                {selectedMeasurement.notes && (
                  <div className="mt-6 bg-gray-750 p-4 rounded-lg">
                    <h4 className="text-white font-medium mb-2">Notes</h4>
                    <p className="text-gray-400 text-sm">{selectedMeasurement.notes}</p>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center py-12">
                <Calendar className="h-16 w-16 text-gray-700 mx-auto mb-4" />
                <h4 className="text-white font-medium mb-2">No Measurements Available</h4>
                <p className="text-gray-400 mb-4">Start tracking your body measurements to see your progress over time</p>
                <Button onClick={onAddMeasurement}>
                  <Plus className="mr-2 h-4 w-4" />
                  Add Your First Measurement
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
        
        {/* Photos Preview if available */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-white font-medium">Progress Photos</h3>
            <Button size="sm" variant="outline" asChild>
              <div>
                <Camera className="h-4 w-4 mr-2" />
                Add Photos
              </div>
            </Button>
          </div>
          
          <div className="text-center py-8 text-gray-400">
            <Camera className="h-16 w-16 mx-auto mb-4 text-gray-700" />
            <p className="mb-3">Add progress photos to visualize your transformation</p>
            <p className="text-sm">Photos are stored securely and visible only to you</p>
          </div>
        </div>
      </Tabs>
    </div>
  );
} 
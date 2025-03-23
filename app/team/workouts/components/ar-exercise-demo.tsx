'use client';

import { useState, useRef, useEffect } from 'react';
import { Camera, Video, RefreshCw, X, Maximize2, Minimize2, Info, PlayCircle, PauseCircle, SkipForward, SkipBack, RotateCw, Check, Zap } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Slider } from '@/components/ui/slider';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

type DemoState = 'inactive' | 'initializing' | 'positioning' | 'demo' | 'paused';
type ViewMode = 'full' | 'side-by-side' | 'picture-in-picture';
type HighlightArea = 'none' | 'shoulders' | 'back' | 'core' | 'hips' | 'knees' | 'feet';
type DemoSpeed = 0.5 | 0.75 | 1 | 1.5 | 2;

interface ARExerciseDemoProps {
  exerciseName?: string;
  exerciseCategory?: string;
  onComplete?: () => void;
}

export default function ARExerciseDemo({
  exerciseName = 'Barbell Squat',
  exerciseCategory = 'Lower Body',
  onComplete
}: ARExerciseDemoProps) {
  const [demoState, setDemoState] = useState<DemoState>('inactive');
  const [viewMode, setViewMode] = useState<ViewMode>('full');
  const [highlightArea, setHighlightArea] = useState<HighlightArea>('none');
  const [demoSpeed, setDemoSpeed] = useState<DemoSpeed>(1);
  const [progress, setProgress] = useState(0);
  const [demoSteps, setDemoSteps] = useState<Array<{
    title: string;
    description: string;
    timeCode: number;
    muscleGroup?: string;
  }>>([
    { 
      title: 'Starting Position', 
      description: 'Stand with feet shoulder-width apart, toes slightly turned out.', 
      timeCode: 0,
      muscleGroup: 'core'
    },
    { 
      title: 'Bar Placement', 
      description: 'Rest the bar on the upper back, supported by shoulder muscles.', 
      timeCode: 2,
      muscleGroup: 'shoulders'
    },
    { 
      title: 'Descent', 
      description: 'Engage core, hinge at hips, bend knees, keeping back neutral.', 
      timeCode: 5,
      muscleGroup: 'hips'
    },
    { 
      title: 'Bottom Position', 
      description: 'Thighs parallel to ground, knees tracking over toes.', 
      timeCode: 8,
      muscleGroup: 'knees'
    },
    { 
      title: 'Ascent', 
      description: 'Drive through heels, keep chest up, maintain back position.', 
      timeCode: 11,
      muscleGroup: 'back'
    },
    { 
      title: 'Lockout', 
      description: 'Fully extend hips and knees, neutral spine, brace core.', 
      timeCode: 14,
      muscleGroup: 'core'
    },
  ]);
  const [currentStep, setCurrentStep] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [cameraPermission, setCameraPermission] = useState<boolean | null>(null);
  const [isMirrored, setIsMirrored] = useState(true);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const demoVideoRef = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number | null>(null);
  
  // Initialize camera when component mounts
  useEffect(() => {
    return () => {
      // Clean up stream when component unmounts
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);
  
  // Handle demo video progress
  useEffect(() => {
    if (demoState === 'demo' && demoVideoRef.current) {
      const updateProgress = () => {
        if (demoVideoRef.current) {
          const currentTime = demoVideoRef.current.currentTime;
          const duration = demoVideoRef.current.duration;
          
          if (duration) {
            // Update progress
            const progressValue = (currentTime / duration) * 100;
            setProgress(progressValue);
            
            // Find current step based on time code
            const step = demoSteps.findIndex(s => 
              currentTime >= s.timeCode && 
              (demoSteps[demoSteps.indexOf(s) + 1] 
                ? currentTime < demoSteps[demoSteps.indexOf(s) + 1].timeCode 
                : true)
            );
            
            if (step !== -1 && step !== currentStep) {
              setCurrentStep(step);
              
              // Update highlight area based on step
              const muscleGroup = demoSteps[step].muscleGroup as HighlightArea | undefined;
              if (muscleGroup) {
                setHighlightArea(muscleGroup);
              }
            }
          }
          
          animationRef.current = requestAnimationFrame(updateProgress);
        }
      };
      
      animationRef.current = requestAnimationFrame(updateProgress);
      
      return () => {
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
        }
      };
    }
  }, [demoState, demoSteps, currentStep]);
  
  // Initialize camera
  const startCamera = async () => {
    try {
      setDemoState('initializing');
      
      // Request camera permission
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'user' },
        audio: false
      });
      
      // Set video source
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      
      streamRef.current = stream;
      setCameraPermission(true);
      
      // After initialization, move to positioning state
      setTimeout(() => {
        setDemoState('positioning');
      }, 2000);
      
    } catch (error) {
      console.error('Error accessing camera:', error);
      setCameraPermission(false);
      setDemoState('inactive');
    }
  };
  
  // Start demo playback
  const startDemo = () => {
    setDemoState('demo');
    
    if (demoVideoRef.current) {
      demoVideoRef.current.playbackRate = demoSpeed;
      demoVideoRef.current.play();
    }
  };
  
  // Pause demo
  const pauseDemo = () => {
    setDemoState('paused');
    
    if (demoVideoRef.current) {
      demoVideoRef.current.pause();
    }
  };
  
  // Resume demo
  const resumeDemo = () => {
    setDemoState('demo');
    
    if (demoVideoRef.current) {
      demoVideoRef.current.play();
    }
  };
  
  // Skip to specific step
  const skipToStep = (stepIndex: number) => {
    if (demoVideoRef.current && stepIndex >= 0 && stepIndex < demoSteps.length) {
      demoVideoRef.current.currentTime = demoSteps[stepIndex].timeCode;
      setCurrentStep(stepIndex);
      
      // If paused, resume playback
      if (demoState === 'paused') {
        resumeDemo();
      }
    }
  };
  
  // Change demo speed
  const changeSpeed = (speed: DemoSpeed) => {
    setDemoSpeed(speed);
    
    if (demoVideoRef.current) {
      demoVideoRef.current.playbackRate = speed;
    }
  };
  
  // Toggle fullscreen
  const toggleFullscreen = () => {
    if (!document.fullscreenElement && containerRef.current) {
      containerRef.current.requestFullscreen().catch(err => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`);
      });
      setIsFullscreen(true);
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
        setIsFullscreen(false);
      }
    }
  };
  
  // Draw overlay on canvas
  useEffect(() => {
    if (
      overlayCanvasRef.current && 
      videoRef.current && 
      (demoState === 'positioning' || demoState === 'demo' || demoState === 'paused')
    ) {
      const canvas = overlayCanvasRef.current;
      const ctx = canvas.getContext('2d');
      
      if (!ctx) return;
      
      const drawOverlay = () => {
        // Match canvas size to video
        canvas.width = videoRef.current?.videoWidth || 640;
        canvas.height = videoRef.current?.videoHeight || 480;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // For positioning mode, draw outline to help user align
        if (demoState === 'positioning') {
          ctx.strokeStyle = 'rgba(0, 255, 0, 0.7)';
          ctx.lineWidth = 2;
          
          // Draw human outline
          const centerX = canvas.width / 2;
          const centerY = canvas.height / 2;
          const height = canvas.height * 0.8;
          
          // Head
          ctx.beginPath();
          ctx.arc(centerX, centerY - height * 0.35, height * 0.08, 0, Math.PI * 2);
          ctx.stroke();
          
          // Body
          ctx.beginPath();
          ctx.moveTo(centerX, centerY - height * 0.27);
          ctx.lineTo(centerX, centerY + height * 0.1);
          ctx.stroke();
          
          // Arms
          ctx.beginPath();
          ctx.moveTo(centerX - height * 0.15, centerY - height * 0.2);
          ctx.lineTo(centerX, centerY - height * 0.25);
          ctx.lineTo(centerX + height * 0.15, centerY - height * 0.2);
          ctx.stroke();
          
          // Legs
          ctx.beginPath();
          ctx.moveTo(centerX, centerY + height * 0.1);
          ctx.lineTo(centerX - height * 0.1, centerY + height * 0.4);
          ctx.stroke();
          
          ctx.beginPath();
          ctx.moveTo(centerX, centerY + height * 0.1);
          ctx.lineTo(centerX + height * 0.1, centerY + height * 0.4);
          ctx.stroke();
          
          // Instruction text
          ctx.fillStyle = 'white';
          ctx.font = '18px Arial';
          ctx.textAlign = 'center';
          ctx.fillText('Position yourself to match the outline', centerX, centerY - height * 0.45);
          ctx.fillText('Click "Start Demo" when ready', centerX, centerY + height * 0.5);
        }
        
        // In demo or paused mode, draw highlights for specific muscle groups if needed
        if ((demoState === 'demo' || demoState === 'paused') && highlightArea !== 'none') {
          const centerX = canvas.width / 2;
          const centerY = canvas.height / 2;
          const height = canvas.height * 0.8;
          
          ctx.fillStyle = 'rgba(255, 165, 0, 0.3)'; // Semi-transparent orange
          
          switch (highlightArea) {
            case 'shoulders':
              ctx.beginPath();
              ctx.ellipse(centerX - height * 0.1, centerY - height * 0.25, height * 0.08, height * 0.04, 0, 0, Math.PI * 2);
              ctx.fill();
              
              ctx.beginPath();
              ctx.ellipse(centerX + height * 0.1, centerY - height * 0.25, height * 0.08, height * 0.04, 0, 0, Math.PI * 2);
              ctx.fill();
              break;
              
            case 'back':
              ctx.beginPath();
              ctx.rect(centerX - height * 0.1, centerY - height * 0.25, height * 0.2, height * 0.25);
              ctx.fill();
              break;
              
            case 'core':
              ctx.beginPath();
              ctx.rect(centerX - height * 0.1, centerY - height * 0.1, height * 0.2, height * 0.2);
              ctx.fill();
              break;
              
            case 'hips':
              ctx.beginPath();
              ctx.ellipse(centerX, centerY + height * 0.1, height * 0.15, height * 0.1, 0, 0, Math.PI * 2);
              ctx.fill();
              break;
              
            case 'knees':
              ctx.beginPath();
              ctx.ellipse(centerX - height * 0.1, centerY + height * 0.25, height * 0.05, height * 0.05, 0, 0, Math.PI * 2);
              ctx.fill();
              
              ctx.beginPath();
              ctx.ellipse(centerX + height * 0.1, centerY + height * 0.25, height * 0.05, height * 0.05, 0, 0, Math.PI * 2);
              ctx.fill();
              break;
              
            case 'feet':
              ctx.beginPath();
              ctx.ellipse(centerX - height * 0.1, centerY + height * 0.4, height * 0.08, height * 0.04, 0, 0, Math.PI * 2);
              ctx.fill();
              
              ctx.beginPath();
              ctx.ellipse(centerX + height * 0.1, centerY + height * 0.4, height * 0.08, height * 0.04, 0, 0, Math.PI * 2);
              ctx.fill();
              break;
          }
        }
        
        requestAnimationFrame(drawOverlay);
      };
      
      drawOverlay();
      
      return () => {
        cancelAnimationFrame(0); // Just to make sure we clean up
      };
    }
  }, [demoState, highlightArea]);
  
  // Render different views based on state
  const renderContent = () => {
    switch (demoState) {
      case 'inactive':
        return (
          <div className="flex flex-col items-center justify-center h-full py-12">
            <Camera className="h-16 w-16 mb-4 text-blue-500" />
            <h3 className="text-white text-xl font-medium mb-2">AR Exercise Demo</h3>
            <p className="text-gray-400 text-center mb-6 max-w-md">
              Get a detailed AR walkthrough of {exerciseName} with proper form guidance and real-time feedback.
            </p>
            <Button onClick={startCamera} size="lg">
              <Camera className="mr-2 h-5 w-5" />
              Start AR Demo
            </Button>
          </div>
        );
        
      case 'initializing':
        return (
          <div className="flex flex-col items-center justify-center h-full py-12">
            <RefreshCw className="h-16 w-16 mb-4 text-blue-500 animate-spin" />
            <h3 className="text-white text-xl font-medium mb-2">Initializing AR</h3>
            <p className="text-gray-400 text-center mb-6 max-w-md">
              Setting up cameras and loading AR models. This may take a few moments.
            </p>
            <Progress value={60} className="w-64 mb-4" />
            <p className="text-gray-500 text-sm">Calibrating AR overlay...</p>
          </div>
        );
        
      case 'positioning':
      case 'demo':
      case 'paused':
        return (
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
            <div className={viewMode === 'full' ? 'lg:col-span-5' : 'lg:col-span-3'}>
              {/* Main View Area */}
              <div className="space-y-4">
                {/* Controls */}
                <div className="flex flex-wrap justify-between gap-2">
                  <div className="flex gap-2">
                    <Select 
                      value={viewMode} 
                      onValueChange={(value) => setViewMode(value as ViewMode)}
                    >
                      <SelectTrigger className="w-[140px]">
                        <SelectValue placeholder="View Mode" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="full">Full View</SelectItem>
                        <SelectItem value="side-by-side">Side by Side</SelectItem>
                        <SelectItem value="picture-in-picture">Picture in Picture</SelectItem>
                      </SelectContent>
                    </Select>
                    
                    <Select 
                      value={String(demoSpeed)} 
                      onValueChange={(value) => changeSpeed(parseFloat(value) as DemoSpeed)}
                    >
                      <SelectTrigger className="w-[110px]">
                        <SelectValue placeholder="Speed" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="0.5">0.5x</SelectItem>
                        <SelectItem value="0.75">0.75x</SelectItem>
                        <SelectItem value="1">1x</SelectItem>
                        <SelectItem value="1.5">1.5x</SelectItem>
                        <SelectItem value="2">2x</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="flex gap-2">
                    <Button 
                      variant="outline" 
                      size="icon" 
                      onClick={() => setIsMirrored(!isMirrored)}
                    >
                      <RotateCw className="h-4 w-4" />
                    </Button>
                    
                    <Button 
                      variant="outline" 
                      size="icon"
                      onClick={toggleFullscreen}
                    >
                      {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
                    </Button>
                  </div>
                </div>
                
                {/* AR View */}
                <div ref={containerRef} className="relative bg-black rounded-lg overflow-hidden aspect-video">
                  {/* Camera Feed */}
                  <video 
                    ref={videoRef} 
                    autoPlay 
                    playsInline 
                    muted
                    className={`w-full h-full object-cover ${isMirrored ? 'scale-x-[-1]' : ''}`}
                  />
                  
                  {/* AR Overlay Canvas */}
                  <canvas 
                    ref={overlayCanvasRef} 
                    className={`absolute inset-0 w-full h-full ${isMirrored ? 'scale-x-[-1]' : ''}`}
                  />
                  
                  {/* Demo Video Overlay - render based on view mode */}
                  {viewMode === 'full' && (demoState === 'demo' || demoState === 'paused') && (
                    <video 
                      ref={demoVideoRef}
                      src="/exercise-demos/squat.mp4" // In a real app, this would be dynamic
                      className="absolute inset-0 w-full h-full object-cover opacity-60"
                      muted
                      loop
                    />
                  )}
                  
                  {viewMode === 'picture-in-picture' && (demoState === 'demo' || demoState === 'paused') && (
                    <div className="absolute top-4 right-4 w-1/3 h-1/3 border-2 border-white rounded overflow-hidden">
                      <video 
                        ref={demoVideoRef}
                        src="/exercise-demos/squat.mp4" // In a real app, this would be dynamic
                        className="w-full h-full object-cover"
                        muted
                        loop
                      />
                    </div>
                  )}
                  
                  {/* Controls overlay */}
                  <div className="absolute bottom-4 left-4 right-4 flex justify-center">
                    {demoState === 'positioning' && (
                      <Button onClick={startDemo}>
                        <PlayCircle className="mr-2 h-4 w-4" />
                        Start Demo
                      </Button>
                    )}
                    
                    {demoState === 'demo' && (
                      <div className="flex gap-2">
                        <Button variant="secondary" size="sm" onClick={() => skipToStep(currentStep - 1)} disabled={currentStep === 0}>
                          <SkipBack className="h-4 w-4" />
                        </Button>
                        
                        <Button onClick={pauseDemo}>
                          <PauseCircle className="mr-2 h-4 w-4" />
                          Pause
                        </Button>
                        
                        <Button variant="secondary" size="sm" onClick={() => skipToStep(currentStep + 1)} disabled={currentStep === demoSteps.length - 1}>
                          <SkipForward className="h-4 w-4" />
                        </Button>
                      </div>
                    )}
                    
                    {demoState === 'paused' && (
                      <div className="flex gap-2">
                        <Button variant="secondary" size="sm" onClick={() => skipToStep(currentStep - 1)} disabled={currentStep === 0}>
                          <SkipBack className="h-4 w-4" />
                        </Button>
                        
                        <Button onClick={resumeDemo}>
                          <PlayCircle className="mr-2 h-4 w-4" />
                          Resume
                        </Button>
                        
                        <Button variant="secondary" size="sm" onClick={() => skipToStep(currentStep + 1)} disabled={currentStep === demoSteps.length - 1}>
                          <SkipForward className="h-4 w-4" />
                        </Button>
                      </div>
                    )}
                  </div>
                  
                  {/* Current step overlay */}
                  {(demoState === 'demo' || demoState === 'paused') && (
                    <div className="absolute top-4 left-4 right-4 bg-black/60 rounded-lg p-3">
                      <h4 className="text-white font-medium mb-1">{demoSteps[currentStep].title}</h4>
                      <p className="text-gray-300 text-sm">{demoSteps[currentStep].description}</p>
                    </div>
                  )}
                </div>
                
                {/* Progress Bar */}
                {(demoState === 'demo' || demoState === 'paused') && (
                  <div className="space-y-2">
                    <Progress value={progress} className="h-2" />
                    
                    <div className="flex justify-between text-xs text-gray-400">
                      {demoSteps.map((step, index) => (
                        <TooltipProvider key={index}>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <button 
                                className={`w-2 h-2 rounded-full ${
                                  currentStep === index ? 'bg-blue-500' : 'bg-gray-600'
                                }`}
                                onClick={() => skipToStep(index)}
                              />
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>{step.title}</p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
            
            {viewMode !== 'full' && (
              <div className="lg:col-span-2 space-y-4">
                {/* Side View Demo Video */}
                {viewMode === 'side-by-side' && (
                  <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
                    <video 
                      ref={demoVideoRef}
                      src="/exercise-demos/squat.mp4" // In a real app, this would be dynamic
                      className="w-full h-full object-cover"
                      muted
                      loop
                    />
                    
                    {/* Step indicator */}
                    <div className="absolute bottom-4 left-4 right-4 flex justify-center">
                      <Badge className="bg-blue-500">
                        Step {currentStep + 1} of {demoSteps.length}
                      </Badge>
                    </div>
                  </div>
                )}
                
                {/* Exercise Info */}
                <Card className="bg-gray-800 border-gray-700">
                  <CardContent className="p-4">
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <h3 className="text-white font-medium">{exerciseName}</h3>
                        <p className="text-gray-400 text-sm">{exerciseCategory}</p>
                      </div>
                      
                      <Badge className="bg-blue-900">
                        AR Demo
                      </Badge>
                    </div>
                    
                    <div className="space-y-3">
                      <div className="space-y-1">
                        <h4 className="text-gray-300 text-sm font-medium">Target Muscles</h4>
                        <div className="flex flex-wrap gap-1">
                          <Badge variant="outline" className="text-xs">Quadriceps</Badge>
                          <Badge variant="outline" className="text-xs">Hamstrings</Badge>
                          <Badge variant="outline" className="text-xs">Glutes</Badge>
                          <Badge variant="outline" className="text-xs">Lower Back</Badge>
                          <Badge variant="outline" className="text-xs">Core</Badge>
                        </div>
                      </div>
                      
                      <div className="space-y-1">
                        <h4 className="text-gray-300 text-sm font-medium">Equipment Needed</h4>
                        <div className="flex flex-wrap gap-1">
                          <Badge variant="outline" className="text-xs">Barbell</Badge>
                          <Badge variant="outline" className="text-xs">Squat Rack</Badge>
                        </div>
                      </div>
                      
                      <div className="space-y-1">
                        <h4 className="text-gray-300 text-sm font-medium">Difficulty</h4>
                        <div className="flex items-center">
                          <div className="w-full bg-gray-700 h-2 rounded-full">
                            <div className="bg-yellow-500 h-2 rounded-full" style={{ width: '70%' }}></div>
                          </div>
                          <span className="ml-2 text-xs text-gray-400">Intermediate</span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                {/* Exercise Steps */}
                <Card className="bg-gray-800 border-gray-700">
                  <CardContent className="p-4">
                    <h3 className="text-white font-medium mb-3">Key Steps</h3>
                    
                    <div className="space-y-3 max-h-[300px] overflow-y-auto pr-2">
                      {demoSteps.map((step, index) => (
                        <div 
                          key={index} 
                          className={`p-3 rounded-lg cursor-pointer transition-colors ${
                            currentStep === index 
                              ? 'bg-blue-900/30 border border-blue-700'
                              : 'bg-gray-750 hover:bg-gray-700'
                          }`}
                          onClick={() => skipToStep(index)}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <h4 className="text-white font-medium">{step.title}</h4>
                            <Badge variant="outline" className="text-xs">
                              {index + 1}/{demoSteps.length}
                            </Badge>
                          </div>
                          <p className="text-gray-400 text-sm">{step.description}</p>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
                
                {/* Action button */}
                <Button onClick={() => onComplete && onComplete()} className="w-full">
                  <Check className="mr-2 h-4 w-4" />
                  Complete Demo
                </Button>
              </div>
            )}
          </div>
        );
    }
  };
  
  return (
    <div className="ar-exercise-demo bg-gray-900 rounded-xl p-4 md:p-6">
      {renderContent()}
    </div>
  );
} 
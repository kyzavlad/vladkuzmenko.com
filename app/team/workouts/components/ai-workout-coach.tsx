'use client';

import { useState, useRef, useEffect } from 'react';
import { Camera, Video, Mic, Volume2, VolumeX, RefreshCw, Check, X, AlertTriangle, Zap, Award, ThumbsUp } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

type CoachStatus = 'inactive' | 'initializing' | 'analyzing' | 'feedback' | 'resting' | 'completed';
type FormQuality = 'perfect' | 'good' | 'needs-improvement' | 'poor';
type ExercisePhase = 'eccentric' | 'isometric' | 'concentric' | 'rest';

interface AIWorkoutCoachProps {
  exerciseName?: string;
  targetReps?: number;
  targetSets?: number;
  restTime?: number; // in seconds
  onComplete?: () => void;
  onFormUpdate?: (quality: FormQuality) => void;
}

export default function AIWorkoutCoach({
  exerciseName = 'Squat',
  targetReps = 12,
  targetSets = 3,
  restTime = 60,
  onComplete,
  onFormUpdate
}: AIWorkoutCoachProps) {
  const [status, setStatus] = useState<CoachStatus>('inactive');
  const [cameraPermission, setCameraPermission] = useState<boolean | null>(null);
  const [micPermission, setMicPermission] = useState<boolean | null>(null);
  const [isMuted, setIsMuted] = useState(false);
  const [currentReps, setCurrentReps] = useState(0);
  const [currentSet, setCurrentSet] = useState(1);
  const [restTimeRemaining, setRestTimeRemaining] = useState(restTime);
  const [formQuality, setFormQuality] = useState<FormQuality>('good');
  const [currentPhase, setCurrentPhase] = useState<ExercisePhase>('rest');
  const [feedbackMessage, setFeedbackMessage] = useState<string>('');
  const [confidenceScore, setConfidenceScore] = useState(95);
  const [cameraMode, setCameraMode] = useState<'front' | 'back'>('front');
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  
  const restTimerRef = useRef<NodeJS.Timeout | null>(null);
  
  // Mock data for keypoints - in a real implementation, this would come from a pose estimation model
  const [keypoints, setKeypoints] = useState<Array<{x: number, y: number, confidence: number}>>([]);
  
  // Form feedback history to display in the log
  const [feedbackHistory, setFeedbackHistory] = useState<Array<{
    timestamp: Date,
    message: string,
    type: 'info' | 'warning' | 'success' | 'error'
  }>>([]);
  
  // Initialize camera when component mounts
  useEffect(() => {
    return () => {
      // Clean up stream when component unmounts
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      
      if (restTimerRef.current) {
        clearInterval(restTimerRef.current);
      }
    };
  }, []);
  
  // Add mock feedback for demonstration purposes
  useEffect(() => {
    if (status === 'analyzing' && currentReps > 0 && currentReps % 3 === 0) {
      // Generate random feedback every 3 reps
      const feedbacks = [
        { message: 'Keep your back straight', type: 'warning' as const },
        { message: 'Excellent depth on that squat!', type: 'success' as const },
        { message: 'Drive through your heels', type: 'info' as const },
        { message: 'Keep your knees in line with your toes', type: 'warning' as const },
        { message: 'Great form on that rep!', type: 'success' as const }
      ];
      
      const randomFeedback = feedbacks[Math.floor(Math.random() * feedbacks.length)];
      setFeedbackMessage(randomFeedback.message);
      
      setFeedbackHistory(prev => [
        { timestamp: new Date(), ...randomFeedback },
        ...prev.slice(0, 9) // Keep the last 10 feedback messages
      ]);
      
      // If warning feedback, adjust form quality
      if (randomFeedback.type === 'warning') {
        setFormQuality('needs-improvement');
        if (onFormUpdate) onFormUpdate('needs-improvement');
      } else if (randomFeedback.type === 'success') {
        setFormQuality('good');
        if (onFormUpdate) onFormUpdate('good');
      }
    }
  }, [currentReps, status, onFormUpdate]);
  
  // Rest timer countdown
  useEffect(() => {
    if (status === 'resting' && restTimeRemaining > 0) {
      restTimerRef.current = setInterval(() => {
        setRestTimeRemaining(prev => {
          if (prev <= 1) {
            clearInterval(restTimerRef.current!);
            setStatus('analyzing');
            return restTime;
          }
          return prev - 1;
        });
      }, 1000);
    }
    
    return () => {
      if (restTimerRef.current) {
        clearInterval(restTimerRef.current);
      }
    };
  }, [status, restTimeRemaining, restTime]);
  
  // Simulate rep counting
  useEffect(() => {
    let repInterval: NodeJS.Timeout | null = null;
    
    if (status === 'analyzing') {
      // Simulate rep counting with random intervals
      repInterval = setInterval(() => {
        setCurrentPhase(prev => {
          // Cycle through phases: rest -> eccentric -> isometric -> concentric -> rest
          if (prev === 'rest') return 'eccentric';
          if (prev === 'eccentric') return 'isometric';
          if (prev === 'isometric') return 'concentric';
          
          // When completing a rep, increment the count
          setCurrentReps(prevReps => {
            const newCount = prevReps + 1;
            
            // If we've reached target reps, move to the next set or complete
            if (newCount >= targetReps) {
              if (currentSet < targetSets) {
                setCurrentSet(prev => prev + 1);
                setCurrentReps(0);
                setStatus('resting');
              } else {
                setStatus('completed');
                if (onComplete) onComplete();
              }
            }
            
            return newCount;
          });
          
          return 'rest';
        });
      }, Math.random() * 1000 + 1000); // Random interval between 1-2 seconds
    }
    
    return () => {
      if (repInterval) clearInterval(repInterval);
    };
  }, [status, currentSet, targetReps, targetSets, onComplete]);
  
  // Initialize the camera
  const startCamera = async () => {
    try {
      setStatus('initializing');
      
      // Request camera permission
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: cameraMode === 'front' ? 'user' : 'environment' },
        audio: !isMuted
      });
      
      // Set video source
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      
      streamRef.current = stream;
      setCameraPermission(true);
      setMicPermission(!isMuted);
      
      // Simulate keypoints generation for visualization
      generateMockKeypoints();
      
      // After initialization, move to analyzing state
      setTimeout(() => {
        setStatus('analyzing');
        setFeedbackHistory(prev => [
          { 
            timestamp: new Date(), 
            message: `Started analyzing ${exerciseName}`, 
            type: 'info'
          },
          ...prev
        ]);
      }, 2000);
      
    } catch (error) {
      console.error('Error accessing camera:', error);
      setCameraPermission(false);
      setStatus('inactive');
      
      setFeedbackHistory(prev => [
        { 
          timestamp: new Date(), 
          message: 'Camera access denied. Please enable camera permissions.', 
          type: 'error'
        },
        ...prev
      ]);
    }
  };
  
  // Toggle mute status
  const toggleMute = () => {
    setIsMuted(!isMuted);
    
    if (streamRef.current) {
      streamRef.current.getAudioTracks().forEach(track => {
        track.enabled = isMuted; // If currently muted, enable audio
      });
    }
  };
  
  // Toggle camera mode (front/back)
  const toggleCameraMode = async () => {
    const newMode = cameraMode === 'front' ? 'back' : 'front';
    setCameraMode(newMode);
    
    // If stream is active, restart with new camera
    if (streamRef.current) {
      // Stop current stream
      streamRef.current.getTracks().forEach(track => track.stop());
      
      try {
        // Start new stream with different camera
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { facingMode: newMode === 'front' ? 'user' : 'environment' },
          audio: !isMuted
        });
        
        // Set video source
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        
        streamRef.current = stream;
      } catch (error) {
        console.error('Error switching camera:', error);
      }
    }
  };
  
  // Reset workout
  const resetWorkout = () => {
    setCurrentReps(0);
    setCurrentSet(1);
    setRestTimeRemaining(restTime);
    setStatus('analyzing');
    
    setFeedbackHistory(prev => [
      { 
        timestamp: new Date(), 
        message: 'Workout reset', 
        type: 'info'
      },
      ...prev
    ]);
  };
  
  // Generate mock keypoints for visualization
  const generateMockKeypoints = () => {
    // In a real implementation, these would come from a pose estimation model
    const mockPoints = [];
    const baseX = 150;
    const baseY = 150;
    
    // Head
    mockPoints.push({ x: baseX, y: baseY - 50, confidence: 0.9 });
    
    // Shoulders
    mockPoints.push({ x: baseX - 30, y: baseY - 20, confidence: 0.85 });
    mockPoints.push({ x: baseX + 30, y: baseY - 20, confidence: 0.85 });
    
    // Elbows
    mockPoints.push({ x: baseX - 45, y: baseY + 10, confidence: 0.8 });
    mockPoints.push({ x: baseX + 45, y: baseY + 10, confidence: 0.8 });
    
    // Wrists
    mockPoints.push({ x: baseX - 50, y: baseY + 40, confidence: 0.75 });
    mockPoints.push({ x: baseX + 50, y: baseY + 40, confidence: 0.75 });
    
    // Hips
    mockPoints.push({ x: baseX - 20, y: baseY + 30, confidence: 0.9 });
    mockPoints.push({ x: baseX + 20, y: baseY + 30, confidence: 0.9 });
    
    // Knees
    mockPoints.push({ x: baseX - 25, y: baseY + 80, confidence: 0.85 });
    mockPoints.push({ x: baseX + 25, y: baseY + 80, confidence: 0.85 });
    
    // Ankles
    mockPoints.push({ x: baseX - 30, y: baseY + 130, confidence: 0.8 });
    mockPoints.push({ x: baseX + 30, y: baseY + 130, confidence: 0.8 });
    
    setKeypoints(mockPoints);
    
    // Animate keypoints for demonstration
    if (status === 'analyzing') {
      setTimeout(generateMockKeypoints, 500);
    }
  };
  
  // Draw keypoints on canvas
  useEffect(() => {
    if (canvasRef.current && keypoints.length > 0) {
      const ctx = canvasRef.current.getContext('2d');
      if (!ctx) return;
      
      // Clear canvas
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      // Draw keypoints
      keypoints.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
        
        // Color based on confidence
        const alpha = point.confidence;
        ctx.fillStyle = `rgba(0, 255, 0, ${alpha})`;
        ctx.fill();
      });
      
      // Draw connections between keypoints to form a skeleton
      // For simplicity, we're just connecting a few points
      ctx.strokeStyle = 'rgba(0, 255, 0, 0.7)';
      ctx.lineWidth = 2;
      
      // Connect head to shoulders
      if (keypoints[0] && keypoints[1]) {
        ctx.beginPath();
        ctx.moveTo(keypoints[0].x, keypoints[0].y);
        ctx.lineTo(keypoints[1].x, keypoints[1].y);
        ctx.stroke();
      }
      
      if (keypoints[0] && keypoints[2]) {
        ctx.beginPath();
        ctx.moveTo(keypoints[0].x, keypoints[0].y);
        ctx.lineTo(keypoints[2].x, keypoints[2].y);
        ctx.stroke();
      }
      
      // Connect shoulders to elbows
      if (keypoints[1] && keypoints[3]) {
        ctx.beginPath();
        ctx.moveTo(keypoints[1].x, keypoints[1].y);
        ctx.lineTo(keypoints[3].x, keypoints[3].y);
        ctx.stroke();
      }
      
      if (keypoints[2] && keypoints[4]) {
        ctx.beginPath();
        ctx.moveTo(keypoints[2].x, keypoints[2].y);
        ctx.lineTo(keypoints[4].x, keypoints[4].y);
        ctx.stroke();
      }
      
      // And so on for other body parts...
    }
  }, [keypoints]);
  
  // Render different views based on status
  const renderContent = () => {
    switch (status) {
      case 'inactive':
        return (
          <div className="flex flex-col items-center justify-center h-full py-12">
            <Camera className="h-16 w-16 mb-4 text-gray-600" />
            <h3 className="text-white text-xl font-medium mb-2">AI Workout Coach</h3>
            <p className="text-gray-400 text-center mb-6 max-w-md">
              Get real-time form feedback, rep counting, and guidance through your workout.
              Position yourself in frame so the camera can see your full body.
            </p>
            <Button onClick={startCamera} size="lg">
              <Camera className="mr-2 h-5 w-5" />
              Start Workout Coach
            </Button>
          </div>
        );
        
      case 'initializing':
        return (
          <div className="flex flex-col items-center justify-center h-full py-12">
            <RefreshCw className="h-16 w-16 mb-4 text-blue-500 animate-spin" />
            <h3 className="text-white text-xl font-medium mb-2">Initializing AI Coach</h3>
            <p className="text-gray-400 text-center mb-6 max-w-md">
              Setting up cameras and loading AI models. This may take a few moments.
            </p>
            <Progress value={40} className="w-64 mb-4" />
            <p className="text-gray-500 text-sm">Calibrating pose estimation...</p>
          </div>
        );
        
      case 'analyzing':
      case 'feedback':
      case 'resting':
        return (
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
            <div className="lg:col-span-3 space-y-4">
              {/* Camera Feed and Overlay */}
              <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
                <video 
                  ref={videoRef} 
                  autoPlay 
                  playsInline 
                  muted={isMuted}
                  className="w-full h-full object-cover"
                />
                
                {/* Pose estimation overlay */}
                <canvas 
                  ref={canvasRef} 
                  width={300} 
                  height={300}
                  className="absolute inset-0 w-full h-full"
                />
                
                {/* Controls overlay */}
                <div className="absolute bottom-4 left-4 right-4 flex justify-between">
                  <div className="flex gap-2">
                    <Button size="icon" variant="secondary" onClick={toggleMute}>
                      {isMuted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
                    </Button>
                    
                    <Button size="icon" variant="secondary" onClick={toggleCameraMode}>
                      <RefreshCw className="h-4 w-4" />
                    </Button>
                  </div>
                  
                  <Badge variant="outline" className="bg-black/50 text-white">
                    {exerciseName} • Form Quality: {formQuality.replace('-', ' ')}
                  </Badge>
                </div>
                
                {/* Rep counter overlay */}
                <div className="absolute top-4 left-4 right-4 flex justify-between items-center">
                  <Badge className="bg-blue-900/70 text-white text-lg py-1 px-3">
                    {currentReps} / {targetReps}
                  </Badge>
                  
                  <Badge variant="outline" className="bg-black/50 text-white">
                    Set {currentSet} of {targetSets}
                  </Badge>
                </div>
                
                {/* Current phase indicator */}
                {status === 'analyzing' && (
                  <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                    <Badge className={`text-lg py-1 px-4 ${
                      currentPhase === 'eccentric' ? 'bg-blue-500' :
                      currentPhase === 'isometric' ? 'bg-yellow-500' :
                      currentPhase === 'concentric' ? 'bg-green-500' :
                      'bg-gray-500'
                    }`}>
                      {currentPhase.charAt(0).toUpperCase() + currentPhase.slice(1)}
                    </Badge>
                  </div>
                )}
                
                {/* Rest timer overlay */}
                {status === 'resting' && (
                  <div className="absolute inset-0 bg-black/70 flex flex-col items-center justify-center">
                    <h3 className="text-white text-2xl font-medium mb-2">Rest Period</h3>
                    <div className="text-white text-4xl font-bold mb-4">
                      {Math.floor(restTimeRemaining / 60)}:{(restTimeRemaining % 60).toString().padStart(2, '0')}
                    </div>
                    <Progress value={(restTimeRemaining / restTime) * 100} className="w-64 mb-4" />
                    <p className="text-gray-300 mb-4">Get ready for set {currentSet} of {targetSets}</p>
                    <Button onClick={() => setStatus('analyzing')}>
                      Skip Rest
                    </Button>
                  </div>
                )}
                
                {/* Feedback overlay */}
                {feedbackMessage && status === 'analyzing' && (
                  <div className="absolute left-4 right-4 bottom-16 bg-black/70 rounded-lg p-3 text-white text-center">
                    {feedbackMessage}
                  </div>
                )}
              </div>
              
              {/* Exercise Information */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <Card className="bg-gray-800 border-gray-700">
                  <CardContent className="p-4">
                    <div className="text-gray-400 text-xs mb-1">Target</div>
                    <div className="text-white text-xl font-medium">{targetReps} × {targetSets}</div>
                    <div className="text-gray-500 text-xs">reps × sets</div>
                  </CardContent>
                </Card>
                
                <Card className="bg-gray-800 border-gray-700">
                  <CardContent className="p-4">
                    <div className="text-gray-400 text-xs mb-1">Form Quality</div>
                    <div className={`text-xl font-medium ${
                      formQuality === 'perfect' ? 'text-green-500' :
                      formQuality === 'good' ? 'text-blue-500' :
                      formQuality === 'needs-improvement' ? 'text-yellow-500' :
                      'text-red-500'
                    }`}>
                      {formQuality.replace('-', ' ')}
                    </div>
                  </CardContent>
                </Card>
                
                <Card className="bg-gray-800 border-gray-700">
                  <CardContent className="p-4">
                    <div className="text-gray-400 text-xs mb-1">AI Confidence</div>
                    <div className="text-white text-xl font-medium">{confidenceScore}%</div>
                    <Progress value={confidenceScore} className="h-1 mt-1" />
                  </CardContent>
                </Card>
                
                <Card className="bg-gray-800 border-gray-700">
                  <CardContent className="p-4">
                    <div className="text-gray-400 text-xs mb-1">Workout Status</div>
                    <div className="text-white text-xl font-medium">
                      {status === 'analyzing' ? 'Active' : 
                       status === 'resting' ? 'Resting' : 
                       status === 'completed' ? 'Complete' : status}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
            
            <div className="lg:col-span-2 space-y-4">
              {/* Feedback Log */}
              <Card className="bg-gray-800 border-gray-700">
                <CardContent className="p-4">
                  <h3 className="text-white font-medium mb-3">Form Feedback Log</h3>
                  
                  <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
                    {feedbackHistory.length > 0 ? (
                      feedbackHistory.map((feedback, index) => (
                        <div key={index} className="flex items-start gap-2 text-sm">
                          <div className="mt-0.5">
                            {feedback.type === 'success' && <Check className="h-4 w-4 text-green-500" />}
                            {feedback.type === 'warning' && <AlertTriangle className="h-4 w-4 text-yellow-500" />}
                            {feedback.type === 'error' && <X className="h-4 w-4 text-red-500" />}
                            {feedback.type === 'info' && <Zap className="h-4 w-4 text-blue-500" />}
                          </div>
                          <div className="flex-1">
                            <div className={`font-medium ${
                              feedback.type === 'success' ? 'text-green-500' :
                              feedback.type === 'warning' ? 'text-yellow-500' :
                              feedback.type === 'error' ? 'text-red-500' :
                              'text-blue-500'
                            }`}>
                              {feedback.message}
                            </div>
                            <div className="text-gray-500 text-xs">
                              {feedback.timestamp.toLocaleTimeString()}
                            </div>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="text-gray-500 text-center py-4">
                        No feedback recorded yet
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
              
              {/* Form Tips */}
              <Card className="bg-gray-800 border-gray-700">
                <CardContent className="p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <ThumbsUp className="h-5 w-5 text-blue-500" />
                    <h3 className="text-white font-medium">Form Tips for {exerciseName}</h3>
                  </div>
                  
                  <ul className="space-y-2 text-sm text-gray-300">
                    <li className="flex items-start gap-2">
                      <Check className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                      <span>Keep your back straight throughout the movement</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Check className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                      <span>Drive through your heels when standing up</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Check className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                      <span>Keep your knees in line with your toes</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Check className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                      <span>Maintain proper depth - thighs parallel to the ground</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Check className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                      <span>Engage your core throughout the movement</span>
                    </li>
                  </ul>
                </CardContent>
              </Card>
              
              {/* Action buttons */}
              <div className="flex gap-3">
                <Button onClick={resetWorkout} variant="outline" className="flex-1">
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Reset
                </Button>
                
                <Button onClick={() => setStatus('completed')} className="flex-1">
                  <Check className="mr-2 h-4 w-4" />
                  Complete Workout
                </Button>
              </div>
            </div>
          </div>
        );
        
      case 'completed':
        return (
          <div className="flex flex-col items-center justify-center h-full py-12">
            <Award className="h-16 w-16 mb-4 text-yellow-500" />
            <h3 className="text-white text-xl font-medium mb-2">Workout Completed!</h3>
            <p className="text-gray-400 text-center mb-6 max-w-md">
              Great job! You've completed {targetReps * targetSets} reps across {targetSets} sets.
              Your form quality was mostly {formQuality.replace('-', ' ')}.
            </p>
            
            <div className="grid grid-cols-3 gap-4 mb-6 w-full max-w-md">
              <Card className="bg-gray-800 border-gray-700">
                <CardContent className="p-4 text-center">
                  <div className="text-gray-400 text-xs mb-1">Total Reps</div>
                  <div className="text-white text-xl font-medium">{targetReps * targetSets}</div>
                </CardContent>
              </Card>
              
              <Card className="bg-gray-800 border-gray-700">
                <CardContent className="p-4 text-center">
                  <div className="text-gray-400 text-xs mb-1">Sets</div>
                  <div className="text-white text-xl font-medium">{targetSets}</div>
                </CardContent>
              </Card>
              
              <Card className="bg-gray-800 border-gray-700">
                <CardContent className="p-4 text-center">
                  <div className="text-gray-400 text-xs mb-1">Form Rating</div>
                  <div className="text-blue-500 text-xl font-medium">
                    {formQuality === 'perfect' ? 'A+' :
                     formQuality === 'good' ? 'A' :
                     formQuality === 'needs-improvement' ? 'B' :
                     'C'}
                  </div>
                </CardContent>
              </Card>
            </div>
            
            <div className="flex gap-3">
              <Button onClick={resetWorkout} variant="outline">
                <RefreshCw className="mr-2 h-4 w-4" />
                Do Another Set
              </Button>
              
              <Button onClick={() => {
                if (onComplete) onComplete();
              }}>
                <Check className="mr-2 h-4 w-4" />
                Finish & Save
              </Button>
            </div>
          </div>
        );
    }
  };
  
  return (
    <div className="ai-workout-coach bg-gray-900 rounded-xl p-4 md:p-6">
      {renderContent()}
    </div>
  );
} 
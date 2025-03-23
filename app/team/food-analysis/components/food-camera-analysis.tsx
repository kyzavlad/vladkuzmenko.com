'use client';

import { useState, useRef, useEffect } from 'react';
import { Camera, RefreshCw, X, Crosshair } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface FoodCameraAnalysisProps {
  onCapture: (imageData: string) => void;
  isAnalyzing: boolean;
}

export function FoodCameraAnalysis({ onCapture, isAnalyzing }: FoodCameraAnalysisProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isCameraSupported, setIsCameraSupported] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('environment');
  const [countdown, setCountdown] = useState<number | null>(null);
  
  // Setup and start the camera feed
  const initializeCamera = async () => {
    try {
      setErrorMessage(null);
      
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera access is not supported in your browser');
      }
      
      const constraints = {
        video: {
          facingMode: facingMode,
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      };
      
      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setIsCameraActive(true);
        };
      }
    } catch (error) {
      console.error('Camera error:', error);
      setIsCameraActive(false);
      setIsCameraSupported(false);
      
      if (error instanceof Error) {
        if (error.name === 'NotAllowedError') {
          setErrorMessage('Camera access denied. Please allow camera access to use this feature.');
        } else if (error.name === 'NotFoundError') {
          setErrorMessage('No camera found on your device.');
        } else {
          setErrorMessage(`Error accessing camera: ${error.message}`);
        }
      } else {
        setErrorMessage('An unknown error occurred when accessing the camera.');
      }
    }
  };
  
  // Stop the camera feed
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsCameraActive(false);
    }
  };
  
  // Switch between front and back cameras
  const toggleFacingMode = () => {
    stopCamera();
    setFacingMode(prev => prev === 'user' ? 'environment' : 'user');
  };
  
  // Start the camera on component mount
  useEffect(() => {
    if (!isAnalyzing) {
      initializeCamera();
    }
    
    // Cleanup function to stop camera when component unmounts
    return () => {
      stopCamera();
    };
  }, [facingMode, isAnalyzing]);
  
  // Capture current frame from video
  const captureFrame = () => {
    if (videoRef.current && canvasRef.current && isCameraActive) {
      const context = canvasRef.current.getContext('2d');
      if (context) {
        // Set canvas dimensions to match video
        canvasRef.current.width = videoRef.current.videoWidth;
        canvasRef.current.height = videoRef.current.videoHeight;
        
        // Draw the current video frame to canvas
        context.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
        
        // Convert to data URL and pass to parent component
        const imageDataURL = canvasRef.current.toDataURL('image/jpeg', 0.8);
        onCapture(imageDataURL);
      }
    }
  };
  
  // Start countdown for capture
  const startCountdown = () => {
    setCountdown(3);
    
    const timer = setInterval(() => {
      setCountdown(prev => {
        if (prev === null || prev <= 1) {
          clearInterval(timer);
          if (prev === 1) {
            captureFrame();
          }
          return null;
        }
        return prev - 1;
      });
    }, 1000);
  };
  
  // Render loading state while analyzing
  if (isAnalyzing) {
    return (
      <div className="flex flex-col items-center justify-center p-8 bg-gray-800 rounded-lg">
        <div className="w-16 h-16 mb-4 relative">
          <RefreshCw className="w-16 h-16 text-blue-400 animate-spin" />
        </div>
        <h3 className="text-lg font-medium text-white mb-2">Analyzing your food...</h3>
        <p className="text-gray-400 text-center">
          Our AI is identifying food items and calculating nutritional information
        </p>
      </div>
    );
  }
  
  // Render camera access error
  if (!isCameraSupported || errorMessage) {
    return (
      <div className="flex flex-col items-center justify-center p-8 bg-gray-800 rounded-lg text-center">
        <div className="w-16 h-16 mb-4 flex items-center justify-center">
          <X className="w-12 h-12 text-red-500" />
        </div>
        <h3 className="text-lg font-medium text-white mb-2">Camera Access Error</h3>
        <p className="text-gray-400 mb-4">
          {errorMessage || "Your device doesn't support camera access or permission was denied."}
        </p>
        <Button 
          onClick={initializeCamera} 
          className="bg-blue-600 mb-2"
        >
          Try Again
        </Button>
        <p className="text-xs text-gray-500 mt-2">
          Alternatively, you can use the Upload feature to analyze food photos.
        </p>
      </div>
    );
  }
  
  return (
    <div className="food-camera-analysis">
      {/* Camera Feed Container */}
      <div className="relative overflow-hidden rounded-lg bg-black">
        {/* Camera overlay when countdown is active */}
        {countdown !== null && (
          <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-10">
            <div className="text-6xl font-bold text-white animate-pulse">
              {countdown}
            </div>
          </div>
        )}
        
        {/* Guide overlay for helping users position food */}
        <div className="absolute inset-0 pointer-events-none flex items-center justify-center z-0">
          <div className="w-5/6 h-5/6 border-2 border-dashed border-blue-400 border-opacity-70 rounded-lg flex items-center justify-center">
            <Crosshair className="text-blue-400 opacity-50 h-12 w-12" />
          </div>
        </div>
        
        {/* Video element for camera feed */}
        <video 
          ref={videoRef}
          autoPlay 
          playsInline 
          muted 
          className="w-full h-auto max-h-[500px] object-cover"
        />
        
        {/* Hidden canvas used for capture */}
        <canvas 
          ref={canvasRef} 
          className="hidden"
        />
        
        {/* Camera controls */}
        <div className="absolute bottom-4 left-0 right-0 flex justify-center space-x-4">
          <Button
            variant="secondary"
            size="icon"
            className="rounded-full w-10 h-10 bg-gray-900 bg-opacity-70 border border-gray-700"
            onClick={toggleFacingMode}
          >
            <RefreshCw className="h-5 w-5" />
          </Button>
          
          <Button
            variant="secondary"
            size="icon"
            className="rounded-full w-14 h-14 bg-white"
            onClick={startCountdown}
            disabled={!isCameraActive || countdown !== null}
          >
            <Camera className="h-7 w-7 text-blue-600" />
          </Button>
          
          <div className="w-10 h-10">
            {/* Spacer for layout balance */}
          </div>
        </div>
      </div>
      
      {/* Instructions */}
      <div className="mt-4 bg-gray-800 bg-opacity-80 p-3 rounded-lg text-center">
        <p className="text-sm text-gray-300">
          Position your plate in the frame and tap the camera button to analyze
        </p>
      </div>
    </div>
  );
} 
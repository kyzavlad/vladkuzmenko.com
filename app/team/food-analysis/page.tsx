'use client';

import { useState, useRef } from 'react';
import Link from 'next/link';
import { 
  Camera, 
  Upload, 
  X, 
  ArrowLeft, 
  Info, 
  Image as ImageIcon,
  Loader2,
  CheckCircle2,
  RefreshCw
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';

// Import our integration component
import FoodAnalysisIntegration from '../nutrition/components/food-analysis-integration';

// Camera states
type CameraState = 'idle' | 'permission-prompt' | 'ready' | 'capture' | 'processing' | 'result' | 'error';

export default function FoodAnalysisPage() {
  const [activeTab, setActiveTab] = useState('camera');
  const [cameraState, setCameraState] = useState<CameraState>('idle');
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [flashMessage, setFlashMessage] = useState<{type: 'info' | 'error' | 'success', message: string} | null>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const photoRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Start camera
  const startCamera = async () => {
    try {
      setCameraState('permission-prompt');
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraState('ready');
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      setCameraState('error');
      setFlashMessage({
        type: 'error', 
        message: 'Camera access was denied. Please grant permission to use this feature.'
      });
    }
  };
  
  // Capture photo
  const capturePhoto = () => {
    if (videoRef.current && photoRef.current) {
      setCameraState('capture');
      
      const video = videoRef.current;
      const canvas = photoRef.current;
      const context = canvas.getContext('2d');
      
      // Set canvas size to match video dimensions
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Draw video frame on canvas
      context?.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Get data URL of the canvas
      const photoSrc = canvas.toDataURL('image/png');
      setImageSrc(photoSrc);
      
      // Move to processing state
      setCameraState('processing');
      
      // Simulate processing (in a real app, this would be an API call to analyze the image)
      setTimeout(() => {
        // Stop the camera stream
        const stream = video.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        
        setCameraState('result');
      }, 2000);
    }
  };
  
  // Handle file upload
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setCameraState('processing');
      
      const reader = new FileReader();
      reader.onload = (e) => {
        setImageSrc(e.target?.result as string);
        
        // Simulate processing (in a real app, this would be an API call to analyze the image)
        setTimeout(() => {
          setCameraState('result');
        }, 2000);
      };
      
      reader.readAsDataURL(file);
    }
  };
  
  // Reset the capture process
  const resetCapture = () => {
    setImageSrc(null);
    setCameraState('idle');
    setFlashMessage(null);
  };
  
  // Handle successful analysis
  const handleSuccessfulAnalysis = () => {
    // In a real app, this would navigate to the nutrition log with analyzed data
    console.log('Food analysis completed successfully');
  };
  
  // Retry with another photo
  const retryWithNewPhoto = () => {
    resetCapture();
  };
  
  // Handle tab change
  const handleTabChange = (value: string) => {
    setActiveTab(value);
    resetCapture();
  };
  
  return (
    <div className="food-analysis-page p-4 max-w-3xl mx-auto">
      <div className="flex items-center mb-6">
        <Button variant="ghost" size="sm" className="mr-2" asChild>
          <Link href="/team/nutrition">
            <ArrowLeft className="h-5 w-5" />
          </Link>
        </Button>
        <div>
          <h1 className="text-3xl font-bold text-white">Food Analysis</h1>
          <p className="text-gray-400">Analyze your food for nutritional content</p>
        </div>
      </div>
      
      {/* Flash Messages */}
      {flashMessage && (
        <Alert 
          className={`mb-6 ${
            flashMessage.type === 'error' ? 'bg-red-950/20 border-red-800' :
            flashMessage.type === 'success' ? 'bg-green-950/20 border-green-800' :
            'bg-blue-950/20 border-blue-800'
          }`}
        >
          {flashMessage.type === 'error' && <X className="h-4 w-4 text-red-400 mr-2" />}
          {flashMessage.type === 'success' && <CheckCircle2 className="h-4 w-4 text-green-400 mr-2" />}
          {flashMessage.type === 'info' && <Info className="h-4 w-4 text-blue-400 mr-2" />}
          <AlertDescription className="text-gray-300">{flashMessage.message}</AlertDescription>
        </Alert>
      )}
      
      {/* Instructions Card */}
      {cameraState === 'idle' && (
        <Card className="bg-gray-800 border-gray-700 mb-6">
          <CardHeader className="pb-2">
            <CardTitle className="text-white">How to Use Food Analysis</CardTitle>
            <CardDescription className="text-gray-400">
              Take a photo of your meal to get nutritional information
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-start">
                <div className="w-8 h-8 rounded-full bg-blue-900 flex items-center justify-center mr-3 flex-shrink-0">
                  <span className="text-white font-medium">1</span>
                </div>
                <p className="text-gray-300">Take a clear photo of your food or upload an existing image</p>
              </div>
              
              <div className="flex items-start">
                <div className="w-8 h-8 rounded-full bg-blue-900 flex items-center justify-center mr-3 flex-shrink-0">
                  <span className="text-white font-medium">2</span>
                </div>
                <p className="text-gray-300">Our AI will analyze the image and identify food items</p>
              </div>
              
              <div className="flex items-start">
                <div className="w-8 h-8 rounded-full bg-blue-900 flex items-center justify-center mr-3 flex-shrink-0">
                  <span className="text-white font-medium">3</span>
                </div>
                <p className="text-gray-300">Verify the identified items and add them to your nutrition log</p>
              </div>
              
              <div className="bg-blue-900/20 border border-blue-800 p-3 rounded-md mt-4 text-sm flex items-start">
                <Info className="h-5 w-5 mr-2 flex-shrink-0 mt-0.5 text-blue-400" />
                <div>
                  <p className="text-blue-300 font-medium">For best results:</p>
                  <ul className="list-disc pl-5 mt-1 text-gray-300 space-y-1">
                    <li>Make sure your food is well-lit and clearly visible</li>
                    <li>Include all items on your plate in the photo</li>
                    <li>Take the photo from above for the most accurate analysis</li>
                  </ul>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
      
      {/* Tabs for Camera and Upload */}
      {cameraState === 'idle' && (
        <Tabs value={activeTab} onValueChange={handleTabChange} className="mb-6">
          <TabsList className="grid grid-cols-2 mb-6">
            <TabsTrigger value="camera">Take Photo</TabsTrigger>
            <TabsTrigger value="upload">Upload Image</TabsTrigger>
          </TabsList>
          
          <TabsContent value="camera">
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader className="pb-2">
                <CardTitle className="text-white">Camera Capture</CardTitle>
                <CardDescription className="text-gray-400">
                  Take a photo of your food with your device's camera
                </CardDescription>
              </CardHeader>
              <CardContent className="flex justify-center">
                <div className="w-full max-w-md aspect-[4/3] bg-gray-900 rounded-md flex items-center justify-center">
                  <Button 
                    size="lg" 
                    className="flex flex-col gap-2 p-6"
                    onClick={startCamera}
                  >
                    <Camera className="h-10 w-10" />
                    <span>Start Camera</span>
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="upload">
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader className="pb-2">
                <CardTitle className="text-white">Upload Image</CardTitle>
                <CardDescription className="text-gray-400">
                  Upload an existing photo of your food
                </CardDescription>
              </CardHeader>
              <CardContent className="flex justify-center">
                <div 
                  className="w-full max-w-md aspect-[4/3] bg-gray-900 rounded-md border-2 border-dashed border-gray-700 flex flex-col items-center justify-center p-6 cursor-pointer"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    className="hidden" 
                    accept="image/*"
                    onChange={handleFileUpload}
                  />
                  <Upload className="h-10 w-10 text-gray-500 mb-3" />
                  <h3 className="text-white font-medium text-center mb-1">Click to upload</h3>
                  <p className="text-gray-400 text-sm text-center">
                    Supports JPG, PNG, HEIC
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}
      
      {/* Camera View */}
      {(cameraState === 'permission-prompt' || cameraState === 'ready' || cameraState === 'capture') && (
        <div className="camera-view mb-6">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="pb-2">
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle className="text-white">Camera View</CardTitle>
                  <CardDescription className="text-gray-400">
                    Position your food in the frame
                  </CardDescription>
                </div>
                <Button variant="outline" size="sm" onClick={resetCapture}>
                  <X className="h-4 w-4 mr-2" />
                  Cancel
                </Button>
              </div>
            </CardHeader>
            <CardContent className="p-0 overflow-hidden rounded-b-md">
              <div className="relative w-full bg-black">
                <video 
                  ref={videoRef} 
                  autoPlay 
                  playsInline
                  className="w-full"
                />
                <canvas ref={photoRef} className="hidden" />
                
                {cameraState === 'permission-prompt' && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/70">
                    <div className="text-center p-6">
                      <Loader2 className="h-8 w-8 text-blue-400 animate-spin mx-auto mb-4" />
                      <h3 className="text-white font-medium mb-2">Camera Permissions</h3>
                      <p className="text-gray-300 mb-4">Please allow access to your camera to continue</p>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
            <CardFooter className="p-4 flex justify-center">
              {cameraState === 'ready' && (
                <Button size="lg" className="rounded-full h-16 w-16 p-0" onClick={capturePhoto}>
                  <span className="sr-only">Take Photo</span>
                  <div className="rounded-full h-12 w-12 border-2 border-white"></div>
                </Button>
              )}
            </CardFooter>
          </Card>
          
          <div className="text-center mt-4">
            <p className="text-gray-400 text-sm">
              For best results, ensure good lighting and position your food clearly in the frame
            </p>
          </div>
        </div>
      )}
      
      {/* Processing View */}
      {cameraState === 'processing' && (
        <div className="processing-view mb-6">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-white">Analyzing Your Food</CardTitle>
              <CardDescription className="text-gray-400">
                Our AI is identifying items and calculating nutrition
              </CardDescription>
            </CardHeader>
            <CardContent className="p-0 overflow-hidden">
              <div className="relative">
                {imageSrc && <img src={imageSrc} alt="Captured food" className="w-full" />}
                <div className="absolute inset-0 bg-black/50 flex flex-col items-center justify-center">
                  <Loader2 className="h-12 w-12 text-blue-400 animate-spin mb-4" />
                  <h3 className="text-white font-medium text-lg">Analyzing Image</h3>
                  <p className="text-gray-300 mt-2">Please wait a moment...</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
      
      {/* Results View */}
      {cameraState === 'result' && imageSrc && (
        <div className="results-view">
          <Card className="bg-gray-800 border-gray-700 mb-6">
            <CardHeader className="pb-2">
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle className="text-white">Analysis Complete</CardTitle>
                  <CardDescription className="text-gray-400">
                    We've identified items in your meal
                  </CardDescription>
                </div>
                <Badge className="bg-green-600">High Confidence</Badge>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <img src={imageSrc} alt="Analyzed food" className="w-full rounded-md" />
            </CardContent>
          </Card>
          
          {/* Integration Component */}
          <FoodAnalysisIntegration 
            onAddToLog={(items, mealType, notes) => {
              console.log('Added to log:', { items, mealType, notes });
              setFlashMessage({
                type: 'success',
                message: 'Food items added to your nutrition log successfully!'
              });
              handleSuccessfulAnalysis();
            }}
            onCreateRecipe={(items) => {
              console.log('Create recipe with:', items);
              // In a real app, this would navigate to the recipe creation page
            }}
          />
          
          {/* Action Buttons */}
          <div className="flex justify-center mt-6 space-x-4">
            <Button variant="outline" onClick={retryWithNewPhoto}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Try Another Photo
            </Button>
            
            <Button asChild>
              <Link href="/team/nutrition">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Dashboard
              </Link>
            </Button>
          </div>
        </div>
      )}
      
      {/* Error State */}
      {cameraState === 'error' && (
        <Card className="bg-gray-800 border-gray-700 mb-6">
          <CardContent className="p-6 text-center">
            <X className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <h3 className="text-white text-lg font-medium mb-2">Camera Access Error</h3>
            <p className="text-gray-400 mb-4">
              We couldn't access your camera. Please check your browser permissions and try again.
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <Button variant="outline" onClick={resetCapture}>
                Try Again
              </Button>
              <Button variant="outline" onClick={() => setActiveTab('upload')}>
                <Upload className="h-4 w-4 mr-2" />
                Upload Instead
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
} 
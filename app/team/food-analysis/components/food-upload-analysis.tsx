'use client';

import { useState, useRef } from 'react';
import { Upload, Image, Trash2, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface FoodUploadAnalysisProps {
  isAnalyzing: boolean;
  onAnalyze: () => void;
}

export function FoodUploadAnalysis({ isAnalyzing, onAnalyze }: FoodUploadAnalysisProps) {
  const [image, setImage] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Handle file selection
  const handleFile = (file: File) => {
    setUploadError(null);
    
    // Check if file is an image
    if (!file.type.startsWith('image/')) {
      setUploadError('Please upload an image file (JPEG, PNG, etc.)');
      return;
    }
    
    // Check file size (limit to 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setUploadError('Image is too large. Please upload an image smaller than 10MB.');
      return;
    }
    
    // Create a URL for the file
    const url = URL.createObjectURL(file);
    setImage(url);
  };
  
  // Handle file selection via input
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };
  
  // Handle drag events
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };
  
  // Handle drop event
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };
  
  // Trigger file input click
  const triggerFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };
  
  // Clear selected image
  const clearImage = () => {
    if (image) {
      URL.revokeObjectURL(image);
    }
    setImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };
  
  // Handle analyze button click
  const handleAnalyze = () => {
    if (!image) return;
    onAnalyze();
  };

  return (
    <div className="food-upload-analysis">
      {/* File upload area */}
      <div 
        className={`relative border-2 border-dashed rounded-lg transition-colors
          ${dragActive ? 'border-blue-400 bg-blue-900/10' : 'border-gray-600 bg-gray-700/50'} 
          ${image ? 'border-opacity-0' : 'p-8'}`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
      >
        {/* Hidden file input */}
        <input 
          ref={fileInputRef}
          type="file" 
          accept="image/*"
          className="hidden"
          onChange={handleInputChange}
        />
        
        {/* Upload placeholder */}
        {!image && (
          <div className="flex flex-col items-center text-center">
            <Upload className="h-12 w-12 text-gray-400 mb-4" />
            <h3 className="text-gray-200 font-medium mb-2">Upload Food Image</h3>
            <p className="text-gray-400 text-sm mb-4">
              Drag and drop an image here, or click to select
            </p>
            <Button variant="outline" onClick={triggerFileInput}>
              <Image className="h-4 w-4 mr-2" /> Select Image
            </Button>
          </div>
        )}
        
        {/* Display uploaded image */}
        {image && (
          <div className="relative">
            <img 
              src={image} 
              alt="Food to analyze" 
              className="w-full rounded-lg"
            />
            
            {/* Image actions overlay */}
            <div className="absolute top-2 right-2 flex space-x-2">
              <Button 
                variant="outline" 
                size="icon" 
                className="bg-gray-900/70 border-gray-800 text-gray-200"
                onClick={clearImage}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}
        
        {/* Error message */}
        {uploadError && (
          <div className="mt-4 text-red-400 text-sm text-center">
            {uploadError}
          </div>
        )}
      </div>
      
      {/* Analysis button */}
      <Button 
        className="w-full mt-4 bg-blue-600 hover:bg-blue-700 text-white"
        disabled={!image || isAnalyzing}
        onClick={handleAnalyze}
      >
        {isAnalyzing ? (
          <span className="flex items-center">
            <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Analyzing...
          </span>
        ) : (
          <>
            <Zap className="mr-2 h-4 w-4" />
            Analyze Food (1 token)
          </>
        )}
      </Button>
      
      {/* Upload guidelines */}
      <div className="mt-4 text-xs text-gray-400">
        <h4 className="font-medium mb-1">For best results:</h4>
        <ul className="list-disc list-inside space-y-1">
          <li>Use clear, well-lit photos</li>
          <li>Include all food items you want to analyze</li>
          <li>Take photos from above for better portion estimation</li>
        </ul>
      </div>
    </div>
  );
} 
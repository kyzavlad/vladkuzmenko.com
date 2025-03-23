'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FiRefreshCw, FiInfo, FiSliders, FiEye, FiMoon, FiSun } from 'react-icons/fi';
import { useCreationContext } from '../contexts/creation-context';

interface AppearanceFeature {
  id: string;
  name: string;
  description: string;
  type: 'slider' | 'color' | 'select';
  options?: {
    id: string;
    name: string;
    thumbnailUrl?: string;
  }[];
  min?: number;
  max?: number;
  step?: number;
  value: number | string;
}

export default function AppearanceControls() {
  const { 
    settings, 
    updateSettings,
    isProcessing,
    generateAppearancePreview,
    previewUrl
  } = useCreationContext();
  
  const [isGeneratingPreview, setIsGeneratingPreview] = useState(false);
  const [selectedAngle, setSelectedAngle] = useState<string>('front');
  const previewAngles = {
    front: previewUrl || undefined,
    side: previewUrl || undefined,
    '45-degree': previewUrl || undefined
  };
  const [features, setFeatures] = useState<AppearanceFeature[]>([
    {
      id: 'skinTone',
      name: 'Skin Tone',
      description: 'Adjust the skin tone of your avatar',
      type: 'color',
      options: [
        { id: 'very-light', name: 'Very Light', thumbnailUrl: '/images/skin-tone-1.png' },
        { id: 'light', name: 'Light', thumbnailUrl: '/images/skin-tone-2.png' },
        { id: 'medium-light', name: 'Medium Light', thumbnailUrl: '/images/skin-tone-3.png' },
        { id: 'medium', name: 'Medium', thumbnailUrl: '/images/skin-tone-4.png' },
        { id: 'medium-dark', name: 'Medium Dark', thumbnailUrl: '/images/skin-tone-5.png' },
        { id: 'dark', name: 'Dark', thumbnailUrl: '/images/skin-tone-6.png' },
        { id: 'very-dark', name: 'Very Dark', thumbnailUrl: '/images/skin-tone-7.png' },
      ],
      value: settings.appearance?.skinTone || 'medium',
    },
    {
      id: 'hairStyle',
      name: 'Hair Style',
      description: 'Select a hair style for your avatar',
      type: 'select',
      options: [
        { id: 'short', name: 'Short', thumbnailUrl: '/images/hair-short.png' },
        { id: 'medium', name: 'Medium', thumbnailUrl: '/images/hair-medium.png' },
        { id: 'long', name: 'Long', thumbnailUrl: '/images/hair-long.png' },
        { id: 'curly', name: 'Curly', thumbnailUrl: '/images/hair-curly.png' },
        { id: 'wavy', name: 'Wavy', thumbnailUrl: '/images/hair-wavy.png' },
        { id: 'bald', name: 'Bald', thumbnailUrl: '/images/hair-bald.png' },
      ],
      value: settings.appearance?.hairStyle || 'short',
    },
    {
      id: 'hairColor',
      name: 'Hair Color',
      description: 'Choose the hair color for your avatar',
      type: 'color',
      options: [
        { id: 'black', name: 'Black', thumbnailUrl: '/images/hair-black.png' },
        { id: 'brown', name: 'Brown', thumbnailUrl: '/images/hair-brown.png' },
        { id: 'blonde', name: 'Blonde', thumbnailUrl: '/images/hair-blonde.png' },
        { id: 'red', name: 'Red', thumbnailUrl: '/images/hair-red.png' },
        { id: 'gray', name: 'Gray', thumbnailUrl: '/images/hair-gray.png' },
        { id: 'white', name: 'White', thumbnailUrl: '/images/hair-white.png' },
      ],
      value: settings.appearance?.hairColor || 'black',
    },
    {
      id: 'eyeColor',
      name: 'Eye Color',
      description: 'Adjust eye color of your avatar',
      type: 'color',
      options: [
        { id: 'brown', name: 'Brown', thumbnailUrl: '/images/eye-brown.png' },
        { id: 'blue', name: 'Blue', thumbnailUrl: '/images/eye-blue.png' },
        { id: 'green', name: 'Green', thumbnailUrl: '/images/eye-green.png' },
        { id: 'hazel', name: 'Hazel', thumbnailUrl: '/images/eye-hazel.png' },
        { id: 'gray', name: 'Gray', thumbnailUrl: '/images/eye-gray.png' },
      ],
      value: settings.appearance?.eyeColor || 'brown',
    },
    {
      id: 'facialFeatures',
      name: 'Facial Features',
      description: 'Adjust how much to enhance or retain your natural features',
      type: 'slider',
      min: 0,
      max: 100,
      step: 5,
      value: settings.appearance?.facialFeatures || 50,
    },
    {
      id: 'age',
      name: 'Age Appearance',
      description: 'Adjust the apparent age of your avatar',
      type: 'slider',
      min: 18,
      max: 80,
      step: 1,
      value: settings.appearance?.age || 30,
    },
    {
      id: 'jawline',
      name: 'Jawline Definition',
      description: 'Adjust the definition of your avatar\'s jawline',
      type: 'slider',
      min: 0,
      max: 100,
      step: 5,
      value: settings.appearance?.jawline || 50,
    },
  ]);
  
  useEffect(() => {
    // Syncing state when settings change
    setFeatures(prevFeatures => 
      prevFeatures.map(feature => {
        const settingValue = settings.appearance?.[feature.id as keyof typeof settings.appearance];
        if (settingValue !== undefined) {
          return { ...feature, value: settingValue };
        }
        return feature;
      })
    );
  }, [settings.appearance]);
  
  const handleFeatureChange = (featureId: string, value: string | number) => {
    // Update local state
    setFeatures(prevFeatures => 
      prevFeatures.map(feature => 
        feature.id === featureId ? { ...feature, value } : feature
      )
    );
    
    // Update global settings
    const updatedAppearance = {
      ...settings.appearance,
      [featureId]: value,
    };
    
    updateSettings({
      ...settings,
      appearance: updatedAppearance,
    });
  };
  
  const handleGeneratePreview = async () => {
    try {
      setIsGeneratingPreview(true);
      await generateAppearancePreview(settings.appearance);
    } catch (error) {
      console.error('Error generating appearance preview:', error);
    } finally {
      setIsGeneratingPreview(false);
    }
  };
  
  const renderFeatureControl = (feature: AppearanceFeature) => {
    switch (feature.type) {
      case 'slider':
        return (
          <div key={feature.id} className="mb-6">
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-medium text-neutral-100">{feature.name}</label>
              <span className="text-xs text-neutral-200">{feature.value}</span>
            </div>
            <input
              type="range"
              min={feature.min}
              max={feature.max}
              step={feature.step}
              value={feature.value}
              onChange={(e) => handleFeatureChange(feature.id, Number(e.target.value))}
              className="w-full h-2 bg-neutral-400 rounded-lg appearance-none cursor-pointer"
            />
            <p className="mt-1 text-xs text-neutral-300">{feature.description}</p>
          </div>
        );
        
      case 'color':
      case 'select':
        return (
          <div key={feature.id} className="mb-6">
            <label className="block text-sm font-medium text-neutral-100 mb-2">{feature.name}</label>
            <div className="grid grid-cols-4 gap-2">
              {feature.options?.map(option => (
                <div
                  key={option.id}
                  onClick={() => handleFeatureChange(feature.id, option.id)}
                  className={`
                    rounded-lg overflow-hidden cursor-pointer relative p-1
                    ${feature.value === option.id ? 'bg-primary bg-opacity-20 ring-1 ring-primary' : 'hover:bg-neutral-400'}
                  `}
                >
                  {option.thumbnailUrl ? (
                    <div className="aspect-square rounded-md overflow-hidden mb-1">
                      <img 
                        src={option.thumbnailUrl} 
                        alt={option.name} 
                        className="w-full h-full object-cover"
                      />
                    </div>
                  ) : (
                    <div 
                      className="aspect-square rounded-md mb-1" 
                      style={{ backgroundColor: option.id }}
                    ></div>
                  )}
                  <p className="text-center text-xs text-neutral-200 truncate">
                    {option.name}
                  </p>
                </div>
              ))}
            </div>
            <p className="mt-1 text-xs text-neutral-300">{feature.description}</p>
          </div>
        );
        
      default:
        return null;
    }
  };
  
  return (
    <div className="bg-neutral-400 rounded-lg overflow-hidden">
      <div className="p-4 border-b border-neutral-400">
        <h2 className="text-lg font-medium text-neutral-100">Appearance Customization</h2>
        <p className="text-sm text-neutral-200">
          Fine-tune the physical appearance of your avatar
        </p>
      </div>
      
      <div className="p-4 grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-1">
          <div className="flex items-center mb-4">
            <FiSliders className="text-primary mr-2" size={18} />
            <h3 className="text-md font-medium text-neutral-100">Appearance Controls</h3>
          </div>
          
          <div className="bg-neutral-400 rounded-lg p-4">
            {features.map(renderFeatureControl)}
            
            <div className="mt-4 flex justify-end">
              <button
                onClick={handleGeneratePreview}
                disabled={isGeneratingPreview || isProcessing}
                className={`
                  flex items-center px-4 py-2 rounded-lg text-sm
                  ${isGeneratingPreview || isProcessing
                    ? 'bg-neutral-300 text-neutral-200 cursor-not-allowed'
                    : 'bg-primary text-white hover:bg-primary-dark'
                  }
                `}
              >
                {isGeneratingPreview || isProcessing ? (
                  <>
                    <motion.span
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      className="mr-2"
                    >
                      <FiRefreshCw size={14} />
                    </motion.span>
                    Generating...
                  </>
                ) : (
                  <>
                    <FiEye className="mr-2" size={14} />
                    Preview Changes
                  </>
                )}
              </button>
            </div>
          </div>
          
          <div className="bg-neutral-400 rounded-lg p-4 mt-4">
            <h4 className="flex items-center text-sm font-medium text-neutral-100 mb-2">
              <FiInfo className="mr-2" size={14} />
              Avatar Appearance Tips
            </h4>
            <ul className="text-xs space-y-1 text-neutral-200 list-disc pl-5">
              <li>Changes to appearance are best previewed with good lighting.</li>
              <li>Some features like hair style may look different depending on your reference photos.</li>
              <li>The AI will attempt to blend your selections with your natural features.</li>
              <li>For the most realistic results, choose settings that are similar to your actual appearance.</li>
            </ul>
          </div>
        </div>
        
        <div>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <FiEye className="text-primary mr-2" size={18} />
              <h3 className="text-md font-medium text-neutral-100">Preview</h3>
            </div>
            
            <div className="flex items-center space-x-2">
              <button 
                className={`p-2 rounded-lg ${settings.appearance?.lightMode === 'day' ? 'bg-primary text-white' : 'bg-neutral-400 text-neutral-200'}`}
                onClick={() => handleFeatureChange('lightMode', 'day')}
                title="Day lighting"
              >
                <FiSun size={14} />
              </button>
              <button 
                className={`p-2 rounded-lg ${settings.appearance?.lightMode === 'night' ? 'bg-primary text-white' : 'bg-neutral-400 text-neutral-200'}`}
                onClick={() => handleFeatureChange('lightMode', 'night')}
                title="Night lighting"
              >
                <FiMoon size={14} />
              </button>
            </div>
          </div>
          
          <div className="aspect-video bg-neutral-400 rounded-lg overflow-hidden relative">
            {previewAngles[selectedAngle as keyof typeof previewAngles] ? (
              <video 
                src={previewAngles[selectedAngle as keyof typeof previewAngles]} 
                className="w-full h-full object-cover"
                autoPlay
                loop
                muted
                playsInline
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-neutral-200">
                <p>Preview will appear here</p>
              </div>
            )}
          </div>
          
          <div className="p-4 border-t border-neutral-400">
            <div className="flex justify-between items-center">
              <div className="flex space-x-2">
                <button
                  className={`p-2 rounded-lg ${selectedAngle === 'front' ? 'bg-primary text-white' : 'bg-neutral-400 text-neutral-200'}`}
                  onClick={() => setSelectedAngle('front')}
                  title="Front view"
                >
                  Front
                </button>
                <button
                  className={`p-2 rounded-lg ${selectedAngle === 'side' ? 'bg-primary text-white' : 'bg-neutral-400 text-neutral-200'}`}
                  onClick={() => setSelectedAngle('side')}
                  title="Side view"
                >
                  Side
                </button>
                <button
                  className={`p-2 rounded-lg ${selectedAngle === '45-degree' ? 'bg-primary text-white' : 'bg-neutral-400 text-neutral-200'}`}
                  onClick={() => setSelectedAngle('45-degree')}
                  title="45° view"
                >
                  45°
                </button>
              </div>
            </div>
          </div>
          
          <div className="mt-3 text-xs text-neutral-200 bg-neutral-400 p-3 rounded-lg">
            <p className="mb-2">This is a preview of how your avatar will look with the current appearance settings.</p>
            <p>Your avatar's final appearance may vary slightly in generated videos based on lighting and animation settings.</p>
          </div>
        </div>
      </div>
    </div>
  );
} 
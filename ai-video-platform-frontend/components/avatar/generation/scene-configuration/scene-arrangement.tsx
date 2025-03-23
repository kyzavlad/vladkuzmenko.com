'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  FiCamera, 
  FiSun, 
  FiImage, 
  FiRotateCw, 
  FiPlus, 
  FiTrash2, 
  FiMove,
  FiRefreshCw,
  FiMaximize2,
  FiInfo,
  FiCheck,
  FiAlertCircle
} from 'react-icons/fi';
import { useGenerationContext } from '../../contexts/generation-context';
import { Background, CameraAngle, LightingSetup, Prop } from '../../contexts/generation-context';
import { SAMPLE_BACKGROUNDS, SAMPLE_CAMERA_ANGLES, SAMPLE_LIGHTING_SETUPS, SAMPLE_PROPS } from '../../contexts/sample-data';

interface IsSelected {
  background: (id: string) => boolean;
  cameraAngle: (id: string) => boolean;
  lightingSetup: (id: string) => boolean;
  prop: (id: string) => boolean;
}

export default function SceneArrangement() {
  const { 
    activeJob,
    updateSceneSetup,
    generateScenePreview,
    isProcessing
  } = useGenerationContext();
  
  const [isGeneratingPreview, setIsGeneratingPreview] = useState(false);
  const [activeTab, setActiveTab] = useState<'background' | 'camera' | 'lighting'>('background');
  
  const handleBackgroundSelect = (background: Background) => {
    if (!activeJob) return;
    updateSceneSetup({
      ...activeJob.sceneSetup,
      background
    });
  };
  
  const handleCameraAngleSelect = (cameraAngle: CameraAngle) => {
    if (!activeJob) return;
    updateSceneSetup({
      ...activeJob.sceneSetup,
      cameraAngle
    });
  };
  
  const handleLightingSelect = (lighting: LightingSetup) => {
    if (!activeJob) return;
    updateSceneSetup({
      ...activeJob.sceneSetup,
      lightingSetup: lighting
    });
  };
  
  const handlePropSelect = (propId: string) => {
    if (!activeJob) return;
    
    const prop = SAMPLE_PROPS.find(p => p.id === propId);
    if (!prop) return;
    
    const updatedProps = [...activeJob.sceneSetup.props];
    const propIndex = updatedProps.findIndex(p => p.id === propId);
    
    if (propIndex === -1) {
      updatedProps.push(prop);
    } else {
      updatedProps.splice(propIndex, 1);
    }
    
    updateSceneSetup({
      ...activeJob.sceneSetup,
      props: updatedProps
    });
  };
  
  const handleGeneratePreview = async () => {
    if (!activeJob) return;
    
    try {
      setIsGeneratingPreview(true);
      await generateScenePreview();
    } catch (error) {
      console.error('Error generating scene preview:', error);
    } finally {
      setIsGeneratingPreview(false);
    }
  };
  
  // Predefined scene elements
  const backgrounds = [
    { id: 'office', name: 'Office', thumbnailUrl: '/images/bg-office.jpg' },
    { id: 'studio', name: 'Studio', thumbnailUrl: '/images/bg-studio.jpg' },
    { id: 'gradient', name: 'Gradient', thumbnailUrl: '/images/bg-gradient.jpg' },
    { id: 'outdoor', name: 'Outdoor', thumbnailUrl: '/images/bg-outdoor.jpg' },
    { id: 'living_room', name: 'Living Room', thumbnailUrl: '/images/bg-living-room.jpg' },
    { id: 'kitchen', name: 'Kitchen', thumbnailUrl: '/images/bg-kitchen.jpg' },
    { id: 'conference', name: 'Conference Room', thumbnailUrl: '/images/bg-conference.jpg' },
    { id: 'custom', name: 'Custom Upload', thumbnailUrl: '/images/bg-custom.jpg' },
  ];
  
  const cameraAngles = [
    { id: 'front', name: 'Front View', thumbnailUrl: '/images/camera-front.jpg', description: 'Standard front-facing camera angle' },
    { id: 'low_angle', name: 'Low Angle', thumbnailUrl: '/images/camera-low.jpg', description: 'Camera positioned below eye level, looking up' },
    { id: 'high_angle', name: 'High Angle', thumbnailUrl: '/images/camera-high.jpg', description: 'Camera positioned above eye level, looking down' },
    { id: 'three_quarter', name: 'Three-Quarter', thumbnailUrl: '/images/camera-three-quarter.jpg', description: 'Camera at 45-degree angle to the subject' },
    { id: 'close_up', name: 'Close-Up', thumbnailUrl: '/images/camera-close-up.jpg', description: 'Tight framing on the face' },
    { id: 'medium', name: 'Medium Shot', thumbnailUrl: '/images/camera-medium.jpg', description: 'From waist up framing' },
  ];
  
  const lightingSetups = [
    {
      id: 'natural',
      name: 'Natural Lighting',
      thumbnail: '/images/lighting/natural.jpg',
      description: 'Soft, balanced lighting that mimics natural daylight',
      brightness: 1.0,
      contrast: 0.8,
      temperature: 5500,
      direction: { x: -1, y: 1, z: -1 }
    },
    {
      id: 'studio',
      name: 'Studio Lighting',
      thumbnail: '/images/lighting/studio.jpg',
      description: 'Professional three-point lighting setup',
      brightness: 1.2,
      contrast: 1.0,
      temperature: 5000,
      direction: { x: 0, y: 0, z: -1 }
    },
    {
      id: 'dramatic',
      name: 'Dramatic Lighting',
      thumbnail: '/images/lighting/dramatic.jpg',
      description: 'High contrast lighting with strong shadows',
      brightness: 0.9,
      contrast: 1.4,
      temperature: 4000,
      direction: { x: 1, y: -0.5, z: -0.5 }
    }
  ];
  
  const props = [
    { id: 'desk', name: 'Desk', thumbnailUrl: '/images/prop-desk.jpg' },
    { id: 'laptop', name: 'Laptop', thumbnailUrl: '/images/prop-laptop.jpg' },
    { id: 'chair', name: 'Chair', thumbnailUrl: '/images/prop-chair.jpg' },
    { id: 'bookshelf', name: 'Bookshelf', thumbnailUrl: '/images/prop-bookshelf.jpg' },
    { id: 'plant', name: 'Plant', thumbnailUrl: '/images/prop-plant.jpg' },
    { id: 'whiteboard', name: 'Whiteboard', thumbnailUrl: '/images/prop-whiteboard.jpg' },
    { id: 'coffee_cup', name: 'Coffee Cup', thumbnailUrl: '/images/prop-coffee-cup.jpg' },
    { id: 'microphone', name: 'Microphone', thumbnailUrl: '/images/prop-microphone.jpg' },
  ];
  
  const isSelected: IsSelected = {
    background: (id: string) => activeJob?.sceneSetup.background?.id === id || false,
    cameraAngle: (id: string) => activeJob?.sceneSetup.cameraAngle?.id === id || false,
    lightingSetup: (id: string) => activeJob?.sceneSetup.lightingSetup?.id === id || false,
    prop: (id: string) => activeJob?.sceneSetup.props.some(p => p.id === id) || false
  };
  
  const renderBackgroundTab = () => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
        {SAMPLE_BACKGROUNDS.map(background => {
          const isSelected = activeJob?.sceneSetup.background?.id === background.id;
          return (
            <div
              key={background.id}
              onClick={() => handleBackgroundSelect(background)}
              className={`
                relative rounded-lg overflow-hidden cursor-pointer
                ${isSelected ? 'ring-2 ring-primary' : ''}
              `}
            >
              <div className="aspect-video">
                <img
                  src={background.thumbnail}
                  alt={background.name}
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
                <span className="text-white text-sm font-medium">{background.name}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
  
  const renderCameraTab = () => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
        {SAMPLE_CAMERA_ANGLES.map(angle => {
          const isSelected = activeJob?.sceneSetup.cameraAngle?.id === angle.id;
          return (
            <div
              key={angle.id}
              onClick={() => handleCameraAngleSelect(angle)}
              className={`
                relative rounded-lg overflow-hidden cursor-pointer
                ${isSelected ? 'ring-2 ring-primary' : ''}
              `}
            >
              <div className="aspect-video">
                <img
                  src={angle.thumbnail}
                  alt={angle.name}
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
                <span className="text-white text-sm font-medium">{angle.name}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
  
  const renderLightingTab = () => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
        {SAMPLE_LIGHTING_SETUPS.map(lighting => {
          const isSelected = activeJob?.sceneSetup.lightingSetup?.id === lighting.id;
          return (
            <div
              key={lighting.id}
              onClick={() => handleLightingSelect(lighting)}
              className={`
                relative rounded-lg overflow-hidden cursor-pointer
                ${isSelected ? 'ring-2 ring-primary' : ''}
              `}
            >
              <div className="aspect-video">
                <img
                  src={lighting.thumbnail}
                  alt={lighting.name}
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
                <span className="text-white text-sm font-medium">{lighting.name}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
  
  const renderPropsTab = () => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
        {SAMPLE_PROPS.map(prop => (
          <div
            key={prop.id}
            onClick={() => handlePropSelect(prop.id)}
            className={`
              relative rounded-lg overflow-hidden cursor-pointer
              ${activeJob?.sceneSetup.props.some(p => p.id === prop.id) ? 'ring-2 ring-primary' : ''}
            `}
          >
            <div className="aspect-video">
              <img
                src={prop.thumbnail}
                alt={prop.name}
                className="w-full h-full object-cover"
              />
            </div>
            <div className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
              <span className="text-white text-sm font-medium">{prop.name}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
  
  const renderPreview = () => (
    <div className="bg-neutral-400 rounded-lg overflow-hidden">
      <div className="p-3 border-b border-neutral-300">
        <h3 className="text-neutral-100 font-medium">Scene Preview</h3>
      </div>
      
      <div className="aspect-video relative">
        {isGeneratingPreview || isProcessing ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-neutral-500">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              className="mb-4"
            >
              <FiRefreshCw size={32} className="text-primary" />
            </motion.div>
            <p className="text-neutral-200 text-sm">Generating preview...</p>
          </div>
        ) : activeJob?.previewUrl ? (
          <img
            src={activeJob.previewUrl}
            alt="Scene Preview"
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-neutral-500">
            <FiImage size={32} className="text-neutral-200 mb-4" />
            <p className="text-neutral-200 text-sm">Click "Generate Preview" to see your scene</p>
          </div>
        )}
      </div>
      
      <div className="p-3">
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleGeneratePreview}
          disabled={isGeneratingPreview || isProcessing}
          className={`
            w-full flex items-center justify-center py-2 rounded-lg text-sm
            ${isGeneratingPreview || isProcessing
              ? 'bg-neutral-300 text-neutral-200 cursor-not-allowed'
              : 'bg-primary text-white hover:bg-primary-dark'}
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
              <FiRefreshCw className="mr-2" size={14} />
              Generate Preview
            </>
          )}
        </motion.button>
      </div>
      
      <div className="p-3 pt-0">
        <div className="text-xs text-neutral-200">
          <p className="flex items-start mb-2">
            <FiInfo className="mt-0.5 mr-1 flex-shrink-0" size={12} />
            This is a low-resolution preview. The final video will be rendered in high definition.
          </p>
        </div>
        
        <div className="flex justify-between text-xs text-neutral-200">
          <div>
            <p>Background: <span className="text-neutral-100">{activeJob?.sceneSetup.background?.name || 'None'}</span></p>
            <p>Camera: <span className="text-neutral-100">{activeJob?.sceneSetup.cameraAngle?.name || 'None'}</span></p>
          </div>
          <div>
            <p>Lighting: <span className="text-neutral-100">{activeJob?.sceneSetup.lightingSetup?.name || 'None'}</span></p>
            <p>Props: <span className="text-neutral-100">{activeJob?.sceneSetup.props?.length || 0} selected</span></p>
          </div>
        </div>
      </div>
    </div>
  );
  
  if (!activeJob) {
    return (
      <div className="bg-neutral-500 rounded-lg p-4 text-center">
        <FiAlertCircle size={24} className="mx-auto mb-2 text-neutral-200" />
        <p className="text-neutral-200">No active generation job. Please start a new video generation.</p>
      </div>
    );
  }
  
  return (
    <div className="bg-neutral-500 rounded-lg overflow-hidden">
      <div className="p-4 border-b border-neutral-400">
        <h2 className="text-lg font-medium text-neutral-100">Scene Configuration</h2>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Don&apos;t worry if you&apos;re not sure what to say - we&apos;ll guide you through the process
        </p>
      </div>
      
      <div className="p-4 grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <div className="flex items-center space-x-2 border-b border-neutral-400">
            <button
              onClick={() => setActiveTab('background')}
              className={`py-2 px-4 text-sm font-medium relative ${
                activeTab === 'background' ? 'text-primary' : 'text-neutral-200 hover:text-neutral-100'
              }`}
            >
              <span className="flex items-center">
                <FiImage className="mr-2" size={16} />
                Background
              </span>
              {activeTab === 'background' && (
                <motion.div 
                  layoutId="activeTabIndicator"
                  className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" 
                />
              )}
            </button>
            
            <button
              onClick={() => setActiveTab('camera')}
              className={`py-2 px-4 text-sm font-medium relative ${
                activeTab === 'camera' ? 'text-primary' : 'text-neutral-200 hover:text-neutral-100'
              }`}
            >
              <span className="flex items-center">
                <FiCamera className="mr-2" size={16} />
                Camera
              </span>
              {activeTab === 'camera' && (
                <motion.div 
                  layoutId="activeTabIndicator"
                  className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" 
                />
              )}
            </button>
            
            <button
              onClick={() => setActiveTab('lighting')}
              className={`py-2 px-4 text-sm font-medium relative ${
                activeTab === 'lighting' ? 'text-primary' : 'text-neutral-200 hover:text-neutral-100'
              }`}
            >
              <span className="flex items-center">
                <FiSun className="mr-2" size={16} />
                Lighting
              </span>
              {activeTab === 'lighting' && (
                <motion.div 
                  layoutId="activeTabIndicator"
                  className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" 
                />
              )}
            </button>
          </div>
          
          {activeTab === 'background' && renderBackgroundTab()}
          {activeTab === 'camera' && renderCameraTab()}
          {activeTab === 'lighting' && renderLightingTab()}
          
          {renderPropsTab()}
        </div>
        
        <div>
          {renderPreview()}
        </div>
      </div>
    </div>
  );
} 
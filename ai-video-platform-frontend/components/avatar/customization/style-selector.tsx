'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { FiCheck, FiInfo, FiRefreshCw } from 'react-icons/fi';
import { useCreationContext } from '../contexts/creation-context';
import { useAvatarContext, AvatarStyle } from '../contexts/avatar-context';

export default function StyleSelector() {
  const { styles: avatarStyles } = useAvatarContext();
  const { 
    settings, 
    updateSettings,
    previewUrl,
    isProcessing,
    generateStylePreview
  } = useCreationContext();
  
  const [selectedStyle, setSelectedStyle] = useState<string | null>(settings.style?.id || null);
  const [previewStyle, setPreviewStyle] = useState<string | null>(null);
  const [isGeneratingPreview, setIsGeneratingPreview] = useState(false);
  
  const handleStyleSelect = useCallback(async (style: AvatarStyle) => {
    setSelectedStyle(style.id);
    setPreviewStyle(style.id);
    setIsGeneratingPreview(true);
    
    try {
      await generateStylePreview(style);
      updateSettings({
        ...settings,
        style
      });
    } catch (error) {
      console.error('Error generating style preview:', error);
    } finally {
      setIsGeneratingPreview(false);
    }
  }, [generateStylePreview, settings, updateSettings]);
  
  // Generate preview for initially selected style
  useEffect(() => {
    if (selectedStyle && !previewUrl && !isProcessing) {
      const generateInitialPreview = async () => {
        try {
          setIsGeneratingPreview(true);
          await generateStylePreview(avatarStyles.find(style => style.id === selectedStyle) as AvatarStyle);
        } catch (error) {
          console.error('Error generating initial preview:', error);
        } finally {
          setIsGeneratingPreview(false);
        }
      };
      
      generateInitialPreview();
    }
  }, [selectedStyle, previewUrl, isProcessing, generateStylePreview, avatarStyles]);
  
  const renderPreview = () => {
    if (!selectedStyle) {
      return (
        <div className="aspect-square w-full bg-neutral-400 rounded-lg flex items-center justify-center">
          <p className="text-neutral-200 text-center p-4">
            Select a style to see a preview of your avatar
          </p>
        </div>
      );
    }
    
    if (isGeneratingPreview || isProcessing) {
      return (
        <div className="aspect-square w-full bg-neutral-400 rounded-lg flex flex-col items-center justify-center p-6">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
            className="mb-4"
          >
            <FiRefreshCw size={32} className="text-primary" />
          </motion.div>
          <p className="text-neutral-200 text-center">
            Generating preview...
          </p>
          <p className="text-neutral-300 text-xs text-center mt-2">
            This may take a few moments as we apply the style to your avatar
          </p>
        </div>
      );
    }
    
    if (previewUrl) {
      return (
        <div className="aspect-square w-full rounded-lg overflow-hidden relative">
          <img 
            src={previewUrl} 
            alt="Avatar preview" 
            className="w-full h-full object-cover" 
          />
          {previewStyle && (
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-3">
              <p className="text-white text-sm">
                {avatarStyles.find(style => style.id === previewStyle)?.name || 'Custom Style'}
              </p>
            </div>
          )}
        </div>
      );
    }
    
    return (
      <div className="aspect-square w-full bg-neutral-400 rounded-lg flex items-center justify-center">
        <p className="text-neutral-200 text-center p-4">
          Failed to generate preview. Please try again.
        </p>
      </div>
    );
  };
  
  return (
    <div className="bg-neutral-500 rounded-lg overflow-hidden">
      <div className="p-4 border-b border-neutral-400">
        <h2 className="text-lg font-medium text-neutral-100">Avatar Style</h2>
        <p className="text-sm text-neutral-200">
          Choose a visual style for your virtual avatar
        </p>
      </div>
      
      <div className="p-4 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {avatarStyles.map(style => (
          <div
            key={style.id}
            className={`
              rounded-lg overflow-hidden cursor-pointer transition-all
              ${selectedStyle === style.id ? 'ring-2 ring-primary' : 'hover:ring-1 hover:ring-primary'}
            `}
            onClick={() => handleStyleSelect(style)}
          >
            <div className="aspect-video relative">
              <img
                src={style.thumbnail}
                alt={style.name}
                className="w-full h-full object-cover"
              />
              {previewStyle === style.id && isGeneratingPreview && (
                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
                  <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent"></div>
                </div>
              )}
            </div>
            <div className="p-3 bg-neutral-400">
              <h3 className="font-medium text-neutral-100">{style.name}</h3>
              <p className="text-sm text-neutral-200">{style.description}</p>
            </div>
          </div>
        ))}
      </div>
      
      <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 className="text-sm font-medium text-neutral-100 mb-3">Preview</h3>
          {renderPreview()}
          
          <div className="mt-3 text-xs text-neutral-200">
            <p className="mb-2">This is a preview of how your avatar will look with the selected style.</p>
            <p>The final result may vary slightly based on additional customization options.</p>
          </div>
          
          {selectedStyle && !isGeneratingPreview && previewUrl && (
            <div className="mt-4 flex justify-end">
              <button
                onClick={async () => {
                  if (selectedStyle) {
                    try {
                      setIsGeneratingPreview(true);
                      await generateStylePreview(avatarStyles.find(style => style.id === selectedStyle) as AvatarStyle);
                    } catch (error) {
                      console.error('Error regenerating preview:', error);
                    } finally {
                      setIsGeneratingPreview(false);
                    }
                  }
                }}
                className="flex items-center text-sm text-primary hover:text-primary-dark transition-colors"
              >
                <FiRefreshCw className="mr-1" size={14} />
                Regenerate Preview
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 
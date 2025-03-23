'use client';

import React, { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FiUploadCloud, 
  FiX, 
  FiLink, 
  FiFile,
  FiCheck,
  FiAlertCircle
} from 'react-icons/fi';
import { useMediaContext } from '../contexts/media-context';
import { formatFileSize } from '../../../lib/utils/formatters';

export default function MediaUploader() {
  const [isOpen, setIsOpen] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isImportingUrl, setIsImportingUrl] = useState(false);
  const [videoUrl, setVideoUrl] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [errors, setErrors] = useState<{[key: string]: string}>({});
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const { uploadMedia } = useMediaContext();
  
  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);
  
  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);
  
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);
  
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      uploadMedia(files);
      setIsOpen(false);
    }
  }, [uploadMedia]);
  
  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      uploadMedia(files);
      setIsOpen(false);
    }
  }, [uploadMedia]);
  
  const openFileDialog = () => {
    fileInputRef.current?.click();
  };
  
  const handleUrlImport = async () => {
    if (!videoUrl) return;
    
    try {
      setIsUploading(true);
      // In a real app, this would validate and process the URL
      await new Promise(resolve => setTimeout(resolve, 1000));
      setVideoUrl('');
      setIsImportingUrl(false);
      setIsOpen(false);
    } catch (error) {
      console.error('Error importing video:', error);
      setErrors({ url: 'Failed to import video' });
    } finally {
      setIsUploading(false);
    }
  };
  
  return (
    <>
      <button
        className="px-4 py-2 bg-primary text-white rounded-md font-medium hover:bg-primary-dark transition-colors flex items-center space-x-2"
        onClick={() => setIsOpen(true)}
      >
        <FiUploadCloud size={18} />
        <span>Upload Media</span>
      </button>
      
      <AnimatePresence>
        {isOpen && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="bg-neutral-400 rounded-xl max-w-lg w-full overflow-hidden shadow-elevation-3"
            >
              <div className="p-4 border-b border-neutral-300 flex justify-between items-center">
                <h2 className="text-lg font-semibold text-neutral-100">Upload Media</h2>
                <button
                  className="text-neutral-200 hover:text-neutral-100"
                  onClick={() => setIsOpen(false)}
                >
                  <FiX size={20} />
                </button>
              </div>
              
              <div className="p-4">
                <div
                  className={`border-2 border-dashed rounded-lg p-8 text-center ${
                    isDragging ? 'border-primary bg-primary bg-opacity-5' : 'border-neutral-300'
                  }`}
                  onDragEnter={handleDragEnter}
                  onDragLeave={handleDragLeave}
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  onClick={openFileDialog}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    accept="video/*,image/*"
                    multiple
                    onChange={handleFileSelect}
                  />
                  
                  <div className="mb-4">
                    <div className="w-12 h-12 rounded-full bg-neutral-300 flex items-center justify-center mx-auto mb-4">
                      <FiUploadCloud size={24} className="text-neutral-100" />
                    </div>
                    <p className="text-neutral-100 font-medium mb-1">
                      Drop your files here or click to browse
                    </p>
                    <p className="text-sm text-neutral-200">
                      Supports video and image files up to 100MB
                    </p>
                  </div>
                </div>
                
                <div className="mt-4 flex items-center">
                  <div className="flex-1 border-t border-neutral-300"></div>
                  <span className="mx-4 text-sm text-neutral-200">or</span>
                  <div className="flex-1 border-t border-neutral-300"></div>
                </div>
                
                <button
                  className="mt-4 w-full py-3 border-2 border-neutral-300 rounded-lg text-neutral-100 font-medium hover:bg-neutral-300 transition-colors flex items-center justify-center space-x-2"
                  onClick={() => setIsImportingUrl(true)}
                >
                  <FiLink size={18} />
                  <span>Import from URL</span>
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
      
      <AnimatePresence>
        {isImportingUrl && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="bg-neutral-400 rounded-xl max-w-md w-full overflow-hidden shadow-elevation-3"
            >
              <div className="p-4 border-b border-neutral-300 flex justify-between items-center">
                <h2 className="text-lg font-semibold text-neutral-100">Import from URL</h2>
                <button
                  className="text-neutral-200 hover:text-neutral-100"
                  onClick={() => setIsImportingUrl(false)}
                >
                  <FiX size={20} />
                </button>
              </div>
              
              <div className="p-4">
                <input
                  type="text"
                  placeholder="Enter video URL..."
                  className="w-full px-4 py-2 bg-neutral-300 border border-neutral-300 rounded-md text-neutral-100 placeholder-neutral-200 focus:outline-none focus:ring-2 focus:ring-primary"
                  value={videoUrl}
                  onChange={(e) => setVideoUrl(e.target.value)}
                />
                
                {errors.url && (
                  <p className="mt-2 text-sm text-red-500">{errors.url}</p>
                )}
                
                <div className="mt-4 flex space-x-3">
                  <button
                    className="flex-1 px-4 py-2 bg-neutral-300 text-neutral-100 rounded-md font-medium hover:bg-neutral-350 transition-colors"
                    onClick={() => setIsImportingUrl(false)}
                  >
                    Cancel
                  </button>
                  <button
                    className="flex-1 px-4 py-2 bg-primary text-white rounded-md font-medium hover:bg-primary-dark transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    onClick={handleUrlImport}
                    disabled={!videoUrl || isUploading}
                  >
                    {isUploading ? 'Importing...' : 'Import'}
                  </button>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </>
  );
} 
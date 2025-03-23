'use client';

import React, { useCallback, useState } from 'react';
import { motion } from 'framer-motion';
import { FiUpload, FiImage, FiTrash2, FiInfo, FiCheckCircle, FiXCircle } from 'react-icons/fi';
import { useCreationContext, CreationReferenceImage } from '../../contexts/creation-context';

export default function ReferenceUploader() {
  const { 
    error, 
    isProcessing,
    settings,
    uploadReferenceImage,
    deleteReferenceImage
  } = useCreationContext();
  
  const [dragActive, setDragActive] = useState(false);
  const [selectedImage, setSelectedImage] = useState<CreationReferenceImage | null>(null);
  
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);
  
  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      await handleFiles(e.dataTransfer.files);
    }
  }, []);
  
  const handleInputChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    
    if (e.target.files && e.target.files.length > 0) {
      await handleFiles(e.target.files);
    }
  }, []);
  
  const handleFiles = async (files: FileList) => {
    const validFiles: File[] = [];
    
    // Filter and validate files
    Array.from(files).forEach(file => {
      // Check if it's an image
      if (file.type.startsWith('image/')) {
        validFiles.push(file);
      }
    });
    
    if (validFiles.length === 0) {
      console.error('No valid image files selected');
      return;
    }
    
    // Upload the first valid file
    try {
      const uploadedImage = await uploadReferenceImage(validFiles[0]);
      setSelectedImage(uploadedImage);
    } catch (err) {
      console.error('Error uploading image:', err);
    }
  };
  
  const handleDelete = useCallback((id: string) => {
    deleteReferenceImage(id);
    if (selectedImage && selectedImage.id === id) {
      setSelectedImage(null);
    }
  }, [deleteReferenceImage, selectedImage]);
  
  const handleSelectImage = useCallback((image: CreationReferenceImage) => {
    setSelectedImage(image);
  }, []);
  
  const renderImageGrid = () => {
    const { referenceImages } = settings;
    
    if (referenceImages.length === 0) {
      return (
        <div className="p-4 text-center text-neutral-200">
          <p>No reference images uploaded yet</p>
        </div>
      );
    }
    
    return (
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 p-4">
        {referenceImages.map(image => (
          <div 
            key={image.id}
            className={`relative rounded-lg overflow-hidden aspect-square cursor-pointer border-2 ${
              selectedImage && selectedImage.id === image.id 
                ? 'border-primary' 
                : 'border-transparent hover:border-neutral-300'
            }`}
            onClick={() => handleSelectImage(image)}
          >
            <img 
              src={image.url} 
              alt="Reference" 
              className="w-full h-full object-cover"
            />
            
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-2">
              <div className="flex justify-between items-center">
                <div className="text-xs text-white">
                  {image.quality >= 85 ? (
                    <span className="flex items-center">
                      <FiCheckCircle className="text-green-500 mr-1" size={12} /> 
                      High quality
                    </span>
                  ) : image.quality >= 70 ? (
                    <span className="flex items-center">
                      <FiCheckCircle className="text-yellow-500 mr-1" size={12} /> 
                      Good quality
                    </span>
                  ) : (
                    <span className="flex items-center">
                      <FiXCircle className="text-red-500 mr-1" size={12} /> 
                      Low quality
                    </span>
                  )}
                </div>
                <button
                  className="text-white hover:text-red-500 transition-colors p-1"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDelete(image.id);
                  }}
                >
                  <FiTrash2 size={14} />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  };
  
  const renderSelectedImage = () => {
    if (!selectedImage) return null;
    
    return (
      <div className="p-4 bg-neutral-400 rounded-lg">
        <h3 className="text-sm font-medium mb-2 text-neutral-100">Selected Reference</h3>
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="w-full sm:w-1/3">
            <div className="aspect-square rounded-lg overflow-hidden">
              <img 
                src={selectedImage.url} 
                alt="Selected reference" 
                className="w-full h-full object-cover"
              />
            </div>
          </div>
          
          <div className="flex-1 flex flex-col justify-between">
            <div>
              <div className="flex justify-between items-center">
                <h4 className="text-neutral-100 font-medium">Image Details</h4>
                <span className="text-xs text-neutral-200">
                  {new Date(selectedImage.createdAt).toLocaleDateString()}
                </span>
              </div>
              
              <div className="mt-2 space-y-2 text-sm text-neutral-200">
                <div className="flex justify-between">
                  <span>Quality Score:</span>
                  <span className="font-medium">{selectedImage.quality}/100</span>
                </div>
                
                {selectedImage.tags && selectedImage.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-2">
                    {selectedImage.tags.map(tag => (
                      <span 
                        key={tag} 
                        className="px-2 py-0.5 bg-neutral-300 rounded-full text-xs"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              
              <div className="mt-4 text-xs text-neutral-200">
                <p>
                  <FiInfo className="inline mr-1" size={14} />
                  This reference image will be used to guide the appearance styling of your avatar.
                </p>
              </div>
            </div>
            
            <div className="mt-4 flex justify-end">
              <button
                className="text-neutral-200 hover:text-red-500 flex items-center"
                onClick={() => handleDelete(selectedImage.id)}
              >
                <FiTrash2 className="mr-1" size={14} />
                <span>Remove</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };
  
  return (
    <div className="bg-neutral-500 rounded-lg overflow-hidden">
      <div className="p-4 border-b border-neutral-400">
        <h2 className="text-lg font-medium text-neutral-100">Reference Images</h2>
        <p className="text-sm text-neutral-200">
          Upload reference photos to guide the appearance of your avatar
        </p>
      </div>
      
      <div 
        className={`border-2 border-dashed rounded-lg m-4 ${
          dragActive 
            ? 'border-primary bg-primary bg-opacity-5' 
            : 'border-neutral-300'
        }`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
      >
        <div className="p-8 text-center">
          <FiUpload 
            className={`mx-auto mb-3 ${dragActive ? 'text-primary' : 'text-neutral-300'}`} 
            size={36} 
          />
          
          <p className="mb-2 text-sm text-neutral-200">
            {dragActive 
              ? 'Drop your images here' 
              : 'Drag and drop or click to upload reference images'
            }
          </p>
          
          <p className="text-xs text-neutral-300 mb-4">
            Recommended: Clear front-facing photos of your face. <br />
            Supported formats: JPG, PNG, WebP (max 5MB)
          </p>
          
          <label className="inline-block">
            <motion.span
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="inline-flex items-center px-4 py-2 bg-primary text-white rounded-lg cursor-pointer"
            >
              <FiImage className="mr-2" />
              Browse Files
            </motion.span>
            <input
              type="file"
              className="hidden"
              accept="image/*"
              onChange={handleInputChange}
              disabled={isProcessing}
            />
          </label>
        </div>
      </div>
      
      {renderImageGrid()}
      
      {selectedImage && renderSelectedImage()}
      
      {error && (
        <div className="m-4 p-3 bg-red-500 bg-opacity-10 border border-red-500 text-red-500 rounded-lg text-sm">
          {error}
        </div>
      )}
      
      <div className="p-4 bg-neutral-400 rounded-lg m-4">
        <h3 className="text-sm font-medium mb-2 flex items-center text-neutral-100">
          <FiInfo className="mr-2" /> Reference Guidelines
        </h3>
        <ul className="text-xs space-y-1 list-disc pl-5 text-neutral-200">
          <li>Upload clear, well-lit photos of your face from different angles</li>
          <li>Neutral expressions work best for the base avatar</li>
          <li>Avoid heavy makeup, glasses, or accessories that obscure facial features</li>
          <li>Photos with solid, neutral backgrounds will produce the best results</li>
          <li>Higher resolution images will result in better quality avatars</li>
        </ul>
      </div>
    </div>
  );
} 
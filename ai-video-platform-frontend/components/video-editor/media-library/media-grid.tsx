'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Image from 'next/image';
import { 
  FiPlay, 
  FiCheck, 
  FiClock, 
  FiAlertCircle, 
  FiInfo,
  FiEdit2
} from 'react-icons/fi';
import { MediaFile } from '../contexts/media-context';
import { formatDuration, formatFileSize, formatDate } from '../../../lib/utils/formatters';

interface MediaGridProps {
  files: MediaFile[];
  onSelect: (file: MediaFile) => void;
  onDetails: (file: MediaFile) => void;
}

export default function MediaGrid({ files, onSelect, onDetails }: MediaGridProps) {
  // Animation variants for staggered appearance
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.05
      }
    }
  };
  
  const itemVariants = {
    hidden: { opacity: 0, scale: 0.95 },
    visible: { 
      opacity: 1, 
      scale: 1,
      transition: {
        duration: 0.2
      }
    }
  };
  
  const getStatusIcon = (status: string) => {
    switch(status) {
      case 'ready':
        return <FiCheck size={16} className="text-green-500" />;
      case 'processing':
        return <FiClock size={16} className="text-yellow-500" />;
      case 'error':
        return <FiAlertCircle size={16} className="text-red-500" />;
      default:
        return null;
    }
  };
  
  return (
    <motion.div
      className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {files.map((file) => (
        <motion.div
          key={file.id}
          className={`bg-neutral-300 rounded-lg overflow-hidden shadow-elevation-1 transition-all group hover:shadow-elevation-2 ${
            file.isSelected ? 'ring-2 ring-primary' : ''
          }`}
          variants={itemVariants}
        >
          {/* Thumbnail / Preview */}
          <div className="relative aspect-video">
            {file.thumbnail ? (
              <Image
                src={file.thumbnail}
                alt={file.name}
                fill
                sizes="(max-width: 640px) 100vw, (max-width: 768px) 50vw, (max-width: 1024px) 33vw, (max-width: 1280px) 25vw, 20vw"
                className="object-cover"
              />
            ) : (
              <div className="w-full h-full bg-neutral-500 flex items-center justify-center">
                <span className="text-neutral-300 text-xl">{file.name.slice(0, 1).toUpperCase()}</span>
              </div>
            )}
            
            {/* Overlay with play button */}
            <div className="absolute inset-0 bg-black bg-opacity-30 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
              <button
                className="bg-primary text-white p-3 rounded-full transform transition-transform hover:scale-110"
                onClick={(e) => {
                  e.stopPropagation();
                  // In a real app, this would open a video preview
                  alert(`Preview ${file.name}`);
                }}
              >
                <FiPlay size={20} />
              </button>
            </div>
            
            {/* Status badge */}
            <div className="absolute top-2 right-2 bg-neutral-400 bg-opacity-80 p-1 rounded-md">
              {getStatusIcon(file.status)}
            </div>
            
            {/* Duration badge */}
            {file.duration && (
              <div className="absolute bottom-2 right-2 bg-black bg-opacity-70 px-2 py-1 rounded text-xs text-white">
                {formatDuration(file.duration)}
              </div>
            )}
            
            {/* Selection checkbox */}
            <div 
              className={`absolute top-2 left-2 w-5 h-5 rounded flex items-center justify-center transition-all ${
                file.isSelected 
                  ? 'bg-primary text-white' 
                  : 'bg-neutral-400 bg-opacity-80 group-hover:bg-neutral-300'
              }`}
              onClick={(e) => {
                e.stopPropagation();
                onSelect(file);
              }}
            >
              {file.isSelected && <FiCheck size={14} />}
            </div>
          </div>
          
          {/* Metadata */}
          <div 
            className="p-3 cursor-pointer"
            onClick={() => onSelect(file)}
          >
            <div className="flex justify-between items-start mb-1">
              <h3 className="font-medium text-neutral-100 text-sm truncate flex-1" title={file.name}>
                {file.name}
              </h3>
              <button
                className="ml-2 text-neutral-200 hover:text-neutral-100"
                onClick={(e) => {
                  e.stopPropagation();
                  onDetails(file);
                }}
              >
                <FiInfo size={16} />
              </button>
            </div>
            
            <div className="flex justify-between text-xs text-neutral-200">
              <span>{formatFileSize(file.size)}</span>
              <span>{formatDate(file.createdAt)}</span>
            </div>
            
            {/* Tags */}
            {file.tags && file.tags.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1">
                {file.tags.slice(0, 2).map((tag) => (
                  <span 
                    key={tag} 
                    className="px-1.5 py-0.5 bg-neutral-400 rounded-sm text-xs text-neutral-200"
                  >
                    {tag}
                  </span>
                ))}
                {file.tags.length > 2 && (
                  <span className="px-1.5 py-0.5 rounded-sm text-xs text-neutral-200">
                    +{file.tags.length - 2}
                  </span>
                )}
              </div>
            )}
          </div>
          
          {/* Quick actions */}
          <div className="px-3 pb-3 pt-0 flex justify-end">
            <button
              className="p-1.5 rounded-md hover:bg-neutral-400 text-neutral-200 hover:text-neutral-100"
              onClick={(e) => {
                e.stopPropagation();
                // In a real app, this would open the editor with this file
                alert(`Edit ${file.name}`);
              }}
            >
              <FiEdit2 size={14} />
            </button>
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
} 
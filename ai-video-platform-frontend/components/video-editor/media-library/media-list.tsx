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
  FiEdit2,
  FiTrash2,
  FiFileText
} from 'react-icons/fi';
import { MediaFile } from '../contexts/media-context';
import { formatDuration, formatFileSize, formatDate } from '../../../lib/utils/formatters';

interface MediaListProps {
  files: MediaFile[];
  onSelect: (file: MediaFile) => void;
  onDetails: (file: MediaFile) => void;
}

export default function MediaList({ files, onSelect, onDetails }: MediaListProps) {
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
    hidden: { opacity: 0, y: 10 },
    visible: { 
      opacity: 1, 
      y: 0,
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
  
  const getFileTypeIcon = (type: string) => {
    if (type.startsWith('video/')) {
      return <FiPlay size={16} className="text-neutral-200" />;
    } else {
      return <FiFileText size={16} className="text-neutral-200" />;
    }
  };
  
  return (
    <motion.div
      className="overflow-hidden rounded-lg border border-neutral-300"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* Table Header */}
      <div className="bg-neutral-300 grid grid-cols-12 gap-2 px-4 py-3 text-sm font-medium text-neutral-100">
        <div className="col-span-1"></div>
        <div className="col-span-4">Name</div>
        <div className="col-span-1 text-right">Type</div>
        <div className="col-span-1 text-right">Size</div>
        <div className="col-span-1 text-right">Duration</div>
        <div className="col-span-2">Tags</div>
        <div className="col-span-1 text-right">Status</div>
        <div className="col-span-1 text-right">Actions</div>
      </div>
      
      {/* Table Body */}
      <div className="divide-y divide-neutral-300">
        {files.map((file) => (
          <motion.div
            key={file.id}
            className={`bg-neutral-400 grid grid-cols-12 gap-2 px-4 py-3 hover:bg-neutral-300 cursor-pointer transition-colors ${
              file.isSelected ? 'bg-primary bg-opacity-10' : ''
            }`}
            onClick={() => onSelect(file)}
            variants={itemVariants}
          >
            {/* Checkbox */}
            <div className="col-span-1 flex items-center">
              <div 
                className={`w-5 h-5 rounded flex items-center justify-center transition-all ${
                  file.isSelected 
                    ? 'bg-primary text-white' 
                    : 'bg-neutral-300'
                }`}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelect(file);
                }}
              >
                {file.isSelected && <FiCheck size={14} />}
              </div>
            </div>
            
            {/* Name with thumbnail */}
            <div className="col-span-4 flex items-center space-x-3">
              <div className="relative w-10 h-10 rounded overflow-hidden flex-shrink-0">
                {file.thumbnail ? (
                  <Image
                    src={file.thumbnail}
                    alt={file.name}
                    fill
                    sizes="40px"
                    className="object-cover"
                  />
                ) : (
                  <div className="w-full h-full bg-neutral-500 flex items-center justify-center">
                    <span className="text-neutral-300 text-sm">{file.name.slice(0, 1).toUpperCase()}</span>
                  </div>
                )}
              </div>
              <span className="truncate font-medium text-neutral-100" title={file.name}>
                {file.name}
              </span>
            </div>
            
            {/* Type */}
            <div className="col-span-1 flex items-center justify-end">
              <div className="flex items-center" title={file.type}>
                {getFileTypeIcon(file.type)}
                <span className="ml-1 text-xs text-neutral-200">
                  {file.type.split('/')[1]?.toUpperCase()}
                </span>
              </div>
            </div>
            
            {/* Size */}
            <div className="col-span-1 text-right text-sm text-neutral-200 flex items-center justify-end">
              {formatFileSize(file.size)}
            </div>
            
            {/* Duration */}
            <div className="col-span-1 text-right text-sm text-neutral-200 flex items-center justify-end">
              {file.duration ? formatDuration(file.duration) : '--:--'}
            </div>
            
            {/* Tags */}
            <div className="col-span-2 flex items-center">
              <div className="flex flex-wrap gap-1">
                {file.tags && file.tags.length > 0 ? (
                  <>
                    {file.tags.slice(0, 2).map((tag) => (
                      <span 
                        key={tag} 
                        className="px-1.5 py-0.5 bg-neutral-300 rounded-sm text-xs text-neutral-200"
                      >
                        {tag}
                      </span>
                    ))}
                    {file.tags.length > 2 && (
                      <span className="px-1.5 py-0.5 rounded-sm text-xs text-neutral-200">
                        +{file.tags.length - 2}
                      </span>
                    )}
                  </>
                ) : (
                  <span className="text-xs text-neutral-200">--</span>
                )}
              </div>
            </div>
            
            {/* Status */}
            <div className="col-span-1 flex items-center justify-end space-x-1">
              {getStatusIcon(file.status)}
              <span className="text-xs text-neutral-200">
                {file.status.charAt(0).toUpperCase() + file.status.slice(1)}
              </span>
            </div>
            
            {/* Actions */}
            <div className="col-span-1 flex items-center justify-end space-x-1">
              <button
                className="p-1.5 rounded-md hover:bg-neutral-300 text-neutral-200 hover:text-neutral-100"
                onClick={(e) => {
                  e.stopPropagation();
                  onDetails(file);
                }}
                title="View Details"
              >
                <FiInfo size={16} />
              </button>
              <button
                className="p-1.5 rounded-md hover:bg-neutral-300 text-neutral-200 hover:text-neutral-100"
                onClick={(e) => {
                  e.stopPropagation();
                  // In a real app, this would open the editor with this file
                  alert(`Edit ${file.name}`);
                }}
                title="Edit File"
              >
                <FiEdit2 size={16} />
              </button>
            </div>
          </motion.div>
        ))}
      </div>
      
      {/* Empty State */}
      {files.length === 0 && (
        <div className="p-8 text-center text-neutral-200">
          <p>No files match the current filters.</p>
        </div>
      )}
    </motion.div>
  );
} 
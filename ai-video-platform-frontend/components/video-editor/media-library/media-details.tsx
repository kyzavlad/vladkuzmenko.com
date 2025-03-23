'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Image from 'next/image';
import { 
  FiX, 
  FiEdit2, 
  FiTrash2, 
  FiDownload, 
  FiTag, 
  FiCalendar,
  FiClock,
  FiList,
  FiInfo,
  FiPlay,
  FiPause,
  FiVolume2,
  FiVolumeX,
  FiMaximize,
  FiVideo,
  FiFileText,
  FiHardDrive,
  FiImage,
  FiMusic,
  FiFile
} from 'react-icons/fi';
import { MediaFile } from '../contexts/media-context';
import { formatDuration, formatFileSize, formatDate } from '../../../lib/utils/formatters';

interface MediaDetailsProps {
  file: MediaFile;
  onClose: () => void;
}

export default function MediaDetails({ file, onClose }: MediaDetailsProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTab, setCurrentTab] = useState<'info'|'metadata'|'history'>('info');
  const [addingTag, setAddingTag] = useState(false);
  const [newTag, setNewTag] = useState('');
  
  // For demonstration purposes
  const metadata = {
    codec: 'H.264',
    audio: 'AAC',
    fps: '30',
    bitrate: '8.5 Mbps',
    pixelFormat: 'yuv420p',
    aspectRatio: '16:9',
    resolution: `${file.width}x${file.height}`,
    colorProfile: 'Rec. 709'
  };
  
  const history = [
    { date: new Date(Date.now() - 86400000 * 2), action: 'File uploaded', user: 'John Doe' },
    { date: new Date(Date.now() - 86400000), action: 'File processed', user: 'System' },
    { date: new Date(Date.now() - 3600000), action: 'Tags added', user: 'Jane Smith' }
  ];
  
  const handleTagAdd = () => {
    if (newTag.trim() === '') return;
    // In a real app, this would update the file with the new tag
    console.log('Adding tag:', newTag);
    setNewTag('');
    setAddingTag(false);
  };
  
  const handleTagDelete = (tag: string) => {
    // In a real app, this would update the file to remove the tag
    console.log('Removing tag:', tag);
  };
  
  const getFileIcon = () => {
    switch (file.type) {
      case 'video':
        return <FiVideo className="text-blue-500" />;
      case 'image':
        return <FiImage className="text-green-500" />;
      case 'audio':
        return <FiMusic className="text-purple-500" />;
      default:
        return <FiFile className="text-gray-500" />;
    }
  };
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <motion.div 
        className="bg-neutral-400 rounded-xl w-full max-w-3xl max-h-[90vh] flex flex-col overflow-hidden shadow-elevation-3"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        transition={{ duration: 0.2 }}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-neutral-300">
          <div className="flex items-center space-x-3">
            {getFileIcon()}
            <h2 className="text-xl font-bold text-neutral-100">{file.name}</h2>
          </div>
          <button 
            className="p-2 text-neutral-200 hover:text-neutral-100 rounded-full hover:bg-neutral-300 transition-colors"
            onClick={onClose}
          >
            <FiX size={20} />
          </button>
        </div>
        
        {/* Preview Section */}
        <div className="relative aspect-video bg-black">
          {file.type.startsWith('video/') ? (
            <>
              {/* Video preview */}
              <video
                src={file.url}
                className="w-full h-full object-contain"
                controls={false}
                muted={isMuted}
                loop
                playsInline
                ref={(ref) => {
                  if (ref && isPlaying) {
                    ref.play();
                  } else if (ref && !isPlaying) {
                    ref.pause();
                  }
                }}
              />
              
              {/* Video controls */}
              <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black to-transparent">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <button 
                      className="text-white p-2 rounded-full bg-black bg-opacity-50 hover:bg-opacity-70"
                      onClick={() => setIsPlaying(!isPlaying)}
                    >
                      {isPlaying ? <FiPause size={18} /> : <FiPlay size={18} />}
                    </button>
                    <button 
                      className="text-white p-2 rounded-full bg-black bg-opacity-50 hover:bg-opacity-70"
                      onClick={() => setIsMuted(!isMuted)}
                    >
                      {isMuted ? <FiVolumeX size={18} /> : <FiVolume2 size={18} />}
                    </button>
                  </div>
                  <button 
                    className="text-white p-2 rounded-full bg-black bg-opacity-50 hover:bg-opacity-70"
                    onClick={() => {
                      // In a real app, this would handle fullscreen
                      alert('Fullscreen video');
                    }}
                  >
                    <FiMaximize size={18} />
                  </button>
                </div>
              </div>
            </>
          ) : (
            // Image preview or fallback
            <div className="w-full h-full flex items-center justify-center">
              {file.thumbnail ? (
                <Image
                  src={file.thumbnail}
                  alt={file.name}
                  fill
                  sizes="(max-width: 1024px) 100vw, 1024px"
                  className="object-contain"
                />
              ) : (
                <div className="flex flex-col items-center justify-center text-neutral-700">
                  {getFileIcon()}
                  <span className="mt-2 text-sm">No preview available</span>
                </div>
              )}
            </div>
          )}
        </div>
        
        {/* Tabs */}
        <div className="flex border-b border-neutral-300">
          <button
            className={`px-4 py-3 font-medium text-sm transition-colors flex items-center space-x-1.5 ${
              currentTab === 'info' 
                ? 'text-primary border-b-2 border-primary' 
                : 'text-neutral-200 hover:text-neutral-100'
            }`}
            onClick={() => setCurrentTab('info')}
          >
            <FiInfo size={16} />
            <span>Information</span>
          </button>
          <button
            className={`px-4 py-3 font-medium text-sm transition-colors flex items-center space-x-1.5 ${
              currentTab === 'metadata' 
                ? 'text-primary border-b-2 border-primary' 
                : 'text-neutral-200 hover:text-neutral-100'
            }`}
            onClick={() => setCurrentTab('metadata')}
          >
            <FiFileText size={16} />
            <span>Metadata</span>
          </button>
          <button
            className={`px-4 py-3 font-medium text-sm transition-colors flex items-center space-x-1.5 ${
              currentTab === 'history' 
                ? 'text-primary border-b-2 border-primary' 
                : 'text-neutral-200 hover:text-neutral-100'
            }`}
            onClick={() => setCurrentTab('history')}
          >
            <FiList size={16} />
            <span>History</span>
          </button>
        </div>
        
        {/* Tab Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {currentTab === 'info' && (
            <div className="space-y-6">
              {/* Basic Info */}
              <div>
                <h3 className="text-sm uppercase font-medium text-neutral-200 mb-3">Basic Information</h3>
                <div className="grid sm:grid-cols-2 gap-3">
                  <div className="bg-neutral-300 p-3 rounded-lg">
                    <div className="text-xs text-neutral-200 mb-1">File Type</div>
                    <div className="text-sm font-medium text-neutral-100">
                      {file.type.split('/')[1]?.toUpperCase() || 'Unknown'}
                    </div>
                  </div>
                  <div className="bg-neutral-300 p-3 rounded-lg">
                    <div className="text-xs text-neutral-200 mb-1">File Size</div>
                    <div className="text-sm font-medium text-neutral-100">{formatFileSize(file.size)}</div>
                  </div>
                  <div className="bg-neutral-300 p-3 rounded-lg">
                    <div className="text-xs text-neutral-200 mb-1">Duration</div>
                    <div className="text-sm font-medium text-neutral-100">
                      {file.duration ? formatDuration(file.duration) : '--:--'}
                    </div>
                  </div>
                  <div className="bg-neutral-300 p-3 rounded-lg">
                    <div className="text-xs text-neutral-200 mb-1">Date Added</div>
                    <div className="text-sm font-medium text-neutral-100">{formatDate(file.createdAt)}</div>
                  </div>
                  {file.width && file.height && (
                    <div className="bg-neutral-300 p-3 rounded-lg">
                      <div className="text-xs text-neutral-200 mb-1">Resolution</div>
                      <div className="text-sm font-medium text-neutral-100">{file.width} × {file.height}</div>
                    </div>
                  )}
                  <div className="bg-neutral-300 p-3 rounded-lg">
                    <div className="text-xs text-neutral-200 mb-1">Status</div>
                    <div className="text-sm font-medium text-neutral-100">
                      {file.status.charAt(0).toUpperCase() + file.status.slice(1)}
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Tags */}
              <div>
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm uppercase font-medium text-neutral-200">Tags</h3>
                  <button 
                    className="text-xs text-primary hover:underline flex items-center"
                    onClick={() => setAddingTag(true)}
                  >
                    <FiTag size={12} className="mr-1" />
                    Add Tag
                  </button>
                </div>
                
                {addingTag ? (
                  <div className="flex items-center space-x-2 mb-3">
                    <input
                      type="text"
                      placeholder="Enter tag name..."
                      className="flex-1 pl-3 pr-4 py-2 bg-neutral-300 border border-neutral-300 rounded-md text-neutral-100 placeholder-neutral-200 focus:outline-none focus:ring-2 focus:ring-primary"
                      value={newTag}
                      onChange={(e) => setNewTag(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleTagAdd()}
                      autoFocus
                    />
                    <button
                      className="px-3 py-2 bg-primary text-white rounded-md text-sm"
                      onClick={handleTagAdd}
                    >
                      Add
                    </button>
                    <button
                      className="px-3 py-2 bg-neutral-300 text-neutral-100 rounded-md text-sm"
                      onClick={() => setAddingTag(false)}
                    >
                      Cancel
                    </button>
                  </div>
                ) : null}
                
                <div className="flex flex-wrap gap-2">
                  {file.tags && file.tags.length > 0 ? (
                    file.tags.map((tag) => (
                      <div 
                        key={tag} 
                        className="px-2 py-1 bg-neutral-300 rounded-md text-sm text-neutral-100 flex items-center"
                      >
                        {tag}
                        <button
                          className="ml-1.5 text-neutral-200 hover:text-neutral-100"
                          onClick={() => handleTagDelete(tag)}
                        >
                          <FiX size={14} />
                        </button>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-neutral-200">No tags have been added yet.</p>
                  )}
                </div>
              </div>
              
              {/* Description */}
              <div>
                <h3 className="text-sm uppercase font-medium text-neutral-200 mb-3">Description</h3>
                {file.description ? (
                  <p className="text-neutral-100 text-sm">{file.description}</p>
                ) : (
                  <div className="bg-neutral-300 rounded-lg p-4 text-center">
                    <p className="text-sm text-neutral-200 mb-2">No description available</p>
                    <button className="text-primary text-sm hover:underline">Add Description</button>
                  </div>
                )}
              </div>
            </div>
          )}
          
          {currentTab === 'metadata' && (
            <div>
              <h3 className="text-sm uppercase font-medium text-neutral-200 mb-3">Technical Metadata</h3>
              <div className="bg-neutral-300 rounded-lg overflow-hidden">
                <table className="w-full text-sm">
                  <tbody className="divide-y divide-neutral-400">
                    {Object.entries(metadata).map(([key, value]) => (
                      <tr key={key} className="border-neutral-400">
                        <td className="py-2.5 px-4 text-neutral-200 font-medium">
                          {key.charAt(0).toUpperCase() + key.slice(1).replace(/([A-Z])/g, ' $1')}
                        </td>
                        <td className="py-2.5 px-4 text-neutral-100 text-right">{value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          
          {currentTab === 'history' && (
            <div>
              <h3 className="text-sm uppercase font-medium text-neutral-200 mb-3">Activity History</h3>
              <div className="space-y-3">
                {history.map((item, index) => (
                  <div 
                    key={index} 
                    className="bg-neutral-300 p-3 rounded-lg flex items-start"
                  >
                    <div className="mt-0.5 mr-3 bg-primary/10 p-2 rounded-full text-primary">
                      <FiCalendar size={16} />
                    </div>
                    <div className="flex-1">
                      <div className="font-medium text-neutral-100 text-sm">{item.action}</div>
                      <div className="flex justify-between mt-1">
                        <span className="text-xs text-neutral-200">{item.user}</span>
                        <span className="text-xs text-neutral-200">{formatDate(item.date)}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        
        {/* Action Buttons */}
        <div className="flex justify-between items-center p-4 border-t border-neutral-300">
          <div>
            <button
              className="px-3 py-2 bg-red-500 hover:bg-red-600 text-white rounded-md text-sm font-medium flex items-center transition-colors"
              onClick={() => {
                // In a real app, this would handle file deletion
                if (confirm(`Are you sure you want to delete "${file.name}"?`)) {
                  alert('File deleted');
                  onClose();
                }
              }}
            >
              <FiTrash2 size={16} className="mr-1.5" />
              Delete
            </button>
          </div>
          <div className="flex space-x-2">
            <button
              className="px-3 py-2 bg-neutral-300 hover:bg-neutral-200 text-neutral-100 rounded-md text-sm font-medium flex items-center transition-colors"
              onClick={() => {
                // In a real app, this would handle file download
                alert(`Downloading ${file.name}`);
              }}
            >
              <FiDownload size={16} className="mr-1.5" />
              Download
            </button>
            <button
              className="px-3 py-2 bg-primary hover:bg-primary-dark text-white rounded-md text-sm font-medium flex items-center transition-colors"
              onClick={() => {
                // In a real app, this would open the editor with this file
                alert(`Edit ${file.name}`);
                onClose();
              }}
            >
              <FiEdit2 size={16} className="mr-1.5" />
              Edit
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
} 
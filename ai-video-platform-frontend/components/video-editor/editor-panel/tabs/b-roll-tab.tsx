'use client';

import React, { useState } from 'react';
import { FiUpload, FiTrash2, FiPlus } from 'react-icons/fi';
import { useProcessingContext } from '../../contexts/processing-context';

interface BRollClip {
  id: string;
  url: string;
  startTime: number;
  duration: number;
}

const BRollTab: React.FC = () => {
  const { activeJob, isProcessing } = useProcessingContext();
  const [clips, setClips] = useState<BRollClip[]>([]);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    const videoFiles = files.filter(file => file.type.startsWith('video/'));

    if (videoFiles.length === 0) return;

    // TODO: Implement file upload and clip creation
    console.log('Video files dropped:', videoFiles);
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    const videoFiles = files.filter(file => file.type.startsWith('video/'));

    if (videoFiles.length === 0) return;

    // TODO: Implement file upload and clip creation
    console.log('Video files selected:', videoFiles);
  };

  const handleRemoveClip = (clipId: string) => {
    setClips(prevClips => prevClips.filter(clip => clip.id !== clipId));
  };

  return (
    <div className="space-y-4">
      <div
        className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
          isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="flex flex-col items-center space-y-2">
          <FiUpload className="w-8 h-8 text-gray-400" />
          <p className="text-sm text-gray-600">
            Drag and drop video files here or{' '}
            <label className="text-blue-500 cursor-pointer hover:text-blue-600">
              browse
              <input
                type="file"
                className="hidden"
                accept="video/*"
                multiple
                onChange={handleFileSelect}
              />
            </label>
          </p>
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <h3 className="text-sm font-medium">B-Roll Clips</h3>
          <button
            className="text-sm text-blue-500 hover:text-blue-600 flex items-center space-x-1"
            onClick={() => {/* TODO: Implement add clip */}}
          >
            <FiPlus className="w-4 h-4" />
            <span>Add Clip</span>
          </button>
        </div>

        {clips.length === 0 ? (
          <p className="text-sm text-gray-500 text-center py-4">
            No B-roll clips added yet
          </p>
        ) : (
          <div className="space-y-2">
            {clips.map(clip => (
              <div
                key={clip.id}
                className="flex items-center justify-between p-2 bg-gray-50 rounded"
              >
                <div className="flex items-center space-x-2">
                  <div className="w-16 h-9 bg-gray-200 rounded" />
                  <div>
                    <p className="text-sm font-medium">Clip {clip.id}</p>
                    <p className="text-xs text-gray-500">
                      {clip.duration.toFixed(1)}s at {clip.startTime.toFixed(1)}s
                    </p>
                  </div>
                </div>
                <button
                  className="text-red-500 hover:text-red-600"
                  onClick={() => handleRemoveClip(clip.id)}
                >
                  <FiTrash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default BRollTab; 
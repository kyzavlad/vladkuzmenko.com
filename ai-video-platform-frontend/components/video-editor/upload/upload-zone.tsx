'use client';

import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

interface UploadZoneProps {
  onUpload: (files: File[]) => void;
}

export default function UploadZone({ onUpload }: UploadZoneProps) {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    onUpload(acceptedFiles);
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mov', '.avi', '.webm']
    },
    maxSize: 1024 * 1024 * 500 // 500MB
  });

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
        isDragActive
          ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/10'
          : 'border-gray-300 dark:border-gray-700 hover:border-purple-500 dark:hover:border-purple-500'
      }`}
    >
      <input {...getInputProps()} />
      <div className="space-y-4">
        <div className="mx-auto w-16 h-16 flex items-center justify-center bg-purple-100 dark:bg-purple-900/20 rounded-full">
          <svg
            className="w-8 h-8 text-purple-600 dark:text-purple-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
        </div>
        <div>
          <p className="text-lg font-medium text-gray-900 dark:text-gray-100">
            {isDragActive ? 'Drop your videos here' : 'Drag & drop your videos here'}
          </p>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            or click to browse (max 500MB)
          </p>
        </div>
      </div>
    </div>
  );
} 
'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ClipProvider, useClipContext, GeneratedClip } from '../../../components/clip-generator/contexts/clip-context';
import { FiFilter, FiDownload, FiEdit, FiTrash, FiEye, FiShare2, FiCheckCircle, FiScissors } from 'react-icons/fi';

function ClipsContent() {
  const { generatedClips } = useClipContext();

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">Generated Clips</h1>
      
      {generatedClips.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-gray-600">No clips generated yet. Start by uploading a video and generating clips.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {generatedClips.map((clip) => (
            <div key={clip.id} className="bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden">
              <div className="aspect-video relative">
                <img
                  src={clip.thumbnail}
                  alt={clip.title}
                  className="object-cover w-full h-full"
                />
              </div>
              <div className="p-4">
                <h3 className="text-lg font-semibold mb-2">{clip.title}</h3>
                <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">
                  Duration: {clip.duration}s
                </p>
                <div className="flex justify-end space-x-2">
                  <button
                    className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
                    onClick={() => window.open(clip.videoUrl, '_blank')}
                  >
                    View
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function ClipsPage() {
  return (
    <ClipProvider>
      <ClipsContent />
    </ClipProvider>
  );
} 
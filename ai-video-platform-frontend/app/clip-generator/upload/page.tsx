'use client';

import React from 'react';
import VideoUpload from '@/components/clip-generator/upload/video-upload';
import { ClipProvider } from '@/components/clip-generator/contexts/clip-context';

export default function UploadPage() {
  return (
    <ClipProvider>
      <main className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <VideoUpload />
      </main>
    </ClipProvider>
  );
} 
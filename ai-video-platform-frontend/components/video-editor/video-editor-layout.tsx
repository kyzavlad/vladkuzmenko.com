'use client';

import React from 'react';
import { MediaProvider } from './contexts/media-context';
import { EditorProvider } from './contexts/editor-context';
import { ProcessingProvider } from './contexts/processing-context';
import MediaLibrary from './media-library/media-library';
import MediaUploader from './media-library/media-uploader';
import EditorPanel from './editor-panel/editor-panel';
import VideoPreview from './preview/video-preview';
import Toolbar from './toolbar/toolbar';
import TimelineEditor from './timeline/timeline-editor';

interface VideoEditorLayoutProps {
  children: React.ReactNode;
}

export default function VideoEditorLayout({ children }: VideoEditorLayoutProps) {
  return (
    <MediaProvider>
      <EditorProvider>
        <ProcessingProvider>
          <div className="min-h-screen bg-neutral-500 flex flex-col">
            {/* Top Toolbar */}
            <div className="border-b border-neutral-400 bg-neutral-500 h-16 flex items-center px-4">
              <Toolbar />
            </div>
            
            {/* Main Content */}
            <div className="flex-1 flex overflow-hidden">
              {/* Left Sidebar - Media Library */}
              <div className="w-72 border-r border-neutral-400 flex flex-col overflow-hidden">
                <div className="flex-1 overflow-y-auto p-4">
                  <MediaLibrary />
                </div>
              </div>
              
              {/* Main Editing Area */}
              <div className="flex-1 flex flex-col overflow-hidden">
                {/* Video Preview */}
                <div className="p-6 flex-1 flex items-center justify-center min-h-0">
                  <div className="w-full max-w-4xl">
                    <VideoPreview />
                  </div>
                </div>
                
                {/* Timeline */}
                <div className="h-48 border-t border-neutral-400 bg-neutral-400 p-4">
                  <TimelineEditor />
                </div>
              </div>
            </div>
            
            {/* Editor Panel Sidebar */}
            <EditorPanel />
            
            {/* Media Uploader (Floating Action Button) */}
            <MediaUploader />
          </div>
        </ProcessingProvider>
      </EditorProvider>
    </MediaProvider>
  );
} 
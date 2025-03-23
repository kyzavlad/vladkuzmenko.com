'use client';

import React from 'react';
import { MediaProvider } from '../../../components/video-editor/contexts/media-context';
import { EditorProvider } from '../../../components/video-editor/contexts/editor-context';
import VideoEditorLayout from '../../../components/video-editor/video-editor-layout';

export default function VideoEditorPage() {
  return (
    <MediaProvider>
      <EditorProvider>
        <VideoEditorLayout>
          <div className="flex-1 flex flex-col">
            {/* Add any additional content here */}
          </div>
        </VideoEditorLayout>
      </EditorProvider>
    </MediaProvider>
  );
} 
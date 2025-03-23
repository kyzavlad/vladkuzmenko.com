'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { 
  FiSave, 
  FiShare2, 
  FiDownload, 
  FiRotateCcw, 
  FiRotateCw,
  FiChevronLeft,
  FiMoreVertical,
  FiUsers,
  FiHelpCircle,
  FiSettings
} from 'react-icons/fi';
import { useEditorContext } from '../contexts/editor-context';
import { useMediaContext } from '../contexts/media-context';

export default function Toolbar() {
  const [showMenu, setShowMenu] = useState(false);
  
  const { activeFile, settings, savePreset } = useEditorContext();
  const { mediaFiles } = useMediaContext();
  
  const handleSaveProject = () => {
    // In a real app, this would save the project to the server
    const timestamp = new Date().toISOString().replace(/[-:.]/g, '').substring(0, 14);
    const projectName = activeFile ? `${activeFile.name.split('.')[0]}_edit_${timestamp}` : `project_${timestamp}`;
    
    alert(`Project saved as: ${projectName}`);
  };
  
  const handleExportVideo = () => {
    // In a real app, this would trigger the export process
    if (!activeFile) {
      alert('Please select a media file to export');
      return;
    }
    
    const options = {
      format: 'mp4',
      resolution: '1080p',
      includeSubtitles: settings.subtitles.enabled,
      quality: 'high'
    };
    
    alert(`Exporting video: ${activeFile.name} with settings: ${JSON.stringify(options, null, 2)}`);
  };
  
  const handleShare = () => {
    // In a real app, this would open a share dialog
    alert('Share feature would open here');
  };
  
  const handleSaveAsPreset = () => {
    const presetName = prompt('Enter a name for this preset:');
    if (presetName) {
      savePreset(presetName);
      alert(`Preset "${presetName}" saved!`);
    }
  };
  
  return (
    <div className="w-full flex items-center justify-between">
      {/* Left side - Logo & Back */}
      <div className="flex items-center space-x-4">
        <Link href="/dashboard" className="flex items-center space-x-2 text-neutral-100 hover:text-neutral-50">
          <FiChevronLeft size={20} />
          <span className="font-medium">Back to Dashboard</span>
        </Link>
        
        <div className="hidden md:block h-6 w-px bg-neutral-400 mx-2"></div>
        
        <div className="text-xl font-bold text-neutral-100">
          {activeFile ? activeFile.name : 'Video Editor'}
        </div>
      </div>
      
      {/* Right side - Actions */}
      <div className="flex items-center space-x-1 md:space-x-2">
        {/* Undo/Redo */}
        <button 
          className="p-2 text-neutral-200 hover:text-neutral-100 rounded-md hover:bg-neutral-400 disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={true} // Would be tied to actual undo stack
          title="Undo"
        >
          <FiRotateCcw size={18} />
        </button>
        
        <button 
          className="p-2 text-neutral-200 hover:text-neutral-100 rounded-md hover:bg-neutral-400 disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={true} // Would be tied to actual redo stack
          title="Redo"
        >
          <FiRotateCw size={18} />
        </button>
        
        <div className="hidden md:block h-6 w-px bg-neutral-400 mx-1"></div>
        
        {/* Save & Export Actions */}
        <button 
          className="hidden md:flex items-center space-x-1.5 p-2 text-neutral-100 hover:bg-neutral-400 rounded-md"
          onClick={handleSaveProject}
          title="Save Project"
        >
          <FiSave size={18} />
          <span className="text-sm font-medium">Save</span>
        </button>
        
        <button 
          className="hidden md:flex items-center space-x-1.5 p-2 text-neutral-100 hover:bg-neutral-400 rounded-md"
          onClick={handleSaveAsPreset}
          title="Save as Preset"
          disabled={!activeFile}
        >
          <FiSettings size={18} />
          <span className="text-sm font-medium">Save as Preset</span>
        </button>
        
        <button 
          className="hidden md:flex items-center space-x-1.5 p-2 text-neutral-100 hover:bg-neutral-400 rounded-md"
          onClick={handleExportVideo}
          disabled={!activeFile}
          title="Export Video"
        >
          <FiDownload size={18} />
          <span className="text-sm font-medium">Export</span>
        </button>
        
        <button 
          className="hidden md:flex items-center space-x-1.5 p-2 text-neutral-100 hover:bg-neutral-400 rounded-md"
          onClick={handleShare}
          disabled={!activeFile}
          title="Share"
        >
          <FiShare2 size={18} />
          <span className="text-sm font-medium">Share</span>
        </button>
        
        {/* Mobile menu button */}
        <div className="relative md:hidden">
          <button 
            className="p-2 text-neutral-100 hover:bg-neutral-400 rounded-md"
            onClick={() => setShowMenu(!showMenu)}
          >
            <FiMoreVertical size={20} />
          </button>
          
          {/* Mobile dropdown menu */}
          {showMenu && (
            <div className="absolute right-0 top-full mt-1 bg-neutral-400 rounded-md shadow-elevation-2 py-1 w-48">
              <button 
                className="w-full flex items-center space-x-2 px-4 py-2 text-neutral-100 hover:bg-neutral-300 text-left"
                onClick={() => {
                  handleSaveProject();
                  setShowMenu(false);
                }}
              >
                <FiSave size={16} />
                <span>Save Project</span>
              </button>
              
              <button 
                className="w-full flex items-center space-x-2 px-4 py-2 text-neutral-100 hover:bg-neutral-300 text-left"
                onClick={() => {
                  handleSaveAsPreset();
                  setShowMenu(false);
                }}
                disabled={!activeFile}
              >
                <FiSettings size={16} />
                <span>Save as Preset</span>
              </button>
              
              <button 
                className="w-full flex items-center space-x-2 px-4 py-2 text-neutral-100 hover:bg-neutral-300 text-left"
                onClick={() => {
                  handleExportVideo();
                  setShowMenu(false);
                }}
                disabled={!activeFile}
              >
                <FiDownload size={16} />
                <span>Export Video</span>
              </button>
              
              <button 
                className="w-full flex items-center space-x-2 px-4 py-2 text-neutral-100 hover:bg-neutral-300 text-left"
                onClick={() => {
                  handleShare();
                  setShowMenu(false);
                }}
                disabled={!activeFile}
              >
                <FiShare2 size={16} />
                <span>Share</span>
              </button>
              
              <div className="h-px bg-neutral-300 my-1"></div>
              
              <button className="w-full flex items-center space-x-2 px-4 py-2 text-neutral-100 hover:bg-neutral-300 text-left">
                <FiHelpCircle size={16} />
                <span>Help</span>
              </button>
            </div>
          )}
        </div>
        
        {/* Additional actions on desktop */}
        <div className="hidden md:block h-6 w-px bg-neutral-400 mx-1"></div>
        
        <button className="p-2 text-neutral-200 hover:text-neutral-100 rounded-md hover:bg-neutral-400">
          <FiHelpCircle size={18} />
        </button>
      </div>
    </div>
  );
} 
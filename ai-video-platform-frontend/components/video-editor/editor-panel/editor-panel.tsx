'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  FiSettings, 
  FiSliders, 
  FiFileText, 
  FiMusic, 
  FiVideo,
  FiClock,
  FiVolume2,
  FiZap,
  FiSave,
  FiChevronRight,
  FiChevronLeft,
  FiShare2,
  FiEye,
  FiEyeOff,
  FiCheckSquare,
  FiHelpCircle
} from 'react-icons/fi';
import { useEditorContext } from '../contexts/editor-context';
import { useProcessingContext } from '../contexts/processing-context';
import SubtitlesTab from './tabs/subtitles-tab';
import BRollTab from './tabs/b-roll-tab';
import AudioTab from './tabs/audio-tab';
import EnhancementsTab from './tabs/enhancements-tab';
import PresetSelector from './presets/preset-selector';
import ProcessingModal from './processing/processing-modal';

export default function EditorPanel() {
  const [currentTab, setCurrentTab] = useState<'subtitles'|'b-roll'|'audio'|'enhancements'>('subtitles');
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [showHelpTooltip, setShowHelpTooltip] = useState(false);
  const [showProcessingModal, setShowProcessingModal] = useState(false);
  
  const { 
    activeFile, 
    settings, 
    updateSettings, 
    resetSettings, 
    beforeAfterMode,
    toggleBeforeAfterMode,
    availablePresets: presets,
    loadPreset
  } = useEditorContext();
  
  const {
    isProcessing,
    activeJob,
    startProcessing,
    cancelProcessing
  } = useProcessingContext();
  
  const handleStartProcessing = async () => {
    if (!activeFile) return;
    
    await startProcessing(activeFile, settings);
    setShowProcessingModal(true);
  };
  
  const handleCloseProcessingModal = () => {
    setShowProcessingModal(false);
  };
  
  const renderTabContent = () => {
    switch (currentTab) {
      case 'subtitles':
        return <SubtitlesTab />;
      case 'b-roll':
        return <BRollTab />;
      case 'audio':
        return <AudioTab />;
      case 'enhancements':
        return <EnhancementsTab />;
      default:
        return <SubtitlesTab />;
    }
  };
  
  if (isCollapsed) {
    return (
      <motion.div
        className="fixed right-0 top-20 bottom-20 w-12 bg-neutral-400 rounded-l-xl shadow-elevation-2 flex flex-col items-center py-4 z-10"
        initial={{ x: 0 }}
        animate={{ x: 0 }}
        transition={{ duration: 0.3 }}
      >
        <button
          className="p-2 text-neutral-100 hover:text-primary rounded-md mb-8"
          onClick={() => setIsCollapsed(false)}
          title="Expand"
        >
          <FiChevronLeft size={20} />
        </button>
        
        <div className="space-y-4 flex-1 flex flex-col items-center">
          <button
            className={`p-2 rounded-md transition-colors ${currentTab === 'subtitles' ? 'text-primary bg-neutral-300' : 'text-neutral-100 hover:text-primary'}`}
            onClick={() => {
              setCurrentTab('subtitles');
              setIsCollapsed(false);
            }}
            title="Subtitles"
          >
            <FiFileText size={20} />
          </button>
          <button
            className={`p-2 rounded-md transition-colors ${currentTab === 'b-roll' ? 'text-primary bg-neutral-300' : 'text-neutral-100 hover:text-primary'}`}
            onClick={() => {
              setCurrentTab('b-roll');
              setIsCollapsed(false);
            }}
            title="B-Roll"
          >
            <FiVideo size={20} />
          </button>
          <button
            className={`p-2 rounded-md transition-colors ${currentTab === 'audio' ? 'text-primary bg-neutral-300' : 'text-neutral-100 hover:text-primary'}`}
            onClick={() => {
              setCurrentTab('audio');
              setIsCollapsed(false);
            }}
            title="Audio"
          >
            <FiMusic size={20} />
          </button>
          <button
            className={`p-2 rounded-md transition-colors ${currentTab === 'enhancements' ? 'text-primary bg-neutral-300' : 'text-neutral-100 hover:text-primary'}`}
            onClick={() => {
              setCurrentTab('enhancements');
              setIsCollapsed(false);
            }}
            title="Enhancements"
          >
            <FiZap size={20} />
          </button>
        </div>
        
        <button
          className="p-2 mt-auto text-neutral-100 hover:text-primary rounded-md"
          onClick={handleStartProcessing}
          disabled={!activeFile || isProcessing}
          title="Process Video"
        >
          <FiZap size={20} />
        </button>
      </motion.div>
    );
  }
  
  return (
    <>
      <motion.div
        className="fixed right-0 top-20 bottom-20 w-80 bg-neutral-400 rounded-l-xl shadow-elevation-2 flex flex-col z-10"
        initial={{ x: 320 }}
        animate={{ x: 0 }}
        transition={{ duration: 0.3 }}
      >
        {/* Header */}
        <div className="p-4 border-b border-neutral-300 flex justify-between items-center">
          <h2 className="text-lg font-bold text-neutral-100">Editor Settings</h2>
          <div className="flex space-x-2">
            <button
              className={`p-1.5 rounded-md text-neutral-100 hover:text-primary hover:bg-neutral-300 transition-colors ${beforeAfterMode ? 'bg-neutral-300 text-primary' : ''}`}
              onClick={toggleBeforeAfterMode}
              title="Before/After Toggle"
            >
              {beforeAfterMode ? <FiEye size={18} /> : <FiEyeOff size={18} />}
            </button>
            <button
              className="p-1.5 text-neutral-100 hover:text-primary hover:bg-neutral-300 rounded-md transition-colors"
              onClick={() => setIsCollapsed(true)}
              title="Collapse"
            >
              <FiChevronRight size={18} />
            </button>
          </div>
        </div>
        
        {/* Preset Selector */}
        <div className="px-4 py-3 border-b border-neutral-300">
          <PresetSelector />
        </div>
        
        {/* Tabs */}
        <div className="flex border-b border-neutral-300">
          <button
            className={`flex-1 px-3 py-2.5 font-medium text-xs transition-colors flex items-center justify-center space-x-1 ${
              currentTab === 'subtitles' 
                ? 'text-primary border-b-2 border-primary' 
                : 'text-neutral-200 hover:text-neutral-100'
            }`}
            onClick={() => setCurrentTab('subtitles')}
          >
            <FiFileText size={14} />
            <span>Subtitles</span>
          </button>
          <button
            className={`flex-1 px-3 py-2.5 font-medium text-xs transition-colors flex items-center justify-center space-x-1 ${
              currentTab === 'b-roll' 
                ? 'text-primary border-b-2 border-primary' 
                : 'text-neutral-200 hover:text-neutral-100'
            }`}
            onClick={() => setCurrentTab('b-roll')}
          >
            <FiVideo size={14} />
            <span>B-Roll</span>
          </button>
          <button
            className={`flex-1 px-3 py-2.5 font-medium text-xs transition-colors flex items-center justify-center space-x-1 ${
              currentTab === 'audio' 
                ? 'text-primary border-b-2 border-primary' 
                : 'text-neutral-200 hover:text-neutral-100'
            }`}
            onClick={() => setCurrentTab('audio')}
          >
            <FiMusic size={14} />
            <span>Audio</span>
          </button>
          <button
            className={`flex-1 px-3 py-2.5 font-medium text-xs transition-colors flex items-center justify-center space-x-1 ${
              currentTab === 'enhancements' 
                ? 'text-primary border-b-2 border-primary' 
                : 'text-neutral-200 hover:text-neutral-100'
            }`}
            onClick={() => setCurrentTab('enhancements')}
          >
            <FiZap size={14} />
            <span>Enhance</span>
          </button>
        </div>
        
        {/* Tab Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {activeFile ? (
            renderTabContent()
          ) : (
            <div className="h-full flex items-center justify-center text-center">
              <div className="max-w-sm p-4">
                <p className="text-neutral-200 mb-2">
                  No media file selected.
                </p>
                <p className="text-sm text-neutral-200">
                  Select a file from your media library to start editing.
                </p>
              </div>
            </div>
          )}
        </div>
        
        {/* Footer */}
        <div className="p-4 border-t border-neutral-300">
          <button
            className="w-full py-2 px-4 bg-primary text-white rounded-md font-medium hover:bg-primary-dark transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
            onClick={handleStartProcessing}
            disabled={!activeFile || isProcessing}
          >
            <FiZap size={18} />
            <span>{isProcessing ? 'Processing...' : 'Process Video'}</span>
          </button>
        </div>
      </motion.div>
      
      {showProcessingModal && (
        <ProcessingModal onClose={handleCloseProcessingModal} />
      )}
    </>
  );
} 
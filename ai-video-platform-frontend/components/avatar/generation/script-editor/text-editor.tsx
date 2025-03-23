'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  FiAlignLeft, 
  FiPlus, 
  FiMic, 
  FiRefreshCw, 
  FiSave, 
  FiTrash2, 
  FiClock,
  FiType,
  FiFileText,
  FiInfo,
  FiAlertCircle,
  FiSmile,
  FiFrown,
  FiThumbsUp
} from 'react-icons/fi';
import { useGenerationContext, ScriptSegment } from '../../contexts/generation-context';

type Emotion = 'neutral' | 'happy' | 'sad' | 'angry' | 'surprised' | 'thoughtful';

interface EmotionTag {
  emotion: Emotion;
  label: string;
  icon: React.ReactNode;
}

export default function TextEditor() {
  const { 
    activeJob,
    updateScript,
    addScriptSegment,
    removeScriptSegment,
    loadScriptTemplate,
    generateScript,
    isProcessing
  } = useGenerationContext();
  
  const [activeSegmentIndex, setActiveSegmentIndex] = useState<number | null>(null);
  const [showTemplates, setShowTemplates] = useState(false);
  const [isGeneratingScript, setIsGeneratingScript] = useState(false);
  const [scriptPrompt, setScriptPrompt] = useState('');
  const [wordCount, setWordCount] = useState<{[key: number]: number}>({});
  const [duration, setDuration] = useState<{[key: number]: number}>({});
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  
  const EMOTIONS: EmotionTag[] = [
    { emotion: 'neutral', label: 'Neutral', icon: <FiSmile /> },
    { emotion: 'happy', label: 'Happy', icon: <FiSmile /> },
    { emotion: 'sad', label: 'Sad', icon: <FiFrown /> },
    { emotion: 'angry', label: 'Angry', icon: <FiAlertCircle /> },
    { emotion: 'surprised', label: 'Surprised', icon: <FiAlertCircle /> },
    { emotion: 'thoughtful', label: 'Thoughtful', icon: <FiThumbsUp /> }
  ];
  
  const scriptTemplates = [
    { id: 'introduction', name: 'Introduction', description: 'A brief introduction of yourself' },
    { id: 'product_demo', name: 'Product Demo', description: 'Demonstration of a product or feature' },
    { id: 'tutorial', name: 'Tutorial', description: 'Step-by-step instructions' },
    { id: 'announcement', name: 'Announcement', description: 'Announcing news or updates' },
    { id: 'testimonial', name: 'Testimonial', description: 'Sharing feedback or experience' },
    { id: 'explainer', name: 'Explainer', description: 'Explaining a concept or process' },
  ];
  
  useEffect(() => {
    // Calculate word count and duration for each segment
    if (activeJob?.script.segments) {
      const newWordCount: {[key: number]: number} = {};
      const newDuration: {[key: number]: number} = {};
      
      activeJob.script.segments.forEach((segment, index) => {
        const text = segment.text || '';
        const count = text.trim().split(/\s+/).filter(Boolean).length;
        newWordCount[index] = count;
        
        // Estimate duration (average speaking rate: ~150 words per minute)
        const durationInSeconds = Math.max(Math.round(count / 2.5), 1);
        newDuration[index] = durationInSeconds;
      });
      
      setWordCount(newWordCount);
      setDuration(newDuration);
    }
  }, [activeJob?.script.segments]);
  
  const handleSegmentUpdate = (index: number, updatedSegment: Partial<ScriptSegment>) => {
    if (!activeJob?.script) return;
    
    const updatedSegments = [...activeJob.script.segments];
    updatedSegments[index] = {
      ...updatedSegments[index],
      ...updatedSegment
    };
    updateScript(updatedSegments);
  };
  
  const handleEmotionChange = (index: number, emotion: Emotion) => {
    handleSegmentUpdate(index, { emotion });
  };
  
  const handleAddSegment = () => {
    if (!activeJob?.script) return;
    
    const newSegment: ScriptSegment = {
      id: `segment-${Date.now()}`,
      text: '',
      emotion: 'neutral',
      emphasis: false
    };
    
    const updatedSegments = [...activeJob.script.segments, newSegment];
    updateScript(updatedSegments);
    
    // Focus on the new segment after it's added
    setTimeout(() => {
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    }, 0);
  };
  
  const handleRemoveSegment = (index: number) => {
    if (!activeJob || activeJob.script.segments.length <= 1) return;
    
    if (window.confirm('Are you sure you want to remove this segment?')) {
      removeScriptSegment(index);
      setActiveSegmentIndex(null);
    }
  };
  
  const handleTemplateSelect = (templateId: string) => {
    loadScriptTemplate(templateId);
    setShowTemplates(false);
    setActiveSegmentIndex(0);
  };
  
  const handleGenerateScript = async () => {
    if (!scriptPrompt.trim()) return;
    
    setIsGeneratingScript(true);
    try {
      await generateScript(scriptPrompt);
      setScriptPrompt('');
      setActiveSegmentIndex(0);
    } catch (error) {
      console.error('Error generating script:', error);
    } finally {
      setIsGeneratingScript(false);
    }
  };
  
  const getTotalWordCount = () => {
    return Object.values(wordCount).reduce((total, count) => total + count, 0);
  };
  
  const getTotalDuration = () => {
    return Object.values(duration).reduce((total, seconds) => total + seconds, 0);
  };
  
  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };
  
  const renderSegmentEditor = () => {
    if (!activeJob) return null;
    
    return (
      <div className="space-y-4">
        {activeJob.script.segments.map((segment, index) => (
          <div 
            key={index}
            className={`
              bg-neutral-400 rounded-lg p-4 border-l-4 cursor-pointer transition-all
              ${activeSegmentIndex === index ? 'border-primary' : 'border-transparent hover:border-neutral-300'}
            `}
            onClick={() => setActiveSegmentIndex(index)}
          >
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-neutral-100 font-medium">
                Segment {index + 1}
              </h3>
              
              <div className="flex items-center space-x-2 text-xs text-neutral-200">
                <div className="flex items-center">
                  <FiType className="mr-1" size={12} />
                  <span>{wordCount[index] || 0} words</span>
                </div>
                <div className="flex items-center">
                  <FiClock className="mr-1" size={12} />
                  <span>{formatTime(duration[index] || 0)}</span>
                </div>
                
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleRemoveSegment(index);
                  }}
                  className="p-1 text-neutral-200 hover:text-red-500 transition-colors"
                  disabled={activeJob.script.segments.length <= 1}
                  title="Remove segment"
                >
                  <FiTrash2 size={14} />
                </button>
              </div>
            </div>
            
            <div>
              <textarea
                ref={activeSegmentIndex === index ? textareaRef : undefined}
                value={segment.text || ''}
                onChange={(e) => handleSegmentUpdate(index, { text: e.target.value })}
                placeholder="Enter the script text for this segment..."
                className="w-full bg-neutral-500 text-neutral-100 rounded-lg p-3 min-h-[100px] focus:outline-none focus:ring-1 focus:ring-primary resize-y"
                onClick={(e) => {
                  e.stopPropagation();
                  setActiveSegmentIndex(index);
                }}
              />
            </div>
            
            <div className="mt-3">
              <label className="block text-sm text-neutral-200 mb-2">Emotion:</label>
              <div className="flex flex-wrap gap-2">
                {EMOTIONS.map(({ emotion, label, icon }) => (
                  <button
                    key={emotion}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleEmotionChange(index, emotion);
                    }}
                    className={`
                      px-3 py-1.5 rounded-full text-xs flex items-center
                      ${segment.emotion === emotion ? 'bg-primary text-white' : 'bg-neutral-300 text-neutral-100 hover:bg-neutral-200'}
                    `}
                  >
                    <span className="mr-1">{icon}</span>
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ))}
        
        <div className="flex justify-center">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleAddSegment}
            className="flex items-center bg-neutral-400 hover:bg-neutral-300 text-neutral-100 px-4 py-2 rounded-lg"
          >
            <FiPlus className="mr-2" size={16} />
            Add Segment
          </motion.button>
        </div>
      </div>
    );
  };
  
  const renderTemplateSelector = () => (
    <div className="bg-neutral-400 rounded-lg p-4">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-neutral-100 font-medium">Script Templates</h3>
        <button
          onClick={() => setShowTemplates(false)}
          className="text-neutral-200 hover:text-neutral-100"
        >
          &times;
        </button>
      </div>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {scriptTemplates.map(template => (
          <div
            key={template.id}
            onClick={() => handleTemplateSelect(template.id)}
            className="bg-neutral-300 hover:bg-neutral-200 rounded-lg p-3 cursor-pointer transition-colors"
          >
            <h4 className="text-neutral-100 font-medium mb-1">{template.name}</h4>
            <p className="text-neutral-200 text-xs">{template.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
  
  const renderAIScriptGenerator = () => (
    <div className="bg-neutral-400 rounded-lg p-4">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-neutral-100 font-medium">AI Script Generator</h3>
        {isGeneratingScript && (
          <div className="flex items-center text-primary text-sm">
            <motion.span
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              className="mr-2"
            >
              <FiRefreshCw size={14} />
            </motion.span>
            Generating...
          </div>
        )}
      </div>
      
      <textarea
        value={scriptPrompt}
        onChange={(e) => setScriptPrompt(e.target.value)}
        placeholder="Describe what kind of script you want to generate..."
        className="w-full bg-neutral-500 text-neutral-100 rounded-lg p-3 min-h-[100px] focus:outline-none focus:ring-1 focus:ring-primary resize-y mb-3"
      />
      
      <div className="flex space-x-2">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={handleGenerateScript}
          disabled={isGeneratingScript || isProcessing || !scriptPrompt.trim()}
          className={`
            flex-1 flex items-center justify-center py-2 rounded-lg text-sm
            ${isGeneratingScript || isProcessing || !scriptPrompt.trim()
              ? 'bg-neutral-300 text-neutral-200 cursor-not-allowed'
              : 'bg-primary text-white hover:bg-primary-dark'}
          `}
        >
          <FiRefreshCw className="mr-2" size={14} />
          Generate Script
        </motion.button>
        
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setScriptPrompt('')}
          disabled={!scriptPrompt.trim()}
          className={`
            flex items-center py-2 px-3 rounded-lg text-sm
            ${!scriptPrompt.trim()
              ? 'bg-neutral-300 text-neutral-200 cursor-not-allowed'
              : 'bg-neutral-300 text-neutral-100 hover:bg-neutral-200'}
          `}
        >
          Clear
        </motion.button>
      </div>
      
      <div className="mt-3 text-xs text-neutral-200">
        <p className="flex items-start">
          <FiInfo className="mt-0.5 mr-1 flex-shrink-0" size={12} />
          Describe your script needs in detail for better results. Include information about tone, purpose, audience, and key points.
        </p>
      </div>
    </div>
  );
  
  if (!activeJob) {
    return (
      <div className="bg-neutral-500 rounded-lg p-4 text-center">
        <FiAlertCircle size={24} className="mx-auto mb-2 text-neutral-200" />
        <p className="text-neutral-200">No active generation job. Please start a new video generation.</p>
      </div>
    );
  }
  
  return (
    <div className="bg-neutral-500 rounded-lg overflow-hidden">
      <div className="p-4 border-b border-neutral-400">
        <h2 className="text-lg font-medium text-neutral-100">Script Editor</h2>
        <p className="text-sm text-neutral-200">
          Write and organize the script for your avatar to speak
        </p>
      </div>
      
      <div className="p-4">
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center">
            <FiAlignLeft className="text-primary mr-2" size={18} />
            <h3 className="text-md font-medium text-neutral-100">Script Content</h3>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3 text-sm text-neutral-200">
              <div className="flex items-center">
                <FiFileText className="mr-1" size={14} />
                <span>{getTotalWordCount()} words</span>
              </div>
              <div className="flex items-center">
                <FiClock className="mr-1" size={14} />
                <span>{formatTime(getTotalDuration())}</span>
              </div>
            </div>
            
            <div className="flex space-x-2">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setShowTemplates(true)}
                className="flex items-center bg-neutral-400 hover:bg-neutral-300 text-neutral-100 px-3 py-1.5 rounded-lg text-sm"
              >
                <FiFileText className="mr-1.5" size={14} />
                Templates
              </motion.button>
              
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setShowTemplates(false)}
                className="flex items-center bg-neutral-400 hover:bg-neutral-300 text-neutral-100 px-3 py-1.5 rounded-lg text-sm"
              >
                <FiMic className="mr-1.5" size={14} />
                AI Generate
              </motion.button>
            </div>
          </div>
        </div>
        
        {showTemplates ? renderTemplateSelector() : null}
        {!showTemplates && activeSegmentIndex === null ? renderAIScriptGenerator() : null}
        
        {!showTemplates && renderSegmentEditor()}
        
        <div className="mt-6 flex justify-end">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="flex items-center bg-primary text-white px-4 py-2 rounded-lg"
          >
            <FiSave className="mr-2" size={16} />
            Save Script
          </motion.button>
        </div>
      </div>
    </div>
  );
} 
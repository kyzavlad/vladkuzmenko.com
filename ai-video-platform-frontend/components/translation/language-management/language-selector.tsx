'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FiCheck, FiChevronDown, FiX, FiPlus, FiLoader, FiGlobe, FiSettings } from 'react-icons/fi';
import { useTranslationContext } from '../contexts/translation-context';

type LanguageSelectorProps = {
  videoId?: string;
};

export const LanguageSelector: React.FC<LanguageSelectorProps> = ({ videoId }) => {
  const {
    availableLanguages,
    selectedSourceLanguage,
    selectedTargetLanguages,
    detectedLanguage,
    isDetectingLanguage,
    detectSourceLanguage,
    setSourceLanguage,
    addTargetLanguage,
    removeTargetLanguage,
    clearTargetLanguages,
    error
  } = useTranslationContext();

  const [sourceDropdownOpen, setSourceDropdownOpen] = useState(false);
  const [targetDropdownOpen, setTargetDropdownOpen] = useState(false);
  const [targetSearch, setTargetSearch] = useState('');
  const [showDetectLanguageButton, setShowDetectLanguageButton] = useState(!!videoId);

  // Filter languages based on search
  const filteredTargetLanguages = availableLanguages.filter(lang => 
    lang.name.toLowerCase().includes(targetSearch.toLowerCase()) ||
    lang.nativeName.toLowerCase().includes(targetSearch.toLowerCase()) ||
    lang.id.toLowerCase().includes(targetSearch.toLowerCase())
  );
  
  // Get source language object
  const sourceLanguage = selectedSourceLanguage 
    ? availableLanguages.find(lang => lang.id === selectedSourceLanguage)
    : null;
  
  // Get target language objects
  const targetLanguages = selectedTargetLanguages
    .map(id => availableLanguages.find(lang => lang.id === id))
    .filter(Boolean) as typeof availableLanguages;
  
  const handleDetectLanguage = async () => {
    if (videoId) {
      try {
        await detectSourceLanguage(videoId);
      } catch (err) {
        console.error('Error detecting language:', err);
      }
    }
  };

  const toggleSourceDropdown = () => {
    setSourceDropdownOpen(!sourceDropdownOpen);
    if (targetDropdownOpen) setTargetDropdownOpen(false);
  };

  const toggleTargetDropdown = () => {
    setTargetDropdownOpen(!targetDropdownOpen);
    if (sourceDropdownOpen) setSourceDropdownOpen(false);
  };

  const handleSourceSelect = (langId: string) => {
    setSourceLanguage(langId);
    setSourceDropdownOpen(false);
  };

  const handleTargetSelect = (langId: string) => {
    if (!selectedTargetLanguages.includes(langId) && langId !== selectedSourceLanguage) {
      addTargetLanguage(langId);
    }
  };

  const handleTargetRemove = (langId: string) => {
    removeTargetLanguage(langId);
  };

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = () => {
      if (sourceDropdownOpen) setSourceDropdownOpen(false);
      if (targetDropdownOpen) setTargetDropdownOpen(false);
    };

    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, [sourceDropdownOpen, targetDropdownOpen]);

  return (
    <div className="w-full bg-white rounded-lg shadow-md p-6 mb-6">
      <h2 className="text-2xl font-bold mb-4">Language Management</h2>
      
      {error && (
        <div className="mb-4 p-3 bg-red-50 text-red-700 rounded-md">
          {error}
        </div>
      )}
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Source Language Section */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Source Language
          </label>
          
          <div className="relative">
            <div 
              className="flex items-center justify-between p-3 border border-gray-300 rounded-md cursor-pointer hover:border-blue-500"
              onClick={(e) => {
                e.stopPropagation();
                toggleSourceDropdown();
              }}
            >
              {sourceLanguage ? (
                <div className="flex items-center space-x-2">
                  <span className="text-xl">{sourceLanguage.flag}</span>
                  <span>{sourceLanguage.name}</span>
                  <span className="text-gray-500 text-sm">({sourceLanguage.nativeName})</span>
                </div>
              ) : (
                <span className="text-gray-500">Select source language</span>
              )}
              <FiChevronDown className={`transition-transform ${sourceDropdownOpen ? 'rotate-180' : ''}`} />
            </div>
            
            {showDetectLanguageButton && (
              <button
                className="mt-2 py-1 px-3 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-md inline-flex items-center space-x-1 text-sm transition-colors"
                onClick={handleDetectLanguage}
                disabled={isDetectingLanguage}
              >
                {isDetectingLanguage ? (
                  <>
                    <FiLoader className="animate-spin" />
                    <span>Detecting language...</span>
                  </>
                ) : (
                  <>
                    <FiGlobe />
                    <span>Auto-detect from video</span>
                  </>
                )}
              </button>
            )}
            
            {detectedLanguage && !selectedSourceLanguage && (
              <div className="mt-2 text-sm text-gray-600">
                <span>Detected language: </span>
                <button 
                  className="text-blue-500 hover:underline"
                  onClick={() => setSourceLanguage(detectedLanguage)}
                >
                  {availableLanguages.find(l => l.id === detectedLanguage)?.name || detectedLanguage}
                </button>
              </div>
            )}
            
            {sourceDropdownOpen && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute z-10 mt-1 w-full bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-y-auto"
                onClick={(e) => e.stopPropagation()}
              >
                <ul>
                  {availableLanguages.map(lang => (
                    <li 
                      key={lang.id}
                      className={`
                        p-3 hover:bg-gray-50 cursor-pointer flex items-center justify-between
                        ${selectedSourceLanguage === lang.id ? 'bg-blue-50' : ''}
                      `}
                      onClick={() => handleSourceSelect(lang.id)}
                    >
                      <div className="flex items-center space-x-2">
                        <span className="text-xl">{lang.flag}</span>
                        <span>{lang.name}</span>
                        <span className="text-gray-500 text-sm">({lang.nativeName})</span>
                      </div>
                      {selectedSourceLanguage === lang.id && <FiCheck className="text-blue-500" />}
                    </li>
                  ))}
                </ul>
              </motion.div>
            )}
          </div>
        </div>
        
        {/* Target Languages Section */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Target Languages
          </label>
          
          <div className="space-y-3">
            {/* Selected target languages list */}
            <div className="flex flex-wrap gap-2">
              {targetLanguages.length === 0 ? (
                <span className="text-gray-500 italic">No target languages selected</span>
              ) : (
                targetLanguages.map(lang => (
                  <div 
                    key={lang.id}
                    className="flex items-center space-x-1 bg-blue-50 text-blue-700 px-2 py-1 rounded-md"
                  >
                    <span className="text-lg">{lang.flag}</span>
                    <span>{lang.name}</span>
                    <button 
                      className="ml-1 text-blue-500 hover:text-blue-700"
                      onClick={() => handleTargetRemove(lang.id)}
                    >
                      <FiX />
                    </button>
                  </div>
                ))
              )}
            </div>
            
            {/* Add target language button */}
            <div className="relative">
              <button
                className="flex items-center space-x-1 py-2 px-3 border border-gray-300 rounded-md hover:border-blue-500 w-full justify-between"
                onClick={(e) => {
                  e.stopPropagation();
                  toggleTargetDropdown();
                }}
              >
                <div className="flex items-center space-x-1">
                  <FiPlus />
                  <span>Add language</span>
                </div>
                <FiChevronDown className={`transition-transform ${targetDropdownOpen ? 'rotate-180' : ''}`} />
              </button>
              
              {targetDropdownOpen && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="absolute z-10 mt-1 w-full bg-white border border-gray-300 rounded-md shadow-lg"
                  onClick={(e) => e.stopPropagation()}
                >
                  <div className="p-2">
                    <input
                      type="text"
                      placeholder="Search languages..."
                      className="w-full p-2 border border-gray-300 rounded-md"
                      value={targetSearch}
                      onChange={(e) => setTargetSearch(e.target.value)}
                    />
                  </div>
                  <ul className="max-h-60 overflow-y-auto">
                    {filteredTargetLanguages.map(lang => {
                      const isSelected = selectedTargetLanguages.includes(lang.id);
                      const isSourceLanguage = selectedSourceLanguage === lang.id;
                      return (
                        <li 
                          key={lang.id}
                          className={`
                            p-3 flex items-center justify-between
                            ${isSelected ? 'bg-blue-50 text-blue-700' : 'hover:bg-gray-50'}
                            ${isSourceLanguage ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                          `}
                          onClick={() => !isSourceLanguage && handleTargetSelect(lang.id)}
                        >
                          <div className="flex items-center space-x-2">
                            <span className="text-xl">{lang.flag}</span>
                            <span>{lang.name}</span>
                            <span className="text-gray-500 text-sm">({lang.nativeName})</span>
                          </div>
                          {isSelected && <FiCheck className="text-blue-500" />}
                          {isSourceLanguage && <span className="text-xs text-gray-500">Source language</span>}
                        </li>
                      );
                    })}
                    {filteredTargetLanguages.length === 0 && (
                      <li className="p-3 text-gray-500 text-center">No languages match your search</li>
                    )}
                  </ul>
                </motion.div>
              )}
            </div>
            
            {/* Clear all button */}
            {targetLanguages.length > 0 && (
              <button
                className="text-sm text-gray-500 hover:text-gray-700"
                onClick={clearTargetLanguages}
              >
                Clear all target languages
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default LanguageSelector; 
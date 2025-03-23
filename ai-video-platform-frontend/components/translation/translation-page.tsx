'use client';

import React, { useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { TranslationProvider } from './contexts/translation-context';
import LanguageSelector from './language-management/language-selector';
import TranslationSettings from './language-management/translation-settings';
import TranslationPreview from './preview/translation-preview';

interface TranslationPageProps {
  videoId?: string;
}

export const TranslationPage: React.FC<TranslationPageProps> = ({ videoId: propVideoId }) => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [videoId, setVideoId] = useState<string | undefined>(propVideoId);
  const [jobId, setJobId] = useState<string | null>(null);
  const [selectedLanguage, setSelectedLanguage] = useState<string | null>(null);
  const [step, setStep] = useState<'language-selection' | 'settings' | 'processing' | 'preview'>('language-selection');
  const [customOptions, setCustomOptions] = useState({});
  
  // Get video ID and job ID from URL if not provided as props
  useEffect(() => {
    const videoIdFromUrl = searchParams.get('videoId');
    const jobIdFromUrl = searchParams.get('jobId');
    const languageFromUrl = searchParams.get('language');
    
    if (!videoId && videoIdFromUrl) {
      setVideoId(videoIdFromUrl);
    }
    
    if (jobIdFromUrl) {
      setJobId(jobIdFromUrl);
      setStep('preview');
    }
    
    if (languageFromUrl) {
      setSelectedLanguage(languageFromUrl);
    }
  }, [searchParams, videoId]);
  
  const handleSettingsComplete = () => {
    setStep('processing');
  };
  
  const handleLanguagesSelected = () => {
    setStep('settings');
  };
  
  const handleCreateJob = (newJobId: string) => {
    setJobId(newJobId);
    // Update URL with job ID
    router.push(`/translation?jobId=${newJobId}`);
    setStep('preview');
  };
  
  const handleLanguageSelect = (languageId: string) => {
    setSelectedLanguage(languageId);
    // Update URL with language
    if (jobId) {
      router.push(`/translation?jobId=${jobId}&language=${languageId}`);
    }
  };
  
  const handleSettingsChange = (options: any) => {
    setCustomOptions(options);
  };

  return (
    <TranslationProvider>
      <div className="container mx-auto py-8 px-4">
        <h1 className="text-3xl font-bold mb-6">Video Translation</h1>
        
        {!videoId && !jobId && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <p className="text-gray-700 text-center">Please select a video to translate.</p>
          </div>
        )}
        
        {step === 'language-selection' && videoId && (
          <>
            <LanguageSelector videoId={videoId} />
            <div className="flex justify-end">
              <button
                className="py-2 px-6 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                onClick={handleLanguagesSelected}
              >
                Continue to Settings
              </button>
            </div>
          </>
        )}
        
        {step === 'settings' && (
          <TranslationSettings 
            onComplete={handleSettingsComplete}
            customOptions={customOptions}
            onChange={handleSettingsChange}
          />
        )}
        
        {step === 'processing' && videoId && (
          <div className="w-full bg-white rounded-lg shadow-md p-6 mb-6">
            <div className="flex flex-col items-center py-12">
              <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
              <h3 className="text-xl font-semibold mb-2">Processing Video Translation</h3>
              <p className="text-gray-700 mb-4">
                We're translating your video. This may take several minutes depending on the length of your video.
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Don&apos;t worry if you&apos;re not sure what to say - we&apos;ll guide you through the process
              </p>
            </div>
          </div>
        )}
        
        {step === 'preview' && jobId && selectedLanguage && (
          <>
            <div className="mb-6">
              <div className="flex justify-between items-center">
                <div>
                  <h2 className="text-xl font-bold">Language Selection</h2>
                  <p className="text-gray-600 text-sm">Select a language to preview the translation</p>
                </div>
                <div className="flex space-x-2">
                  <select
                    className="p-2 border border-gray-300 rounded-md"
                    value={selectedLanguage || ''}
                    onChange={(e) => handleLanguageSelect(e.target.value)}
                  >
                    <option value="" disabled>Select a language</option>
                    <option value="es">Spanish (Español)</option>
                    <option value="fr">French (Français)</option>
                    <option value="de">German (Deutsch)</option>
                    <option value="ja">Japanese (日本語)</option>
                    <option value="zh">Chinese (中文)</option>
                  </select>
                </div>
              </div>
            </div>
            <TranslationPreview jobId={jobId} languageId={selectedLanguage} />
          </>
        )}
      </div>
    </TranslationProvider>
  );
};

export default TranslationPage; 
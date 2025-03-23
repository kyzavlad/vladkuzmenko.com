import React, { useState } from 'react';
import { ChakraProvider, Container, VStack, useToast } from '@chakra-ui/react';
import { VideoUpload } from './components/VideoUpload';
import { ConfigurationPanel } from './components/ConfigurationPanel';
import { ProcessingStatus } from './components/ProcessingStatus';
import { ResultsGallery } from './components/ResultsGallery';
import { validateVideoForPlatform } from './utils/VideoPresets';

interface VideoFile extends File {
  preview?: string;
}

interface VideoConfiguration {
  duration: {
    min: number;
    max: number;
  };
  faceTracking: boolean;
  silenceRemoval: boolean;
  momentDetection: boolean;
  targetPlatform: string;
  maxClips: number;
  outputQuality: string;
}

interface ProcessingStage {
  id: string;
  name: string;
  progress: number;
  status: 'pending' | 'processing' | 'completed' | 'error';
  estimatedTimeRemaining?: number;
}

interface VideoClip {
  id: string;
  title: string;
  thumbnail: string;
  duration: number;
  engagementScore: number;
  videoUrl: string;
  platform: string;
  category: string;
  createdAt: Date;
}

function App() {
  const [selectedVideos, setSelectedVideos] = useState<VideoFile[]>([]);
  const [configuration, setConfiguration] = useState<VideoConfiguration>({
    duration: { min: 15, max: 30 },
    faceTracking: true,
    silenceRemoval: true,
    momentDetection: true,
    targetPlatform: 'tiktok',
    maxClips: 10,
    outputQuality: '1080p',
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStages, setProcessingStages] = useState<ProcessingStage[]>([
    {
      id: 'analysis',
      name: 'Video Analysis',
      progress: 0,
      status: 'pending',
    },
    {
      id: 'detection',
      name: 'Moment Detection',
      progress: 0,
      status: 'pending',
    },
    {
      id: 'generation',
      name: 'Clip Generation',
      progress: 0,
      status: 'pending',
    },
    {
      id: 'optimization',
      name: 'Platform Optimization',
      progress: 0,
      status: 'pending',
    },
  ]);
  const [processedClips, setProcessedClips] = useState<VideoClip[]>([]);
  const [isBackgroundProcessing, setIsBackgroundProcessing] = useState(false);
  const toast = useToast();

  const handleVideosSelected = (files: VideoFile[]) => {
    // Validate each video against the selected platform's requirements
    const invalidVideos = files.filter((file) => {
      const validation = validateVideoForPlatform(
        configuration.targetPlatform,
        0, // We'll need to get actual duration
        file.size
      );
      return !validation.isValid;
    });

    if (invalidVideos.length > 0) {
      toast({
        title: 'Invalid videos detected',
        description: `${invalidVideos.length} videos do not meet the platform requirements`,
        status: 'warning',
        duration: 5000,
        isClosable: true,
      });
    }

    setSelectedVideos(files);
  };

  const handleConfigurationChange = (newConfig: VideoConfiguration) => {
    setConfiguration(newConfig);

    // Revalidate existing videos with new configuration
    if (selectedVideos.length > 0) {
      const invalidVideos = selectedVideos.filter((file) => {
        const validation = validateVideoForPlatform(
          newConfig.targetPlatform,
          0, // We'll need to get actual duration
          file.size
        );
        return !validation.isValid;
      });

      if (invalidVideos.length > 0) {
        toast({
          title: 'Configuration Warning',
          description: `${invalidVideos.length} selected videos may not meet the new platform requirements`,
          status: 'warning',
          duration: 5000,
          isClosable: true,
        });
      }
    }
  };

  const handleProcessingCancel = () => {
    // Implement cancellation logic
    setIsProcessing(false);
    toast({
      title: 'Processing Cancelled',
      status: 'info',
      duration: 3000,
      isClosable: true,
    });
  };

  const handleSavePartial = () => {
    // Implement partial save logic
    toast({
      title: 'Progress Saved',
      description: 'Partial results have been saved',
      status: 'success',
      duration: 3000,
      isClosable: true,
    });
  };

  const handleBackgroundProcessing = (enabled: boolean) => {
    setIsBackgroundProcessing(enabled);
  };

  const handleExport = (clips: VideoClip[]) => {
    // Implement export logic
    toast({
      title: 'Exporting Clips',
      description: `Exporting ${clips.length} clips...`,
      status: 'info',
      duration: 3000,
      isClosable: true,
    });
  };

  // Calculate total progress
  const totalProgress = processingStages.reduce(
    (acc, stage) => acc + stage.progress,
    0
  ) / processingStages.length;

  return (
    <ChakraProvider>
      <Container maxW="container.xl" py={8}>
        <VStack spacing={8} w="full">
          <VideoUpload onVideosSelected={handleVideosSelected} />
          <ConfigurationPanel onChange={handleConfigurationChange} />
          
          {isProcessing && (
            <ProcessingStatus
              stages={processingStages}
              onCancel={handleProcessingCancel}
              onSavePartial={handleSavePartial}
              onToggleBackground={handleBackgroundProcessing}
              isBackgroundProcessing={isBackgroundProcessing}
              totalProgress={totalProgress}
              estimatedTimeRemaining={300} // Example: 5 minutes
              accuracyScore={0.85} // Example: 85% accurate
            />
          )}

          {processedClips.length > 0 && (
            <ResultsGallery
              clips={processedClips}
              onExport={handleExport}
            />
          )}
        </VStack>
      </Container>
    </ChakraProvider>
  );
}

export default App; 
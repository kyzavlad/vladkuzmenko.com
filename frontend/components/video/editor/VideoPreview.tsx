import React, { useEffect, useRef, useState } from 'react';
import {
  AspectRatio,
  Box,
  HStack,
  IconButton,
  Progress,
  Text,
  VStack,
  useToast,
} from '@chakra-ui/react';
import {
  FiPlay,
  FiPause,
  FiVolume2,
  FiVolumeX,
  FiMaximize,
  FiMinimize,
} from 'react-icons/fi';
import { formatDuration } from '@/utils/format';

interface VideoPreviewProps {
  videoUrl: string;
  subtitles?: Array<{
    id: string;
    text: string;
    startTime: number;
    endTime: number;
  }>;
  effects?: Array<{
    id: string;
    type: 'transition' | 'filter' | 'overlay';
    startTime: number;
    endTime: number;
    params: Record<string, any>;
  }>;
  onTimeUpdate?: (currentTime: number) => void;
}

export function VideoPreview({
  videoUrl,
  subtitles = [],
  effects = [],
  onTimeUpdate,
}: VideoPreviewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [progress, setProgress] = useState(0);
  const [currentSubtitle, setCurrentSubtitle] = useState<string>('');
  const toast = useToast();

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadedMetadata = () => {
      setDuration(video.duration);
    };

    const handleTimeUpdate = () => {
      const time = video.currentTime;
      setCurrentTime(time);
      setProgress((time / video.duration) * 100);
      onTimeUpdate?.(time);

      // Update subtitles
      const activeSubtitle = subtitles.find(
        (sub) => time >= sub.startTime && time <= sub.endTime
      );
      setCurrentSubtitle(activeSubtitle?.text || '');

      // Apply effects
      effects.forEach((effect) => {
        if (time >= effect.startTime && time <= effect.endTime) {
          applyEffect(effect);
        }
      });
    };

    const handleEnded = () => {
      setIsPlaying(false);
      video.currentTime = 0;
    };

    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('ended', handleEnded);

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('ended', handleEnded);
    };
  }, [subtitles, effects, onTimeUpdate]);

  const applyEffect = (effect: VideoPreviewProps['effects'][0]) => {
    const video = videoRef.current;
    if (!video) return;

    switch (effect.type) {
      case 'filter':
        // Apply CSS filters
        const filters = Object.entries(effect.params)
          .map(([key, value]) => `${key}(${value})`)
          .join(' ');
        video.style.filter = filters;
        break;

      case 'transition':
        // Handle transitions
        break;

      case 'overlay':
        // Handle overlays
        break;
    }
  };

  const togglePlayPause = () => {
    if (!videoRef.current) return;

    if (isPlaying) {
      videoRef.current.pause();
    } else {
      videoRef.current.play().catch((error) => {
        toast({
          title: 'Playback Error',
          description: 'Failed to play video. Please try again.',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
      });
    }
    setIsPlaying(!isPlaying);
  };

  const toggleMute = () => {
    if (!videoRef.current) return;
    videoRef.current.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  const toggleFullscreen = () => {
    if (!containerRef.current) return;

    if (!isFullscreen) {
      if (containerRef.current.requestFullscreen) {
        containerRef.current.requestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
    setIsFullscreen(!isFullscreen);
  };

  const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!videoRef.current) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = x / rect.width;
    const time = percentage * duration;

    videoRef.current.currentTime = time;
    setCurrentTime(time);
    setProgress(percentage * 100);
  };

  return (
    <VStack spacing={4} w="full" ref={containerRef}>
      <AspectRatio ratio={16 / 9} w="full">
        <Box bg="black" position="relative">
          <video
            ref={videoRef}
            src={videoUrl}
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'contain',
            }}
          />
          {currentSubtitle && (
            <Box
              position="absolute"
              bottom="10%"
              left="50%"
              transform="translateX(-50%)"
              bg="blackAlpha.700"
              color="white"
              px={4}
              py={2}
              borderRadius="md"
              textAlign="center"
              maxW="80%"
            >
              <Text fontSize="lg">{currentSubtitle}</Text>
            </Box>
          )}
        </Box>
      </AspectRatio>

      <Box w="full" px={4}>
        <Progress
          value={progress}
          h="4px"
          cursor="pointer"
          onClick={handleProgressClick}
          mb={4}
        />

        <HStack spacing={4} justify="space-between">
          <HStack spacing={2}>
            <IconButton
              aria-label={isPlaying ? 'Pause' : 'Play'}
              icon={isPlaying ? <FiPause /> : <FiPlay />}
              onClick={togglePlayPause}
              colorScheme="primary"
              size="sm"
            />
            <IconButton
              aria-label={isMuted ? 'Unmute' : 'Mute'}
              icon={isMuted ? <FiVolumeX /> : <FiVolume2 />}
              onClick={toggleMute}
              size="sm"
            />
            <Text fontSize="sm" color="gray.500">
              {formatDuration(currentTime)} / {formatDuration(duration)}
            </Text>
          </HStack>

          <IconButton
            aria-label={isFullscreen ? 'Exit Fullscreen' : 'Enter Fullscreen'}
            icon={isFullscreen ? <FiMinimize /> : <FiMaximize />}
            onClick={toggleFullscreen}
            size="sm"
          />
        </HStack>
      </Box>
    </VStack>
  );
} 
import React, { useEffect, useRef, useState } from 'react';
import {
  Box,
  HStack,
  IconButton,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Text,
  VStack,
  useToast,
} from '@chakra-ui/react';
import { FiPlay, FiPause, FiVolume2, FiVolumeX } from 'react-icons/fi';
import WaveSurfer from 'wavesurfer.js';
import Timeline from 'wavesurfer.js/dist/plugins/timeline';
import Regions from 'wavesurfer.js/dist/plugins/regions';
import { formatDuration } from '@/utils/format';

interface TimelineEditorProps {
  videoUrl: string;
  audioUrl: string;
  onTimeUpdate?: (currentTime: number) => void;
  onRegionUpdate?: (regions: any[]) => void;
}

export function TimelineEditor({
  videoUrl,
  audioUrl,
  onTimeUpdate,
  onRegionUpdate,
}: TimelineEditorProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const waveformRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [volume, setVolume] = useState(1);
  const toast = useToast();

  useEffect(() => {
    if (!waveformRef.current) return;

    const wavesurfer = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#4A5568',
      progressColor: '#3182CE',
      cursorColor: '#E53E3E',
      height: 100,
      normalize: true,
      plugins: [
        Timeline.create({
          container: '#timeline',
          primaryLabelInterval: 5,
          secondaryLabelInterval: 1,
        }),
        Regions.create(),
      ],
    });

    wavesurfer.load(audioUrl);

    wavesurfer.on('ready', () => {
      wavesurferRef.current = wavesurfer;
      setDuration(wavesurfer.getDuration());
    });

    wavesurfer.on('timeupdate', (time: number) => {
      setCurrentTime(time);
      if (videoRef.current) {
        videoRef.current.currentTime = time;
      }
      onTimeUpdate?.(time);
    });

    wavesurfer.on('region-update-end', () => {
      const regions = wavesurfer.regions.list;
      onRegionUpdate?.(Object.values(regions));
    });

    return () => {
      wavesurfer.destroy();
    };
  }, [audioUrl, onTimeUpdate, onRegionUpdate]);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.volume = volume;
      videoRef.current.muted = isMuted;
    }
  }, [volume, isMuted]);

  const togglePlayPause = () => {
    if (!wavesurferRef.current) return;

    if (isPlaying) {
      wavesurferRef.current.pause();
      if (videoRef.current) {
        videoRef.current.pause();
      }
    } else {
      wavesurferRef.current.play();
      if (videoRef.current) {
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
    }
    setIsPlaying(!isPlaying);
  };

  const toggleMute = () => {
    setIsMuted(!isMuted);
  };

  const handleVolumeChange = (value: number) => {
    setVolume(value);
  };

  const handleTimelineClick = (time: number) => {
    if (!wavesurferRef.current || !videoRef.current) return;

    wavesurferRef.current.setTime(time);
    videoRef.current.currentTime = time;
    setCurrentTime(time);
  };

  const addRegion = () => {
    if (!wavesurferRef.current) return;

    const currentTime = wavesurferRef.current.getCurrentTime();
    wavesurferRef.current.addRegion({
      start: currentTime,
      end: Math.min(currentTime + 2, duration),
      color: 'rgba(49, 130, 206, 0.3)',
      drag: true,
      resize: true,
    });
  };

  return (
    <VStack spacing={4} w="full">
      <Box
        w="full"
        h="400px"
        bg="black"
        position="relative"
        overflow="hidden"
        borderRadius="lg"
      >
        <video
          ref={videoRef}
          src={videoUrl}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'contain',
          }}
        />
      </Box>

      <Box w="full" px={4}>
        <HStack spacing={4} mb={4}>
          <IconButton
            aria-label={isPlaying ? 'Pause' : 'Play'}
            icon={isPlaying ? <FiPause /> : <FiPlay />}
            onClick={togglePlayPause}
            colorScheme="primary"
          />
          <IconButton
            aria-label={isMuted ? 'Unmute' : 'Mute'}
            icon={isMuted ? <FiVolumeX /> : <FiVolume2 />}
            onClick={toggleMute}
          />
          <Box w="100px">
            <Slider
              value={volume}
              onChange={handleVolumeChange}
              min={0}
              max={1}
              step={0.1}
              isDisabled={isMuted}
            >
              <SliderTrack>
                <SliderFilledTrack />
              </SliderTrack>
              <SliderThumb />
            </Slider>
          </Box>
          <Text fontSize="sm" color="gray.500">
            {formatDuration(currentTime)} / {formatDuration(duration)}
          </Text>
        </HStack>

        <Box w="full" ref={waveformRef} />
        <Box id="timeline" />
      </Box>
    </VStack>
  );
} 
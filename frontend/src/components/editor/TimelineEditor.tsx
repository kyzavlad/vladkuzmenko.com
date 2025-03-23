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
  useColorModeValue,
} from '@chakra-ui/react';
import {
  FiPlay,
  FiPause,
  FiSkipBack,
  FiSkipForward,
  FiZoomIn,
  FiZoomOut,
} from 'react-icons/fi';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions';
import TimelinePlugin from 'wavesurfer.js/dist/plugins/timeline';

interface TimelineEditorProps {
  videoUrl: string;
  audioUrl: string;
  duration: number;
  currentTime: number;
  startTime: number;
  endTime: number;
  engagementMarkers: Array<{ time: number; score: number }>;
  faceDetectionMarkers: Array<{ time: number; faces: number }>;
  onTimeUpdate: (time: number) => void;
  onRangeUpdate: (start: number, end: number) => void;
}

export const TimelineEditor: React.FC<TimelineEditorProps> = ({
  videoUrl,
  audioUrl,
  duration,
  currentTime,
  startTime,
  endTime,
  engagementMarkers,
  faceDetectionMarkers,
  onTimeUpdate,
  onRangeUpdate,
}) => {
  const waveformRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer>();
  const [isPlaying, setIsPlaying] = useState(false);
  const [zoom, setZoom] = useState(50);

  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const waveColor = useColorModeValue('blue.500', 'blue.300');
  const progressColor = useColorModeValue('blue.700', 'blue.500');

  useEffect(() => {
    if (!waveformRef.current) return;

    const wavesurfer = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: waveColor,
      progressColor: progressColor,
      height: 100,
      normalize: true,
      splitChannels: false,
      plugins: [
        RegionsPlugin.create(),
        TimelinePlugin.create({
          container: '#timeline',
          primaryLabelInterval: 10,
          secondaryLabelInterval: 1,
        }),
      ],
    });

    wavesurfer.load(audioUrl);

    wavesurfer.on('ready', () => {
      wavesurfer.zoom(zoom);
      wavesurfer.addRegion({
        start: startTime,
        end: endTime,
        color: 'rgba(0, 123, 255, 0.2)',
        drag: true,
        resize: true,
      });
    });

    wavesurfer.on('timeupdate', (time: number) => {
      onTimeUpdate(time);
    });

    wavesurfer.on('region-updated', (region: any) => {
      onRangeUpdate(region.start, region.end);
    });

    wavesurferRef.current = wavesurfer;

    return () => {
      wavesurfer.destroy();
    };
  }, [audioUrl, startTime, endTime, onTimeUpdate, onRangeUpdate, zoom, waveColor, progressColor]);

  useEffect(() => {
    if (!wavesurferRef.current) return;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = waveformRef.current?.clientWidth || 1000;
    canvas.height = 30;

    // Draw engagement markers
    ctx.fillStyle = 'rgba(75, 192, 192, 0.5)';
    engagementMarkers.forEach(({ time, score }) => {
      const x = (time / duration) * canvas.width;
      const height = score * canvas.height;
      ctx.fillRect(x - 1, canvas.height - height, 2, height);
    });

    // Draw face detection markers
    ctx.fillStyle = 'rgba(255, 159, 64, 0.5)';
    faceDetectionMarkers.forEach(({ time, faces }) => {
      const x = (time / duration) * canvas.width;
      const height = (faces / 5) * canvas.height; // Normalize for up to 5 faces
      ctx.fillRect(x - 1, 0, 2, height);
    });

    const overlay = document.createElement('div');
    overlay.style.position = 'absolute';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.right = '0';
    overlay.style.bottom = '0';
    overlay.style.pointerEvents = 'none';
    overlay.appendChild(canvas);

    waveformRef.current?.appendChild(overlay);

    return () => {
      overlay.remove();
    };
  }, [engagementMarkers, faceDetectionMarkers, duration]);

  const handlePlayPause = () => {
    if (!wavesurferRef.current) return;
    wavesurferRef.current.playPause();
    setIsPlaying(!isPlaying);
  };

  const handleSkipBack = () => {
    if (!wavesurferRef.current) return;
    wavesurferRef.current.skip(-5);
  };

  const handleSkipForward = () => {
    if (!wavesurferRef.current) return;
    wavesurferRef.current.skip(5);
  };

  const handleZoomIn = () => {
    setZoom((prev) => Math.min(prev + 20, 150));
    wavesurferRef.current?.zoom(zoom + 20);
  };

  const handleZoomOut = () => {
    setZoom((prev) => Math.max(prev - 20, 20));
    wavesurferRef.current?.zoom(zoom - 20);
  };

  const formatTime = (time: number): string => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <Box
      bg={bgColor}
      borderRadius="lg"
      shadow="sm"
      p={4}
      borderWidth={1}
      borderColor={borderColor}
    >
      <Box ref={waveformRef} mb={4} />
      <Box id="timeline" mb={4} />

      <HStack spacing={4} justify="space-between">
        <HStack>
          <IconButton
            aria-label="Play/Pause"
            icon={isPlaying ? <FiPause /> : <FiPlay />}
            onClick={handlePlayPause}
          />
          <IconButton
            aria-label="Skip Back"
            icon={<FiSkipBack />}
            onClick={handleSkipBack}
          />
          <IconButton
            aria-label="Skip Forward"
            icon={<FiSkipForward />}
            onClick={handleSkipForward}
          />
          <Text fontSize="sm">
            {formatTime(currentTime)} / {formatTime(duration)}
          </Text>
        </HStack>

        <HStack>
          <IconButton
            aria-label="Zoom Out"
            icon={<FiZoomOut />}
            onClick={handleZoomOut}
            isDisabled={zoom <= 20}
          />
          <IconButton
            aria-label="Zoom In"
            icon={<FiZoomIn />}
            onClick={handleZoomIn}
            isDisabled={zoom >= 150}
          />
        </HStack>
      </HStack>
    </Box>
  );
};

export default TimelineEditor; 
import React, { useRef, useState, useEffect } from 'react';
import {
  Box,
  Button,
  Grid,
  HStack,
  Text,
  VStack,
  Progress,
  useToast,
  IconButton,
  Tooltip,
} from '@chakra-ui/react';
import {
  FiVideo,
  FiMic,
  FiStopCircle,
  FiTrash2,
  FiCheck,
  FiX,
} from 'react-icons/fi';
import { VideoGuidelines } from './VideoGuidelines';
import { AudioVisualizer } from './AudioVisualizer';
import { SampleQualityIndicator } from './SampleQualityIndicator';
import { SampleLibrary } from './SampleLibrary';

interface Sample {
  id: string;
  type: 'video' | 'audio';
  url: string;
  duration: number;
  quality: number;
  timestamp: Date;
}

export function SampleCollection() {
  const [isRecording, setIsRecording] = useState(false);
  const [sampleType, setSampleType] = useState<'video' | 'audio' | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [currentQuality, setCurrentQuality] = useState(0);
  const [recordingTime, setRecordingTime] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const toast = useToast();

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRecording) {
      interval = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isRecording]);

  const startRecording = async (type: 'video' | 'audio') => {
    try {
      const constraints = {
        video: type === 'video',
        audio: true,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (videoRef.current && type === 'video') {
        videoRef.current.srcObject = stream;
      }

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, {
          type: type === 'video' ? 'video/webm' : 'audio/webm',
        });
        const url = URL.createObjectURL(blob);
        const newSample: Sample = {
          id: `sample-${Date.now()}`,
          type,
          url,
          duration: recordingTime,
          quality: currentQuality,
          timestamp: new Date(),
        };
        setSamples((prev) => [...prev, newSample]);
        setRecordingTime(0);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setSampleType(type);
      setIsRecording(true);
    } catch (error) {
      toast({
        title: 'Recording Error',
        description: 'Failed to start recording. Please check your device permissions.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setSampleType(null);
    }
  };

  const deleteSample = (sampleId: string) => {
    setSamples((prev) => prev.filter((sample) => sample.id !== sampleId));
  };

  const handleQualityUpdate = (quality: number) => {
    setCurrentQuality(quality);
  };

  return (
    <VStack spacing={6} w="full">
      <Grid templateColumns={{ base: '1fr', md: '2fr 1fr' }} gap={6} w="full">
        <Box
          bg="neutral.900"
          borderRadius="lg"
          overflow="hidden"
          position="relative"
        >
          {sampleType === 'video' && (
            <>
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
              <VideoGuidelines />
            </>
          )}
          {sampleType === 'audio' && (
            <Box p={8}>
              <AudioVisualizer isRecording={isRecording} />
            </Box>
          )}
          {isRecording && (
            <Box position="absolute" top={4} right={4}>
              <SampleQualityIndicator
                quality={currentQuality}
                onQualityUpdate={handleQualityUpdate}
              />
            </Box>
          )}
        </Box>

        <VStack spacing={4}>
          <HStack spacing={4} w="full">
            <Button
              leftIcon={<FiVideo />}
              colorScheme="primary"
              isDisabled={isRecording}
              onClick={() => startRecording('video')}
              flex={1}
            >
              Record Video
            </Button>
            <Button
              leftIcon={<FiMic />}
              colorScheme="primary"
              isDisabled={isRecording}
              onClick={() => startRecording('audio')}
              flex={1}
            >
              Record Audio
            </Button>
          </HStack>

          {isRecording && (
            <VStack spacing={4} w="full">
              <Progress
                value={(recordingTime / 60) * 100}
                w="full"
                colorScheme="primary"
                hasStripe
                isAnimated
              />
              <Text>{recordingTime}s / 60s</Text>
              <Button
                leftIcon={<FiStopCircle />}
                colorScheme="red"
                onClick={stopRecording}
                w="full"
              >
                Stop Recording
              </Button>
            </VStack>
          )}
        </VStack>
      </Grid>

      <Box w="full">
        <Text fontSize="lg" fontWeight="semibold" mb={4}>
          Sample Library
        </Text>
        <SampleLibrary
          samples={samples}
          onDelete={deleteSample}
          onSelect={() => {}}
        />
      </Box>
    </VStack>
  );
} 
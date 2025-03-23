import React, { useEffect, useRef } from 'react';
import { Box } from '@chakra-ui/react';

interface AudioVisualizerProps {
  isRecording: boolean;
}

export function AudioVisualizer({ isRecording }: AudioVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number>();

  useEffect(() => {
    let audioContext: AudioContext;
    let mediaStream: MediaStream;

    const initializeAudioAnalyser = async () => {
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new AudioContext();
        const source = audioContext.createMediaStreamSource(mediaStream);
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);
        analyserRef.current = analyser;
        draw();
      } catch (error) {
        console.error('Error initializing audio analyser:', error);
      }
    };

    const draw = () => {
      if (!canvasRef.current || !analyserRef.current) return;

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const width = canvas.width;
      const height = canvas.height;
      const analyser = analyserRef.current;
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const drawFrame = () => {
        animationFrameRef.current = requestAnimationFrame(drawFrame);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#1A202C'; // Dark background
        ctx.fillRect(0, 0, width, height);

        const barWidth = (width / bufferLength) * 2.5;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const barHeight = (dataArray[i] / 255) * height;
          const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
          gradient.addColorStop(0, '#3182CE'); // Primary blue
          gradient.addColorStop(1, '#63B3ED'); // Lighter blue

          ctx.fillStyle = gradient;
          ctx.fillRect(
            x,
            height - barHeight,
            barWidth,
            barHeight
          );

          x += barWidth + 1;
        }
      };

      drawFrame();
    };

    if (isRecording) {
      initializeAudioAnalyser();
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [isRecording]);

  return (
    <Box
      w="full"
      h="200px"
      bg="neutral.900"
      borderRadius="lg"
      overflow="hidden"
    >
      <canvas
        ref={canvasRef}
        width={800}
        height={200}
        style={{
          width: '100%',
          height: '100%',
        }}
      />
    </Box>
  );
} 
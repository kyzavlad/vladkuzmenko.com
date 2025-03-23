import React, { useEffect } from 'react';
import {
  Box,
  CircularProgress,
  CircularProgressLabel,
  Tooltip,
  VStack,
  Text,
} from '@chakra-ui/react';
import { FiCheckCircle, FiAlertCircle } from 'react-icons/fi';

interface SampleQualityIndicatorProps {
  quality: number;
  onQualityUpdate: (quality: number) => void;
}

interface QualityMetrics {
  lighting: number;
  noise: number;
  stability: number;
  faceDetection: number;
}

export function SampleQualityIndicator({
  quality,
  onQualityUpdate,
}: SampleQualityIndicatorProps) {
  useEffect(() => {
    // Simulate real-time quality analysis
    const interval = setInterval(() => {
      const metrics = analyzeQuality();
      const overallQuality = calculateOverallQuality(metrics);
      onQualityUpdate(overallQuality);
    }, 500);

    return () => clearInterval(interval);
  }, [onQualityUpdate]);

  const analyzeQuality = (): QualityMetrics => {
    // Simulate quality analysis with random values
    // In a real implementation, this would analyze the actual video/audio stream
    return {
      lighting: Math.random() * 100,
      noise: Math.random() * 100,
      stability: Math.random() * 100,
      faceDetection: Math.random() * 100,
    };
  };

  const calculateOverallQuality = (metrics: QualityMetrics): number => {
    const weights = {
      lighting: 0.3,
      noise: 0.2,
      stability: 0.2,
      faceDetection: 0.3,
    };

    return (
      metrics.lighting * weights.lighting +
      metrics.noise * weights.noise +
      metrics.stability * weights.stability +
      metrics.faceDetection * weights.faceDetection
    );
  };

  const getQualityColor = (value: number): string => {
    if (value >= 80) return 'green.500';
    if (value >= 60) return 'yellow.500';
    return 'red.500';
  };

  const getQualityLabel = (value: number): string => {
    if (value >= 80) return 'Good';
    if (value >= 60) return 'Fair';
    return 'Poor';
  };

  return (
    <VStack spacing={2} align="center">
      <Tooltip
        label={`Quality: ${getQualityLabel(quality)}`}
        placement="left"
        hasArrow
      >
        <Box position="relative">
          <CircularProgress
            value={quality}
            color={getQualityColor(quality)}
            size="60px"
            thickness="8px"
          >
            <CircularProgressLabel>
              {quality >= 80 ? (
                <FiCheckCircle size="24px" color="#48BB78" />
              ) : (
                <FiAlertCircle
                  size="24px"
                  color={quality >= 60 ? '#ECC94B' : '#E53E3E'}
                />
              )}
            </CircularProgressLabel>
          </CircularProgress>
        </Box>
      </Tooltip>
      <Text
        fontSize="sm"
        fontWeight="medium"
        color={getQualityColor(quality)}
      >
        {Math.round(quality)}%
      </Text>
    </VStack>
  );
} 
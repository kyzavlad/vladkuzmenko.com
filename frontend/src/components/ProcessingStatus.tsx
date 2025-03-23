import React, { useState, useEffect } from 'react';
import {
  Box,
  VStack,
  HStack,
  Progress,
  Text,
  Button,
  Badge,
  useToast,
  Icon,
  Tooltip,
  Flex,
  Switch,
  CircularProgress,
  CircularProgressLabel,
} from '@chakra-ui/react';
import {
  FiClock,
  FiAlertCircle,
  FiCheckCircle,
  FiXCircle,
  FiPause,
  FiPlay,
  FiSave,
} from 'react-icons/fi';

interface ProcessingStage {
  id: string;
  name: string;
  progress: number;
  status: 'pending' | 'processing' | 'completed' | 'error';
  estimatedTimeRemaining?: number;
}

interface ProcessingStatusProps {
  stages: ProcessingStage[];
  onCancel: () => void;
  onSavePartial: () => void;
  onToggleBackground: (enabled: boolean) => void;
  isBackgroundProcessing: boolean;
  totalProgress: number;
  estimatedTimeRemaining: number;
  accuracyScore: number;
}

export const ProcessingStatus: React.FC<ProcessingStatusProps> = ({
  stages,
  onCancel,
  onSavePartial,
  onToggleBackground,
  isBackgroundProcessing,
  totalProgress,
  estimatedTimeRemaining,
  accuracyScore,
}) => {
  const toast = useToast();
  const [isPaused, setIsPaused] = useState(false);

  const formatTime = (seconds: number): string => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  };

  const getStageIcon = (status: ProcessingStage['status']) => {
    switch (status) {
      case 'completed':
        return FiCheckCircle;
      case 'error':
        return FiAlertCircle;
      case 'processing':
        return FiClock;
      default:
        return undefined;
    }
  };

  const handleBackgroundToggle = (enabled: boolean) => {
    onToggleBackground(enabled);
    if (enabled) {
      toast({
        title: 'Background Processing Enabled',
        description: "We'll notify you when processing is complete",
        status: 'info',
        duration: 3000,
        isClosable: true,
      });
    }
  };

  const handlePauseResume = () => {
    setIsPaused(!isPaused);
    toast({
      title: isPaused ? 'Processing Resumed' : 'Processing Paused',
      status: 'info',
      duration: 2000,
      isClosable: true,
    });
  };

  return (
    <Box w="full" bg="white" borderRadius="lg" shadow="sm" p={6}>
      <VStack spacing={6} align="stretch">
        {/* Overall Progress */}
        <HStack spacing={4} justify="space-between">
          <VStack align="start" flex={1}>
            <Text fontSize="lg" fontWeight="semibold">
              Processing Videos
            </Text>
            <Progress
              value={totalProgress}
              size="lg"
              width="full"
              borderRadius="full"
              hasStripe
              isAnimated={!isPaused}
            />
          </VStack>
          <CircularProgress
            value={accuracyScore * 100}
            color="green.400"
            size="60px"
          >
            <CircularProgressLabel>
              {Math.round(accuracyScore * 100)}%
            </CircularProgressLabel>
          </CircularProgress>
        </HStack>

        {/* Estimated Time */}
        <HStack spacing={2}>
          <Icon as={FiClock} color="gray.500" />
          <Text color="gray.600">
            Estimated time remaining:{' '}
            <Text as="span" fontWeight="semibold">
              {formatTime(estimatedTimeRemaining)}
            </Text>
          </Text>
          <Tooltip label="Accuracy score for time estimation">
            <Badge colorScheme="green" ml={2}>
              {Math.round(accuracyScore * 100)}% accurate
            </Badge>
          </Tooltip>
        </HStack>

        {/* Individual Stages */}
        <VStack spacing={4} align="stretch">
          {stages.map((stage) => (
            <Box
              key={stage.id}
              p={4}
              borderRadius="md"
              bg="gray.50"
              borderLeft="4px solid"
              borderLeftColor={
                stage.status === 'completed'
                  ? 'green.400'
                  : stage.status === 'error'
                  ? 'red.400'
                  : stage.status === 'processing'
                  ? 'blue.400'
                  : 'gray.200'
              }
            >
              <HStack justify="space-between">
                <HStack>
                  {getStageIcon(stage.status) && (
                    <Icon
                      as={getStageIcon(stage.status)}
                      color={
                        stage.status === 'completed'
                          ? 'green.400'
                          : stage.status === 'error'
                          ? 'red.400'
                          : stage.status === 'processing'
                          ? 'blue.400'
                          : 'gray.400'
                      }
                    />
                  )}
                  <Text fontWeight="medium">{stage.name}</Text>
                </HStack>
                <Text color="gray.600">{Math.round(stage.progress)}%</Text>
              </HStack>
              <Progress
                value={stage.progress}
                size="sm"
                mt={2}
                borderRadius="full"
                colorScheme={
                  stage.status === 'completed'
                    ? 'green'
                    : stage.status === 'error'
                    ? 'red'
                    : 'blue'
                }
              />
            </Box>
          ))}
        </VStack>

        {/* Controls */}
        <Flex justify="space-between" align="center">
          <HStack spacing={4}>
            <Button
              leftIcon={<Icon as={isPaused ? FiPlay : FiPause} />}
              onClick={handlePauseResume}
              size="sm"
            >
              {isPaused ? 'Resume' : 'Pause'}
            </Button>
            <Button
              leftIcon={<Icon as={FiSave} />}
              onClick={onSavePartial}
              size="sm"
              variant="outline"
            >
              Save Current Progress
            </Button>
            <Button
              leftIcon={<Icon as={FiXCircle} />}
              onClick={onCancel}
              size="sm"
              colorScheme="red"
              variant="ghost"
            >
              Cancel
            </Button>
          </HStack>
          <HStack spacing={2}>
            <Text fontSize="sm" color="gray.600">
              Background Processing
            </Text>
            <Switch
              isChecked={isBackgroundProcessing}
              onChange={(e) => handleBackgroundToggle(e.target.checked)}
            />
          </HStack>
        </Flex>
      </VStack>
    </Box>
  );
};

export default ProcessingStatus; 
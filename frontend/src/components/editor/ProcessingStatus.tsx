import React from 'react';
import {
  Box,
  VStack,
  HStack,
  Text,
  Progress,
  Button,
  Badge,
  useColorModeValue,
  Tooltip,
  IconButton,
} from '@chakra-ui/react';
import {
  FiPlay,
  FiPause,
  FiX,
  FiClock,
  FiCheckCircle,
  FiAlertCircle,
} from 'react-icons/fi';

export interface ProcessingStage {
  id: string;
  name: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress: number;
  estimatedTimeRemaining?: number;
  error?: string;
}

interface ProcessingStatusProps {
  stages: ProcessingStage[];
  totalProgress: number;
  isPaused: boolean;
  isBackgroundProcessing: boolean;
  onPauseResume: () => void;
  onCancel: () => void;
  onBackgroundProcessingToggle: () => void;
}

export const ProcessingStatus: React.FC<ProcessingStatusProps> = ({
  stages,
  totalProgress,
  isPaused,
  isBackgroundProcessing,
  onPauseResume,
  onCancel,
  onBackgroundProcessingToggle,
}) => {
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  const formatTime = (seconds?: number): string => {
    if (!seconds) return 'Calculating...';
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const getStageIcon = (status: ProcessingStage['status']) => {
    switch (status) {
      case 'completed':
        return <FiCheckCircle color="green.500" />;
      case 'error':
        return <FiAlertCircle color="red.500" />;
      case 'processing':
        return <FiClock />;
      default:
        return null;
    }
  };

  return (
    <Box
      w="full"
      bg={bgColor}
      borderRadius="lg"
      shadow="sm"
      p={4}
      borderWidth={1}
      borderColor={borderColor}
    >
      <VStack spacing={4} align="stretch">
        <HStack justify="space-between">
          <Text fontSize="lg" fontWeight="semibold">
            Processing Status
          </Text>
          <HStack spacing={2}>
            <Tooltip
              label={isPaused ? 'Resume Processing' : 'Pause Processing'}
              placement="top"
            >
              <IconButton
                aria-label={isPaused ? 'Resume' : 'Pause'}
                icon={isPaused ? <FiPlay /> : <FiPause />}
                size="sm"
                onClick={onPauseResume}
              />
            </Tooltip>
            <Tooltip label="Cancel Processing" placement="top">
              <IconButton
                aria-label="Cancel"
                icon={<FiX />}
                size="sm"
                colorScheme="red"
                variant="ghost"
                onClick={onCancel}
              />
            </Tooltip>
          </HStack>
        </HStack>

        <Box>
          <HStack justify="space-between" mb={2}>
            <Text fontSize="sm">Overall Progress</Text>
            <Text fontSize="sm" fontWeight="medium">
              {Math.round(totalProgress)}%
            </Text>
          </HStack>
          <Progress
            value={totalProgress}
            size="sm"
            borderRadius="full"
            hasStripe
            isAnimated={!isPaused}
          />
        </Box>

        <VStack spacing={3} align="stretch">
          {stages.map((stage) => (
            <Box
              key={stage.id}
              p={3}
              borderWidth={1}
              borderRadius="md"
              borderColor={borderColor}
            >
              <HStack justify="space-between" mb={2}>
                <HStack>
                  <Text fontSize="sm" fontWeight="medium">
                    {stage.name}
                  </Text>
                  <Badge
                    colorScheme={
                      stage.status === 'completed'
                        ? 'green'
                        : stage.status === 'error'
                        ? 'red'
                        : stage.status === 'processing'
                        ? 'blue'
                        : 'gray'
                    }
                  >
                    {stage.status.charAt(0).toUpperCase() +
                      stage.status.slice(1)}
                  </Badge>
                </HStack>
                {getStageIcon(stage.status)}
              </HStack>

              {stage.status === 'processing' && (
                <>
                  <Progress
                    value={stage.progress}
                    size="xs"
                    borderRadius="full"
                    mb={1}
                  />
                  <HStack justify="space-between">
                    <Text fontSize="xs" color="gray.500">
                      {stage.progress}% complete
                    </Text>
                    {stage.estimatedTimeRemaining && (
                      <Text fontSize="xs" color="gray.500">
                        {formatTime(stage.estimatedTimeRemaining)} remaining
                      </Text>
                    )}
                  </HStack>
                </>
              )}

              {stage.status === 'error' && stage.error && (
                <Text fontSize="xs" color="red.500" mt={1}>
                  {stage.error}
                </Text>
              )}
            </Box>
          ))}
        </VStack>

        <Button
          size="sm"
          variant="ghost"
          onClick={onBackgroundProcessingToggle}
          isActive={isBackgroundProcessing}
        >
          {isBackgroundProcessing
            ? 'Processing in Background'
            : 'Process in Background'}
        </Button>
      </VStack>
    </Box>
  );
};

export default ProcessingStatus; 
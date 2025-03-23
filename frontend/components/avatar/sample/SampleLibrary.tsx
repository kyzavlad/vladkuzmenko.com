import React from 'react';
import {
  Box,
  Grid,
  HStack,
  IconButton,
  Text,
  VStack,
  Badge,
} from '@chakra-ui/react';
import { FiPlay, FiTrash2, FiCheck } from 'react-icons/fi';
import { formatDuration } from '@/utils/format';

interface Sample {
  id: string;
  type: 'video' | 'audio';
  url: string;
  duration: number;
  quality: number;
  timestamp: Date;
}

interface SampleLibraryProps {
  samples: Sample[];
  onDelete: (sampleId: string) => void;
  onSelect: (sample: Sample) => void;
}

export function SampleLibrary({
  samples,
  onDelete,
  onSelect,
}: SampleLibraryProps) {
  const getQualityColor = (quality: number): string => {
    if (quality >= 80) return 'green';
    if (quality >= 60) return 'yellow';
    return 'red';
  };

  const getQualityLabel = (quality: number): string => {
    if (quality >= 80) return 'Good';
    if (quality >= 60) return 'Fair';
    return 'Poor';
  };

  return (
    <Grid
      templateColumns={{
        base: '1fr',
        md: 'repeat(2, 1fr)',
        lg: 'repeat(3, 1fr)',
      }}
      gap={4}
    >
      {samples.map((sample) => (
        <Box
          key={sample.id}
          bg="neutral.800"
          borderRadius="lg"
          overflow="hidden"
          position="relative"
        >
          {sample.type === 'video' ? (
            <Box position="relative" pb="56.25%">
              <video
                src={sample.url}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                }}
              />
            </Box>
          ) : (
            <Box
              h="100px"
              bg="neutral.700"
              display="flex"
              alignItems="center"
              justifyContent="center"
            >
              <IconButton
                aria-label="Play audio"
                icon={<FiPlay />}
                size="lg"
                variant="ghost"
                colorScheme="primary"
                onClick={() => onSelect(sample)}
              />
            </Box>
          )}

          <VStack spacing={2} p={4}>
            <HStack w="full" justify="space-between">
              <Badge
                colorScheme={getQualityColor(sample.quality)}
                variant="subtle"
              >
                {getQualityLabel(sample.quality)}
              </Badge>
              <Text fontSize="sm" color="neutral.400">
                {formatDuration(sample.duration)}
              </Text>
            </HStack>

            <HStack w="full" justify="space-between">
              <Text fontSize="sm" color="neutral.300">
                {new Date(sample.timestamp).toLocaleString()}
              </Text>
              <HStack spacing={2}>
                <IconButton
                  aria-label="Select sample"
                  icon={<FiCheck />}
                  size="sm"
                  colorScheme="primary"
                  variant="ghost"
                  onClick={() => onSelect(sample)}
                />
                <IconButton
                  aria-label="Delete sample"
                  icon={<FiTrash2 />}
                  size="sm"
                  colorScheme="red"
                  variant="ghost"
                  onClick={() => onDelete(sample.id)}
                />
              </HStack>
            </HStack>
          </VStack>
        </Box>
      ))}
    </Grid>
  );
} 
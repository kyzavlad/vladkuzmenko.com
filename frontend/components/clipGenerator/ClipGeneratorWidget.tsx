import React from 'react';
import {
  Box,
  Button,
  HStack,
  Text,
  VStack,
  Image,
  Badge,
  useToken,
  IconButton,
} from '@chakra-ui/react';
import { FiChevronLeft, FiChevronRight, FiZap } from 'react-icons/fi';

interface Sample {
  id: string;
  title: string;
  thumbnailUrl: string;
  duration: number;
  views: number;
}

interface ClipGeneratorWidgetProps {
  samples: Sample[];
  estimatedTokens: number;
  onTryNow: () => void;
}

export function ClipGeneratorWidget({
  samples,
  estimatedTokens,
  onTryNow,
}: ClipGeneratorWidgetProps) {
  const [currentIndex, setCurrentIndex] = React.useState(0);
  const [primary500] = useToken('colors', ['primary.500']);

  const nextSample = () => {
    setCurrentIndex((prev) =>
      prev === samples.length - 1 ? 0 : prev + 1
    );
  };

  const prevSample = () => {
    setCurrentIndex((prev) =>
      prev === 0 ? samples.length - 1 : prev - 1
    );
  };

  return (
    <Box
      bg="neutral.800"
      borderRadius="xl"
      overflow="hidden"
      position="relative"
    >
      <VStack spacing={0}>
        <Box position="relative" w="full" pb="56.25%">
          <Image
            src={samples[currentIndex].thumbnailUrl}
            alt={samples[currentIndex].title}
            position="absolute"
            top={0}
            left={0}
            w="full"
            h="full"
            objectFit="cover"
          />
          <Box
            position="absolute"
            bottom={0}
            left={0}
            right={0}
            bg="blackAlpha.700"
            p={4}
          >
            <Text color="white" fontWeight="semibold">
              {samples[currentIndex].title}
            </Text>
            <HStack spacing={4} mt={2}>
              <Text color="neutral.300" fontSize="sm">
                {formatDuration(samples[currentIndex].duration)}
              </Text>
              <Text color="neutral.300" fontSize="sm">
                {formatViews(samples[currentIndex].views)} views
              </Text>
            </HStack>
          </Box>
          <HStack
            position="absolute"
            top="50%"
            left={0}
            right={0}
            transform="translateY(-50%)"
            justify="space-between"
            px={4}
          >
            <IconButton
              aria-label="Previous sample"
              icon={<FiChevronLeft />}
              onClick={prevSample}
              variant="solid"
              colorScheme="blackAlpha"
              rounded="full"
            />
            <IconButton
              aria-label="Next sample"
              icon={<FiChevronRight />}
              onClick={nextSample}
              variant="solid"
              colorScheme="blackAlpha"
              rounded="full"
            />
          </HStack>
        </Box>

        <Box p={6} w="full">
          <VStack spacing={4} align="stretch">
            <Text fontSize="xl" fontWeight="semibold">
              AI Clip Generator
            </Text>
            <Text color="neutral.400">
              Transform your long-form content into engaging short-form clips
              optimized for social media.
            </Text>
            <HStack justify="space-between">
              <Button
                leftIcon={<FiZap />}
                colorScheme="primary"
                size="lg"
                onClick={onTryNow}
              >
                Try Now
              </Button>
              <Badge
                colorScheme="primary"
                variant="subtle"
                px={3}
                py={1}
                borderRadius="full"
              >
                ~{estimatedTokens} tokens per clip
              </Badge>
            </HStack>
          </VStack>
        </Box>
      </VStack>
    </Box>
  );
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function formatViews(views: number): string {
  if (views >= 1000000) {
    return `${(views / 1000000).toFixed(1)}M`;
  }
  if (views >= 1000) {
    return `${(views / 1000).toFixed(1)}K`;
  }
  return views.toString();
} 
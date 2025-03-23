import React, { useState } from 'react';
import {
  Box,
  Button,
  Grid,
  HStack,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Text,
  VStack,
  IconButton,
  useToast,
} from '@chakra-ui/react';
import { FiPlay, FiStop, FiRotateCcw } from 'react-icons/fi';

interface VoiceControl {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
}

interface VoiceControlsProps {
  controls: VoiceControl[];
  sampleText: string;
  onChange: (controlId: string, value: number) => void;
  onPreview: (settings: Record<string, number>) => Promise<string>;
  onReset: () => void;
}

export function VoiceControls({
  controls,
  sampleText,
  onChange,
  onPreview,
  onReset,
}: VoiceControlsProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const toast = useToast();

  const handlePreview = async () => {
    try {
      setIsLoading(true);
      const settings = controls.reduce(
        (acc, control) => ({
          ...acc,
          [control.id]: control.value,
        }),
        {}
      );

      const url = await onPreview(settings);
      setAudioUrl(url);
      
      const audio = new Audio(url);
      audio.onended = () => setIsPlaying(false);
      audio.play();
      setIsPlaying(true);
    } catch (error) {
      toast({
        title: 'Preview Error',
        description: 'Failed to generate voice preview',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const stopPreview = () => {
    if (audioUrl) {
      const audio = new Audio(audioUrl);
      audio.pause();
      setIsPlaying(false);
    }
  };

  return (
    <VStack spacing={6} w="full">
      <HStack w="full" justify="space-between">
        <Text fontSize="xl" fontWeight="semibold">
          Voice Settings
        </Text>
        <Button
          leftIcon={<FiRotateCcw />}
          variant="ghost"
          size="sm"
          onClick={onReset}
        >
          Reset
        </Button>
      </HStack>

      <Grid
        templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)' }}
        gap={6}
        w="full"
      >
        {controls.map((control) => (
          <Box key={control.id}>
            <HStack justify="space-between" mb={2}>
              <Text fontSize="sm" fontWeight="medium">
                {control.label}
              </Text>
              <Text fontSize="sm" color="neutral.400">
                {control.value}
              </Text>
            </HStack>
            <Slider
              value={control.value}
              min={control.min}
              max={control.max}
              step={control.step}
              onChange={(value) => onChange(control.id, value)}
            >
              <SliderTrack>
                <SliderFilledTrack />
              </SliderTrack>
              <SliderThumb />
            </Slider>
          </Box>
        ))}
      </Grid>

      <Box
        w="full"
        p={4}
        bg="neutral.800"
        borderRadius="lg"
        position="relative"
      >
        <Text fontSize="sm" color="neutral.400" mb={4}>
          Preview Text:
        </Text>
        <Text fontSize="md" mb={4}>
          {sampleText}
        </Text>
        <HStack>
          <IconButton
            aria-label={isPlaying ? 'Stop preview' : 'Play preview'}
            icon={isPlaying ? <FiStop /> : <FiPlay />}
            onClick={isPlaying ? stopPreview : handlePreview}
            isLoading={isLoading}
            colorScheme="primary"
          />
          <Text fontSize="sm" color="neutral.400">
            {isPlaying
              ? 'Playing preview...'
              : 'Click to preview voice settings'}
          </Text>
        </HStack>
      </Box>
    </VStack>
  );
} 
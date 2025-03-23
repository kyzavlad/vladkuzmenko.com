import React from 'react';
import {
  Box,
  Grid,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Text,
  VStack,
  HStack,
  Select,
  Button,
  useToken,
} from '@chakra-ui/react';
import { FiRotateCcw } from 'react-icons/fi';

interface AppearanceControl {
  id: string;
  label: string;
  type: 'slider' | 'select';
  value: number | string;
  options?: { value: string; label: string }[];
  min?: number;
  max?: number;
  step?: number;
}

interface AppearanceControlsProps {
  controls: AppearanceControl[];
  onChange: (controlId: string, value: number | string) => void;
  onReset: () => void;
}

export function AppearanceControls({
  controls,
  onChange,
  onReset,
}: AppearanceControlsProps) {
  const [primary500] = useToken('colors', ['primary.500']);

  const renderControl = (control: AppearanceControl) => {
    switch (control.type) {
      case 'slider':
        return (
          <Slider
            value={control.value as number}
            min={control.min || 0}
            max={control.max || 100}
            step={control.step || 1}
            onChange={(value) => onChange(control.id, value)}
          >
            <SliderTrack>
              <SliderFilledTrack />
            </SliderTrack>
            <SliderThumb
              boxSize={6}
              bg={primary500}
              _focus={{ boxShadow: 'outline' }}
            />
          </Slider>
        );
      case 'select':
        return (
          <Select
            value={control.value as string}
            onChange={(e) => onChange(control.id, e.target.value)}
          >
            {control.options?.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </Select>
        );
      default:
        return null;
    }
  };

  return (
    <VStack spacing={6} w="full">
      <HStack w="full" justify="space-between">
        <Text fontSize="xl" fontWeight="semibold">
          Customize Appearance
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
              {control.type === 'slider' && (
                <Text fontSize="sm" color="neutral.400">
                  {control.value}
                </Text>
              )}
            </HStack>
            {renderControl(control)}
          </Box>
        ))}
      </Grid>
    </VStack>
  );
} 
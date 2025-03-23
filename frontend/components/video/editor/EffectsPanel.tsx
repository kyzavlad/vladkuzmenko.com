import React, { useState } from 'react';
import {
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Box,
  Button,
  FormControl,
  FormLabel,
  HStack,
  IconButton,
  Input,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Select,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Text,
  VStack,
  useToast,
} from '@chakra-ui/react';
import { FiPlus, FiTrash2 } from 'react-icons/fi';

interface Effect {
  id: string;
  type: 'transition' | 'filter' | 'overlay';
  name: string;
  startTime: number;
  endTime: number;
  params: Record<string, number | string>;
}

interface EffectsPanelProps {
  currentTime: number;
  duration: number;
  effects: Effect[];
  onEffectAdd: (effect: Effect) => void;
  onEffectUpdate: (effectId: string, updates: Partial<Effect>) => void;
  onEffectDelete: (effectId: string) => void;
}

const EFFECT_TEMPLATES = {
  transitions: [
    { name: 'Fade', params: { duration: 1 } },
    { name: 'Dissolve', params: { duration: 1, intensity: 50 } },
    { name: 'Slide', params: { duration: 1, direction: 'left' } },
  ],
  filters: [
    { name: 'Brightness', params: { value: 100 } },
    { name: 'Contrast', params: { value: 100 } },
    { name: 'Saturation', params: { value: 100 } },
    { name: 'Blur', params: { radius: 0 } },
  ],
  overlays: [
    { name: 'Text', params: { text: '', size: 24, color: '#ffffff' } },
    { name: 'Logo', params: { opacity: 100, scale: 100 } },
    { name: 'Shape', params: { type: 'rectangle', color: '#000000', opacity: 50 } },
  ],
};

export function EffectsPanel({
  currentTime,
  duration,
  effects,
  onEffectAdd,
  onEffectUpdate,
  onEffectDelete,
}: EffectsPanelProps) {
  const [selectedType, setSelectedType] = useState<Effect['type']>('filter');
  const [selectedTemplate, setSelectedTemplate] = useState('');
  const toast = useToast();

  const handleAddEffect = () => {
    if (!selectedTemplate) {
      toast({
        title: 'Select Effect',
        description: 'Please select an effect template first.',
        status: 'warning',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    const template = [...EFFECT_TEMPLATES.transitions, ...EFFECT_TEMPLATES.filters, ...EFFECT_TEMPLATES.overlays].find(
      (t) => t.name === selectedTemplate
    );

    if (!template) return;

    const newEffect: Effect = {
      id: `effect-${Date.now()}`,
      type: selectedType,
      name: template.name,
      startTime: currentTime,
      endTime: Math.min(currentTime + 2, duration),
      params: { ...template.params },
    };

    onEffectAdd(newEffect);
  };

  const renderParamControl = (effect: Effect, param: string, value: number | string) => {
    const isNumber = typeof value === 'number';

    if (isNumber) {
      return (
        <FormControl key={param}>
          <FormLabel>{param}</FormLabel>
          <Slider
            value={value as number}
            onChange={(newValue) =>
              onEffectUpdate(effect.id, {
                params: { ...effect.params, [param]: newValue },
              })
            }
            min={0}
            max={param === 'duration' ? 5 : 100}
            step={0.1}
          >
            <SliderTrack>
              <SliderFilledTrack />
            </SliderTrack>
            <SliderThumb />
          </Slider>
        </FormControl>
      );
    }

    if (param === 'direction') {
      return (
        <FormControl key={param}>
          <FormLabel>{param}</FormLabel>
          <Select
            value={value as string}
            onChange={(e) =>
              onEffectUpdate(effect.id, {
                params: { ...effect.params, [param]: e.target.value },
              })
            }
          >
            <option value="left">Left</option>
            <option value="right">Right</option>
            <option value="top">Top</option>
            <option value="bottom">Bottom</option>
          </Select>
        </FormControl>
      );
    }

    if (param === 'type') {
      return (
        <FormControl key={param}>
          <FormLabel>{param}</FormLabel>
          <Select
            value={value as string}
            onChange={(e) =>
              onEffectUpdate(effect.id, {
                params: { ...effect.params, [param]: e.target.value },
              })
            }
          >
            <option value="rectangle">Rectangle</option>
            <option value="circle">Circle</option>
            <option value="triangle">Triangle</option>
          </Select>
        </FormControl>
      );
    }

    return (
      <FormControl key={param}>
        <FormLabel>{param}</FormLabel>
        <Input
          value={value as string}
          onChange={(e) =>
            onEffectUpdate(effect.id, {
              params: { ...effect.params, [param]: e.target.value },
            })
          }
          type={param === 'color' ? 'color' : 'text'}
        />
      </FormControl>
    );
  };

  return (
    <VStack spacing={4} w="full">
      <Box w="full" p={4} bg="neutral.800" borderRadius="lg">
        <VStack spacing={4} align="stretch">
          <FormControl>
            <FormLabel>Effect Type</FormLabel>
            <Select
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value as Effect['type'])}
            >
              <option value="transition">Transition</option>
              <option value="filter">Filter</option>
              <option value="overlay">Overlay</option>
            </Select>
          </FormControl>

          <FormControl>
            <FormLabel>Effect Template</FormLabel>
            <Select
              value={selectedTemplate}
              onChange={(e) => setSelectedTemplate(e.target.value)}
            >
              <option value="">Select a template...</option>
              {selectedType === 'transition' &&
                EFFECT_TEMPLATES.transitions.map((t) => (
                  <option key={t.name} value={t.name}>
                    {t.name}
                  </option>
                ))}
              {selectedType === 'filter' &&
                EFFECT_TEMPLATES.filters.map((t) => (
                  <option key={t.name} value={t.name}>
                    {t.name}
                  </option>
                ))}
              {selectedType === 'overlay' &&
                EFFECT_TEMPLATES.overlays.map((t) => (
                  <option key={t.name} value={t.name}>
                    {t.name}
                  </option>
                ))}
            </Select>
          </FormControl>

          <Button
            leftIcon={<FiPlus />}
            colorScheme="primary"
            onClick={handleAddEffect}
          >
            Add Effect
          </Button>
        </VStack>
      </Box>

      <Accordion allowMultiple w="full">
        {effects.map((effect) => (
          <AccordionItem key={effect.id}>
            <AccordionButton>
              <Box flex="1" textAlign="left">
                <Text fontWeight="medium">{effect.name}</Text>
                <Text fontSize="sm" color="gray.500">
                  {formatTime(effect.startTime)} - {formatTime(effect.endTime)}
                </Text>
              </Box>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel>
              <VStack spacing={4} align="stretch">
                <HStack>
                  <FormControl flex={1}>
                    <FormLabel>Start Time</FormLabel>
                    <NumberInput
                      value={effect.startTime}
                      onChange={(_, value) =>
                        onEffectUpdate(effect.id, { startTime: value })
                      }
                      min={0}
                      max={effect.endTime}
                      step={0.1}
                      precision={2}
                    >
                      <NumberInputField />
                      <NumberInputStepper>
                        <NumberIncrementStepper />
                        <NumberDecrementStepper />
                      </NumberInputStepper>
                    </NumberInput>
                  </FormControl>
                  <FormControl flex={1}>
                    <FormLabel>End Time</FormLabel>
                    <NumberInput
                      value={effect.endTime}
                      onChange={(_, value) =>
                        onEffectUpdate(effect.id, { endTime: value })
                      }
                      min={effect.startTime}
                      max={duration}
                      step={0.1}
                      precision={2}
                    >
                      <NumberInputField />
                      <NumberInputStepper>
                        <NumberIncrementStepper />
                        <NumberDecrementStepper />
                      </NumberInputStepper>
                    </NumberInput>
                  </FormControl>
                  <IconButton
                    aria-label="Delete effect"
                    icon={<FiTrash2 />}
                    colorScheme="red"
                    variant="ghost"
                    onClick={() => onEffectDelete(effect.id)}
                  />
                </HStack>

                {Object.entries(effect.params).map(([param, value]) =>
                  renderParamControl(effect, param, value)
                )}
              </VStack>
            </AccordionPanel>
          </AccordionItem>
        ))}
      </Accordion>
    </VStack>
  );
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
} 
import React from 'react';
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Grid,
  HStack,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Select,
  Switch,
  Text,
  VStack,
  Tooltip,
  IconButton,
} from '@chakra-ui/react';
import {
  FiInfo,
  FiClock,
  FiMaximize,
  FiMusic,
  FiType,
  FiZap,
} from 'react-icons/fi';

interface ClipConfig {
  duration: number;
  aspectRatio: '9:16' | '16:9' | '1:1';
  addCaptions: boolean;
  addBackground: boolean;
  addIntro: boolean;
  addOutro: boolean;
  musicStyle: string;
  captionStyle: string;
}

interface ClipConfigProps {
  config: ClipConfig;
  onChange: (config: Partial<ClipConfig>) => void;
  onGenerate: () => void;
  isGenerating: boolean;
}

export function ClipConfig({
  config,
  onChange,
  onGenerate,
  isGenerating,
}: ClipConfigProps) {
  const handleNumberChange = (field: keyof ClipConfig, value: number) => {
    onChange({ [field]: value });
  };

  const handleSelectChange = (
    field: keyof ClipConfig,
    event: React.ChangeEvent<HTMLSelectElement>
  ) => {
    onChange({ [field]: event.target.value });
  };

  const handleSwitchChange = (field: keyof ClipConfig) => {
    onChange({ [field]: !config[field] });
  };

  return (
    <Box
      bg="neutral.800"
      borderRadius="xl"
      p={6}
    >
      <VStack spacing={6} align="stretch">
        <Text fontSize="xl" fontWeight="semibold">
          Clip Configuration
        </Text>

        <Grid
          templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)' }}
          gap={6}
        >
          <FormControl>
            <HStack justify="space-between">
              <FormLabel>
                <HStack>
                  <FiClock />
                  <Text>Duration</Text>
                </HStack>
              </FormLabel>
              <Tooltip label="Recommended duration for social media platforms">
                <IconButton
                  aria-label="Duration info"
                  icon={<FiInfo />}
                  size="sm"
                  variant="ghost"
                />
              </Tooltip>
            </HStack>
            <NumberInput
              value={config.duration}
              onChange={(_, value) => handleNumberChange('duration', value)}
              min={15}
              max={60}
              step={5}
            >
              <NumberInputField />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
          </FormControl>

          <FormControl>
            <HStack justify="space-between">
              <FormLabel>
                <HStack>
                  <FiMaximize />
                  <Text>Aspect Ratio</Text>
                </HStack>
              </FormLabel>
            </HStack>
            <Select
              value={config.aspectRatio}
              onChange={(e) => handleSelectChange('aspectRatio', e)}
            >
              <option value="9:16">Vertical (9:16)</option>
              <option value="16:9">Horizontal (16:9)</option>
              <option value="1:1">Square (1:1)</option>
            </Select>
          </FormControl>

          <FormControl>
            <HStack justify="space-between">
              <FormLabel>
                <HStack>
                  <FiMusic />
                  <Text>Background Music</Text>
                </HStack>
              </FormLabel>
              <Switch
                isChecked={config.addBackground}
                onChange={() => handleSwitchChange('addBackground')}
              />
            </HStack>
            {config.addBackground && (
              <Select
                value={config.musicStyle}
                onChange={(e) => handleSelectChange('musicStyle', e)}
                mt={2}
              >
                <option value="upbeat">Upbeat</option>
                <option value="calm">Calm</option>
                <option value="dramatic">Dramatic</option>
                <option value="inspirational">Inspirational</option>
              </Select>
            )}
          </FormControl>

          <FormControl>
            <HStack justify="space-between">
              <FormLabel>
                <HStack>
                  <FiType />
                  <Text>Captions</Text>
                </HStack>
              </FormLabel>
              <Switch
                isChecked={config.addCaptions}
                onChange={() => handleSwitchChange('addCaptions')}
              />
            </HStack>
            {config.addCaptions && (
              <Select
                value={config.captionStyle}
                onChange={(e) => handleSelectChange('captionStyle', e)}
                mt={2}
              >
                <option value="modern">Modern</option>
                <option value="minimal">Minimal</option>
                <option value="bold">Bold</option>
                <option value="subtitle">Subtitle</option>
              </Select>
            )}
          </FormControl>

          <FormControl>
            <HStack justify="space-between">
              <FormLabel>Add Intro</FormLabel>
              <Switch
                isChecked={config.addIntro}
                onChange={() => handleSwitchChange('addIntro')}
              />
            </HStack>
          </FormControl>

          <FormControl>
            <HStack justify="space-between">
              <FormLabel>Add Outro</FormLabel>
              <Switch
                isChecked={config.addOutro}
                onChange={() => handleSwitchChange('addOutro')}
              />
            </HStack>
          </FormControl>
        </Grid>

        <Button
          leftIcon={<FiZap />}
          colorScheme="primary"
          size="lg"
          onClick={onGenerate}
          isLoading={isGenerating}
          loadingText="Generating Clip..."
        >
          Generate Clip
        </Button>
      </VStack>
    </Box>
  );
} 
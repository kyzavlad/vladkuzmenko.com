import React, { useState } from 'react';
import {
  Box,
  VStack,
  HStack,
  FormControl,
  FormLabel,
  Switch,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Select,
  Text,
  Collapse,
  Button,
  useDisclosure,
  Tooltip,
  Icon,
} from '@chakra-ui/react';
import { FiChevronDown, FiChevronUp, FiInfo } from 'react-icons/fi';

interface ConfigurationPanelProps {
  onChange: (config: VideoConfiguration) => void;
}

interface VideoConfiguration {
  duration: {
    min: number;
    max: number;
  };
  faceTracking: boolean;
  silenceRemoval: boolean;
  momentDetection: boolean;
  targetPlatform: string;
  maxClips: number;
  outputQuality: string;
}

const qualityPresets = [
  { label: 'High (1080p)', value: '1080p' },
  { label: 'Medium (720p)', value: '720p' },
  { label: 'Low (480p)', value: '480p' },
];

const platformPresets = [
  { label: 'YouTube Shorts', value: 'youtube_shorts' },
  { label: 'TikTok', value: 'tiktok' },
  { label: 'Instagram Reels', value: 'instagram_reels' },
  { label: 'Twitch Clips', value: 'twitch_clips' },
];

const durationPresets = [
  { label: 'Short (15s)', min: 5, max: 15 },
  { label: 'Medium (30s)', min: 15, max: 30 },
  { label: 'Long (60s)', min: 30, max: 60 },
  { label: 'Custom', min: 5, max: 120 },
];

export const ConfigurationPanel: React.FC<ConfigurationPanelProps> = ({ onChange }) => {
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: false });
  const [config, setConfig] = useState<VideoConfiguration>({
    duration: { min: 15, max: 30 },
    faceTracking: true,
    silenceRemoval: true,
    momentDetection: true,
    targetPlatform: 'tiktok',
    maxClips: 10,
    outputQuality: '1080p',
  });

  const handleConfigChange = (updates: Partial<VideoConfiguration>) => {
    const newConfig = { ...config, ...updates };
    setConfig(newConfig);
    onChange(newConfig);
  };

  return (
    <Box w="full" bg="white" borderRadius="lg" shadow="sm" p={6}>
      <VStack spacing={6} align="stretch">
        <HStack justify="space-between" onClick={onToggle} cursor="pointer">
          <Text fontSize="lg" fontWeight="semibold">
            Configuration Settings
          </Text>
          <Icon as={isOpen ? FiChevronUp : FiChevronDown} />
        </HStack>

        <Collapse in={isOpen}>
          <VStack spacing={6} align="stretch">
            {/* Duration Range */}
            <FormControl>
              <FormLabel>
                <HStack>
                  <Text>Clip Duration Range</Text>
                  <Tooltip label="Set the minimum and maximum duration for generated clips">
                    <Icon as={FiInfo} color="gray.400" />
                  </Tooltip>
                </HStack>
              </FormLabel>
              <Select
                mb={4}
                onChange={(e) => {
                  const preset = durationPresets.find(p => 
                    p.label === e.target.value
                  );
                  if (preset) {
                    handleConfigChange({
                      duration: { min: preset.min, max: preset.max }
                    });
                  }
                }}
              >
                {durationPresets.map((preset) => (
                  <option key={preset.label} value={preset.label}>
                    {preset.label}
                  </option>
                ))}
              </Select>
              <HStack>
                <Text w="60px">{config.duration.min}s</Text>
                <Slider
                  min={5}
                  max={120}
                  value={[config.duration.min, config.duration.max]}
                  onChange={([min, max]) =>
                    handleConfigChange({ duration: { min, max } })
                  }
                >
                  <SliderTrack>
                    <SliderFilledTrack />
                  </SliderTrack>
                  <SliderThumb index={0} />
                  <SliderThumb index={1} />
                </Slider>
                <Text w="60px">{config.duration.max}s</Text>
              </HStack>
            </FormControl>

            {/* Feature Toggles */}
            <FormControl>
              <VStack align="stretch" spacing={4}>
                <HStack justify="space-between">
                  <FormLabel mb={0}>Face Tracking</FormLabel>
                  <Switch
                    isChecked={config.faceTracking}
                    onChange={(e) =>
                      handleConfigChange({ faceTracking: e.target.checked })
                    }
                  />
                </HStack>
                <HStack justify="space-between">
                  <FormLabel mb={0}>Silence Removal</FormLabel>
                  <Switch
                    isChecked={config.silenceRemoval}
                    onChange={(e) =>
                      handleConfigChange({ silenceRemoval: e.target.checked })
                    }
                  />
                </HStack>
                <HStack justify="space-between">
                  <FormLabel mb={0}>Moment Detection</FormLabel>
                  <Switch
                    isChecked={config.momentDetection}
                    onChange={(e) =>
                      handleConfigChange({ momentDetection: e.target.checked })
                    }
                  />
                </HStack>
              </VStack>
            </FormControl>

            {/* Platform Selection */}
            <FormControl>
              <FormLabel>Target Platform</FormLabel>
              <Select
                value={config.targetPlatform}
                onChange={(e) =>
                  handleConfigChange({ targetPlatform: e.target.value })
                }
              >
                {platformPresets.map((platform) => (
                  <option key={platform.value} value={platform.value}>
                    {platform.label}
                  </option>
                ))}
              </Select>
            </FormControl>

            {/* Clip Count */}
            <FormControl>
              <FormLabel>Maximum Number of Clips</FormLabel>
              <HStack>
                <Text w="40px">{config.maxClips}</Text>
                <Slider
                  value={config.maxClips}
                  min={1}
                  max={50}
                  onChange={(value) => handleConfigChange({ maxClips: value })}
                >
                  <SliderTrack>
                    <SliderFilledTrack />
                  </SliderTrack>
                  <SliderThumb />
                </Slider>
              </HStack>
            </FormControl>

            {/* Output Quality */}
            <FormControl>
              <FormLabel>Output Quality</FormLabel>
              <Select
                value={config.outputQuality}
                onChange={(e) =>
                  handleConfigChange({ outputQuality: e.target.value })
                }
              >
                {qualityPresets.map((quality) => (
                  <option key={quality.value} value={quality.value}>
                    {quality.label}
                  </option>
                ))}
              </Select>
            </FormControl>
          </VStack>
        </Collapse>
      </VStack>
    </Box>
  );
};

export default ConfigurationPanel; 
import {
  Box,
  VStack,
  Heading,
  Switch,
  FormControl,
  FormLabel,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Select,
  Divider,
  Text,
  HStack,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Tooltip,
} from '@chakra-ui/react';
import { useState } from 'react';

interface SettingsPanelProps {
  onSettingsChange: (settings: VideoEditSettings) => void;
}

export interface VideoEditSettings {
  subtitles: {
    enabled: boolean;
    style: 'overlay' | 'caption';
    fontSize: number;
    position: 'top' | 'bottom';
  };
  bRoll: {
    enabled: boolean;
    intensity: number;
    timing: 'auto' | 'manual';
  };
  audio: {
    music: {
      enabled: boolean;
      genre: string;
      volume: number;
    };
    soundEffects: {
      enabled: boolean;
      categories: string[];
    };
    voiceEnhancement: {
      enabled: boolean;
      clarity: number;
      denoising: number;
    };
  };
  timing: {
    pauseThreshold: number;
    transitionDuration: number;
  };
}

const defaultSettings: VideoEditSettings = {
  subtitles: {
    enabled: true,
    style: 'overlay',
    fontSize: 24,
    position: 'bottom',
  },
  bRoll: {
    enabled: true,
    intensity: 50,
    timing: 'auto',
  },
  audio: {
    music: {
      enabled: true,
      genre: 'ambient',
      volume: 30,
    },
    soundEffects: {
      enabled: true,
      categories: ['transitions', 'emphasis'],
    },
    voiceEnhancement: {
      enabled: true,
      clarity: 70,
      denoising: 50,
    },
  },
  timing: {
    pauseThreshold: 0.5,
    transitionDuration: 0.5,
  },
};

export function SettingsPanel({ onSettingsChange }: SettingsPanelProps) {
  const [settings, setSettings] = useState<VideoEditSettings>(defaultSettings);

  const updateSettings = (path: string[], value: any) => {
    const newSettings = { ...settings };
    let current = newSettings;
    for (let i = 0; i < path.length - 1; i++) {
      current = current[path[i]];
    }
    current[path[path.length - 1]] = value;
    setSettings(newSettings);
    onSettingsChange(newSettings);
  };

  return (
    <Box p={6} bg="neutral.800" borderRadius="lg">
      <VStack spacing={6} align="stretch">
        {/* Subtitles Section */}
        <Box>
          <Heading size="sm" mb={4}>
            Subtitles
          </Heading>
          <VStack spacing={4}>
            <FormControl display="flex" alignItems="center">
              <FormLabel mb={0}>Enable Subtitles</FormLabel>
              <Switch
                isChecked={settings.subtitles.enabled}
                onChange={(e) =>
                  updateSettings(['subtitles', 'enabled'], e.target.checked)
                }
                colorScheme="primary"
              />
            </FormControl>
            {settings.subtitles.enabled && (
              <>
                <FormControl>
                  <FormLabel>Style</FormLabel>
                  <Select
                    value={settings.subtitles.style}
                    onChange={(e) =>
                      updateSettings(['subtitles', 'style'], e.target.value)
                    }
                  >
                    <option value="overlay">Overlay</option>
                    <option value="caption">Caption</option>
                  </Select>
                </FormControl>
                <FormControl>
                  <FormLabel>Font Size</FormLabel>
                  <NumberInput
                    value={settings.subtitles.fontSize}
                    onChange={(value) =>
                      updateSettings(['subtitles', 'fontSize'], parseInt(value))
                    }
                    min={12}
                    max={48}
                  >
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                </FormControl>
              </>
            )}
          </VStack>
        </Box>

        <Divider />

        {/* B-Roll Section */}
        <Box>
          <Heading size="sm" mb={4}>
            B-Roll
          </Heading>
          <VStack spacing={4}>
            <FormControl display="flex" alignItems="center">
              <FormLabel mb={0}>Enable B-Roll</FormLabel>
              <Switch
                isChecked={settings.bRoll.enabled}
                onChange={(e) =>
                  updateSettings(['bRoll', 'enabled'], e.target.checked)
                }
                colorScheme="primary"
              />
            </FormControl>
            {settings.bRoll.enabled && (
              <>
                <FormControl>
                  <FormLabel>Intensity</FormLabel>
                  <Slider
                    value={settings.bRoll.intensity}
                    onChange={(value) =>
                      updateSettings(['bRoll', 'intensity'], value)
                    }
                    min={0}
                    max={100}
                  >
                    <SliderTrack>
                      <SliderFilledTrack />
                    </SliderTrack>
                    <SliderThumb />
                  </Slider>
                </FormControl>
                <FormControl>
                  <FormLabel>Timing</FormLabel>
                  <Select
                    value={settings.bRoll.timing}
                    onChange={(e) =>
                      updateSettings(['bRoll', 'timing'], e.target.value)
                    }
                  >
                    <option value="auto">Automatic</option>
                    <option value="manual">Manual</option>
                  </Select>
                </FormControl>
              </>
            )}
          </VStack>
        </Box>

        <Divider />

        {/* Audio Section */}
        <Box>
          <Heading size="sm" mb={4}>
            Audio
          </Heading>
          <VStack spacing={6}>
            {/* Background Music */}
            <Box w="full">
              <FormControl display="flex" alignItems="center" mb={4}>
                <FormLabel mb={0}>Background Music</FormLabel>
                <Switch
                  isChecked={settings.audio.music.enabled}
                  onChange={(e) =>
                    updateSettings(['audio', 'music', 'enabled'], e.target.checked)
                  }
                  colorScheme="primary"
                />
              </FormControl>
              {settings.audio.music.enabled && (
                <VStack spacing={4}>
                  <FormControl>
                    <FormLabel>Genre</FormLabel>
                    <Select
                      value={settings.audio.music.genre}
                      onChange={(e) =>
                        updateSettings(['audio', 'music', 'genre'], e.target.value)
                      }
                    >
                      <option value="ambient">Ambient</option>
                      <option value="upbeat">Upbeat</option>
                      <option value="dramatic">Dramatic</option>
                      <option value="corporate">Corporate</option>
                    </Select>
                  </FormControl>
                  <FormControl>
                    <FormLabel>Volume</FormLabel>
                    <Slider
                      value={settings.audio.music.volume}
                      onChange={(value) =>
                        updateSettings(['audio', 'music', 'volume'], value)
                      }
                      min={0}
                      max={100}
                    >
                      <SliderTrack>
                        <SliderFilledTrack />
                      </SliderTrack>
                      <SliderThumb />
                    </Slider>
                  </FormControl>
                </VStack>
              )}
            </Box>

            {/* Voice Enhancement */}
            <Box w="full">
              <FormControl display="flex" alignItems="center" mb={4}>
                <FormLabel mb={0}>Voice Enhancement</FormLabel>
                <Switch
                  isChecked={settings.audio.voiceEnhancement.enabled}
                  onChange={(e) =>
                    updateSettings(
                      ['audio', 'voiceEnhancement', 'enabled'],
                      e.target.checked
                    )
                  }
                  colorScheme="primary"
                />
              </FormControl>
              {settings.audio.voiceEnhancement.enabled && (
                <VStack spacing={4}>
                  <FormControl>
                    <FormLabel>Clarity</FormLabel>
                    <Slider
                      value={settings.audio.voiceEnhancement.clarity}
                      onChange={(value) =>
                        updateSettings(
                          ['audio', 'voiceEnhancement', 'clarity'],
                          value
                        )
                      }
                      min={0}
                      max={100}
                    >
                      <SliderTrack>
                        <SliderFilledTrack />
                      </SliderTrack>
                      <SliderThumb />
                    </Slider>
                  </FormControl>
                  <FormControl>
                    <FormLabel>Noise Reduction</FormLabel>
                    <Slider
                      value={settings.audio.voiceEnhancement.denoising}
                      onChange={(value) =>
                        updateSettings(
                          ['audio', 'voiceEnhancement', 'denoising'],
                          value
                        )
                      }
                      min={0}
                      max={100}
                    >
                      <SliderTrack>
                        <SliderFilledTrack />
                      </SliderTrack>
                      <SliderThumb />
                    </Slider>
                  </FormControl>
                </VStack>
              )}
            </Box>
          </VStack>
        </Box>

        <Divider />

        {/* Timing Section */}
        <Box>
          <Heading size="sm" mb={4}>
            Timing
          </Heading>
          <VStack spacing={4}>
            <FormControl>
              <FormLabel>
                <Tooltip label="Minimum duration of silence to trigger a cut">
                  Pause Threshold (seconds)
                </Tooltip>
              </FormLabel>
              <NumberInput
                value={settings.timing.pauseThreshold}
                onChange={(value) =>
                  updateSettings(['timing', 'pauseThreshold'], parseFloat(value))
                }
                min={0.1}
                max={2}
                step={0.1}
                precision={1}
              >
                <NumberInputField />
                <NumberInputStepper>
                  <NumberIncrementStepper />
                  <NumberDecrementStepper />
                </NumberInputStepper>
              </NumberInput>
            </FormControl>
            <FormControl>
              <FormLabel>
                <Tooltip label="Duration of transition effects between cuts">
                  Transition Duration (seconds)
                </Tooltip>
              </FormLabel>
              <NumberInput
                value={settings.timing.transitionDuration}
                onChange={(value) =>
                  updateSettings(
                    ['timing', 'transitionDuration'],
                    parseFloat(value)
                  )
                }
                min={0.2}
                max={1.5}
                step={0.1}
                precision={1}
              >
                <NumberInputField />
                <NumberInputStepper>
                  <NumberIncrementStepper />
                  <NumberDecrementStepper />
                </NumberInputStepper>
              </NumberInput>
            </FormControl>
          </VStack>
        </Box>
      </VStack>
    </Box>
  );
} 
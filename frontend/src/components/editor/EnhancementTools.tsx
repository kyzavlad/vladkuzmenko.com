import React, { useState } from 'react';
import {
  Box,
  VStack,
  HStack,
  Text,
  Button,
  Input,
  Select,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  IconButton,
  Tabs,
  TabList,
  TabPanels,
  TabPanel,
  Tab,
  Grid,
  GridItem,
  Image,
  useColorModeValue,
  Tooltip,
} from '@chakra-ui/react';
import {
  FiType,
  FiMusic,
  FiImage,
  FiUpload,
  FiPlus,
  FiMinus,
  FiMove,
} from 'react-icons/fi';
import { SketchPicker } from 'react-color';

interface EnhancementToolsProps {
  onCaptionStyleChange: (style: CaptionStyle) => void;
  onMusicAdd: (music: MusicTrack) => void;
  onFilterChange: (filter: VideoFilter) => void;
  onWatermarkChange: (watermark: WatermarkSettings) => void;
  onEndCardChange: (template: EndCardTemplate) => void;
}

interface CaptionStyle {
  fontFamily: string;
  fontSize: number;
  color: string;
  backgroundColor: string;
  position: 'top' | 'center' | 'bottom';
}

interface MusicTrack {
  url: string;
  name: string;
  duration: number;
  volume: number;
  fadeIn: number;
  fadeOut: number;
}

interface VideoFilter {
  name: string;
  intensity: number;
  settings: {
    brightness: number;
    contrast: number;
    saturation: number;
    temperature: number;
  };
}

interface WatermarkSettings {
  image: string;
  position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  size: number;
  opacity: number;
}

interface EndCardTemplate {
  id: string;
  name: string;
  preview: string;
}

const fontOptions = [
  'Arial',
  'Helvetica',
  'Roboto',
  'Open Sans',
  'Montserrat',
  'Inter',
];

const filterPresets = [
  { name: 'None', preview: 'none.jpg' },
  { name: 'Vibrant', preview: 'vibrant.jpg' },
  { name: 'Cinematic', preview: 'cinematic.jpg' },
  { name: 'Vintage', preview: 'vintage.jpg' },
  { name: 'Dramatic', preview: 'dramatic.jpg' },
];

const endCardTemplates = [
  { id: 'simple', name: 'Simple', preview: 'simple.jpg' },
  { id: 'social', name: 'Social Links', preview: 'social.jpg' },
  { id: 'subscribe', name: 'Subscribe', preview: 'subscribe.jpg' },
  { id: 'custom', name: 'Custom', preview: 'custom.jpg' },
];

export const EnhancementTools: React.FC<EnhancementToolsProps> = ({
  onCaptionStyleChange,
  onMusicAdd,
  onFilterChange,
  onWatermarkChange,
  onEndCardChange,
}) => {
  const [captionStyle, setCaptionStyle] = useState<CaptionStyle>({
    fontFamily: 'Arial',
    fontSize: 24,
    color: '#FFFFFF',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    position: 'bottom',
  });

  const [musicTrack, setMusicTrack] = useState<MusicTrack>({
    url: '',
    name: '',
    duration: 0,
    volume: 1,
    fadeIn: 0,
    fadeOut: 0,
  });

  const [filter, setFilter] = useState<VideoFilter>({
    name: 'None',
    intensity: 1,
    settings: {
      brightness: 0,
      contrast: 0,
      saturation: 0,
      temperature: 0,
    },
  });

  const [watermark, setWatermark] = useState<WatermarkSettings>({
    image: '',
    position: 'bottom-right',
    size: 48,
    opacity: 0.8,
  });

  const handleCaptionStyleChange = (updates: Partial<CaptionStyle>) => {
    const newStyle = { ...captionStyle, ...updates };
    setCaptionStyle(newStyle);
    onCaptionStyleChange(newStyle);
  };

  const handleMusicChange = (updates: Partial<MusicTrack>) => {
    const newTrack = { ...musicTrack, ...updates };
    setMusicTrack(newTrack);
    onMusicAdd(newTrack);
  };

  const handleFilterChange = (updates: Partial<VideoFilter>) => {
    const newFilter = { ...filter, ...updates };
    setFilter(newFilter);
    onFilterChange(newFilter);
  };

  const handleWatermarkChange = (updates: Partial<WatermarkSettings>) => {
    const newWatermark = { ...watermark, ...updates };
    setWatermark(newWatermark);
    onWatermarkChange(newWatermark);
  };

  const bgColor = useColorModeValue('white', 'gray.800');

  return (
    <Box w="full" bg={bgColor} borderRadius="lg" shadow="sm" p={4}>
      <Tabs isFitted variant="enclosed">
        <TabList mb={4}>
          <Tab>
            <HStack>
              <FiType />
              <Text>Captions</Text>
            </HStack>
          </Tab>
          <Tab>
            <HStack>
              <FiMusic />
              <Text>Music</Text>
            </HStack>
          </Tab>
          <Tab>
            <HStack>
              <FiImage />
              <Text>Visual</Text>
            </HStack>
          </Tab>
        </TabList>

        <TabPanels>
          {/* Captions Panel */}
          <TabPanel>
            <VStack spacing={4} align="stretch">
              <Select
                value={captionStyle.fontFamily}
                onChange={(e) =>
                  handleCaptionStyleChange({ fontFamily: e.target.value })
                }
              >
                {fontOptions.map((font) => (
                  <option key={font} value={font}>
                    {font}
                  </option>
                ))}
              </Select>

              <HStack justify="space-between">
                <Text>Font Size</Text>
                <HStack w="200px">
                  <IconButton
                    aria-label="Decrease font size"
                    icon={<FiMinus />}
                    size="sm"
                    onClick={() =>
                      handleCaptionStyleChange({
                        fontSize: Math.max(12, captionStyle.fontSize - 2),
                      })
                    }
                  />
                  <Text>{captionStyle.fontSize}px</Text>
                  <IconButton
                    aria-label="Increase font size"
                    icon={<FiPlus />}
                    size="sm"
                    onClick={() =>
                      handleCaptionStyleChange({
                        fontSize: Math.min(72, captionStyle.fontSize + 2),
                      })
                    }
                  />
                </HStack>
              </HStack>

              <Box>
                <Text mb={2}>Colors</Text>
                <HStack>
                  <Box flex={1}>
                    <Text fontSize="sm">Text</Text>
                    <Box position="relative">
                      <Box
                        w="full"
                        h="36px"
                        borderRadius="md"
                        bg={captionStyle.color}
                        cursor="pointer"
                      />
                      <Box position="absolute" top="100%" zIndex={1}>
                        <SketchPicker
                          color={captionStyle.color}
                          onChange={(color) =>
                            handleCaptionStyleChange({ color: color.hex })
                          }
                        />
                      </Box>
                    </Box>
                  </Box>
                  <Box flex={1}>
                    <Text fontSize="sm">Background</Text>
                    <Box position="relative">
                      <Box
                        w="full"
                        h="36px"
                        borderRadius="md"
                        bg={captionStyle.backgroundColor}
                        cursor="pointer"
                      />
                      <Box position="absolute" top="100%" zIndex={1}>
                        <SketchPicker
                          color={captionStyle.backgroundColor}
                          onChange={(color) =>
                            handleCaptionStyleChange({
                              backgroundColor: color.hex,
                            })
                          }
                        />
                      </Box>
                    </Box>
                  </Box>
                </HStack>
              </Box>

              <Box>
                <Text mb={2}>Position</Text>
                <HStack spacing={2}>
                  {(['top', 'center', 'bottom'] as const).map((pos) => (
                    <Button
                      key={pos}
                      variant={captionStyle.position === pos ? 'solid' : 'outline'}
                      onClick={() => handleCaptionStyleChange({ position: pos })}
                      flex={1}
                    >
                      {pos.charAt(0).toUpperCase() + pos.slice(1)}
                    </Button>
                  ))}
                </HStack>
              </Box>
            </VStack>
          </TabPanel>

          {/* Music Panel */}
          <TabPanel>
            <VStack spacing={4} align="stretch">
              <Button
                leftIcon={<FiUpload />}
                onClick={() => {
                  // Implement music upload
                }}
              >
                Upload Music
              </Button>

              <Box>
                <Text mb={2}>Volume</Text>
                <Slider
                  value={musicTrack.volume * 100}
                  onChange={(value) =>
                    handleMusicChange({ volume: value / 100 })
                  }
                >
                  <SliderTrack>
                    <SliderFilledTrack />
                  </SliderTrack>
                  <SliderThumb />
                </Slider>
              </Box>

              <Grid templateColumns="repeat(2, 1fr)" gap={4}>
                <Box>
                  <Text mb={2}>Fade In</Text>
                  <Slider
                    value={musicTrack.fadeIn}
                    max={5}
                    onChange={(value) => handleMusicChange({ fadeIn: value })}
                  >
                    <SliderTrack>
                      <SliderFilledTrack />
                    </SliderTrack>
                    <SliderThumb />
                  </Slider>
                </Box>
                <Box>
                  <Text mb={2}>Fade Out</Text>
                  <Slider
                    value={musicTrack.fadeOut}
                    max={5}
                    onChange={(value) => handleMusicChange({ fadeOut: value })}
                  >
                    <SliderTrack>
                      <SliderFilledTrack />
                    </SliderTrack>
                    <SliderThumb />
                  </Slider>
                </Box>
              </Grid>
            </VStack>
          </TabPanel>

          {/* Visual Panel */}
          <TabPanel>
            <VStack spacing={6} align="stretch">
              {/* Filters */}
              <Box>
                <Text fontWeight="semibold" mb={2}>
                  Filters
                </Text>
                <Grid templateColumns="repeat(5, 1fr)" gap={2}>
                  {filterPresets.map((preset) => (
                    <GridItem key={preset.name}>
                      <Box
                        cursor="pointer"
                        onClick={() =>
                          handleFilterChange({ name: preset.name })
                        }
                        borderWidth={2}
                        borderColor={
                          filter.name === preset.name
                            ? 'blue.500'
                            : 'transparent'
                        }
                        borderRadius="md"
                        overflow="hidden"
                      >
                        <Image src={preset.preview} alt={preset.name} />
                        <Text
                          fontSize="sm"
                          textAlign="center"
                          p={1}
                        >
                          {preset.name}
                        </Text>
                      </Box>
                    </GridItem>
                  ))}
                </Grid>
              </Box>

              {/* Watermark */}
              <Box>
                <Text fontWeight="semibold" mb={2}>
                  Watermark
                </Text>
                <VStack spacing={4} align="stretch">
                  <Button
                    leftIcon={<FiUpload />}
                    onClick={() => {
                      // Implement watermark upload
                    }}
                  >
                    Upload Watermark
                  </Button>

                  <Grid templateColumns="repeat(2, 1fr)" gap={4}>
                    <Box>
                      <Text mb={2}>Size</Text>
                      <Slider
                        value={watermark.size}
                        min={24}
                        max={120}
                        onChange={(value) =>
                          handleWatermarkChange({ size: value })
                        }
                      >
                        <SliderTrack>
                          <SliderFilledTrack />
                        </SliderTrack>
                        <SliderThumb />
                      </Slider>
                    </Box>
                    <Box>
                      <Text mb={2}>Opacity</Text>
                      <Slider
                        value={watermark.opacity * 100}
                        onChange={(value) =>
                          handleWatermarkChange({ opacity: value / 100 })
                        }
                      >
                        <SliderTrack>
                          <SliderFilledTrack />
                        </SliderTrack>
                        <SliderThumb />
                      </Slider>
                    </Box>
                  </Grid>

                  <Box>
                    <Text mb={2}>Position</Text>
                    <Grid templateColumns="repeat(2, 1fr)" gap={2}>
                      {(['top-left', 'top-right', 'bottom-left', 'bottom-right'] as const).map(
                        (pos) => (
                          <Button
                            key={pos}
                            variant={
                              watermark.position === pos ? 'solid' : 'outline'
                            }
                            onClick={() =>
                              handleWatermarkChange({ position: pos })
                            }
                          >
                            {pos
                              .split('-')
                              .map(
                                (word) =>
                                  word.charAt(0).toUpperCase() + word.slice(1)
                              )
                              .join(' ')}
                          </Button>
                        )
                      )}
                    </Grid>
                  </Box>
                </VStack>
              </Box>

              {/* End Card Templates */}
              <Box>
                <Text fontWeight="semibold" mb={2}>
                  End Card Template
                </Text>
                <Grid templateColumns="repeat(2, 1fr)" gap={4}>
                  {endCardTemplates.map((template) => (
                    <GridItem key={template.id}>
                      <Box
                        cursor="pointer"
                        onClick={() => onEndCardChange(template)}
                        borderWidth={2}
                        borderColor="transparent"
                        borderRadius="md"
                        overflow="hidden"
                        _hover={{ borderColor: 'blue.500' }}
                      >
                        <Image
                          src={template.preview}
                          alt={template.name}
                          w="full"
                        />
                        <Text
                          fontSize="sm"
                          textAlign="center"
                          p={2}
                        >
                          {template.name}
                        </Text>
                      </Box>
                    </GridItem>
                  ))}
                </Grid>
              </Box>
            </VStack>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Box>
  );
};

export default EnhancementTools; 
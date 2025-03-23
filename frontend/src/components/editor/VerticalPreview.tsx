import React, { useEffect, useRef, useState } from 'react';
import {
  Box,
  IconButton,
  VStack,
  useColorModeValue,
  Text,
  HStack,
} from '@chakra-ui/react';
import {
  FiMaximize,
  FiMinimize,
  FiZoomIn,
  FiZoomOut,
} from 'react-icons/fi';

interface Caption {
  text: string;
  startTime: number;
  endTime: number;
  style: CaptionStyle;
}

interface CaptionStyle {
  fontFamily: string;
  fontSize: number;
  color: string;
  backgroundColor: string;
  position: 'top' | 'center' | 'bottom';
}

interface WatermarkSettings {
  image: string;
  position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  size: number;
  opacity: number;
}

interface VerticalPreviewProps {
  videoUrl: string;
  currentTime: number;
  aspectRatio: number;
  smartCrop: boolean;
  captions: Caption[];
  watermark?: WatermarkSettings;
}

export const VerticalPreview: React.FC<VerticalPreviewProps> = ({
  videoUrl,
  currentTime,
  aspectRatio,
  smartCrop,
  captions,
  watermark,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [zoom, setZoom] = useState(1);

  const bgColor = useColorModeValue('gray.100', 'gray.900');
  const deviceColor = useColorModeValue('gray.200', 'gray.700');
  const statusBarColor = useColorModeValue('gray.300', 'gray.600');

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.currentTime = currentTime;
    }
  }, [currentTime]);

  const handleFullscreenToggle = () => {
    if (!document.fullscreenElement) {
      videoRef.current?.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const handleZoomIn = () => {
    setZoom((prev) => Math.min(prev + 0.1, 2));
  };

  const handleZoomOut = () => {
    setZoom((prev) => Math.max(prev - 0.1, 0.5));
  };

  const getActiveCaptions = () => {
    return captions.filter(
      (caption) =>
        currentTime >= caption.startTime && currentTime <= caption.endTime
    );
  };

  const getWatermarkStyles = () => {
    if (!watermark) return {};

    const position: { [key: string]: string } = {
      'top-left': 'top: 16px; left: 16px;',
      'top-right': 'top: 16px; right: 16px;',
      'bottom-left': 'bottom: 16px; left: 16px;',
      'bottom-right': 'bottom: 16px; right: 16px;',
    };

    return {
      position: 'absolute',
      ...position[watermark.position],
      width: `${watermark.size}px`,
      height: `${watermark.size}px`,
      opacity: watermark.opacity,
    };
  };

  return (
    <Box
      bg={bgColor}
      p={6}
      borderRadius="lg"
      display="flex"
      justifyContent="center"
      alignItems="center"
    >
      {/* Mobile Device Frame */}
      <Box
        bg={deviceColor}
        borderRadius="3xl"
        p={3}
        maxW="360px"
        w="full"
        position="relative"
        transform={`scale(${zoom})`}
        transformOrigin="center"
        transition="transform 0.2s"
      >
        {/* Status Bar */}
        <Box
          bg={statusBarColor}
          h="24px"
          borderTopRadius="xl"
          mb={1}
          px={4}
          display="flex"
          alignItems="center"
          justifyContent="space-between"
        >
          <Text fontSize="xs">9:41</Text>
          <HStack spacing={2}>
            <Text fontSize="xs">5G</Text>
            <Text fontSize="xs">100%</Text>
          </HStack>
        </Box>

        {/* Video Container */}
        <Box
          position="relative"
          paddingTop={`${(1 / aspectRatio) * 100}%`}
          bg="black"
          borderRadius="lg"
          overflow="hidden"
        >
          <video
            ref={videoRef}
            src={videoUrl}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              objectFit: smartCrop ? 'cover' : 'contain',
            }}
          />

          {/* Smart Crop Overlay */}
          {smartCrop && (
            <Box
              position="absolute"
              top="50%"
              left="50%"
              transform="translate(-50%, -50%)"
              border="2px dashed white"
              w="80%"
              h="80%"
              opacity={0.5}
              pointerEvents="none"
            />
          )}

          {/* Captions */}
          {getActiveCaptions().map((caption, index) => (
            <Box
              key={index}
              position="absolute"
              left="50%"
              transform="translateX(-50%)"
              {...(caption.style.position === 'top' && { top: '16px' })}
              {...(caption.style.position === 'center' && {
                top: '50%',
                transform: 'translate(-50%, -50%)',
              })}
              {...(caption.style.position === 'bottom' && { bottom: '16px' })}
              maxW="90%"
              textAlign="center"
            >
              <Text
                fontSize={caption.style.fontSize}
                fontFamily={caption.style.fontFamily}
                color={caption.style.color}
                bg={caption.style.backgroundColor}
                px={3}
                py={1}
                borderRadius="md"
              >
                {caption.text}
              </Text>
            </Box>
          ))}

          {/* Watermark */}
          {watermark && (
            <Box
              as="img"
              src={watermark.image}
              sx={getWatermarkStyles()}
              alt="Watermark"
            />
          )}
        </Box>

        {/* Home Indicator */}
        <Box
          w="32px"
          h="4px"
          bg={statusBarColor}
          borderRadius="full"
          mx="auto"
          mt={2}
        />
      </Box>

      {/* Controls */}
      <VStack position="absolute" right={4} top="50%" transform="translateY(-50%)">
        <IconButton
          aria-label={isFullscreen ? 'Exit Fullscreen' : 'Enter Fullscreen'}
          icon={isFullscreen ? <FiMinimize /> : <FiMaximize />}
          onClick={handleFullscreenToggle}
          size="sm"
        />
        <IconButton
          aria-label="Zoom In"
          icon={<FiZoomIn />}
          onClick={handleZoomIn}
          size="sm"
          isDisabled={zoom >= 2}
        />
        <IconButton
          aria-label="Zoom Out"
          icon={<FiZoomOut />}
          onClick={handleZoomOut}
          size="sm"
          isDisabled={zoom <= 0.5}
        />
      </VStack>
    </Box>
  );
};

export default VerticalPreview; 
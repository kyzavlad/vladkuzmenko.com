import React from 'react';
import { Box, Text, VStack } from '@chakra-ui/react';

export function VideoGuidelines() {
  return (
    <Box
      position="absolute"
      top={0}
      left={0}
      right={0}
      bottom={0}
      pointerEvents="none"
    >
      {/* Face outline guide */}
      <Box
        position="absolute"
        top="50%"
        left="50%"
        transform="translate(-50%, -50%)"
        w="280px"
        h="320px"
        border="2px dashed"
        borderColor="primary.500"
        borderRadius="full"
        opacity={0.6}
      />

      {/* Center crosshair */}
      <Box
        position="absolute"
        top="50%"
        left="50%"
        transform="translate(-50%, -50%)"
        w="20px"
        h="20px"
      >
        <Box
          position="absolute"
          top="50%"
          left={0}
          right={0}
          h="2px"
          bg="primary.500"
          transform="translateY(-50%)"
        />
        <Box
          position="absolute"
          left="50%"
          top={0}
          bottom={0}
          w="2px"
          bg="primary.500"
          transform="translateX(-50%)"
        />
      </Box>

      {/* Instructions */}
      <VStack
        position="absolute"
        bottom={8}
        left={0}
        right={0}
        spacing={2}
        px={4}
      >
        <Text
          color="white"
          fontSize="lg"
          fontWeight="semibold"
          textAlign="center"
          textShadow="0 0 8px rgba(0,0,0,0.6)"
        >
          Position your face within the circle
        </Text>
        <Text
          color="white"
          fontSize="sm"
          textAlign="center"
          textShadow="0 0 8px rgba(0,0,0,0.6)"
        >
          Ensure good lighting and a neutral expression
        </Text>
      </VStack>

      {/* Recording indicators */}
      <Box
        position="absolute"
        top={4}
        left={4}
        px={3}
        py={1}
        bg="red.500"
        borderRadius="full"
      >
        <Text color="white" fontSize="sm" fontWeight="medium">
          REC
        </Text>
      </Box>
    </Box>
  );
} 
import React, { useState } from 'react';
import {
  Box,
  Button,
  HStack,
  Image,
  Text,
  VStack,
  useBreakpointValue,
} from '@chakra-ui/react';
import { FiChevronLeft, FiChevronRight } from 'react-icons/fi';

interface AvatarStyle {
  id: string;
  name: string;
  description: string;
  previewUrl: string;
  category: 'realistic' | 'stylized' | 'anime' | 'cartoon';
}

interface StyleSelectionProps {
  styles: AvatarStyle[];
  selectedStyle: string;
  onStyleSelect: (styleId: string) => void;
}

export function StyleSelection({
  styles,
  selectedStyle,
  onStyleSelect,
}: StyleSelectionProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const itemsPerPage = useBreakpointValue({ base: 1, md: 2, lg: 3 }) || 1;

  const nextSlide = () => {
    setCurrentIndex((prev) =>
      Math.min(prev + 1, styles.length - itemsPerPage)
    );
  };

  const prevSlide = () => {
    setCurrentIndex((prev) => Math.max(prev - 1, 0));
  };

  const visibleStyles = styles.slice(
    currentIndex,
    currentIndex + itemsPerPage
  );

  return (
    <VStack spacing={6} w="full">
      <Text fontSize="xl" fontWeight="semibold">
        Choose Your Avatar Style
      </Text>

      <Box position="relative" w="full">
        <HStack spacing={4} justify="center" w="full">
          <Button
            leftIcon={<FiChevronLeft />}
            onClick={prevSlide}
            isDisabled={currentIndex === 0}
            variant="ghost"
          />

          <HStack spacing={6} overflow="hidden" flex={1}>
            {visibleStyles.map((style) => (
              <Box
                key={style.id}
                flex={1}
                cursor="pointer"
                onClick={() => onStyleSelect(style.id)}
                position="relative"
                transition="transform 0.2s"
                _hover={{ transform: 'scale(1.02)' }}
              >
                <Box
                  borderRadius="lg"
                  overflow="hidden"
                  borderWidth={2}
                  borderColor={
                    selectedStyle === style.id
                      ? 'primary.500'
                      : 'transparent'
                  }
                >
                  <Image
                    src={style.previewUrl}
                    alt={style.name}
                    w="full"
                    h="200px"
                    objectFit="cover"
                  />
                  <Box p={4} bg="neutral.800">
                    <Text
                      fontSize="lg"
                      fontWeight="semibold"
                      mb={1}
                    >
                      {style.name}
                    </Text>
                    <Text
                      fontSize="sm"
                      color="neutral.400"
                      noOfLines={2}
                    >
                      {style.description}
                    </Text>
                  </Box>
                </Box>

                {selectedStyle === style.id && (
                  <Box
                    position="absolute"
                    top={2}
                    right={2}
                    w={6}
                    h={6}
                    borderRadius="full"
                    bg="primary.500"
                    display="flex"
                    alignItems="center"
                    justifyContent="center"
                  >
                    <Box
                      as="span"
                      w={2}
                      h={2}
                      borderRadius="full"
                      bg="white"
                    />
                  </Box>
                )}
              </Box>
            ))}
          </HStack>

          <Button
            rightIcon={<FiChevronRight />}
            onClick={nextSlide}
            isDisabled={currentIndex >= styles.length - itemsPerPage}
            variant="ghost"
          />
        </HStack>
      </Box>

      <HStack spacing={2} mt={4}>
        {Array.from({ length: Math.ceil(styles.length / itemsPerPage) }).map(
          (_, index) => (
            <Box
              key={index}
              w={2}
              h={2}
              borderRadius="full"
              bg={
                index === Math.floor(currentIndex / itemsPerPage)
                  ? 'primary.500'
                  : 'neutral.600'
              }
              cursor="pointer"
              onClick={() =>
                setCurrentIndex(index * itemsPerPage)
              }
            />
          )
        )}
      </HStack>
    </VStack>
  );
} 
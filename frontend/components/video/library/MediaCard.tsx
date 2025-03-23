import { useState } from 'react';
import {
  Box,
  Image,
  Text,
  IconButton,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  AspectRatio,
  HStack,
  VStack,
  useDisclosure,
} from '@chakra-ui/react';
import { FiMoreVertical, FiEdit2, FiTrash2, FiDownload } from 'react-icons/fi';
import { Video } from '@/lib/redux/services/videoService';
import { formatDuration, formatFileSize, formatDate } from '@/utils/format';

interface MediaCardProps {
  video: Video;
}

export function MediaCard({ video }: MediaCardProps) {
  const [isHovered, setIsHovered] = useState(false);
  const { isOpen, onOpen, onClose } = useDisclosure();

  const handleEdit = () => {
    // TODO: Implement edit functionality
  };

  const handleDelete = () => {
    // TODO: Implement delete functionality
  };

  const handleDownload = () => {
    // TODO: Implement download functionality
  };

  return (
    <Box
      borderRadius="lg"
      overflow="hidden"
      bg="neutral.800"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      transition="all 0.2s"
      _hover={{
        transform: 'translateY(-2px)',
        shadow: 'lg',
      }}
    >
      <AspectRatio ratio={16 / 9}>
        <Box position="relative">
          <Image
            src={video.thumbnailUrl}
            alt={video.title}
            objectFit="cover"
            w="full"
            h="full"
            fallbackSrc="/images/video-placeholder.png"
          />
          {isHovered && (
            <Box
              position="absolute"
              top={0}
              left={0}
              right={0}
              bottom={0}
              bg="blackAlpha.600"
              display="flex"
              alignItems="center"
              justifyContent="center"
            >
              <IconButton
                aria-label="Play video"
                icon={<FiEdit2 />}
                colorScheme="primary"
                size="lg"
                onClick={handleEdit}
              />
            </Box>
          )}
          <Box
            position="absolute"
            bottom={2}
            right={2}
            bg="blackAlpha.700"
            px={2}
            py={1}
            borderRadius="md"
          >
            <Text fontSize="sm" color="white">
              {formatDuration(video.duration)}
            </Text>
          </Box>
        </Box>
      </AspectRatio>

      <Box p={4}>
        <HStack justify="space-between" align="start" mb={2}>
          <VStack align="start" spacing={1}>
            <Text fontWeight="semibold" noOfLines={1}>
              {video.title}
            </Text>
            <Text fontSize="sm" color="neutral.400">
              {formatDate(video.createdAt)}
            </Text>
          </VStack>
          <Menu isOpen={isOpen} onClose={onClose}>
            <MenuButton
              as={IconButton}
              icon={<FiMoreVertical />}
              variant="ghost"
              size="sm"
              onClick={onOpen}
            />
            <MenuList>
              <MenuItem icon={<FiEdit2 />} onClick={handleEdit}>
                Edit
              </MenuItem>
              <MenuItem icon={<FiDownload />} onClick={handleDownload}>
                Download
              </MenuItem>
              <MenuItem icon={<FiTrash2 />} onClick={handleDelete} color="red.500">
                Delete
              </MenuItem>
            </MenuList>
          </Menu>
        </HStack>

        <Text fontSize="sm" color="neutral.400">
          {formatFileSize(video.metadata.size)} • {video.metadata.resolution}
        </Text>
      </Box>
    </Box>
  );
} 
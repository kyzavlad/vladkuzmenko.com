import {
  Box,
  HStack,
  Image,
  Text,
  IconButton,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Badge,
  AspectRatio,
} from '@chakra-ui/react';
import { FiMoreVertical, FiEdit2, FiTrash2, FiDownload } from 'react-icons/fi';
import { Video } from '@/lib/redux/services/videoService';
import { formatDuration, formatFileSize, formatDate } from '@/utils/format';

interface MediaListItemProps {
  video: Video;
}

export function MediaListItem({ video }: MediaListItemProps) {
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
    <HStack
      spacing={4}
      p={4}
      bg="neutral.800"
      borderRadius="lg"
      transition="all 0.2s"
      _hover={{
        bg: 'neutral.700',
      }}
    >
      <Box w="200px" flexShrink={0}>
        <AspectRatio ratio={16 / 9}>
          <Image
            src={video.thumbnailUrl}
            alt={video.title}
            objectFit="cover"
            borderRadius="md"
            fallbackSrc="/images/video-placeholder.png"
          />
        </AspectRatio>
      </Box>

      <Box flex={1} minW={0}>
        <HStack justify="space-between" mb={2}>
          <Text fontWeight="semibold" noOfLines={1}>
            {video.title}
          </Text>
          <Menu>
            <MenuButton
              as={IconButton}
              icon={<FiMoreVertical />}
              variant="ghost"
              size="sm"
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

        <Text fontSize="sm" color="neutral.400" noOfLines={2} mb={2}>
          {video.description}
        </Text>

        <HStack spacing={4} fontSize="sm" color="neutral.400">
          <Text>{formatDate(video.createdAt)}</Text>
          <Text>{formatDuration(video.duration)}</Text>
          <Text>{formatFileSize(video.metadata.size)}</Text>
          <Text>{video.metadata.resolution}</Text>
          <Badge
            colorScheme={
              video.status === 'ready'
                ? 'green'
                : video.status === 'processing'
                ? 'yellow'
                : 'red'
            }
          >
            {video.status}
          </Badge>
        </HStack>
      </Box>
    </HStack>
  );
} 
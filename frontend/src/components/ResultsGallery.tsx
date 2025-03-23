import React, { useState, useRef } from 'react';
import {
  Box,
  SimpleGrid,
  VStack,
  HStack,
  Text,
  Button,
  IconButton,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Checkbox,
  Badge,
  useDisclosure,
  Icon,
  Tooltip,
  Input,
  InputGroup,
  InputLeftElement,
  Select,
} from '@chakra-ui/react';
import {
  FiGrid,
  FiList,
  FiFilter,
  FiDownload,
  FiClock,
  FiTrendingUp,
  FiSearch,
  FiMoreVertical,
  FiPlay,
  FiPause,
} from 'react-icons/fi';

interface VideoClip {
  id: string;
  title: string;
  thumbnail: string;
  duration: number;
  engagementScore: number;
  videoUrl: string;
  platform: string;
  category: string;
  createdAt: Date;
}

interface ResultsGalleryProps {
  clips: VideoClip[];
  onExport: (clips: VideoClip[]) => void;
}

type ViewMode = 'grid' | 'list';
type SortOption = 'date' | 'duration' | 'engagement';

export const ResultsGallery: React.FC<ResultsGalleryProps> = ({
  clips,
  onExport,
}) => {
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [selectedClips, setSelectedClips] = useState<Set<string>>(new Set());
  const [sortBy, setSortBy] = useState<SortOption>('date');
  const [filterText, setFilterText] = useState('');
  const [hoveredClip, setHoveredClip] = useState<string | null>(null);
  const [playingClip, setPlayingClip] = useState<string | null>(null);
  const videoRefs = useRef<{ [key: string]: HTMLVideoElement }>({});

  const handleClipSelect = (clipId: string) => {
    const newSelected = new Set(selectedClips);
    if (newSelected.has(clipId)) {
      newSelected.delete(clipId);
    } else {
      newSelected.add(clipId);
    }
    setSelectedClips(newSelected);
  };

  const handleSelectAll = () => {
    if (selectedClips.size === filteredClips.length) {
      setSelectedClips(new Set());
    } else {
      setSelectedClips(new Set(filteredClips.map((clip) => clip.id)));
    }
  };

  const handleSort = (option: SortOption) => {
    setSortBy(option);
  };

  const handleExport = () => {
    const clipsToExport = clips.filter((clip) => selectedClips.has(clip.id));
    onExport(clipsToExport);
  };

  const handleClipHover = (clipId: string | null, isEnter: boolean) => {
    setHoveredClip(isEnter ? clipId : null);
    if (clipId && isEnter && videoRefs.current[clipId]) {
      videoRefs.current[clipId].play().catch(() => {});
    } else if (!isEnter && clipId && videoRefs.current[clipId]) {
      videoRefs.current[clipId].pause();
      videoRefs.current[clipId].currentTime = 0;
    }
  };

  const formatDuration = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  // Filter and sort clips
  const filteredClips = clips
    .filter((clip) =>
      filterText
        ? clip.title.toLowerCase().includes(filterText.toLowerCase()) ||
          clip.platform.toLowerCase().includes(filterText.toLowerCase()) ||
          clip.category.toLowerCase().includes(filterText.toLowerCase())
        : true
    )
    .sort((a, b) => {
      switch (sortBy) {
        case 'duration':
          return a.duration - b.duration;
        case 'engagement':
          return b.engagementScore - a.engagementScore;
        default:
          return b.createdAt.getTime() - a.createdAt.getTime();
      }
    });

  const renderClip = (clip: VideoClip) => {
    const isSelected = selectedClips.has(clip.id);
    const isHovered = hoveredClip === clip.id;

    return (
      <Box
        key={clip.id}
        position="relative"
        borderRadius="lg"
        overflow="hidden"
        bg="white"
        shadow="sm"
        onMouseEnter={() => handleClipHover(clip.id, true)}
        onMouseLeave={() => handleClipHover(clip.id, false)}
      >
        <Box position="relative" pb={viewMode === 'grid' ? '56.25%' : '0'}>
          <Box
            position={viewMode === 'grid' ? 'absolute' : 'relative'}
            top="0"
            left="0"
            width="100%"
            height="100%"
          >
            <video
              ref={(el) => {
                if (el) videoRefs.current[clip.id] = el;
              }}
              src={clip.videoUrl}
              poster={clip.thumbnail}
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
              }}
              muted
              playsInline
            />
          </Box>
        </Box>

        <Box
          position="absolute"
          top={2}
          right={2}
          zIndex={1}
        >
          <Checkbox
            isChecked={isSelected}
            onChange={() => handleClipSelect(clip.id)}
            colorScheme="blue"
            size="lg"
          />
        </Box>

        <VStack
          p={4}
          align="stretch"
          spacing={2}
          bg={isHovered ? 'blackAlpha.700' : 'blackAlpha.600'}
          position="absolute"
          bottom={0}
          left={0}
          right={0}
          color="white"
          transition="all 0.2s"
        >
          <HStack justify="space-between">
            <Text fontWeight="semibold" isTruncated>
              {clip.title}
            </Text>
            <Badge colorScheme="green">
              {Math.round(clip.engagementScore * 100)}%
            </Badge>
          </HStack>
          <HStack spacing={4} fontSize="sm">
            <HStack>
              <Icon as={FiClock} />
              <Text>{formatDuration(clip.duration)}</Text>
            </HStack>
            <HStack>
              <Icon as={FiTrendingUp} />
              <Text>{clip.platform}</Text>
            </HStack>
          </HStack>
        </VStack>
      </Box>
    );
  };

  return (
    <VStack spacing={6} w="full">
      {/* Controls */}
      <HStack w="full" justify="space-between" p={4} bg="white" borderRadius="lg" shadow="sm">
        <HStack spacing={4}>
          <IconButton
            aria-label="Grid view"
            icon={<Icon as={FiGrid} />}
            onClick={() => setViewMode('grid')}
            colorScheme={viewMode === 'grid' ? 'blue' : 'gray'}
            variant="ghost"
          />
          <IconButton
            aria-label="List view"
            icon={<Icon as={FiList} />}
            onClick={() => setViewMode('list')}
            colorScheme={viewMode === 'list' ? 'blue' : 'gray'}
            variant="ghost"
          />
          <InputGroup maxW="300px">
            <InputLeftElement>
              <Icon as={FiSearch} color="gray.400" />
            </InputLeftElement>
            <Input
              placeholder="Search clips..."
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
            />
          </InputGroup>
        </HStack>

        <HStack spacing={4}>
          <Select
            value={sortBy}
            onChange={(e) => handleSort(e.target.value as SortOption)}
            w="200px"
          >
            <option value="date">Sort by Date</option>
            <option value="duration">Sort by Duration</option>
            <option value="engagement">Sort by Engagement</option>
          </Select>

          <Button
            leftIcon={<Icon as={FiDownload} />}
            colorScheme="blue"
            isDisabled={selectedClips.size === 0}
            onClick={handleExport}
          >
            Export Selected
          </Button>
        </HStack>
      </HStack>

      {/* Gallery */}
      {viewMode === 'grid' ? (
        <SimpleGrid columns={[1, 2, 3, 4]} spacing={6} w="full">
          {filteredClips.map(renderClip)}
        </SimpleGrid>
      ) : (
        <VStack spacing={4} w="full">
          {filteredClips.map(renderClip)}
        </VStack>
      )}
    </VStack>
  );
};

export default ResultsGallery; 
import { useState, useMemo } from 'react';
import {
  Box,
  Grid,
  HStack,
  Icon,
  Input,
  Select,
  IconButton,
  useBreakpointValue,
  Text,
  Flex,
} from '@chakra-ui/react';
import { FiGrid, FiList, FiSearch } from 'react-icons/fi';
import { useGetVideosQuery } from '@/lib/redux/services/videoService';
import { MediaCard } from './MediaCard';
import { MediaListItem } from './MediaListItem';

type ViewMode = 'grid' | 'list';
type SortOption = 'date' | 'name' | 'duration' | 'size';

export function MediaLibrary() {
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<SortOption>('date');

  const { data: videos = [], isLoading, error } = useGetVideosQuery();

  const columns = useBreakpointValue({ base: 1, sm: 2, md: 3, lg: 4 }) || 1;

  const filteredAndSortedVideos = useMemo(() => {
    let result = [...videos];

    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        (video) =>
          video.title.toLowerCase().includes(query) ||
          video.description.toLowerCase().includes(query)
      );
    }

    // Apply sorting
    result.sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.title.localeCompare(b.title);
        case 'duration':
          return b.duration - a.duration;
        case 'size':
          return b.metadata.size - a.metadata.size;
        case 'date':
        default:
          return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
      }
    });

    return result;
  }, [videos, searchQuery, sortBy]);

  if (isLoading) {
    return <Text>Loading...</Text>;
  }

  if (error) {
    return <Text color="red.500">Error loading videos</Text>;
  }

  return (
    <Box>
      <HStack spacing={4} mb={6}>
        <Input
          placeholder="Search videos..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          leftElement={<Icon as={FiSearch} color="neutral.400" />}
          maxW="sm"
        />
        <Select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as SortOption)}
          maxW="xs"
        >
          <option value="date">Sort by Date</option>
          <option value="name">Sort by Name</option>
          <option value="duration">Sort by Duration</option>
          <option value="size">Sort by Size</option>
        </Select>
        <HStack>
          <IconButton
            aria-label="Grid view"
            icon={<Icon as={FiGrid} />}
            variant={viewMode === 'grid' ? 'solid' : 'ghost'}
            colorScheme="primary"
            onClick={() => setViewMode('grid')}
          />
          <IconButton
            aria-label="List view"
            icon={<Icon as={FiList} />}
            variant={viewMode === 'list' ? 'solid' : 'ghost'}
            colorScheme="primary"
            onClick={() => setViewMode('list')}
          />
        </HStack>
      </HStack>

      {viewMode === 'grid' ? (
        <Grid
          templateColumns={`repeat(${columns}, 1fr)`}
          gap={6}
          px={2}
        >
          {filteredAndSortedVideos.map((video) => (
            <MediaCard key={video.id} video={video} />
          ))}
        </Grid>
      ) : (
        <Flex direction="column" gap={4}>
          {filteredAndSortedVideos.map((video) => (
            <MediaListItem key={video.id} video={video} />
          ))}
        </Flex>
      )}

      {filteredAndSortedVideos.length === 0 && (
        <Text textAlign="center" color="neutral.500" mt={8}>
          No videos found
        </Text>
      )}
    </Box>
  );
} 
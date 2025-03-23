import React, { useEffect, useState } from 'react';
import {
  Box,
  Button,
  Grid,
  HStack,
  Icon,
  Input,
  InputGroup,
  InputLeftElement,
  Select,
  Text,
  VStack,
  useToast,
} from '@chakra-ui/react';
import { FiGrid, FiList, FiSearch } from 'react-icons/fi';
import { MediaCard } from './MediaCard';
import { MediaListItem } from './MediaListItem';
import { formatDuration } from '@/utils/format';

interface Video {
  id: string;
  filename: string;
  size: number;
  mimeType: string;
  uploadedAt: string;
  duration?: number;
  resolution?: {
    width: number;
    height: number;
  };
  url: string;
}

interface VideoListProps {
  onVideoSelect: (video: Video) => void;
}

type ViewMode = 'grid' | 'list';
type SortField = 'uploadedAt' | 'filename' | 'duration' | 'size';
type SortOrder = 'asc' | 'desc';

export function VideoList({ onVideoSelect }: VideoListProps) {
  const [videos, setVideos] = useState<Video[]>([]);
  const [filteredVideos, setFilteredVideos] = useState<Video[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortField, setSortField] = useState<SortField>('uploadedAt');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
  const toast = useToast();

  useEffect(() => {
    fetchVideos();
  }, []);

  useEffect(() => {
    const filtered = filterVideos(videos, searchQuery);
    const sorted = sortVideos(filtered, sortField, sortOrder);
    setFilteredVideos(sorted);
  }, [videos, searchQuery, sortField, sortOrder]);

  const fetchVideos = async () => {
    try {
      const response = await fetch('/api/videos/upload');
      if (!response.ok) {
        throw new Error('Failed to fetch videos');
      }
      const data = await response.json();
      setVideos(data);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to load videos. Please try again.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const filterVideos = (videos: Video[], query: string): Video[] => {
    if (!query) return videos;

    const lowercaseQuery = query.toLowerCase();
    return videos.filter((video) =>
      video.filename.toLowerCase().includes(lowercaseQuery)
    );
  };

  const sortVideos = (
    videos: Video[],
    field: SortField,
    order: SortOrder
  ): Video[] => {
    return [...videos].sort((a, b) => {
      let comparison = 0;

      switch (field) {
        case 'uploadedAt':
          comparison = new Date(a.uploadedAt).getTime() - new Date(b.uploadedAt).getTime();
          break;
        case 'filename':
          comparison = a.filename.localeCompare(b.filename);
          break;
        case 'duration':
          comparison = (a.duration || 0) - (b.duration || 0);
          break;
        case 'size':
          comparison = a.size - b.size;
          break;
      }

      return order === 'asc' ? comparison : -comparison;
    });
  };

  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(event.target.value);
  };

  const handleSortFieldChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSortField(event.target.value as SortField);
  };

  const handleSortOrderChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSortOrder(event.target.value as SortOrder);
  };

  const toggleViewMode = () => {
    setViewMode((current) => (current === 'grid' ? 'list' : 'grid'));
  };

  if (isLoading) {
    return (
      <Box p={8} textAlign="center">
        <Text>Loading videos...</Text>
      </Box>
    );
  }

  return (
    <VStack spacing={6} align="stretch" w="full">
      <HStack spacing={4} justify="space-between">
        <InputGroup maxW="400px">
          <InputLeftElement pointerEvents="none">
            <Icon as={FiSearch} color="gray.400" />
          </InputLeftElement>
          <Input
            placeholder="Search videos..."
            value={searchQuery}
            onChange={handleSearchChange}
          />
        </InputGroup>

        <HStack spacing={4}>
          <Select value={sortField} onChange={handleSortFieldChange} w="150px">
            <option value="uploadedAt">Upload Date</option>
            <option value="filename">Filename</option>
            <option value="duration">Duration</option>
            <option value="size">Size</option>
          </Select>

          <Select value={sortOrder} onChange={handleSortOrderChange} w="100px">
            <option value="desc">Desc</option>
            <option value="asc">Asc</option>
          </Select>

          <Button
            leftIcon={<Icon as={viewMode === 'grid' ? FiList : FiGrid} />}
            onClick={toggleViewMode}
            variant="ghost"
          >
            {viewMode === 'grid' ? 'List View' : 'Grid View'}
          </Button>
        </HStack>
      </HStack>

      {filteredVideos.length === 0 ? (
        <Box p={8} textAlign="center">
          <Text color="gray.500">No videos found</Text>
        </Box>
      ) : viewMode === 'grid' ? (
        <Grid
          templateColumns="repeat(auto-fill, minmax(280px, 1fr))"
          gap={6}
          w="full"
        >
          {filteredVideos.map((video) => (
            <MediaCard
              key={video.id}
              video={video}
              onClick={() => onVideoSelect(video)}
            />
          ))}
        </Grid>
      ) : (
        <VStack spacing={4} align="stretch">
          {filteredVideos.map((video) => (
            <MediaListItem
              key={video.id}
              video={video}
              onClick={() => onVideoSelect(video)}
            />
          ))}
        </VStack>
      )}
    </VStack>
  );
} 
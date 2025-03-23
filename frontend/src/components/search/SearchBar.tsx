import React, { useState, useRef, useCallback } from 'react';
import {
  Box,
  Input,
  InputGroup,
  InputLeftElement,
  InputRightElement,
  Popover,
  PopoverTrigger,
  PopoverContent,
  VStack,
  Text,
  HStack,
  Icon,
  Kbd,
  useColorModeValue,
  Spinner,
  Badge,
  IconButton,
} from '@chakra-ui/react';
import {
  FiSearch,
  FiVideo,
  FiImage,
  FiSettings,
  FiX,
  FiClock,
} from 'react-icons/fi';
import { useNavigate } from 'react-router-dom';
import debounce from 'lodash/debounce';

interface SearchResult {
  id: string;
  type: 'video' | 'clip' | 'asset' | 'preset' | 'setting';
  title: string;
  description?: string;
  path: string;
  timestamp?: string;
}

const SearchBar: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const bgColor = useColorModeValue('white', 'gray.800');
  const hoverBgColor = useColorModeValue('gray.50', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  // Mock search function - replace with actual API call
  const searchContent = async (searchQuery: string) => {
    setIsLoading(true);
    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 500));
      
      // Mock results
      const mockResults: SearchResult[] = [
        {
          id: '1',
          type: 'video',
          title: 'Project Video 1',
          description: 'Main project video file',
          path: '/editor/1',
          timestamp: '2 hours ago',
        },
        {
          id: '2',
          type: 'clip',
          title: 'Generated Clip',
          description: 'Auto-generated highlight clip',
          path: '/clips/2',
          timestamp: '1 day ago',
        },
        {
          id: '3',
          type: 'asset',
          title: 'Background Music',
          description: 'Audio asset',
          path: '/assets/3',
          timestamp: '3 days ago',
        },
      ].filter((result) =>
        result.title.toLowerCase().includes(searchQuery.toLowerCase())
      );

      setResults(mockResults);
    } catch (error) {
      console.error('Search error:', error);
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  const debouncedSearch = useCallback(
    debounce((value: string) => {
      if (value.trim()) {
        searchContent(value);
      } else {
        setResults([]);
      }
    }, 300),
    []
  );

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
    debouncedSearch(value);
    setIsOpen(true);
  };

  const handleResultClick = (result: SearchResult) => {
    navigate(result.path);
    setIsOpen(false);
    setQuery('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      inputRef.current?.focus();
    }
  };

  React.useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  const getIcon = (type: SearchResult['type']) => {
    switch (type) {
      case 'video':
        return FiVideo;
      case 'clip':
        return FiVideo;
      case 'asset':
        return FiImage;
      case 'setting':
        return FiSettings;
      default:
        return FiVideo;
    }
  };

  return (
    <Popover
      isOpen={isOpen}
      onClose={() => setIsOpen(false)}
      placement="bottom"
      autoFocus={false}
    >
      <PopoverTrigger>
        <InputGroup>
          <InputLeftElement pointerEvents="none">
            <Icon as={FiSearch} color="gray.400" />
          </InputLeftElement>
          <Input
            ref={inputRef}
            placeholder="Search videos, clips, assets... (⌘K)"
            value={query}
            onChange={handleInputChange}
            onFocus={() => setIsOpen(true)}
            pr="4.5rem"
          />
          <InputRightElement>
            {query ? (
              <IconButton
                size="sm"
                icon={<FiX />}
                aria-label="Clear search"
                variant="ghost"
                onClick={() => {
                  setQuery('');
                  setResults([]);
                }}
              />
            ) : (
              <HStack spacing={1} mr={2}>
                <Kbd>⌘</Kbd>
                <Kbd>K</Kbd>
              </HStack>
            )}
          </InputRightElement>
        </InputGroup>
      </PopoverTrigger>

      <PopoverContent
        bg={bgColor}
        borderColor={borderColor}
        boxShadow="lg"
        maxH="400px"
        overflowY="auto"
      >
        <Box p={4}>
          {isLoading ? (
            <Box textAlign="center" py={4}>
              <Spinner />
            </Box>
          ) : results.length > 0 ? (
            <VStack align="stretch" spacing={2}>
              {results.map((result) => (
                <Box
                  key={result.id}
                  p={2}
                  cursor="pointer"
                  borderRadius="md"
                  onClick={() => handleResultClick(result)}
                  _hover={{ bg: hoverBgColor }}
                >
                  <HStack spacing={3}>
                    <Icon as={getIcon(result.type)} />
                    <Box flex={1}>
                      <Text fontWeight="medium">{result.title}</Text>
                      {result.description && (
                        <Text fontSize="sm" color="gray.500">
                          {result.description}
                        </Text>
                      )}
                    </Box>
                    <HStack spacing={2}>
                      <Badge colorScheme="blue">{result.type}</Badge>
                      {result.timestamp && (
                        <HStack spacing={1} color="gray.500" fontSize="sm">
                          <Icon as={FiClock} />
                          <Text>{result.timestamp}</Text>
                        </HStack>
                      )}
                    </HStack>
                  </HStack>
                </Box>
              ))}
            </VStack>
          ) : query ? (
            <Text color="gray.500" textAlign="center">
              No results found
            </Text>
          ) : (
            <Text color="gray.500" textAlign="center">
              Start typing to search
            </Text>
          )}
        </Box>
      </PopoverContent>
    </Popover>
  );
};

export default SearchBar; 
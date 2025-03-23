import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Button,
  Checkbox,
  FormControl,
  FormLabel,
  Select,
  Text,
  VStack,
  HStack,
  useToast,
  Icon,
  Flex,
} from '@chakra-ui/react';
import { FiUpload, FiVideo, FiTrash2 } from 'react-icons/fi';

interface VideoFile extends File {
  preview?: string;
}

interface VideoUploadProps {
  onVideosSelected: (files: VideoFile[]) => void;
}

const contentCategories = [
  'Gaming',
  'Vlog',
  'Tutorial',
  'Entertainment',
  'Sports',
  'Music',
  'Education',
  'Other',
];

const platformPresets = [
  { label: 'YouTube Shorts', value: 'youtube_shorts' },
  { label: 'TikTok', value: 'tiktok' },
  { label: 'Instagram Reels', value: 'instagram_reels' },
  { label: 'Twitch Clips', value: 'twitch_clips' },
];

export const VideoUpload: React.FC<VideoUploadProps> = ({ onVideosSelected }) => {
  const [selectedFiles, setSelectedFiles] = useState<VideoFile[]>([]);
  const [selectedCategory, setSelectedCategory] = useState('');
  const [selectedPlatform, setSelectedPlatform] = useState('');
  const [generateClips, setGenerateClips] = useState(false);
  const toast = useToast();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles = acceptedFiles.map((file) => Object.assign(file, {
      preview: URL.createObjectURL(file),
    }));
    setSelectedFiles((prev) => [...prev, ...newFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mov', '.avi', '.mkv']
    },
    multiple: true,
  });

  const removeFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUpload = () => {
    if (selectedFiles.length === 0) {
      toast({
        title: 'No files selected',
        description: 'Please select at least one video file to upload.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    onVideosSelected(selectedFiles);
  };

  return (
    <VStack spacing={6} w="full" p={6} bg="white" borderRadius="lg" shadow="sm">
      <Box
        {...getRootProps()}
        w="full"
        h="200px"
        border="2px dashed"
        borderColor={isDragActive ? 'blue.400' : 'gray.200'}
        borderRadius="lg"
        display="flex"
        alignItems="center"
        justifyContent="center"
        bg={isDragActive ? 'blue.50' : 'gray.50'}
        transition="all 0.2s"
        cursor="pointer"
        _hover={{ borderColor: 'blue.400', bg: 'blue.50' }}
      >
        <input {...getInputProps()} />
        <VStack spacing={2}>
          <Icon as={FiUpload} w={8} h={8} color="gray.400" />
          <Text color="gray.500">
            {isDragActive
              ? 'Drop your videos here'
              : 'Drag & drop videos here or click to select'}
          </Text>
        </VStack>
      </Box>

      {selectedFiles.length > 0 && (
        <VStack w="full" spacing={4}>
          {selectedFiles.map((file, index) => (
            <HStack key={index} w="full" p={2} bg="gray.50" borderRadius="md">
              <Icon as={FiVideo} color="gray.500" />
              <Text flex={1} isTruncated>{file.name}</Text>
              <Button
                size="sm"
                colorScheme="red"
                variant="ghost"
                onClick={() => removeFile(index)}
              >
                <Icon as={FiTrash2} />
              </Button>
            </HStack>
          ))}
        </VStack>
      )}

      <FormControl>
        <FormLabel>Content Category</FormLabel>
        <Select
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
          placeholder="Select category"
        >
          {contentCategories.map((category) => (
            <option key={category} value={category.toLowerCase()}>
              {category}
            </option>
          ))}
        </Select>
      </FormControl>

      <FormControl>
        <FormLabel>Target Platform</FormLabel>
        <Select
          value={selectedPlatform}
          onChange={(e) => setSelectedPlatform(e.target.value)}
          placeholder="Select platform"
        >
          {platformPresets.map((platform) => (
            <option key={platform.value} value={platform.value}>
              {platform.label}
            </option>
          ))}
        </Select>
      </FormControl>

      <Flex w="full" alignItems="center">
        <Checkbox
          isChecked={generateClips}
          onChange={(e) => setGenerateClips(e.target.checked)}
        >
          Generate Clips
        </Checkbox>
      </Flex>

      <Button
        colorScheme="blue"
        size="lg"
        w="full"
        onClick={handleUpload}
        isDisabled={selectedFiles.length === 0}
      >
        Upload {selectedFiles.length} {selectedFiles.length === 1 ? 'Video' : 'Videos'}
      </Button>
    </VStack>
  );
};

export default VideoUpload; 
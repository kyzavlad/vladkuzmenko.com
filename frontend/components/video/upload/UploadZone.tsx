import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Box, Text, VStack, Progress, Icon, useToast } from '@chakra-ui/react';
import { FiUploadCloud } from 'react-icons/fi';
import { useCreateVideoMutation } from '@/lib/redux/services/videoService';

const MAX_FILE_SIZE = parseInt(process.env.NEXT_PUBLIC_MAX_VIDEO_SIZE || '100') * 1024 * 1024; // Convert MB to bytes
const SUPPORTED_FORMATS = (process.env.NEXT_PUBLIC_SUPPORTED_VIDEO_FORMATS || 'mp4,webm,mov').split(',');

export function UploadZone() {
  const [uploadProgress, setUploadProgress] = useState(0);
  const toast = useToast();
  const [createVideo] = useCreateVideoMutation();

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    try {
      await createVideo({
        title: file.name.replace(/\.[^/.]+$/, ''),
        file,
      }).unwrap();

      toast({
        title: 'Upload successful',
        description: 'Your video has been uploaded and is being processed.',
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    } catch (error) {
      toast({
        title: 'Upload failed',
        description: error instanceof Error ? error.message : 'An error occurred during upload',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  }, [createVideo, toast]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': SUPPORTED_FORMATS.map(format => `.${format}`),
    },
    maxSize: MAX_FILE_SIZE,
    multiple: false,
  });

  return (
    <Box
      {...getRootProps()}
      p={8}
      border="2px"
      borderStyle="dashed"
      borderColor={isDragActive ? 'primary.500' : 'neutral.200'}
      borderRadius="lg"
      bg={isDragActive ? 'primary.50' : 'neutral.50'}
      cursor="pointer"
      transition="all 0.2s"
      _hover={{
        borderColor: 'primary.500',
        bg: 'primary.50',
      }}
    >
      <input {...getInputProps()} />
      <VStack spacing={4}>
        <Icon as={FiUploadCloud} w={12} h={12} color="primary.500" />
        <Text fontSize="lg" fontWeight="medium" textAlign="center">
          {isDragActive
            ? 'Drop your video here'
            : 'Drag and drop your video here, or click to browse'}
        </Text>
        <Text fontSize="sm" color="neutral.500" textAlign="center">
          Supported formats: {SUPPORTED_FORMATS.join(', ').toUpperCase()}
          <br />
          Maximum file size: {process.env.NEXT_PUBLIC_MAX_VIDEO_SIZE || '100'}MB
        </Text>
        {uploadProgress > 0 && uploadProgress < 100 && (
          <Box w="full">
            <Progress
              value={uploadProgress}
              size="sm"
              colorScheme="primary"
              borderRadius="full"
            />
          </Box>
        )}
      </VStack>
    </Box>
  );
} 
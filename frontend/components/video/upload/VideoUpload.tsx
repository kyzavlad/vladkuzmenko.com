import React, { useCallback, useState } from 'react';
import {
  Box,
  Button,
  Center,
  Icon,
  Progress,
  Text,
  VStack,
  useToast,
} from '@chakra-ui/react';
import { useDropzone } from 'react-dropzone';
import { FiUpload, FiVideo } from 'react-icons/fi';

interface VideoUploadProps {
  onUploadComplete: (videoId: string, url: string) => void;
  maxSize?: number; // in bytes
  acceptedTypes?: string[];
}

export function VideoUpload({
  onUploadComplete,
  maxSize = 1024 * 1024 * 100, // 100MB default
  acceptedTypes = ['video/mp4', 'video/webm', 'video/quicktime'],
}: VideoUploadProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const toast = useToast();

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];

      if (!file) return;

      try {
        setIsUploading(true);
        setUploadProgress(0);

        const formData = new FormData();
        formData.append('file', file);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/videos/upload', true);

        // Track upload progress
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            const progress = (event.loaded / event.total) * 100;
            setUploadProgress(progress);
          }
        };

        // Handle response
        xhr.onload = () => {
          if (xhr.status === 201) {
            const response = JSON.parse(xhr.responseText);
            onUploadComplete(response.id, response.url);
            toast({
              title: 'Upload Complete',
              description: 'Your video has been uploaded successfully.',
              status: 'success',
              duration: 5000,
              isClosable: true,
            });
          } else {
            throw new Error('Upload failed');
          }
        };

        // Handle errors
        xhr.onerror = () => {
          throw new Error('Upload failed');
        };

        // Send the request
        xhr.send(formData);
      } catch (error) {
        toast({
          title: 'Upload Failed',
          description:
            error instanceof Error
              ? error.message
              : 'An error occurred while uploading your video.',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      } finally {
        setIsUploading(false);
      }
    },
    [onUploadComplete, toast]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: acceptedTypes.reduce(
      (acc, type) => ({ ...acc, [type]: [] }),
      {}
    ),
    maxSize,
    multiple: false,
  });

  return (
    <Box w="full">
      <Box
        {...getRootProps()}
        w="full"
        h="300px"
        border="2px"
        borderStyle="dashed"
        borderColor={isDragActive ? 'primary.500' : 'gray.300'}
        borderRadius="lg"
        bg={isDragActive ? 'primary.50' : 'transparent'}
        transition="all 0.2s"
        cursor="pointer"
        _hover={{
          borderColor: 'primary.500',
          bg: 'primary.50',
        }}
      >
        <input {...getInputProps()} />
        <Center h="full">
          <VStack spacing={4}>
            <Icon
              as={isUploading ? FiVideo : FiUpload}
              boxSize={12}
              color={isDragActive ? 'primary.500' : 'gray.400'}
            />
            <Text
              fontSize="lg"
              color={isDragActive ? 'primary.500' : 'gray.500'}
              textAlign="center"
            >
              {isDragActive
                ? 'Drop your video here'
                : isUploading
                ? 'Uploading...'
                : 'Drag and drop your video here, or click to select'}
            </Text>
            <Text fontSize="sm" color="gray.500">
              Supported formats: {acceptedTypes.join(', ')}
              <br />
              Maximum size: {Math.round(maxSize / (1024 * 1024))}MB
            </Text>
            {!isDragActive && !isUploading && (
              <Button colorScheme="primary" size="sm">
                Select Video
              </Button>
            )}
          </VStack>
        </Center>
      </Box>

      {isUploading && (
        <Box mt={4}>
          <Progress
            value={uploadProgress}
            size="sm"
            colorScheme="primary"
            hasStripe
            isAnimated
          />
          <Text mt={2} fontSize="sm" color="gray.500" textAlign="center">
            {Math.round(uploadProgress)}% uploaded
          </Text>
        </Box>
      )}
    </Box>
  );
} 
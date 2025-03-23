import React, { useState } from 'react';
import {
  Box,
  Button,
  Grid,
  HStack,
  Text,
  VStack,
  IconButton,
  Textarea,
  useToken,
} from '@chakra-ui/react';
import {
  FiPlay,
  FiPause,
  FiSkipBack,
  FiSkipForward,
  FiEdit,
  FiSave,
} from 'react-icons/fi';

interface Subtitle {
  id: string;
  startTime: number;
  endTime: number;
  text: string;
  translation: string;
}

interface TranslationPreviewProps {
  videoUrl: string;
  subtitles: Subtitle[];
  currentTime: number;
  isPlaying: boolean;
  onPlayPause: () => void;
  onSeek: (time: number) => void;
  onSubtitleUpdate: (subtitle: Subtitle) => void;
}

export function TranslationPreview({
  videoUrl,
  subtitles,
  currentTime,
  isPlaying,
  onPlayPause,
  onSeek,
  onSubtitleUpdate,
}: TranslationPreviewProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState('');
  const [primary500] = useToken('colors', ['primary.500']);

  const handleEdit = (subtitle: Subtitle) => {
    setEditingId(subtitle.id);
    setEditText(subtitle.translation);
  };

  const handleSave = (subtitle: Subtitle) => {
    onSubtitleUpdate({
      ...subtitle,
      translation: editText,
    });
    setEditingId(null);
  };

  const getCurrentSubtitle = () => {
    return subtitles.find(
      (sub) => currentTime >= sub.startTime && currentTime <= sub.endTime
    );
  };

  const seekToSubtitle = (direction: 'prev' | 'next') => {
    const current = getCurrentSubtitle();
    if (!current) return;

    const currentIndex = subtitles.findIndex((sub) => sub.id === current.id);
    const targetIndex =
      direction === 'prev'
        ? Math.max(0, currentIndex - 1)
        : Math.min(subtitles.length - 1, currentIndex + 1);

    onSeek(subtitles[targetIndex].startTime);
  };

  return (
    <Grid templateColumns="1fr 1fr" gap={6} w="full">
      <Box>
        <video
          src={videoUrl}
          style={{ width: '100%', borderRadius: '8px' }}
          controls={false}
        />
        <HStack justify="center" mt={4} spacing={4}>
          <IconButton
            aria-label="Previous subtitle"
            icon={<FiSkipBack />}
            onClick={() => seekToSubtitle('prev')}
          />
          <IconButton
            aria-label={isPlaying ? 'Pause' : 'Play'}
            icon={isPlaying ? <FiPause /> : <FiPlay />}
            onClick={onPlayPause}
            colorScheme="primary"
          />
          <IconButton
            aria-label="Next subtitle"
            icon={<FiSkipForward />}
            onClick={() => seekToSubtitle('next')}
          />
        </HStack>
      </Box>

      <VStack spacing={4} align="stretch">
        <Text fontSize="lg" fontWeight="semibold">
          Subtitles
        </Text>
        <Box
          maxH="400px"
          overflowY="auto"
          borderRadius="lg"
          borderWidth={1}
          borderColor="neutral.700"
        >
          {subtitles.map((subtitle) => (
            <Box
              key={subtitle.id}
              p={4}
              borderBottomWidth={1}
              borderColor="neutral.700"
              bg={
                currentTime >= subtitle.startTime &&
                currentTime <= subtitle.endTime
                  ? 'neutral.800'
                  : 'transparent'
              }
            >
              <HStack justify="space-between" mb={2}>
                <Text fontSize="sm" color="neutral.400">
                  {formatTime(subtitle.startTime)} -{' '}
                  {formatTime(subtitle.endTime)}
                </Text>
                {editingId === subtitle.id ? (
                  <IconButton
                    aria-label="Save"
                    icon={<FiSave />}
                    size="sm"
                    colorScheme="primary"
                    onClick={() => handleSave(subtitle)}
                  />
                ) : (
                  <IconButton
                    aria-label="Edit"
                    icon={<FiEdit />}
                    size="sm"
                    variant="ghost"
                    onClick={() => handleEdit(subtitle)}
                  />
                )}
              </HStack>
              <Text mb={2}>{subtitle.text}</Text>
              {editingId === subtitle.id ? (
                <Textarea
                  value={editText}
                  onChange={(e) => setEditText(e.target.value)}
                  size="sm"
                  rows={2}
                />
              ) : (
                <Text color={primary500}>{subtitle.translation}</Text>
              )}
            </Box>
          ))}
        </Box>
      </VStack>
    </Grid>
  );
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
} 
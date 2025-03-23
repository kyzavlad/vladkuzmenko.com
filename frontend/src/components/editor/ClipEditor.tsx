import React, { useState, useCallback } from 'react';
import {
  Box,
  Grid,
  GridItem,
  useColorModeValue,
  VStack,
} from '@chakra-ui/react';
import { ProcessingStage } from './ProcessingStatus';
import ProcessingStatus from './ProcessingStatus';
import EnhancementTools from './EnhancementTools';
import { TimelineEditor } from './TimelineEditor';
import { VerticalPreview } from './VerticalPreview';

interface ClipEditorProps {
  videoUrl: string;
  audioUrl: string;
  duration: number;
  engagementMarkers: Array<{ time: number; score: number }>;
  faceDetectionMarkers: Array<{ time: number; faces: number }>;
  onSave: (clipData: ClipData) => void;
  onCancel: () => void;
}

interface ClipData {
  startTime: number;
  endTime: number;
  captionStyle: CaptionStyle;
  musicTrack?: MusicTrack;
  filter?: VideoFilter;
  watermark?: WatermarkSettings;
  endCard?: EndCardTemplate;
}

interface CaptionStyle {
  fontFamily: string;
  fontSize: number;
  color: string;
  backgroundColor: string;
  position: 'top' | 'center' | 'bottom';
}

interface MusicTrack {
  url: string;
  name: string;
  volume: number;
  fadeIn: number;
  fadeOut: number;
}

interface VideoFilter {
  name: string;
  intensity: number;
  settings: {
    brightness: number;
    contrast: number;
    saturation: number;
    temperature: number;
  };
}

interface WatermarkSettings {
  image: string;
  position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  size: number;
  opacity: number;
}

interface EndCardTemplate {
  id: string;
  name: string;
  preview: string;
}

export const ClipEditor: React.FC<ClipEditorProps> = ({
  videoUrl,
  audioUrl,
  duration,
  engagementMarkers,
  faceDetectionMarkers,
  onSave,
  onCancel,
}) => {
  const [currentTime, setCurrentTime] = useState(0);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(duration);
  const [captionStyle, setCaptionStyle] = useState<CaptionStyle>({
    fontFamily: 'Arial',
    fontSize: 24,
    color: '#FFFFFF',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    position: 'bottom',
  });
  const [musicTrack, setMusicTrack] = useState<MusicTrack | undefined>();
  const [filter, setFilter] = useState<VideoFilter | undefined>();
  const [watermark, setWatermark] = useState<WatermarkSettings | undefined>();
  const [endCard, setEndCard] = useState<EndCardTemplate | undefined>();

  // Processing state
  const [isPaused, setIsPaused] = useState(false);
  const [isBackgroundProcessing, setIsBackgroundProcessing] = useState(false);
  const [processingStages, setProcessingStages] = useState<ProcessingStage[]>([
    {
      id: 'trim',
      name: 'Trimming Video',
      status: 'processing',
      progress: 35,
      estimatedTimeRemaining: 120,
    },
    {
      id: 'captions',
      name: 'Generating Captions',
      status: 'pending',
      progress: 0,
    },
    {
      id: 'music',
      name: 'Adding Background Music',
      status: 'pending',
      progress: 0,
    },
    {
      id: 'effects',
      name: 'Applying Visual Effects',
      status: 'pending',
      progress: 0,
    },
  ]);

  const totalProgress = processingStages.reduce(
    (acc, stage) => acc + stage.progress,
    0
  ) / processingStages.length;

  const handleTimeUpdate = useCallback((time: number) => {
    setCurrentTime(time);
  }, []);

  const handleRangeUpdate = useCallback((start: number, end: number) => {
    setStartTime(start);
    setEndTime(end);
  }, []);

  const handleCaptionStyleChange = useCallback((style: CaptionStyle) => {
    setCaptionStyle(style);
  }, []);

  const handleMusicAdd = useCallback((track: MusicTrack) => {
    setMusicTrack(track);
  }, []);

  const handleFilterChange = useCallback((newFilter: VideoFilter) => {
    setFilter(newFilter);
  }, []);

  const handleWatermarkChange = useCallback((newWatermark: WatermarkSettings) => {
    setWatermark(newWatermark);
  }, []);

  const handleEndCardChange = useCallback((template: EndCardTemplate) => {
    setEndCard(template);
  }, []);

  const handleSave = useCallback(() => {
    onSave({
      startTime,
      endTime,
      captionStyle,
      musicTrack,
      filter,
      watermark,
      endCard,
    });
  }, [
    startTime,
    endTime,
    captionStyle,
    musicTrack,
    filter,
    watermark,
    endCard,
    onSave,
  ]);

  const bgColor = useColorModeValue('gray.50', 'gray.900');

  return (
    <Box w="full" minH="100vh" bg={bgColor} p={4}>
      <Grid
        templateColumns="1fr 400px"
        templateRows="auto 1fr"
        gap={4}
        maxW="1800px"
        mx="auto"
      >
        <GridItem colSpan={1}>
          <TimelineEditor
            videoUrl={videoUrl}
            audioUrl={audioUrl}
            duration={duration}
            currentTime={currentTime}
            startTime={startTime}
            endTime={endTime}
            engagementMarkers={engagementMarkers}
            faceDetectionMarkers={faceDetectionMarkers}
            onTimeUpdate={handleTimeUpdate}
            onRangeUpdate={handleRangeUpdate}
          />
        </GridItem>

        <GridItem colSpan={1} rowSpan={2}>
          <VStack spacing={4} align="stretch" position="sticky" top={4}>
            <ProcessingStatus
              stages={processingStages}
              totalProgress={totalProgress}
              isPaused={isPaused}
              isBackgroundProcessing={isBackgroundProcessing}
              onPauseResume={() => setIsPaused(!isPaused)}
              onCancel={onCancel}
              onBackgroundProcessingToggle={() =>
                setIsBackgroundProcessing(!isBackgroundProcessing)
              }
            />
            <EnhancementTools
              onCaptionStyleChange={handleCaptionStyleChange}
              onMusicAdd={handleMusicAdd}
              onFilterChange={handleFilterChange}
              onWatermarkChange={handleWatermarkChange}
              onEndCardChange={handleEndCardChange}
            />
          </VStack>
        </GridItem>

        <GridItem colSpan={1}>
          <VerticalPreview
            videoUrl={videoUrl}
            currentTime={currentTime}
            aspectRatio={9 / 16}
            smartCrop={true}
            captions={[
              {
                text: 'Sample caption text',
                startTime: currentTime,
                endTime: currentTime + 3,
                style: captionStyle,
              },
            ]}
            watermark={watermark}
          />
        </GridItem>
      </Grid>
    </Box>
  );
};

export default ClipEditor; 
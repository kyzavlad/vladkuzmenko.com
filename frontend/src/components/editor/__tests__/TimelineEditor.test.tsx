import React from 'react';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { render } from '../../../utils/test-utils';
import { TimelineEditor } from '../TimelineEditor';
import WaveSurfer from 'wavesurfer.js';

// Mock WaveSurfer
jest.mock('wavesurfer.js', () => {
  return {
    __esModule: true,
    default: jest.fn().mockImplementation(() => ({
      load: jest.fn(),
      on: jest.fn(),
      play: jest.fn(),
      pause: jest.fn(),
      destroy: jest.fn(),
      getCurrentTime: jest.fn().mockReturnValue(0),
      getDuration: jest.fn().mockReturnValue(60),
      setCurrentTime: jest.fn(),
      zoom: jest.fn(),
    })),
  };
});

const mockProps = {
  videoUrl: 'https://example.com/test-video.mp4',
  audioUrl: 'https://example.com/test-audio.mp3',
  duration: 60,
  engagementMarkers: [
    { time: 10, score: 0.8 },
    { time: 30, score: 0.6 },
  ],
  faceDetectionMarkers: [
    { time: 5, faces: 2 },
    { time: 15, faces: 1 },
  ],
  onTimeUpdate: jest.fn(),
};

describe('TimelineEditor Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  it('initializes WaveSurfer with correct options', () => {
    render(<TimelineEditor {...mockProps} />);
    expect(WaveSurfer).toHaveBeenCalledWith(
      expect.objectContaining({
        container: expect.any(HTMLElement),
        waveColor: expect.any(String),
        progressColor: expect.any(String),
        responsive: true,
        height: expect.any(Number),
      })
    );
  });

  it('loads audio when component mounts', async () => {
    render(<TimelineEditor {...mockProps} />);
    const wavesurfer = (WaveSurfer as jest.Mock).mock.results[0].value;
    
    await waitFor(() => {
      expect(wavesurfer.load).toHaveBeenCalledWith(mockProps.audioUrl);
    });
  });

  it('handles play/pause toggle', () => {
    render(<TimelineEditor {...mockProps} />);
    const playButton = screen.getByLabelText('Play/Pause');
    const wavesurfer = (WaveSurfer as jest.Mock).mock.results[0].value;

    fireEvent.click(playButton);
    expect(wavesurfer.play).toHaveBeenCalled();

    fireEvent.click(playButton);
    expect(wavesurfer.pause).toHaveBeenCalled();
  });

  it('updates current time on slider change', () => {
    render(<TimelineEditor {...mockProps} />);
    const timeSlider = screen.getByRole('slider');
    const wavesurfer = (WaveSurfer as jest.Mock).mock.results[0].value;

    fireEvent.change(timeSlider, { target: { value: '30' } });
    expect(wavesurfer.setCurrentTime).toHaveBeenCalledWith(30);
    expect(mockProps.onTimeUpdate).toHaveBeenCalledWith(30);
  });

  it('renders engagement markers', () => {
    render(<TimelineEditor {...mockProps} />);
    const markers = screen.getAllByTestId('engagement-marker');
    expect(markers).toHaveLength(mockProps.engagementMarkers.length);
  });

  it('renders face detection markers', () => {
    render(<TimelineEditor {...mockProps} />);
    const markers = screen.getAllByTestId('face-marker');
    expect(markers).toHaveLength(mockProps.faceDetectionMarkers.length);
  });

  it('handles skip forward/backward', () => {
    render(<TimelineEditor {...mockProps} />);
    const skipForwardButton = screen.getByLabelText('Skip forward');
    const skipBackwardButton = screen.getByLabelText('Skip backward');
    const wavesurfer = (WaveSurfer as jest.Mock).mock.results[0].value;

    fireEvent.click(skipForwardButton);
    expect(wavesurfer.setCurrentTime).toHaveBeenCalledWith(5); // Default skip time

    fireEvent.click(skipBackwardButton);
    expect(wavesurfer.setCurrentTime).toHaveBeenCalledWith(-5);
  });

  it('cleans up WaveSurfer instance on unmount', () => {
    const { unmount } = render(<TimelineEditor {...mockProps} />);
    const wavesurfer = (WaveSurfer as jest.Mock).mock.results[0].value;
    
    unmount();
    expect(wavesurfer.destroy).toHaveBeenCalled();
  });
}); 
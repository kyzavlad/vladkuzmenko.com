import React from 'react';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { render } from '../../../utils/test-utils';
import { TimelineWaveform } from '../TimelineWaveform';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions';
import TimelinePlugin from 'wavesurfer.js/dist/plugins/timeline';

// Mock WaveSurfer and its plugins
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
      addRegion: jest.fn(),
      clearRegions: jest.fn(),
      regions: {
        list: {},
      },
    })),
  };
});

jest.mock('wavesurfer.js/dist/plugins/regions', () => {
  return {
    __esModule: true,
    default: jest.fn().mockImplementation(() => ({
      on: jest.fn(),
      add: jest.fn(),
      clear: jest.fn(),
    })),
  };
});

jest.mock('wavesurfer.js/dist/plugins/timeline', () => {
  return {
    __esModule: true,
    default: jest.fn().mockImplementation(() => ({
      on: jest.fn(),
    })),
  };
});

const mockProps = {
  audioUrl: 'https://example.com/test-audio.mp3',
  duration: 60,
  markers: [
    { id: '1', startTime: 10, endTime: 15, type: 'highlight' },
    { id: '2', startTime: 30, endTime: 35, type: 'segment' },
  ],
  onMarkerUpdate: jest.fn(),
  onTimeUpdate: jest.fn(),
  zoom: 50,
};

describe('TimelineWaveform Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  it('initializes WaveSurfer with correct options', () => {
    render(<TimelineWaveform {...mockProps} />);
    
    expect(WaveSurfer).toHaveBeenCalledWith(
      expect.objectContaining({
        container: expect.any(HTMLElement),
        waveColor: expect.any(String),
        progressColor: expect.any(String),
        responsive: true,
        height: expect.any(Number),
        plugins: expect.arrayContaining([
          expect.any(RegionsPlugin),
          expect.any(TimelinePlugin),
        ]),
      })
    );
  });

  it('loads audio when component mounts', async () => {
    render(<TimelineWaveform {...mockProps} />);
    const wavesurfer = (WaveSurfer as jest.Mock).mock.results[0].value;
    
    await waitFor(() => {
      expect(wavesurfer.load).toHaveBeenCalledWith(mockProps.audioUrl);
    });
  });

  it('creates regions for markers', async () => {
    render(<TimelineWaveform {...mockProps} />);
    const wavesurfer = (WaveSurfer as jest.Mock).mock.results[0].value;
    
    await waitFor(() => {
      expect(wavesurfer.addRegion).toHaveBeenCalledTimes(mockProps.markers.length);
      mockProps.markers.forEach((marker) => {
        expect(wavesurfer.addRegion).toHaveBeenCalledWith(
          expect.objectContaining({
            id: marker.id,
            start: marker.startTime,
            end: marker.endTime,
            color: expect.any(String),
            drag: true,
            resize: true,
          })
        );
      });
    });
  });

  it('updates zoom level when prop changes', async () => {
    const { rerender } = render(<TimelineWaveform {...mockProps} />);
    const wavesurfer = (WaveSurfer as jest.Mock).mock.results[0].value;
    
    rerender(<TimelineWaveform {...mockProps} zoom={75} />);
    
    expect(wavesurfer.zoom).toHaveBeenCalledWith(75);
  });

  it('handles region updates', async () => {
    render(<TimelineWaveform {...mockProps} />);
    const wavesurfer = (WaveSurfer as jest.Mock).mock.results[0].value;
    
    // Simulate region update event
    const regionUpdateEvent = {
      id: '1',
      start: 12,
      end: 17,
    };
    
    const onHandler = wavesurfer.on.mock.calls.find(
      ([event]) => event === 'region-update-end'
    )[1];
    
    onHandler(regionUpdateEvent);
    
    expect(mockProps.onMarkerUpdate).toHaveBeenCalledWith({
      id: regionUpdateEvent.id,
      startTime: regionUpdateEvent.start,
      endTime: regionUpdateEvent.end,
    });
  });

  it('cleans up WaveSurfer instance on unmount', () => {
    const { unmount } = render(<TimelineWaveform {...mockProps} />);
    const wavesurfer = (WaveSurfer as jest.Mock).mock.results[0].value;
    
    unmount();
    expect(wavesurfer.destroy).toHaveBeenCalled();
  });

  it('handles playback controls', () => {
    render(<TimelineWaveform {...mockProps} />);
    const wavesurfer = (WaveSurfer as jest.Mock).mock.results[0].value;
    
    const playButton = screen.getByLabelText('Play/Pause');
    fireEvent.click(playButton);
    expect(wavesurfer.play).toHaveBeenCalled();
    
    fireEvent.click(playButton);
    expect(wavesurfer.pause).toHaveBeenCalled();
  });

  it('updates current time on seek', async () => {
    render(<TimelineWaveform {...mockProps} />);
    const wavesurfer = (WaveSurfer as jest.Mock).mock.results[0].value;
    
    const onHandler = wavesurfer.on.mock.calls.find(
      ([event]) => event === 'seek'
    )[1];
    
    onHandler(0.5); // 50% of duration
    
    expect(mockProps.onTimeUpdate).toHaveBeenCalledWith(30); // 50% of 60 seconds
  });
}); 
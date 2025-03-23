import React from 'react';
import { screen, fireEvent } from '@testing-library/react';
import { render } from '../../../utils/test-utils';
import { VerticalPreview } from '../VerticalPreview';

const mockProps = {
  videoUrl: 'https://example.com/test-video.mp4',
  currentTime: 0,
  aspectRatio: '9:16',
  smartCrop: {
    x: 0,
    y: 0,
    width: 100,
    height: 100,
  },
  captions: [
    {
      id: '1',
      text: 'Test caption',
      startTime: 0,
      endTime: 5,
    },
  ],
  watermark: {
    url: 'https://example.com/watermark.png',
    position: 'bottom-right' as const,
    opacity: 0.8,
  },
};

describe('VerticalPreview Component', () => {
  beforeEach(() => {
    // Mock video element methods
    window.HTMLMediaElement.prototype.load = jest.fn();
    window.HTMLMediaElement.prototype.play = jest.fn();
    window.HTMLMediaElement.prototype.pause = jest.fn();
  });

  it('renders video player with correct source', () => {
    render(<VerticalPreview {...mockProps} />);
    const videoElement = screen.getByTestId('preview-video') as HTMLVideoElement;
    expect(videoElement).toBeInTheDocument();
    expect(videoElement.src).toBe(mockProps.videoUrl);
  });

  it('displays device frame with correct aspect ratio', () => {
    render(<VerticalPreview {...mockProps} />);
    const frame = screen.getByTestId('device-frame');
    expect(frame).toHaveStyle({ aspectRatio: mockProps.aspectRatio });
  });

  it('shows captions at correct time', () => {
    render(<VerticalPreview {...mockProps} />);
    const videoElement = screen.getByTestId('preview-video');
    fireEvent.timeUpdate(videoElement, { target: { currentTime: 2 } });
    expect(screen.getByText(mockProps.captions[0].text)).toBeInTheDocument();
  });

  it('applies smart crop overlay', () => {
    render(<VerticalPreview {...mockProps} />);
    const cropOverlay = screen.getByTestId('smart-crop-overlay');
    expect(cropOverlay).toHaveStyle({
      left: `${mockProps.smartCrop.x}%`,
      top: `${mockProps.smartCrop.y}%`,
      width: `${mockProps.smartCrop.width}%`,
      height: `${mockProps.smartCrop.height}%`,
    });
  });

  it('displays watermark with correct properties', () => {
    render(<VerticalPreview {...mockProps} />);
    const watermark = screen.getByTestId('watermark');
    expect(watermark).toHaveAttribute('src', mockProps.watermark.url);
    expect(watermark).toHaveStyle({
      opacity: mockProps.watermark.opacity,
    });
  });

  it('toggles fullscreen mode', () => {
    render(<VerticalPreview {...mockProps} />);
    const fullscreenButton = screen.getByLabelText('Toggle fullscreen');
    
    document.documentElement.requestFullscreen = jest.fn();
    document.exitFullscreen = jest.fn();

    fireEvent.click(fullscreenButton);
    expect(document.documentElement.requestFullscreen).toHaveBeenCalled();

    // Mock fullscreen state
    Object.defineProperty(document, 'fullscreenElement', {
      value: document.documentElement,
      writable: true,
    });

    fireEvent.click(fullscreenButton);
    expect(document.exitFullscreen).toHaveBeenCalled();
  });

  it('handles zoom controls', () => {
    render(<VerticalPreview {...mockProps} />);
    const zoomInButton = screen.getByLabelText('Zoom in');
    const zoomOutButton = screen.getByLabelText('Zoom out');
    const preview = screen.getByTestId('preview-container');

    fireEvent.click(zoomInButton);
    expect(preview).toHaveStyle({ transform: 'scale(1.1)' });

    fireEvent.click(zoomOutButton);
    expect(preview).toHaveStyle({ transform: 'scale(1)' });
  });
}); 
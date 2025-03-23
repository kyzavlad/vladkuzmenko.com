import React from 'react';
import { screen, fireEvent } from '@testing-library/react';
import { render } from '../../../utils/test-utils';
import { DeviceFrame } from '../DeviceFrame';
import { motion } from 'framer-motion';

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: jest.fn().mockImplementation(({ children, ...props }) => (
      <div {...props}>{children}</div>
    )),
  },
  useAnimation: jest.fn().mockReturnValue({
    start: jest.fn(),
    set: jest.fn(),
  }),
}));

const mockProps = {
  aspectRatio: '9:16',
  deviceType: 'iphone-13',
  orientation: 'portrait',
  scale: 1,
  notchStyle: 'dynamic-island',
  statusBarContent: {
    time: '9:41',
    batteryLevel: 100,
    signalStrength: 4,
    carrier: 'Carrier',
  },
  children: <div data-testid="frame-content">Content</div>,
  onScaleChange: jest.fn(),
};

describe('DeviceFrame Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders device frame with correct aspect ratio', () => {
    render(<DeviceFrame {...mockProps} />);
    const frame = screen.getByTestId('device-frame');
    
    expect(frame).toHaveStyle({
      aspectRatio: mockProps.aspectRatio,
    });
  });

  it('renders status bar with correct content', () => {
    render(<DeviceFrame {...mockProps} />);
    
    expect(screen.getByText(mockProps.statusBarContent.time)).toBeInTheDocument();
    expect(screen.getByText(mockProps.statusBarContent.carrier)).toBeInTheDocument();
    expect(screen.getByTestId('battery-indicator')).toHaveAttribute(
      'data-level',
      mockProps.statusBarContent.batteryLevel.toString()
    );
    expect(screen.getByTestId('signal-strength')).toHaveAttribute(
      'data-level',
      mockProps.statusBarContent.signalStrength.toString()
    );
  });

  it('applies correct device-specific styles', () => {
    render(<DeviceFrame {...mockProps} />);
    const frame = screen.getByTestId('device-frame');
    
    expect(frame).toHaveClass(`device-${mockProps.deviceType}`);
    expect(frame).toHaveClass(mockProps.orientation);
    expect(frame).toHaveClass(`notch-${mockProps.notchStyle}`);
  });

  it('handles zoom gestures', () => {
    render(<DeviceFrame {...mockProps} />);
    const frame = screen.getByTestId('device-frame');
    
    // Simulate pinch zoom gesture
    fireEvent.pointerDown(frame, { pointerId: 1, clientX: 0, clientY: 0 });
    fireEvent.pointerDown(frame, { pointerId: 2, clientX: 100, clientY: 0 });
    
    // Move pointers apart
    fireEvent.pointerMove(frame, { pointerId: 1, clientX: -50, clientY: 0 });
    fireEvent.pointerMove(frame, { pointerId: 2, clientX: 150, clientY: 0 });
    
    expect(mockProps.onScaleChange).toHaveBeenCalledWith(expect.any(Number));
  });

  it('renders device frame corners and edges', () => {
    render(<DeviceFrame {...mockProps} />);
    
    expect(screen.getByTestId('top-left-corner')).toBeInTheDocument();
    expect(screen.getByTestId('top-right-corner')).toBeInTheDocument();
    expect(screen.getByTestId('bottom-left-corner')).toBeInTheDocument();
    expect(screen.getByTestId('bottom-right-corner')).toBeInTheDocument();
    
    expect(screen.getByTestId('left-edge')).toBeInTheDocument();
    expect(screen.getByTestId('right-edge')).toBeInTheDocument();
    expect(screen.getByTestId('top-edge')).toBeInTheDocument();
    expect(screen.getByTestId('bottom-edge')).toBeInTheDocument();
  });

  it('renders device buttons and ports', () => {
    render(<DeviceFrame {...mockProps} />);
    
    expect(screen.getByTestId('volume-up')).toBeInTheDocument();
    expect(screen.getByTestId('volume-down')).toBeInTheDocument();
    expect(screen.getByTestId('power-button')).toBeInTheDocument();
    expect(screen.getByTestId('charging-port')).toBeInTheDocument();
  });

  it('handles orientation change', () => {
    const { rerender } = render(<DeviceFrame {...mockProps} />);
    
    rerender(<DeviceFrame {...mockProps} orientation="landscape" />);
    const frame = screen.getByTestId('device-frame');
    
    expect(frame).toHaveClass('landscape');
    expect(frame).not.toHaveClass('portrait');
  });

  it('applies scale transform', () => {
    const scale = 1.5;
    render(<DeviceFrame {...mockProps} scale={scale} />);
    const frame = screen.getByTestId('device-frame');
    
    expect(frame).toHaveStyle({
      transform: `scale(${scale})`,
    });
  });

  it('renders children content', () => {
    render(<DeviceFrame {...mockProps} />);
    expect(screen.getByTestId('frame-content')).toBeInTheDocument();
  });

  it('handles device type change', () => {
    const { rerender } = render(<DeviceFrame {...mockProps} />);
    
    rerender(<DeviceFrame {...mockProps} deviceType="iphone-14-pro" />);
    const frame = screen.getByTestId('device-frame');
    
    expect(frame).toHaveClass('device-iphone-14-pro');
    expect(frame).not.toHaveClass('device-iphone-13');
  });
}); 
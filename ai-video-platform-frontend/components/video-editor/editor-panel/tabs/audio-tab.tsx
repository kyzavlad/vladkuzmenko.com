'use client';

import React, { useState } from 'react';
import { FiVolume2, FiVolumeX, FiMusic, FiMic, FiTrash2, FiPlus } from 'react-icons/fi';
import { useProcessingContext } from '../../contexts/processing-context';

interface AudioTrack {
  id: string;
  type: 'music' | 'sfx' | 'voice';
  name: string;
  volume: number;
  isMuted: boolean;
}

const AudioTab: React.FC = () => {
  const { activeJob, isProcessing } = useProcessingContext();
  const [tracks, setTracks] = useState<AudioTrack[]>([
    {
      id: '1',
      type: 'voice',
      name: 'Main Audio',
      volume: 100,
      isMuted: false
    }
  ]);

  const handleVolumeChange = (trackId: string, volume: number) => {
    setTracks(prevTracks =>
      prevTracks.map(track =>
        track.id === trackId ? { ...track, volume } : track
      )
    );
  };

  const handleMuteToggle = (trackId: string) => {
    setTracks(prevTracks =>
      prevTracks.map(track =>
        track.id === trackId ? { ...track, isMuted: !track.isMuted } : track
      )
    );
  };

  const handleRemoveTrack = (trackId: string) => {
    setTracks(prevTracks => prevTracks.filter(track => track.id !== trackId));
  };

  const getTrackIcon = (type: AudioTrack['type']) => {
    switch (type) {
      case 'music':
        return <FiMusic className="w-4 h-4" />;
      case 'sfx':
        return <FiVolume2 className="w-4 h-4" />;
      case 'voice':
        return <FiMic className="w-4 h-4" />;
      default:
        return <FiVolume2 className="w-4 h-4" />;
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-sm font-medium">Audio Tracks</h3>
        <button
          className="text-sm text-blue-500 hover:text-blue-600 flex items-center space-x-1"
          onClick={() => {/* TODO: Implement add track */}}
        >
          <FiPlus className="w-4 h-4" />
          <span>Add Track</span>
        </button>
      </div>

      <div className="space-y-2">
        {tracks.map(track => (
          <div
            key={track.id}
            className="flex items-center justify-between p-2 bg-gray-50 rounded"
          >
            <div className="flex items-center space-x-2">
              {getTrackIcon(track.type)}
              <div>
                <p className="text-sm font-medium">{track.name}</p>
                <p className="text-xs text-gray-500 capitalize">{track.type}</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <button
                  className="text-gray-500 hover:text-gray-700"
                  onClick={() => handleMuteToggle(track.id)}
                >
                  {track.isMuted ? (
                    <FiVolumeX className="w-4 h-4" />
                  ) : (
                    <FiVolume2 className="w-4 h-4" />
                  )}
                </button>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={track.volume}
                  onChange={(e) => handleVolumeChange(track.id, Number(e.target.value))}
                  className="w-24"
                  disabled={track.isMuted}
                />
              </div>
              <button
                className="text-red-500 hover:text-red-600"
                onClick={() => handleRemoveTrack(track.id)}
              >
                <FiTrash2 className="w-4 h-4" />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AudioTab; 
'use client';

import React, { useState } from 'react';
import { FiSave, FiTrash2, FiPlus } from 'react-icons/fi';
import { useProcessingContext } from '../../contexts/processing-context';

interface Preset {
  id: string;
  name: string;
  description: string;
  settings: {
    [key: string]: any;
  };
}

const PresetSelector: React.FC = () => {
  const { activeJob, isProcessing } = useProcessingContext();
  const [presets, setPresets] = useState<Preset[]>([
    {
      id: '1',
      name: 'Professional Look',
      description: 'High-quality settings for professional videos',
      settings: {
        colorCorrection: true,
        stabilization: true,
        audioEnhancement: true,
        noiseReduction: true
      }
    },
    {
      id: '2',
      name: 'Social Media',
      description: 'Optimized for social media platforms',
      settings: {
        colorCorrection: true,
        stabilization: true,
        audioEnhancement: true,
        noiseReduction: false
      }
    }
  ]);
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [newPresetName, setNewPresetName] = useState('');
  const [newPresetDescription, setNewPresetDescription] = useState('');

  const handlePresetSelect = (presetId: string) => {
    setSelectedPreset(presetId);
    // TODO: Apply preset settings to the editor
  };

  const handleCreatePreset = () => {
    if (!newPresetName.trim()) return;

    const newPreset: Preset = {
      id: `preset-${Date.now()}`,
      name: newPresetName,
      description: newPresetDescription,
      settings: {
        // TODO: Get current editor settings
        colorCorrection: true,
        stabilization: true,
        audioEnhancement: true,
        noiseReduction: true
      }
    };

    setPresets(prev => [...prev, newPreset]);
    setIsCreating(false);
    setNewPresetName('');
    setNewPresetDescription('');
  };

  const handleDeletePreset = (presetId: string) => {
    setPresets(prev => prev.filter(preset => preset.id !== presetId));
    if (selectedPreset === presetId) {
      setSelectedPreset(null);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-sm font-medium">Presets</h3>
        <button
          className="text-sm text-blue-500 hover:text-blue-600 flex items-center space-x-1"
          onClick={() => setIsCreating(true)}
        >
          <FiPlus className="w-4 h-4" />
          <span>New Preset</span>
        </button>
      </div>

      {isCreating ? (
        <div className="space-y-3 p-4 bg-gray-50 rounded-lg">
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Preset Name
            </label>
            <input
              type="text"
              value={newPresetName}
              onChange={(e) => setNewPresetName(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
              placeholder="Enter preset name"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Description
            </label>
            <textarea
              value={newPresetDescription}
              onChange={(e) => setNewPresetDescription(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
              placeholder="Enter preset description"
              rows={2}
            />
          </div>
          <div className="flex justify-end space-x-2">
            <button
              className="px-3 py-1 text-sm text-gray-600 hover:text-gray-800"
              onClick={() => setIsCreating(false)}
            >
              Cancel
            </button>
            <button
              className="px-3 py-1 text-sm text-white bg-blue-500 rounded-md hover:bg-blue-600"
              onClick={handleCreatePreset}
            >
              Create
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-2">
          {presets.map(preset => (
            <div
              key={preset.id}
              className={`p-3 rounded-lg cursor-pointer transition-colors ${
                selectedPreset === preset.id
                  ? 'bg-blue-50 border border-blue-200'
                  : 'bg-gray-50 hover:bg-gray-100'
              }`}
              onClick={() => handlePresetSelect(preset.id)}
            >
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-sm font-medium">{preset.name}</h4>
                  <p className="text-xs text-gray-500">{preset.description}</p>
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    className="p-1 text-gray-400 hover:text-gray-600"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeletePreset(preset.id);
                    }}
                  >
                    <FiTrash2 className="w-4 h-4" />
                  </button>
                  <button
                    className="p-1 text-gray-400 hover:text-gray-600"
                    onClick={(e) => {
                      e.stopPropagation();
                      // TODO: Implement save current settings as preset
                    }}
                  >
                    <FiSave className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default PresetSelector; 
'use client';

import React, { useState } from 'react';
import { FiSliders, FiCheck, FiX } from 'react-icons/fi';
import { useProcessingContext } from '../../contexts/processing-context';

interface Enhancement {
  id: string;
  name: string;
  description: string;
  isEnabled: boolean;
  settings: {
    [key: string]: any;
  };
}

const EnhancementsTab: React.FC = () => {
  const { activeJob, isProcessing } = useProcessingContext();
  const [enhancements, setEnhancements] = useState<Enhancement[]>([
    {
      id: '1',
      name: 'Auto Color Correction',
      description: 'Automatically adjust colors for better visual quality',
      isEnabled: true,
      settings: {
        intensity: 75,
        preserveSkinTones: true
      }
    },
    {
      id: '2',
      name: 'Stabilization',
      description: 'Reduce camera shake and stabilize footage',
      isEnabled: false,
      settings: {
        strength: 50,
        smoothing: 75
      }
    },
    {
      id: '3',
      name: 'Background Removal',
      description: 'Remove or blur background for better focus',
      isEnabled: false,
      settings: {
        blurStrength: 50,
        edgeSmoothing: 75
      }
    }
  ]);

  const handleToggleEnhancement = (enhancementId: string) => {
    setEnhancements(prevEnhancements =>
      prevEnhancements.map(enhancement =>
        enhancement.id === enhancementId
          ? { ...enhancement, isEnabled: !enhancement.isEnabled }
          : enhancement
      )
    );
  };

  const handleSettingChange = (
    enhancementId: string,
    settingKey: string,
    value: any
  ) => {
    setEnhancements(prevEnhancements =>
      prevEnhancements.map(enhancement =>
        enhancement.id === enhancementId
          ? {
              ...enhancement,
              settings: {
                ...enhancement.settings,
                [settingKey]: value
              }
            }
          : enhancement
      )
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-sm font-medium">Video Enhancements</h3>
        <button
          className="text-sm text-blue-500 hover:text-blue-600 flex items-center space-x-1"
          onClick={() => {/* TODO: Implement add enhancement */}}
        >
          <FiSliders className="w-4 h-4" />
          <span>Add Enhancement</span>
        </button>
      </div>

      <div className="space-y-4">
        {enhancements.map(enhancement => (
          <div
            key={enhancement.id}
            className="p-4 bg-gray-50 rounded-lg space-y-3"
          >
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-medium">{enhancement.name}</h4>
                <p className="text-xs text-gray-500">{enhancement.description}</p>
              </div>
              <button
                className={`p-1 rounded-full ${
                  enhancement.isEnabled
                    ? 'bg-green-100 text-green-600'
                    : 'bg-gray-100 text-gray-400'
                }`}
                onClick={() => handleToggleEnhancement(enhancement.id)}
              >
                {enhancement.isEnabled ? (
                  <FiCheck className="w-4 h-4" />
                ) : (
                  <FiX className="w-4 h-4" />
                )}
              </button>
            </div>

            {enhancement.isEnabled && (
              <div className="space-y-2 pt-2 border-t border-gray-200">
                {Object.entries(enhancement.settings).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between">
                    <label className="text-xs text-gray-600 capitalize">
                      {key.replace(/([A-Z])/g, ' $1').trim()}
                    </label>
                    {typeof value === 'boolean' ? (
                      <button
                        className={`p-1 rounded-full ${
                          value
                            ? 'bg-green-100 text-green-600'
                            : 'bg-gray-100 text-gray-400'
                        }`}
                        onClick={() =>
                          handleSettingChange(enhancement.id, key, !value)
                        }
                      >
                        {value ? (
                          <FiCheck className="w-3 h-3" />
                        ) : (
                          <FiX className="w-3 h-3" />
                        )}
                      </button>
                    ) : (
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={value}
                        onChange={(e) =>
                          handleSettingChange(
                            enhancement.id,
                            key,
                            Number(e.target.value)
                          )
                        }
                        className="w-32"
                      />
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default EnhancementsTab; 
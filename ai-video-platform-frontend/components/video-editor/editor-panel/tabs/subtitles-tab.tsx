'use client';

import React from 'react';
import { useEditorContext } from '../../contexts/editor-context';
import SettingSlider from '../controls/setting-slider';
import SettingToggle from '../controls/setting-toggle';
import SettingSelect from '../controls/setting-select';
import SettingColorPicker from '../controls/setting-color-picker';
import InfoTooltip from '../controls/info-tooltip';

export default function SubtitlesTab() {
  const { settings, updateSettings } = useEditorContext();
  
  const fontOptions = [
    { value: 'Arial', label: 'Arial' },
    { value: 'Roboto', label: 'Roboto' },
    { value: 'Poppins', label: 'Poppins' },
    { value: 'Montserrat', label: 'Montserrat' },
    { value: 'Open Sans', label: 'Open Sans' }
  ];
  
  const positionOptions = [
    { value: 'bottom', label: 'Bottom' },
    { value: 'top', label: 'Top' },
    { value: 'middle', label: 'Middle' }
  ];
  
  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <span className="text-sm font-medium text-neutral-100">Enable Subtitles</span>
          <InfoTooltip text="Generate and display subtitles for your video" />
        </div>
        <SettingToggle
          value={settings.subtitles.enabled}
          onChange={(value) => updateSettings('subtitles', { enabled: value })}
        />
      </div>
      
      {settings.subtitles.enabled && (
        <>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <span className="text-sm font-medium text-neutral-100">Font Family</span>
              <InfoTooltip text="Choose the font for your subtitles" />
            </div>
            <div className="w-40">
              <SettingSelect
                value={settings.subtitles.style.font}
                options={fontOptions}
                onChange={(value) => updateSettings('subtitles', { 
                  style: { ...settings.subtitles.style, font: value } 
                })}
              />
            </div>
          </div>
          
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <span className="text-sm font-medium text-neutral-100">Font Size</span>
              <InfoTooltip text="Adjust the size of your subtitles" />
            </div>
            <div className="w-40">
              <SettingSlider
                value={settings.subtitles.style.size}
                min={12}
                max={48}
                step={1}
                onChange={(value) => updateSettings('subtitles', { 
                  style: { ...settings.subtitles.style, size: value } 
                })}
              />
            </div>
          </div>
          
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <span className="text-sm font-medium text-neutral-100">Font Color</span>
              <InfoTooltip text="Choose the color of your subtitles" />
            </div>
            <div className="w-40">
              <SettingColorPicker
                value={settings.subtitles.style.color}
                onChange={(value) => updateSettings('subtitles', { 
                  style: { ...settings.subtitles.style, color: value } 
                })}
              />
            </div>
          </div>
          
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <span className="text-sm font-medium text-neutral-100">Position</span>
              <InfoTooltip text="Choose where to display the subtitles" />
            </div>
            <div className="w-40">
              <SettingSelect
                value={settings.subtitles.style.position}
                options={positionOptions}
                onChange={(value) => updateSettings('subtitles', { 
                  style: { ...settings.subtitles.style, position: value as 'top' | 'bottom' | 'middle' } 
                })}
              />
            </div>
          </div>
          
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <span className="text-sm font-medium text-neutral-100">Outline</span>
              <InfoTooltip text="Add an outline to your subtitles" />
            </div>
            <SettingToggle
              value={settings.subtitles.style.outline}
              onChange={(value) => updateSettings('subtitles', { 
                style: { ...settings.subtitles.style, outline: value } 
              })}
            />
          </div>
          
          {settings.subtitles.style.outline && (
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center">
                <span className="text-sm font-medium text-neutral-100">Outline Color</span>
                <InfoTooltip text="Choose the outline color for your subtitles" />
              </div>
              <div className="w-40">
                <SettingColorPicker
                  value={settings.subtitles.style.outlineColor}
                  onChange={(value) => updateSettings('subtitles', { 
                    style: { ...settings.subtitles.style, outlineColor: value } 
                  })}
                />
              </div>
            </div>
          )}
          
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <span className="text-sm font-medium text-neutral-100">Background Color</span>
              <InfoTooltip text="Choose the background color for your subtitles" />
            </div>
            <div className="w-40">
              <SettingColorPicker
                value={settings.subtitles.style.backgroundColor}
                onChange={(value) => updateSettings('subtitles', { 
                  style: { ...settings.subtitles.style, backgroundColor: value } 
                })}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
} 
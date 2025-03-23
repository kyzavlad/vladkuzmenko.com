'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FiCheck, FiChevronDown, FiEdit, FiPlus, FiTrash } from 'react-icons/fi';
import { useTranslationContext, TranslationTerminology } from '../contexts/translation-context';

interface TranslationSettingsProps {
  onComplete?: () => void;
  customOptions?: Partial<{
    preserveNames: boolean;
    preserveFormatting: boolean;
    preserveTechnicalTerms: boolean;
    formality: 'formal' | 'neutral' | 'informal';
    keepOriginalVoice: boolean;
    useTranslationMemory: boolean;
  }>;
  onChange?: (options: TranslationSettingsProps['customOptions']) => void;
}

export const TranslationSettings: React.FC<TranslationSettingsProps> = ({ 
  onComplete, 
  customOptions,
  onChange 
}) => {
  const { 
    terminologies, 
    selectedTerminologies, 
    toggleTerminology,
    addTerminology,
    updateTerminology,
    deleteTerminology
  } = useTranslationContext();

  const [settings, setSettings] = useState<Required<NonNullable<TranslationSettingsProps['customOptions']>>>({
    preserveNames: customOptions?.preserveNames ?? true,
    preserveFormatting: customOptions?.preserveFormatting ?? true,
    preserveTechnicalTerms: customOptions?.preserveTechnicalTerms ?? true,
    formality: customOptions?.formality ?? 'neutral',
    keepOriginalVoice: customOptions?.keepOriginalVoice ?? false,
    useTranslationMemory: customOptions?.useTranslationMemory ?? true
  });

  const [showNewTermPanel, setShowNewTermPanel] = useState(false);
  const [editingTerminology, setEditingTerminology] = useState<TranslationTerminology | null>(null);
  const [newTermName, setNewTermName] = useState('');
  const [newTermPairs, setNewTermPairs] = useState<Array<{ source: string; target: string }>>([{ source: '', target: '' }]);

  const handleSettingChange = (key: keyof typeof settings, value: any) => {
    const updatedSettings = { ...settings, [key]: value };
    setSettings(updatedSettings);
    onChange?.(updatedSettings);
  };

  const handleToggleTerm = (id: string) => {
    toggleTerminology(id);
  };

  const handleCreateTerm = async () => {
    if (!newTermName.trim()) return;
    
    // Filter out empty pairs
    const validPairs = newTermPairs.filter(pair => pair.source.trim() && pair.target.trim());
    if (validPairs.length === 0) return;

    try {
      await addTerminology({
        name: newTermName,
        terms: validPairs
      });
      
      // Reset form
      setNewTermName('');
      setNewTermPairs([{ source: '', target: '' }]);
      setShowNewTermPanel(false);
    } catch (error) {
      console.error('Error creating terminology:', error);
    }
  };

  const handleUpdateTerm = async () => {
    if (!editingTerminology || !newTermName.trim()) return;
    
    // Filter out empty pairs
    const validPairs = newTermPairs.filter(pair => pair.source.trim() && pair.target.trim());
    if (validPairs.length === 0) return;

    try {
      await updateTerminology(editingTerminology.id, {
        name: newTermName,
        terms: validPairs
      });
      
      // Reset form
      setNewTermName('');
      setNewTermPairs([{ source: '', target: '' }]);
      setEditingTerminology(null);
    } catch (error) {
      console.error('Error updating terminology:', error);
    }
  };

  const startEditingTerm = (term: TranslationTerminology) => {
    setEditingTerminology(term);
    setNewTermName(term.name);
    setNewTermPairs([...term.terms]);
  };

  const handleDeleteTerm = async (id: string) => {
    try {
      await deleteTerminology(id);
    } catch (error) {
      console.error('Error deleting terminology:', error);
    }
  };

  const addNewPair = () => {
    setNewTermPairs([...newTermPairs, { source: '', target: '' }]);
  };

  const updatePair = (index: number, field: 'source' | 'target', value: string) => {
    const updatedPairs = [...newTermPairs];
    updatedPairs[index][field] = value;
    setNewTermPairs(updatedPairs);
  };

  const removePair = (index: number) => {
    if (newTermPairs.length <= 1) return;
    const updatedPairs = [...newTermPairs];
    updatedPairs.splice(index, 1);
    setNewTermPairs(updatedPairs);
  };

  return (
    <div className="w-full bg-white rounded-lg shadow-md p-6 mb-6">
      <h2 className="text-2xl font-bold mb-4">Translation Settings</h2>
      
      <div className="space-y-6">
        {/* General Options */}
        <div>
          <h3 className="text-lg font-semibold mb-3">General Options</h3>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={settings.preserveNames}
                  onChange={(e) => handleSettingChange('preserveNames', e.target.checked)}
                  className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500"
                />
                <span>Preserve proper names</span>
              </label>
              <div className="text-sm text-gray-500">
                Keep names of people, places, and organizations in original form
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={settings.preserveFormatting}
                  onChange={(e) => handleSettingChange('preserveFormatting', e.target.checked)}
                  className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500"
                />
                <span>Preserve formatting</span>
              </label>
              <div className="text-sm text-gray-500">
                Keep formatting elements like line breaks and emphasis
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={settings.preserveTechnicalTerms}
                  onChange={(e) => handleSettingChange('preserveTechnicalTerms', e.target.checked)}
                  className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500"
                />
                <span>Preserve technical terms</span>
              </label>
              <div className="text-sm text-gray-500">
                Keep industry-specific terminology in their standard translations
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={settings.keepOriginalVoice}
                  onChange={(e) => handleSettingChange('keepOriginalVoice', e.target.checked)}
                  className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500"
                />
                <span>Keep original voice</span>
              </label>
              <div className="text-sm text-gray-500">
                Preserve the original speaker's voice characteristics in the translation
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={settings.useTranslationMemory}
                  onChange={(e) => handleSettingChange('useTranslationMemory', e.target.checked)}
                  className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500"
                />
                <span>Use translation memory</span>
              </label>
              <div className="text-sm text-gray-500">
                Apply previous translations for consistency across videos
              </div>
            </div>
            
            <div>
              <div className="flex justify-between items-center mb-2">
                <label>Formality level</label>
                <div className="text-sm text-gray-500">
                  Choose how formal or casual the translation should be
                </div>
              </div>
              <div className="flex">
                <button
                  type="button"
                  className={`flex-1 py-2 border ${settings.formality === 'formal' ? 'bg-blue-50 border-blue-500 text-blue-700' : 'border-gray-300 hover:bg-gray-50'} rounded-l-md`}
                  onClick={() => handleSettingChange('formality', 'formal')}
                >
                  Formal
                </button>
                <button
                  type="button"
                  className={`flex-1 py-2 border-t border-b ${settings.formality === 'neutral' ? 'bg-blue-50 border-blue-500 text-blue-700' : 'border-gray-300 hover:bg-gray-50'}`}
                  onClick={() => handleSettingChange('formality', 'neutral')}
                >
                  Neutral
                </button>
                <button
                  type="button"
                  className={`flex-1 py-2 border ${settings.formality === 'informal' ? 'bg-blue-50 border-blue-500 text-blue-700' : 'border-gray-300 hover:bg-gray-50'} rounded-r-md`}
                  onClick={() => handleSettingChange('formality', 'informal')}
                >
                  Informal
                </button>
              </div>
            </div>
          </div>
        </div>
        
        {/* Terminology Management */}
        <div>
          <h3 className="text-lg font-semibold mb-3">Terminology Management</h3>
          
          <div className="space-y-4">
            {/* Terminology List */}
            <div className="border border-gray-200 rounded-md divide-y">
              {terminologies.length === 0 ? (
                <div className="p-4 text-gray-500 text-center">
                  No terminology lists created yet
                </div>
              ) : (
                terminologies.map(term => (
                  <div key={term.id} className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          checked={selectedTerminologies.includes(term.id)}
                          onChange={() => handleToggleTerm(term.id)}
                          className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500"
                        />
                        <span className="font-medium">{term.name}</span>
                        <span className="text-gray-500 text-sm">
                          ({term.terms.length} terms)
                        </span>
                      </div>
                      <div className="flex space-x-2">
                        <button
                          className="text-gray-500 hover:text-blue-600"
                          onClick={() => startEditingTerm(term)}
                        >
                          <FiEdit size={16} />
                        </button>
                        <button
                          className="text-gray-500 hover:text-red-600"
                          onClick={() => handleDeleteTerm(term.id)}
                        >
                          <FiTrash size={16} />
                        </button>
                      </div>
                    </div>
                    
                    {/* Preview of terms */}
                    {selectedTerminologies.includes(term.id) && (
                      <div className="mt-2 pl-6 text-sm text-gray-600 grid grid-cols-2 gap-2">
                        {term.terms.slice(0, 3).map((pair, idx) => (
                          <div key={idx} className="flex space-x-2">
                            <span className="font-medium">{pair.source}</span>
                            <span>→</span>
                            <span>{pair.target}</span>
                          </div>
                        ))}
                        {term.terms.length > 3 && (
                          <div className="text-gray-500">
                            +{term.terms.length - 3} more terms
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
            
            {/* Add New Terminology Button */}
            {!showNewTermPanel && !editingTerminology && (
              <button
                className="flex items-center space-x-1 py-2 px-3 border border-gray-300 rounded-md hover:bg-gray-50 hover:border-blue-500"
                onClick={() => setShowNewTermPanel(true)}
              >
                <FiPlus />
                <span>Add New Terminology</span>
              </button>
            )}
            
            {/* New Terminology Form */}
            {(showNewTermPanel || editingTerminology) && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="border border-gray-200 rounded-md p-4 space-y-4"
              >
                <h4 className="font-medium">
                  {editingTerminology ? 'Edit Terminology' : 'New Terminology'}
                </h4>
                
                <div>
                  <label className="block text-sm text-gray-700 mb-1">Name</label>
                  <input
                    type="text"
                    value={newTermName}
                    onChange={(e) => setNewTermName(e.target.value)}
                    placeholder="e.g., Technical Terms, Brand Names"
                    className="w-full p-2 border border-gray-300 rounded-md"
                  />
                </div>
                
                <div>
                  <div className="flex justify-between mb-1">
                    <label className="block text-sm text-gray-700">Term Pairs</label>
                    <button
                      className="text-sm text-blue-600 hover:underline"
                      onClick={addNewPair}
                    >
                      + Add Pair
                    </button>
                  </div>
                  
                  <div className="space-y-2">
                    {newTermPairs.map((pair, idx) => (
                      <div key={idx} className="flex space-x-2">
                        <input
                          type="text"
                          value={pair.source}
                          onChange={(e) => updatePair(idx, 'source', e.target.value)}
                          placeholder="Source term"
                          className="flex-1 p-2 border border-gray-300 rounded-md"
                        />
                        <input
                          type="text"
                          value={pair.target}
                          onChange={(e) => updatePair(idx, 'target', e.target.value)}
                          placeholder="Target term"
                          className="flex-1 p-2 border border-gray-300 rounded-md"
                        />
                        <button
                          className="text-gray-500 hover:text-red-600"
                          onClick={() => removePair(idx)}
                          disabled={newTermPairs.length <= 1}
                        >
                          <FiTrash size={16} />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div className="flex justify-end space-x-2">
                  <button
                    className="py-2 px-4 border border-gray-300 rounded-md hover:bg-gray-50"
                    onClick={() => {
                      setShowNewTermPanel(false);
                      setEditingTerminology(null);
                      setNewTermName('');
                      setNewTermPairs([{ source: '', target: '' }]);
                    }}
                  >
                    Cancel
                  </button>
                  <button
                    className="py-2 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-blue-300"
                    onClick={editingTerminology ? handleUpdateTerm : handleCreateTerm}
                    disabled={!newTermName.trim() || newTermPairs.every(p => !p.source.trim() || !p.target.trim())}
                  >
                    {editingTerminology ? 'Update' : 'Create'}
                  </button>
                </div>
              </motion.div>
            )}
          </div>
        </div>
        
        {/* Action Buttons */}
        {onComplete && (
          <div className="pt-4 flex justify-end">
            <button
              className="py-2 px-6 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              onClick={onComplete}
            >
              Apply Settings
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default TranslationSettings; 
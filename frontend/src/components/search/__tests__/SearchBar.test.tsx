import React from 'react';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { render } from '../../../utils/test-utils';
import { SearchBar } from '../SearchBar';
import userEvent from '@testing-library/user-event';

const mockSearchResults = [
  {
    id: '1',
    type: 'video',
    title: 'Test Video',
    description: 'A test video description',
    path: '/videos/test',
    timestamp: '2024-01-20T12:00:00Z',
  },
];

jest.mock('../../../services/api', () => ({
  searchContent: jest.fn().mockResolvedValue(mockSearchResults),
}));

describe('SearchBar Component', () => {
  beforeEach(() => {
    render(<SearchBar />);
  });

  it('renders search input', () => {
    const searchInput = screen.getByPlaceholderText('Search content...');
    expect(searchInput).toBeInTheDocument();
  });

  it('shows loading state while searching', async () => {
    const searchInput = screen.getByPlaceholderText('Search content...');
    await userEvent.type(searchInput, 'test');
    
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    });
  });

  it('displays search results', async () => {
    const searchInput = screen.getByPlaceholderText('Search content...');
    await userEvent.type(searchInput, 'test');

    await waitFor(() => {
      expect(screen.getByText('Test Video')).toBeInTheDocument();
      expect(screen.getByText('A test video description')).toBeInTheDocument();
    });
  });

  it('clears search results when input is cleared', async () => {
    const searchInput = screen.getByPlaceholderText('Search content...');
    await userEvent.type(searchInput, 'test');
    
    await waitFor(() => {
      expect(screen.getByText('Test Video')).toBeInTheDocument();
    });

    await userEvent.clear(searchInput);
    expect(screen.queryByText('Test Video')).not.toBeInTheDocument();
  });

  it('handles search error state', async () => {
    jest.spyOn(console, 'error').mockImplementation(() => {});
    const mockError = new Error('Search failed');
    jest.spyOn(require('../../../services/api'), 'searchContent')
      .mockRejectedValueOnce(mockError);

    const searchInput = screen.getByPlaceholderText('Search content...');
    await userEvent.type(searchInput, 'error');

    await waitFor(() => {
      expect(screen.getByText('Error: Search failed')).toBeInTheDocument();
    });
  });
}); 
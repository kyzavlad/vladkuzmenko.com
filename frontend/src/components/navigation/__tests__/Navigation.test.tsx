import React from 'react';
import { screen, fireEvent } from '@testing-library/react';
import { render } from '../../../utils/test-utils';
import { Navigation } from '../Navigation';

describe('Navigation Component', () => {
  beforeEach(() => {
    render(<Navigation />);
  });

  it('renders logo and main navigation links', () => {
    expect(screen.getByAltText('Logo')).toBeInTheDocument();
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Editor')).toBeInTheDocument();
    expect(screen.getByText('Library')).toBeInTheDocument();
  });

  it('shows search bar', () => {
    const searchInput = screen.getByPlaceholderText('Search...');
    expect(searchInput).toBeInTheDocument();
  });

  it('toggles mobile menu', () => {
    const menuButton = screen.getByLabelText('Toggle Navigation');
    fireEvent.click(menuButton);
    expect(screen.getByRole('navigation')).toHaveAttribute('aria-expanded', 'true');
  });

  it('toggles color mode', () => {
    const colorModeButton = screen.getByLabelText('Toggle color mode');
    fireEvent.click(colorModeButton);
    expect(document.documentElement).toHaveAttribute('data-theme', 'dark');
  });

  it('shows user menu when logged in', () => {
    const userMenuButton = screen.getByLabelText('User menu');
    fireEvent.click(userMenuButton);
    expect(screen.getByText('Profile')).toBeInTheDocument();
    expect(screen.getByText('Settings')).toBeInTheDocument();
    expect(screen.getByText('Logout')).toBeInTheDocument();
  });
}); 
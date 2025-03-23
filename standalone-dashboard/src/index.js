import React from 'react';
import ReactDOM from 'react-dom';
import Dashboard from './Dashboard';

// Create global object for integration
window.PlatformDashboard = {
  init: function(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
      ReactDOM.render(<Dashboard />, container);
      return true;
    }
    return false;
  }
}; 
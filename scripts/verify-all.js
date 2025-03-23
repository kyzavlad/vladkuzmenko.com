const { spawn } = require('child_process');
const path = require('path');

async function runCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, {
      stdio: 'inherit',
      ...options
    });

    proc.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Command failed with code ${code}`));
      }
    });
  });
}

async function verifyAll() {
  console.log('\n1. Running Feature Verification...');
  try {
    await runCommand('node', ['scripts/verify-features.js']);
  } catch (error) {
    console.error('Feature verification failed:', error);
    process.exit(1);
  }

  console.log('\n2. Running Performance Monitoring...');
  try {
    await runCommand('node', ['scripts/monitor-performance.js']);
  } catch (error) {
    console.error('Performance monitoring failed:', error);
    process.exit(1);
  }

  console.log('\n3. Running Integration Tests...');
  try {
    await runCommand('npm', ['run', 'test:integration']);
  } catch (error) {
    console.error('Integration tests failed:', error);
    process.exit(1);
  }

  console.log('\n4. Running E2E Tests...');
  try {
    await runCommand('npm', ['run', 'test:e2e']);
  } catch (error) {
    console.error('E2E tests failed:', error);
    process.exit(1);
  }

  console.log('\n5. Checking API Health...');
  try {
    const response = await fetch(`${process.env.BASE_URL || 'http://localhost:3000'}/api/health`);
    if (!response.ok) {
      throw new Error(`API health check failed: ${response.status}`);
    }
    console.log('API health check passed');
  } catch (error) {
    console.error('API health check failed:', error);
    process.exit(1);
  }

  console.log('\n6. Checking WebSocket Connection...');
  try {
    const WebSocket = require('ws');
    const ws = new WebSocket(`${(process.env.BASE_URL || 'http://localhost:3000').replace('http', 'ws')}/ws`);
    
    await new Promise((resolve, reject) => {
      ws.on('open', resolve);
      ws.on('error', reject);
      setTimeout(() => reject(new Error('WebSocket connection timeout')), 5000);
    });
    
    ws.close();
    console.log('WebSocket connection check passed');
  } catch (error) {
    console.error('WebSocket connection check failed:', error);
    process.exit(1);
  }

  console.log('\n7. Checking Token Management...');
  try {
    const response = await fetch(`${process.env.BASE_URL || 'http://localhost:3000'}/api/tokens/balance`, {
      headers: {
        Authorization: `Bearer ${process.env.TEST_AUTH_TOKEN}`
      }
    });
    if (!response.ok) {
      throw new Error(`Token management check failed: ${response.status}`);
    }
    console.log('Token management check passed');
  } catch (error) {
    console.error('Token management check failed:', error);
    process.exit(1);
  }

  console.log('\nAll verifications completed successfully! 🎉');
}

// Run verification if called directly
if (require.main === module) {
  verifyAll()
    .catch((error) => {
      console.error('\nVerification failed:', error);
      process.exit(1);
    });
}

module.exports = verifyAll; 
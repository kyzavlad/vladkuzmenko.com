const axios = require('axios');
const WebSocket = require('ws');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

async function verifyFeatures(baseUrl, authToken) {
  const api = axios.create({
    baseURL: baseUrl,
    headers: { Authorization: `Bearer ${authToken}` }
  });

  console.log('Starting comprehensive feature verification...');

  // 1. Verify Authentication & User Settings
  console.log('\n1. Verifying authentication and user settings...');
  const authResponse = await api.get('/api/auth/verify');
  console.assert(authResponse.status === 200, 'Authentication failed');

  const settings = await api.get('/api/user/settings');
  console.assert(settings.data.preferences, 'Failed to fetch user settings');

  // 2. Token Management
  console.log('\n2. Verifying token management...');
  const initialBalance = await api.get('/api/tokens/balance');
  console.assert(initialBalance.data.balance >= 0, 'Invalid token balance');

  // 3. Video Upload & Processing
  console.log('\n3. Verifying video upload and processing...');
  const form = new FormData();
  form.append('file', fs.createReadStream(path.join(__dirname, '../test-assets/sample.mp4')));
  
  const uploadResponse = await api.post('/api/videos/upload', form, {
    headers: { ...form.getHeaders() }
  });
  const videoId = uploadResponse.data.videoId;
  console.assert(videoId, 'Failed to upload video');

  // 4. Translation Pipeline
  console.log('\n4. Verifying translation pipeline...');
  const translationJob = await api.post('/api/videos/translate', {
    videoId,
    sourceLang: 'en',
    targetLang: 'es'
  });
  console.assert(translationJob.data.jobId, 'Failed to create translation job');

  // 5. WebSocket Connection & Real-time Updates
  console.log('\n5. Verifying WebSocket connection...');
  const ws = new WebSocket(`${baseUrl.replace('http', 'ws')}/ws`, {
    headers: { Authorization: `Bearer ${authToken}` }
  });
  
  const wsPromise = new Promise((resolve, reject) => {
    let messageReceived = false;
    
    ws.on('open', () => {
      console.log('WebSocket connected successfully');
      ws.send(JSON.stringify({ type: 'SUBSCRIBE', payload: { videoId } }));
    });

    ws.on('message', (data) => {
      const message = JSON.parse(data);
      if (message.type === 'PROGRESS_UPDATE') {
        messageReceived = true;
        console.log('Received progress update:', message.payload.progress);
      }
    });

    setTimeout(() => {
      if (messageReceived) resolve();
      else reject(new Error('No WebSocket messages received'));
    }, 5000);
  });

  await wsPromise;

  // 6. Avatar Creation & Integration
  console.log('\n6. Verifying avatar creation and integration...');
  const avatarJob = await api.post('/api/avatar/create', {
    name: 'Test Avatar',
    style: 'professional',
    videoId // Use the uploaded video for avatar creation
  });
  const avatarId = avatarJob.data.avatarId;
  console.assert(avatarId, 'Failed to create avatar');

  // Wait for avatar processing
  await new Promise(resolve => setTimeout(resolve, 5000));

  // Verify avatar status
  const avatarStatus = await api.get(`/api/avatar/${avatarId}`);
  console.assert(avatarStatus.data.status === 'ready', 'Avatar not ready');

  // 7. Clip Generation with Integration
  console.log('\n7. Verifying clip generation with integration...');
  const clipJob = await api.post('/api/clips/generate', {
    videoId,
    duration: 30,
    includeAvatar: true,
    avatarId
  });
  console.assert(clipJob.data.clipId, 'Failed to create clip');

  // 8. Token Usage Verification
  console.log('\n8. Verifying token usage...');
  const finalBalance = await api.get('/api/tokens/balance');
  console.assert(
    finalBalance.data.balance < initialBalance.data.balance,
    'Token usage not reflected in balance'
  );

  // 9. Feature Integration Check
  console.log('\n9. Verifying feature integration...');
  const projectStatus = await api.get(`/api/projects/${videoId}`);
  console.assert(
    projectStatus.data.translations?.length > 0 &&
    projectStatus.data.avatars?.length > 0 &&
    projectStatus.data.clips?.length > 0,
    'Feature integration incomplete'
  );

  // 10. Cleanup
  console.log('\n10. Cleaning up test resources...');
  await api.delete(`/api/videos/${videoId}`);
  await api.delete(`/api/avatar/${avatarId}`);

  console.log('\nAll features verified successfully!');
  return {
    videoId,
    avatarId,
    translationJobId: translationJob.data.jobId,
    clipJobId: clipJob.data.clipId
  };
}

// Run verification if called directly
if (require.main === module) {
  const baseUrl = process.env.BASE_URL || 'http://localhost:3000';
  const authToken = process.env.AUTH_TOKEN;
  
  if (!authToken) {
    console.error('AUTH_TOKEN environment variable is required');
    process.exit(1);
  }

  verifyFeatures(baseUrl, authToken)
    .then((results) => {
      console.log('\nVerification Results:', results);
      process.exit(0);
    })
    .catch((error) => {
      console.error('\nVerification failed:', error);
      process.exit(1);
    });
}

module.exports = verifyFeatures; 
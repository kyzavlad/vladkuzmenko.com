const axios = require('axios');
const WebSocket = require('ws');
const { performance } = require('perf_hooks');

class PerformanceMonitor {
  constructor(baseUrl, authToken) {
    this.api = axios.create({
      baseURL: baseUrl,
      headers: { Authorization: `Bearer ${authToken}` }
    });
    this.metrics = {
      api: {},
      websocket: {
        connectionTime: 0,
        messageLatency: []
      },
      processing: {},
      memory: []
    };
  }

  async measureApiCall(endpoint, method = 'GET', data = null) {
    const start = performance.now();
    try {
      const response = await this.api[method.toLowerCase()](endpoint, data);
      const duration = performance.now() - start;
      
      if (!this.metrics.api[endpoint]) {
        this.metrics.api[endpoint] = [];
      }
      this.metrics.api[endpoint].push({
        duration,
        status: response.status,
        timestamp: new Date().toISOString()
      });
      
      return response;
    } catch (error) {
      const duration = performance.now() - start;
      if (!this.metrics.api[endpoint]) {
        this.metrics.api[endpoint] = [];
      }
      this.metrics.api[endpoint].push({
        duration,
        status: error.response?.status || 500,
        error: error.message,
        timestamp: new Date().toISOString()
      });
      throw error;
    }
  }

  async measureWebSocketPerformance(duration = 60000) {
    return new Promise((resolve, reject) => {
      const wsStart = performance.now();
      const ws = new WebSocket(`${this.api.defaults.baseURL.replace('http', 'ws')}/ws`, {
        headers: { Authorization: this.api.defaults.headers.Authorization }
      });

      ws.on('open', () => {
        this.metrics.websocket.connectionTime = performance.now() - wsStart;
        console.log(`WebSocket connected in ${this.metrics.websocket.connectionTime}ms`);
      });

      ws.on('message', (data) => {
        const message = JSON.parse(data);
        if (message.timestamp) {
          const latency = performance.now() - new Date(message.timestamp).getTime();
          this.metrics.websocket.messageLatency.push(latency);
        }
      });

      setTimeout(() => {
        ws.close();
        resolve();
      }, duration);

      ws.on('error', reject);
    });
  }

  async measureFeatureInteraction(scenario) {
    const start = performance.now();
    try {
      switch (scenario) {
        case 'video-translation': {
          // Upload video
          const uploadResponse = await this.measureApiCall('/api/videos/upload', 'POST', {
            name: 'test.mp4',
            size: 1024
          });
          const videoId = uploadResponse.data.videoId;

          // Start translation
          const translationResponse = await this.measureApiCall('/api/videos/translate', 'POST', {
            videoId,
            sourceLang: 'en',
            targetLang: 'es'
          });

          // Monitor progress
          await this.measureWebSocketPerformance(30000);

          return {
            scenario,
            duration: performance.now() - start,
            videoId,
            translationId: translationResponse.data.jobId
          };
        }

        case 'avatar-integration': {
          // Create avatar
          const avatarResponse = await this.measureApiCall('/api/avatar/create', 'POST', {
            name: 'Test Avatar',
            style: 'professional'
          });

          // Use in video
          const videoResponse = await this.measureApiCall('/api/videos/upload', 'POST', {
            name: 'test.mp4',
            size: 1024
          });

          // Integrate
          await this.measureApiCall('/api/videos/integrate', 'POST', {
            videoId: videoResponse.data.videoId,
            avatarId: avatarResponse.data.avatarId
          });

          return {
            scenario,
            duration: performance.now() - start,
            avatarId: avatarResponse.data.avatarId,
            videoId: videoResponse.data.videoId
          };
        }

        default:
          throw new Error(`Unknown scenario: ${scenario}`);
      }
    } catch (error) {
      this.metrics.processing[scenario] = {
        error: error.message,
        duration: performance.now() - start
      };
      throw error;
    }
  }

  recordMemoryUsage() {
    const usage = process.memoryUsage();
    this.metrics.memory.push({
      timestamp: new Date().toISOString(),
      heapUsed: usage.heapUsed,
      heapTotal: usage.heapTotal,
      external: usage.external,
      rss: usage.rss
    });
  }

  async monitorSystem(duration = 300000) {
    console.log('Starting system monitoring...');
    
    // Record memory usage every 10 seconds
    const interval = setInterval(() => this.recordMemoryUsage(), 10000);

    try {
      // Test common feature interactions
      await this.measureFeatureInteraction('video-translation');
      await this.measureFeatureInteraction('avatar-integration');

      // Monitor WebSocket performance
      await this.measureWebSocketPerformance();

      // Test API endpoints under load
      const endpoints = [
        '/api/health',
        '/api/tokens/balance',
        '/api/user/settings'
      ];

      for (const endpoint of endpoints) {
        for (let i = 0; i < 10; i++) {
          await this.measureApiCall(endpoint);
        }
      }

    } finally {
      clearInterval(interval);
    }

    return this.generateReport();
  }

  generateReport() {
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        apiCalls: Object.keys(this.metrics.api).reduce((acc, endpoint) => {
          const calls = this.metrics.api[endpoint];
          acc[endpoint] = {
            count: calls.length,
            averageDuration: calls.reduce((sum, call) => sum + call.duration, 0) / calls.length,
            errors: calls.filter(call => call.error).length
          };
          return acc;
        }, {}),
        websocket: {
          connectionTime: this.metrics.websocket.connectionTime,
          averageLatency: this.metrics.websocket.messageLatency.reduce((a, b) => a + b, 0) / 
                         this.metrics.websocket.messageLatency.length || 0
        },
        memory: {
          averageHeapUsed: this.metrics.memory.reduce((sum, m) => sum + m.heapUsed, 0) / 
                          this.metrics.memory.length,
          maxHeapUsed: Math.max(...this.metrics.memory.map(m => m.heapUsed))
        }
      },
      details: this.metrics
    };

    return report;
  }
}

// Run monitoring if called directly
if (require.main === module) {
  const baseUrl = process.env.BASE_URL || 'http://localhost:3000';
  const authToken = process.env.AUTH_TOKEN;
  
  if (!authToken) {
    console.error('AUTH_TOKEN environment variable is required');
    process.exit(1);
  }

  const monitor = new PerformanceMonitor(baseUrl, authToken);
  
  monitor.monitorSystem()
    .then((report) => {
      console.log('\nPerformance Monitoring Report:');
      console.log(JSON.stringify(report, null, 2));
      process.exit(0);
    })
    .catch((error) => {
      console.error('\nMonitoring failed:', error);
      process.exit(1);
    });
}

module.exports = PerformanceMonitor; 
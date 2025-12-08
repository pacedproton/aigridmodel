// API Configuration
// For production, use relative URLs so nginx can proxy to backend
const API_BASE_URL = process.env.REACT_APP_API_URL || (process.env.NODE_ENV === 'production' ? '' : 'http://localhost:5001');

export const config = {
  api: {
    baseUrl: API_BASE_URL,
    endpoints: {
      health: `${API_BASE_URL}/api/health`,
      generateData: `${API_BASE_URL}/api/generate-data`,
      dataStats: `${API_BASE_URL}/api/data/stats`,
      timeSeries: `${API_BASE_URL}/api/data/time-series`,
      modelsStatus: `${API_BASE_URL}/api/models/status`,
      trainingHistory: (modelType: string) => `${API_BASE_URL}/api/training/history/${modelType}`,
      trainModel: (modelType: string) => `${API_BASE_URL}/api/train/${modelType}`,
      predict: (modelType: string) => `${API_BASE_URL}/api/predict/${modelType}`,
      anomalyDetection: `${API_BASE_URL}/api/anomaly-detection`,
      networkPlot: `${API_BASE_URL}/api/network/plot`,
      // UI Button Endpoints
      uiNeuralNetwork: (useCase: string) => `${API_BASE_URL}/api/ui/neural-network/${useCase}`,
      uiClassicalModel: (useCase: string) => `${API_BASE_URL}/api/ui/classical-model/${useCase}`,
      uiAdvancedMath: (useCase: string) => `${API_BASE_URL}/api/ui/advanced-math/${useCase}`,
      uiShowComparison: (useCase: string) => `${API_BASE_URL}/api/ui/show-comparison/${useCase}`,
      testStatus: `${API_BASE_URL}/api/ui/test-status`
    }
  }
};

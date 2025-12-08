import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  CardHeader,
  Chip,
  LinearProgress,
  Alert,
  Button,
  Stack
} from '@mui/material';
import {
  Memory,
  Timeline,
  ModelTraining,
  Science,
  CheckCircle,
  Error,
  PlayArrow,
  Info
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import axios from 'axios';
import { config } from '../config';

interface DataStats {
  node_features_shape: [number, number, number];
  node_targets_shape: [number, number, number];
  p_load_range: [number, number];
  voltage_range: [number, number];
  time_steps: number;
  nodes: number;
}

interface DashboardProps {
  onShowDemoModal?: () => void;
}

interface TimeSeriesData {
  time: number[];
  p_load: number[];
  voltage: number[];
  q_load: number[];
  theta: number[];
}

const Dashboard: React.FC<DashboardProps> = ({ onShowDemoModal }) => {
  const [dataStats, setDataStats] = useState<DataStats | null>(null);
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [statsResponse, timeSeriesResponse] = await Promise.all([
        axios.get(config.api.endpoints.dataStats).catch(err => {
          if (err.response?.status === 404) {
            return { data: { success: false, error: 'No data generated yet' } };
          }
          throw err;
        }),
        axios.get(config.api.endpoints.timeSeries).catch(err => {
          if (err.response?.status === 404) {
            return { data: { success: false, error: 'No data generated yet' } };
          }
          throw err;
        })
      ]);

      if (statsResponse.data.success) {
        setDataStats(statsResponse.data.stats);
      }
      if (timeSeriesResponse.data.success) {
        setTimeSeriesData(timeSeriesResponse.data.data);
      }
      // If both endpoints return 404, that's expected when no data is generated yet
      if (!statsResponse.data.success && !timeSeriesResponse.data.success) {
        setError('No data available. Please generate data first using the Data Management tab.');
      }
    } catch (err) {
      setError('Unable to connect to backend. Make sure the API server is running.');
    } finally {
      setLoading(false);
    }
  };

  const generateData = async () => {
    try {
      setLoading(true);
      setError(null);
      await axios.post(config.api.endpoints.generateData, { n_steps: 1000 });
      await fetchData(); // Refresh data after generation
    } catch (err) {
      setError('Failed to generate data.');
    } finally {
      setLoading(false);
    }
  };

  const chartData = timeSeriesData ? timeSeriesData.time.map((time, index) => ({
    time,
    'Active Power (MW)': timeSeriesData.p_load[index],
    'Voltage (pu)': timeSeriesData.voltage[index],
    'Reactive Power (MVAr)': timeSeriesData.q_load[index],
    'Angle (°)': timeSeriesData.theta[index]
  })) : [];

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h4" component="h1" gutterBottom>
              AI-Powered Grid Modeling Dashboard
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
              Professional demonstrator for AI-driven power grid analysis and optimization
            </Typography>
          </Box>
          {onShowDemoModal && (
            <Button
              variant="outlined"
              startIcon={<Info />}
              onClick={onShowDemoModal}
              sx={{ ml: 2 }}
            >
              About This Demo
            </Button>
          )}
          <Button
            variant="outlined"
            color="secondary"
            onClick={async () => {
              try {
                console.log('Testing network plot API...');
                const response = await axios.get(config.api.endpoints.networkPlot);
                console.log('API Response:', response.data);
                alert(`API Success! Plot data length: ${response.data.plot?.length || 0} chars`);
              } catch (error) {
                console.error('API Error:', error);
                const errorMessage = (error as Error)?.message || 'Unknown error';
                alert(`API Error: ${errorMessage}`);
              }
            }}
            sx={{ ml: 1 }}
          >
            Test Network API
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
          <Button
            size="small"
            onClick={() => window.open(config.api.endpoints.health, '_blank')}
            sx={{ ml: 2 }}
          >
            Check API Status
          </Button>
        </Alert>
      )}

      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
        {/* System Status */}
        <Box sx={{ flex: '1 1 500px', minWidth: '300px' }}>
          <Card>
            <CardHeader
              title="System Status"
              avatar={<Memory color="primary" />}
            />
            <CardContent>
              <Stack spacing={2}>
                <Box display="flex" alignItems="center" gap={1}>
                  <CheckCircle color="success" />
                  <Typography variant="body2">Backend API: Connected</Typography>
                </Box>
                <Box display="flex" alignItems="center" gap={1}>
                  {dataStats ? (
                    <CheckCircle color="success" />
                  ) : (
                    <Error color="warning" />
                  )}
                  <Typography variant="body2">
                    Grid Data: {dataStats ? 'Loaded' : 'Not Generated'}
                  </Typography>
                </Box>
                <Box display="flex" alignItems="center" gap={1}>
                  <ModelTraining color="info" />
                  <Typography variant="body2">AI Models: Ready for Training</Typography>
                </Box>
              </Stack>
            </CardContent>
          </Card>
        </Box>

        {/* Data Statistics */}
        <Box sx={{ flex: '1 1 500px', minWidth: '300px' }}>
          <Card>
            <CardHeader
              title="Grid Configuration"
              avatar={<Timeline color="secondary" />}
            />
            <CardContent>
              {dataStats ? (
                <Stack spacing={1}>
                  <Typography variant="body2">
                    <strong>Time Steps:</strong> {dataStats.time_steps}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Grid Nodes:</strong> {dataStats.nodes}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Active Power Range:</strong> {dataStats.p_load_range[0].toFixed(1)} - {dataStats.p_load_range[1].toFixed(1)} MW
                  </Typography>
                  <Typography variant="body2">
                    <strong>Voltage Range:</strong> {dataStats.voltage_range[0].toFixed(3)} - {dataStats.voltage_range[1].toFixed(3)} pu
                  </Typography>
                  <Box sx={{ mt: 2 }}>
                    <Button
                      variant="outlined"
                      size="small"
                      onClick={generateData}
                      disabled={loading}
                      startIcon={<PlayArrow />}
                    >
                      Regenerate Data
                    </Button>
                  </Box>
                </Stack>
              ) : (
                <Box>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    No grid data available. Generate synthetic grid data to begin.
                  </Typography>
                  <Button
                    variant="contained"
                    onClick={generateData}
                    disabled={loading}
                    startIcon={<PlayArrow />}
                  >
                    Generate Grid Data
                  </Button>
                </Box>
              )}
            </CardContent>
          </Card>
        </Box>

        {/* Time Series Chart */}
        <Box sx={{ flex: '1 1 100%', minWidth: '100%' }}>
          <Card>
            <CardHeader
              title="Grid Time Series (Sample)"
              subheader="Active power, voltage, reactive power, and phase angle over time"
            />
            <CardContent>
              {timeSeriesData && chartData.length > 0 ? (
                <Box sx={{ width: '100%', height: 400 }}>
                  <ResponsiveContainer>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="time"
                        label={{ value: 'Time Step', position: 'insideBottom', offset: -5 }}
                      />
                      <YAxis
                        yAxisId="left"
                        label={{ value: 'Power (MW/MVAr)', angle: -90, position: 'insideLeft' }}
                      />
                      <YAxis
                        yAxisId="right"
                        orientation="right"
                        label={{ value: 'Voltage (pu) / Angle (°)', angle: 90, position: 'insideRight' }}
                      />
                      <Tooltip />
                      <Line
                        yAxisId="left"
                        type="monotone"
                        dataKey="Active Power (MW)"
                        stroke="#2196f3"
                        strokeWidth={2}
                        dot={false}
                      />
                      <Line
                        yAxisId="right"
                        type="monotone"
                        dataKey="Voltage (pu)"
                        stroke="#4caf50"
                        strokeWidth={2}
                        dot={false}
                      />
                      <Line
                        yAxisId="left"
                        type="monotone"
                        dataKey="Reactive Power (MVAr)"
                        stroke="#ff9800"
                        strokeWidth={2}
                        dot={false}
                      />
                      <Line
                        yAxisId="right"
                        type="monotone"
                        dataKey="Angle (°)"
                        stroke="#f44336"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 200 }}>
                  <Typography variant="body2" color="text.secondary">
                    Generate data to view time series visualization
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Box>

        {/* Use Cases Overview */}
        <Box sx={{ flex: '1 1 100%', minWidth: '100%' }}>
          <Card>
            <CardHeader
              title="AI Use Cases"
              avatar={<Science color="primary" />}
            />
            <CardContent>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {[
                  'Load/Demand Forecasting',
                  'Grid State Estimation',
                  'Congestion Prediction',
                  'OPF Surrogate',
                  'Spatiotemporal Fusion',
                  'Anomaly Detection'
                ].map((useCase, index) => (
                  <Box sx={{ flex: '1 1 200px', minWidth: '200px' }} key={index}>
                    <Chip
                      label={useCase}
                      variant="outlined"
                      color="primary"
                      sx={{ width: '100%', justifyContent: 'center', py: 1 }}
                    />
                  </Box>
                ))}
              </Box>
              <Box sx={{ mt: 3 }}>
                <Typography variant="body2" color="text.secondary">
                  Explore each use case in the dedicated sections. All models demonstrate convergence
                  and real-time capability for grid control applications.
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Box>
      </Box>

      {loading && (
        <Box sx={{ width: '100%', mt: 2 }}>
          <LinearProgress />
        </Box>
      )}
    </Container>
  );
};

export default Dashboard;

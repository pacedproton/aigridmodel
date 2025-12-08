import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  CardHeader,
  Button,
  TextField,
  Grid,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip
} from '@mui/material';
import { DataUsage, PlayArrow, Refresh } from '@mui/icons-material';
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

const DataManagement: React.FC = () => {
  const [dataStats, setDataStats] = useState<DataStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [nSteps, setNSteps] = useState(1000);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    fetchDataStats();
  }, []);

  const fetchDataStats = async () => {
    try {
      setLoading(true);
      const response = await axios.get(config.api.endpoints.dataStats);
      if (response.data.success) {
        setDataStats(response.data.stats);
      }
    } catch (err) {
      setError('Failed to fetch data statistics');
    } finally {
      setLoading(false);
    }
  };

  const generateData = async () => {
    try {
      setGenerating(true);
      setError(null);
      setSuccess(null);

      const response = await axios.post(config.api.endpoints.generateData, {
        n_steps: nSteps
      });

      if (response.data.success) {
        setSuccess('Grid data generated successfully!');
        await fetchDataStats(); // Refresh stats
      } else {
        setError(response.data.error || 'Failed to generate data');
      }
    } catch (err) {
      setError('Failed to generate data. Make sure the backend is running.');
    } finally {
      setGenerating(false);
    }
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Data Management
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Generate and manage synthetic grid data for AI model training and evaluation
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 2 }}>
          {success}
        </Alert>
      )}

      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
        {/* Data Generation */}
        <Box sx={{ flex: '1 1 500px', minWidth: '300px' }}>
          <Card>
            <CardHeader
              title="Generate Grid Data"
              avatar={<DataUsage color="primary" />}
            />
            <CardContent>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Create synthetic time series data representing a power grid with realistic load patterns,
                voltage fluctuations, and power flow characteristics.
              </Typography>

              <TextField
                label="Number of Time Steps"
                type="number"
                value={nSteps}
                onChange={(e) => setNSteps(parseInt(e.target.value) || 1000)}
                fullWidth
                sx={{ mb: 2 }}
                helperText="Typical: 1000-10000 steps (1-10 years of 30-min data)"
              />

              <Button
                variant="contained"
                fullWidth
                onClick={generateData}
                disabled={generating}
                startIcon={generating ? <CircularProgress size={20} /> : <PlayArrow />}
              >
                {generating ? 'Generating...' : 'Generate Data'}
              </Button>
            </CardContent>
          </Card>
        </Box>

        {/* Data Statistics */}
        <Box sx={{ flex: '1 1 500px', minWidth: '300px' }}>
          <Card>
            <CardHeader
              title="Data Overview"
              avatar={<Refresh color="secondary" />}
            />
            <CardContent>
              {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                  <CircularProgress />
                </Box>
              ) : dataStats ? (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Grid Configuration
                  </Typography>
                  <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell><strong>Property</strong></TableCell>
                          <TableCell><strong>Value</strong></TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        <TableRow>
                          <TableCell>Time Steps</TableCell>
                          <TableCell>{dataStats.time_steps}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Grid Nodes</TableCell>
                          <TableCell>{dataStats.nodes}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Features Shape</TableCell>
                          <TableCell>{dataStats.node_features_shape.join(' × ')}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Targets Shape</TableCell>
                          <TableCell>{dataStats.node_targets_shape.join(' × ')}</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>

                  <Typography variant="h6" gutterBottom>
                    Data Ranges
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    <Chip
                      label={`P: ${dataStats.p_load_range[0].toFixed(1)}-${dataStats.p_load_range[1].toFixed(1)} MW`}
                      variant="outlined"
                      color="primary"
                    />
                    <Chip
                      label={`V: ${dataStats.voltage_range[0].toFixed(3)}-${dataStats.voltage_range[1].toFixed(3)} pu`}
                      variant="outlined"
                      color="secondary"
                    />
                  </Box>
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No data available. Generate grid data to see statistics.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Box>

        {/* Data Description */}
        <Box sx={{ flex: '1 1 100%', minWidth: '100%' }}>
          <Card>
            <CardHeader title="Data Structure" />
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Features (Input)
              </Typography>
              <Typography variant="body2" paragraph>
                Each time step contains node-level features representing the current grid state:
              </Typography>
              <Box sx={{ ml: 2, mb: 2 }}>
                <Typography variant="body2" component="div">
                  • <strong>P_load</strong>: Active power demand at each node (MW)
                </Typography>
                <Typography variant="body2" component="div">
                  • <strong>Q_load</strong>: Reactive power demand at each node (MVAr)
                </Typography>
                <Typography variant="body2" component="div">
                  • <strong>P_gen</strong>: Active power generation (MW)
                </Typography>
                <Typography variant="body2" component="div">
                  • <strong>Time_sin/cos</strong>: Temporal encoding for hour-of-day patterns
                </Typography>
              </Box>

              <Typography variant="h6" gutterBottom>
                Targets (Output)
              </Typography>
              <Typography variant="body2" paragraph>
                Corresponding physical quantities that AI models learn to predict:
              </Typography>
              <Box sx={{ ml: 2 }}>
                <Typography variant="body2" component="div">
                  • <strong>Voltage magnitude</strong>: Per-unit voltage at each node
                </Typography>
                <Typography variant="body2" component="div">
                  • <strong>Voltage angle</strong>: Phase angle in degrees
                </Typography>
                <Typography variant="body2" component="div">
                  • <strong>Power flows</strong>: Active/reactive power on transmission lines
                </Typography>
                <Typography variant="body2" component="div">
                  • <strong>Congestion flags</strong>: Binary indicators of line overloads
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Box>
      </Box>
    </Container>
  );
};

export default DataManagement;

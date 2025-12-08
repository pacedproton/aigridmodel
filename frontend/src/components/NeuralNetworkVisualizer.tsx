import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  IconButton,
  Tooltip,
  Paper,
  Fade,
  Grow,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  Timeline,
  ElectricBolt as Flash,
  Warning,
  TrendingUp,
  Science,
  Security,
  ExpandMore,
  Info,
  PlayArrow,
  Pause,
  SkipNext,
  Layers,
  Transform,
  Output
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';

interface LayerConfig {
  id: string;
  type: 'input' | 'lstm' | 'attention' | 'gnn' | 'dense' | 'output' | 'transformer' | 'graph';
  label: string;
  size: number[];
  description: string;
  color: string;
  icon: React.ReactNode;
}

interface Connection {
  from: string;
  to: string;
  label?: string;
  animated?: boolean;
}

interface ProcessingStep {
  step: number;
  title: string;
  description: string;
  inputShape: number[];
  outputShape: number[];
  operation: string;
  duration: number;
  highlight: boolean;
}

interface UseCaseArchitecture {
  id: string;
  title: string;
  icon: React.ReactNode;
  description: string;
  layers: LayerConfig[];
  connections: Connection[];
  processingSteps: ProcessingStep[];
  keyInsights: string[];
}

const useCaseArchitectures: UseCaseArchitecture[] = [
  {
    id: 'forecasting',
    title: 'Load/Demand Forecasting',
    icon: <Timeline color="primary" />,
    description: 'Predict future electricity demand using temporal patterns',
    layers: [
      {
        id: 'input',
        type: 'input',
        label: 'Time Series Data',
        size: [100, 14, 3],
        description: 'Historical load, weather, and time features',
        color: '#2196F3',
        icon: <Layers />
      },
      {
        id: 'temporal_encoder',
        type: 'lstm',
        label: 'Temporal Encoder',
        size: [64],
        description: 'LSTM layers capture temporal dependencies',
        color: '#FF9800',
        icon: <Timeline />
      },
      {
        id: 'attention',
        type: 'attention',
        label: 'Attention Layer',
        size: [4, 64],
        description: 'Multi-head attention for important time steps',
        color: '#FFC107',
        icon: <Flash />
      },
      {
        id: 'dense',
        type: 'dense',
        label: 'Feature Extractor',
        size: [32],
        description: 'Dense layers for final feature extraction',
        color: '#4CAF50',
        icon: <Transform />
      },
      {
        id: 'output',
        type: 'output',
        label: '24h Forecast',
        size: [24],
        description: '24-hour ahead load predictions',
        color: '#9C27B0',
        icon: <Output />
      }
    ],
    connections: [
      { from: 'input', to: 'temporal_encoder', animated: true },
      { from: 'temporal_encoder', to: 'attention', animated: true },
      { from: 'attention', to: 'dense', animated: true },
      { from: 'dense', to: 'output', animated: true }
    ],
    processingSteps: [
      {
        step: 1,
        title: 'Data Input',
        description: 'Raw time series data enters the network',
        inputShape: [100, 14, 3],
        outputShape: [100, 14, 3],
        operation: 'Input Processing',
        duration: 500,
        highlight: true
      },
      {
        step: 2,
        title: 'Temporal Encoding',
        description: 'LSTM layers learn temporal patterns',
        inputShape: [100, 14, 3],
        outputShape: [100, 64],
        operation: 'LSTM Forward Pass',
        duration: 1000,
        highlight: true
      },
      {
        step: 3,
        title: 'Attention Mechanism',
        description: 'Model focuses on relevant time periods',
        inputShape: [100, 64],
        outputShape: [64],
        operation: 'Attention Computation',
        duration: 800,
        highlight: true
      },
      {
        step: 4,
        title: 'Feature Extraction',
        description: 'Dense layers create final representations',
        inputShape: [64],
        outputShape: [32],
        operation: 'Dense Transformation',
        duration: 600,
        highlight: true
      },
      {
        step: 5,
        title: 'Forecast Generation',
        description: 'Generate 24-hour ahead predictions',
        inputShape: [32],
        outputShape: [24],
        operation: 'Output Generation',
        duration: 400,
        highlight: false
      }
    ],
    keyInsights: [
      'LSTM captures weekly and daily patterns',
      'Attention focuses on recent high-impact periods',
      'Multi-step ahead prediction with uncertainty'
    ]
  },
  {
    id: 'state_estimation',
    title: 'Grid State Estimation',
    icon: <Flash color="secondary" />,
    description: 'Reconstruct full grid state from partial measurements',
    layers: [
      {
        id: 'measurements',
        type: 'input',
        label: 'Sensor Measurements',
        size: [20, 2],
        description: 'Partial voltage and current measurements',
        color: '#2196F3',
        icon: <Layers />
      },
      {
        id: 'topology',
        type: 'input',
        label: 'Grid Topology',
        size: [14, 14],
        description: 'Electrical network connectivity matrix',
        color: '#2196F3',
        icon: <Science />
      },
      {
        id: 'gnn_encoder',
        type: 'gnn',
        label: 'Graph Neural Network',
        size: [64],
        description: 'Message passing between connected buses',
        color: '#FF5722',
        icon: <Transform />
      },
      {
        id: 'decoder',
        type: 'dense',
        label: 'State Decoder',
        size: [28],
        description: 'Reconstruct complete grid state',
        color: '#4CAF50',
        icon: <Output />
      }
    ],
    connections: [
      { from: 'measurements', to: 'gnn_encoder', label: 'measurements' },
      { from: 'topology', to: 'gnn_encoder', label: 'topology' },
      { from: 'gnn_encoder', to: 'decoder', animated: true }
    ],
    processingSteps: [
      {
        step: 1,
        title: 'Sensor Data',
        description: 'Partial measurements from grid sensors',
        inputShape: [20, 2],
        outputShape: [20, 2],
        operation: 'Data Acquisition',
        duration: 400,
        highlight: true
      },
      {
        step: 2,
        title: 'Graph Construction',
        description: 'Build electrical network graph',
        inputShape: [14, 14],
        outputShape: [14, 14],
        operation: 'Topology Processing',
        duration: 600,
        highlight: true
      },
      {
        step: 3,
        title: 'Message Passing',
        description: 'GNN propagates information through network',
        inputShape: [14, 64],
        outputShape: [14, 64],
        operation: 'Neighborhood Aggregation',
        duration: 1000,
        highlight: true
      },
      {
        step: 4,
        title: 'State Reconstruction',
        description: 'Estimate complete voltage and current state',
        inputShape: [64],
        outputShape: [28],
        operation: 'State Estimation',
        duration: 800,
        highlight: false
      }
    ],
    keyInsights: [
      'Graph structure captures electrical connectivity',
      'Message passing shares information between buses',
      'Handles missing sensor data gracefully'
    ]
  },
  {
    id: 'congestion',
    title: 'Congestion Prediction',
    icon: <Warning color="warning" />,
    description: 'Predict transmission line overloads in advance',
    layers: [
      {
        id: 'power_flow',
        type: 'input',
        label: 'Power Flow Data',
        size: [20, 4],
        description: 'Current line flows and capacities',
        color: '#2196F3',
        icon: <Layers />
      },
      {
        id: 'temporal_features',
        type: 'input',
        label: 'Temporal Features',
        size: [24, 6],
        description: 'Recent power flow history',
        color: '#2196F3',
        icon: <Timeline />
      },
      {
        id: 'risk_assessment',
        type: 'dense',
        label: 'Risk Assessment',
        size: [32],
        description: 'Evaluate congestion risk factors',
        color: '#FF9800',
        icon: <Warning />
      },
      {
        id: 'prediction_head',
        type: 'dense',
        label: 'Prediction Head',
        size: [20],
        description: 'Predict congestion probabilities',
        color: '#F44336',
        icon: <Output />
      }
    ],
    connections: [
      { from: 'power_flow', to: 'risk_assessment', animated: true },
      { from: 'temporal_features', to: 'risk_assessment', animated: true },
      { from: 'risk_assessment', to: 'prediction_head', animated: true }
    ],
    processingSteps: [
      {
        step: 1,
        title: 'Current State',
        description: 'Real-time power flow measurements',
        inputShape: [20, 4],
        outputShape: [20, 4],
        operation: 'State Monitoring',
        duration: 400,
        highlight: true
      },
      {
        step: 2,
        title: 'Historical Context',
        description: 'Recent power flow patterns',
        inputShape: [24, 6],
        outputShape: [24, 6],
        operation: 'Temporal Analysis',
        duration: 600,
        highlight: true
      },
      {
        step: 3,
        title: 'Risk Evaluation',
        description: 'Assess line loading and stress factors',
        inputShape: [44],
        outputShape: [32],
        operation: 'Risk Computation',
        duration: 800,
        highlight: true
      },
      {
        step: 4,
        title: 'Congestion Forecast',
        description: 'Predict which lines may congest',
        inputShape: [32],
        outputShape: [20],
        operation: 'Probability Estimation',
        duration: 600,
        highlight: false
      }
    ],
    keyInsights: [
      'Combines real-time and historical data',
      'Focuses on line capacity utilization',
      'Enables preventive control actions'
    ]
  },
  {
    id: 'opf',
    title: 'OPF Surrogate',
    icon: <TrendingUp color="success" />,
    description: 'Fast approximation of Optimal Power Flow solutions',
    layers: [
      {
        id: 'system_state',
        type: 'input',
        label: 'System State',
        size: [14, 5],
        description: 'Loads, generations, and constraints',
        color: '#2196F3',
        icon: <Layers />
      },
      {
        id: 'physics_layer',
        type: 'dense',
        label: 'Physics Layer',
        size: [64],
        description: 'Enforce physical constraints',
        color: '#9C27B0',
        icon: <Science />
      },
      {
        id: 'optimization_layer',
        type: 'dense',
        label: 'Optimization Layer',
        size: [32],
        description: 'Minimize cost subject to constraints',
        color: '#FF5722',
        icon: <TrendingUp />
      },
      {
        id: 'feasibility_check',
        type: 'dense',
        label: 'Feasibility Check',
        size: [1],
        description: 'Verify solution feasibility',
        color: '#4CAF50',
        icon: <Output />
      },
      {
        id: 'dispatch_output',
        type: 'output',
        label: 'Optimal Dispatch',
        size: [14],
        description: 'Generator setpoints and costs',
        color: '#2196F3',
        icon: <Output />
      }
    ],
    connections: [
      { from: 'system_state', to: 'physics_layer', animated: true },
      { from: 'physics_layer', to: 'optimization_layer', animated: true },
      { from: 'optimization_layer', to: 'feasibility_check', animated: true },
      { from: 'optimization_layer', to: 'dispatch_output', animated: true }
    ],
    processingSteps: [
      {
        step: 1,
        title: 'Problem Formulation',
        description: 'Define loads, costs, and constraints',
        inputShape: [14, 5],
        outputShape: [14, 5],
        operation: 'Input Processing',
        duration: 400,
        highlight: true
      },
      {
        step: 2,
        title: 'Physics Integration',
        description: 'Incorporate electrical laws and limits',
        inputShape: [70],
        outputShape: [64],
        operation: 'Constraint Encoding',
        duration: 800,
        highlight: true
      },
      {
        step: 3,
        title: 'Optimization',
        description: 'Find cost-minimizing solution',
        inputShape: [64],
        outputShape: [32],
        operation: 'Cost Minimization',
        duration: 1000,
        highlight: true
      },
      {
        step: 4,
        title: 'Validation',
        description: 'Check solution feasibility',
        inputShape: [32],
        outputShape: [1],
        operation: 'Feasibility Test',
        duration: 400,
        highlight: true
      },
      {
        step: 5,
        title: 'Dispatch Generation',
        description: 'Output optimal generator settings',
        inputShape: [32],
        outputShape: [14],
        operation: 'Solution Output',
        duration: 500,
        highlight: false
      }
    ],
    keyInsights: [
      'Learns physics-based optimization patterns',
      '1000x faster than traditional OPF solvers',
      'Maintains feasibility guarantees'
    ]
  },
  {
    id: 'spatiotemporal',
    title: 'Spatiotemporal Fusion',
    icon: <Science color="info" />,
    description: 'Combined spatial and temporal grid analysis',
    layers: [
      {
        id: 'spatial_data',
        type: 'input',
        label: 'Spatial Features',
        size: [14, 8],
        description: 'Node features and connectivity',
        color: '#2196F3',
        icon: <Layers />
      },
      {
        id: 'temporal_data',
        type: 'input',
        label: 'Temporal Sequence',
        size: [50, 14, 8],
        description: 'Time series of spatial features',
        color: '#2196F3',
        icon: <Timeline />
      },
      {
        id: 'graph_encoder',
        type: 'gnn',
        label: 'Graph Encoder',
        size: [64],
        description: 'Spatial feature extraction',
        color: '#FF5722',
        icon: <Transform />
      },
      {
        id: 'temporal_encoder',
        type: 'transformer',
        label: 'Temporal Encoder',
        size: [64],
        description: 'Temporal pattern learning',
        color: '#FF9800',
        icon: <Flash />
      },
      {
        id: 'fusion_layer',
        type: 'dense',
        label: 'Fusion Layer',
        size: [32],
        description: 'Combine spatial and temporal features',
        color: '#9C27B0',
        icon: <Science />
      },
      {
        id: 'prediction_output',
        type: 'output',
        label: 'Spatiotemporal Prediction',
        size: [14, 3],
        description: 'Multi-node, multi-step predictions',
        color: '#4CAF50',
        icon: <Output />
      }
    ],
    connections: [
      { from: 'spatial_data', to: 'graph_encoder', animated: true },
      { from: 'temporal_data', to: 'temporal_encoder', animated: true },
      { from: 'graph_encoder', to: 'fusion_layer', label: 'spatial features' },
      { from: 'temporal_encoder', to: 'fusion_layer', label: 'temporal features' },
      { from: 'fusion_layer', to: 'prediction_output', animated: true }
    ],
    processingSteps: [
      {
        step: 1,
        title: 'Spatial Input',
        description: 'Grid topology and node features',
        inputShape: [14, 8],
        outputShape: [14, 8],
        operation: 'Spatial Data',
        duration: 400,
        highlight: true
      },
      {
        step: 2,
        title: 'Temporal Input',
        description: 'Time series of grid measurements',
        inputShape: [50, 14, 8],
        outputShape: [50, 14, 8],
        operation: 'Temporal Data',
        duration: 500,
        highlight: true
      },
      {
        step: 3,
        title: 'Graph Processing',
        description: 'GNN learns spatial relationships',
        inputShape: [14, 8],
        outputShape: [64],
        operation: 'Spatial Encoding',
        duration: 800,
        highlight: true
      },
      {
        step: 4,
        title: 'Sequence Modeling',
        description: 'Transformer processes temporal patterns',
        inputShape: [50, 64],
        outputShape: [64],
        operation: 'Temporal Encoding',
        duration: 900,
        highlight: true
      },
      {
        step: 5,
        title: 'Feature Fusion',
        description: 'Combine spatial and temporal features',
        inputShape: [128],
        outputShape: [32],
        operation: 'Multimodal Fusion',
        duration: 600,
        highlight: true
      },
      {
        step: 6,
        title: 'Joint Prediction',
        description: 'Predict spatiotemporal grid behavior',
        inputShape: [32],
        outputShape: [42],
        operation: 'Prediction Output',
        duration: 500,
        highlight: false
      }
    ],
    keyInsights: [
      'Captures both spatial connectivity and temporal dynamics',
      'Handles cascading failures and propagation patterns',
      'Enables whole-grid stability analysis'
    ]
  },
  {
    id: 'anomaly',
    title: 'Anomaly Detection',
    icon: <Security color="error" />,
    description: 'Identify unusual grid behavior and cyber threats',
    layers: [
      {
        id: 'normal_patterns',
        type: 'input',
        label: 'Normal Patterns',
        size: [1000, 14, 6],
        description: 'Training data of normal operation',
        color: '#2196F3',
        icon: <Layers />
      },
      {
        id: 'real_time_data',
        type: 'input',
        label: 'Real-time Data',
        size: [1, 14, 6],
        description: 'Current grid measurements',
        color: '#FF5722',
        icon: <Timeline />
      },
      {
        id: 'autoencoder',
        type: 'dense',
        label: 'Autoencoder',
        size: [32],
        description: 'Learn normal behavior patterns',
        color: '#9C27B0',
        icon: <Transform />
      },
      {
        id: 'reconstruction',
        type: 'dense',
        label: 'Reconstruction',
        size: [84],
        description: 'Attempt to reconstruct input',
        color: '#FF9800',
        icon: <Output />
      },
      {
        id: 'anomaly_score',
        type: 'dense',
        label: 'Anomaly Score',
        size: [1],
        description: 'Measure reconstruction error',
        color: '#F44336',
        icon: <Security />
      }
    ],
    connections: [
      { from: 'normal_patterns', to: 'autoencoder', label: 'training' },
      { from: 'real_time_data', to: 'autoencoder', label: 'inference', animated: true },
      { from: 'autoencoder', to: 'reconstruction', animated: true },
      { from: 'reconstruction', to: 'anomaly_score', animated: true }
    ],
    processingSteps: [
      {
        step: 1,
        title: 'Training Data',
        description: 'Learn patterns of normal grid operation',
        inputShape: [1000, 14, 6],
        outputShape: [1000, 14, 6],
        operation: 'Pattern Learning',
        duration: 600,
        highlight: true
      },
      {
        step: 2,
        title: 'Real-time Input',
        description: 'Current grid measurements to analyze',
        inputShape: [1, 14, 6],
        outputShape: [1, 14, 6],
        operation: 'Live Monitoring',
        duration: 300,
        highlight: true
      },
      {
        step: 3,
        title: 'Feature Encoding',
        description: 'Compress input into latent representation',
        inputShape: [84],
        outputShape: [32],
        operation: 'Dimensionality Reduction',
        duration: 700,
        highlight: true
      },
      {
        step: 4,
        title: 'Pattern Reconstruction',
        description: 'Attempt to reconstruct normal pattern',
        inputShape: [32],
        outputShape: [84],
        operation: 'Pattern Reconstruction',
        duration: 800,
        highlight: true
      },
      {
        step: 5,
        title: 'Anomaly Detection',
        description: 'Compare reconstruction with actual input',
        inputShape: [84],
        outputShape: [1],
        operation: 'Error Analysis',
        duration: 500,
        highlight: false
      }
    ],
    keyInsights: [
      'Unsupervised learning of normal behavior',
      'Detects both physical and cyber anomalies',
      'Handles concept drift and seasonal changes'
    ]
  }
];

interface NeuralNetworkVisualizerProps {
  useCaseId: string;
  showAnimation?: boolean;
}

export const NeuralNetworkVisualizer: React.FC<NeuralNetworkVisualizerProps> = ({
  useCaseId,
  showAnimation = true
}) => {
  const [selectedArchitecture, setSelectedArchitecture] = useState<UseCaseArchitecture | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const architecture = useCaseArchitectures.find(arch => arch.id === useCaseId);
    setSelectedArchitecture(architecture || null);
    setCurrentStep(0);
    setProgress(0);
  }, [useCaseId]);

  useEffect(() => {
    if (!showAnimation || !selectedArchitecture || !isPlaying) return;

    const step = selectedArchitecture.processingSteps[currentStep];
    if (!step) {
      setIsPlaying(false);
      return;
    }

    const timer = setTimeout(() => {
      setProgress(0);
      const nextStep = (currentStep + 1) % selectedArchitecture.processingSteps.length;
      setCurrentStep(nextStep);
    }, step.duration);

    return () => clearTimeout(timer);
  }, [currentStep, isPlaying, selectedArchitecture, showAnimation]);

  useEffect(() => {
    if (!isPlaying || !selectedArchitecture) return;

    const step = selectedArchitecture.processingSteps[currentStep];
    const duration = step?.duration || 1000;

    const progressTimer = setInterval(() => {
      setProgress(prev => {
        const newProgress = prev + (100 / (duration / 100));
        return newProgress >= 100 ? 100 : newProgress;
      });
    }, 100);

    return () => clearInterval(progressTimer);
  }, [currentStep, isPlaying, selectedArchitecture]);

  if (!selectedArchitecture) {
    return <Typography>Loading architecture...</Typography>;
  }

  const currentProcessingStep = selectedArchitecture.processingSteps[currentStep];

  return (
    <Box sx={{ width: '100%', p: 2 }}>
      <Card elevation={3}>
        <CardContent>
          <Box display="flex" alignItems="center" mb={3}>
            {selectedArchitecture.icon}
            <Typography variant="h5" sx={{ ml: 2, fontWeight: 'bold' }}>
              {selectedArchitecture.title}
            </Typography>
          </Box>

          <Typography variant="body2" color="text.secondary" mb={3}>
            {selectedArchitecture.description}
          </Typography>

          {/* Architecture Visualization */}
          <Box sx={{ mb: 4 }}>
            <Typography variant="h6" gutterBottom>
              Neural Network Architecture
            </Typography>

            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: 2,
                p: 3,
                bgcolor: 'grey.50',
                borderRadius: 2,
                minHeight: 120
              }}
            >
              {selectedArchitecture.layers.map((layer, index) => (
                <Grow in key={layer.id} timeout={500 + index * 200}>
                  <Paper
                    elevation={2}
                    sx={{
                      p: 2,
                      minWidth: 140,
                      textAlign: 'center',
                      bgcolor: layer.color + '20',
                      border: `2px solid ${layer.color}`,
                      position: 'relative'
                    }}
                  >
                    <Box color={layer.color} mb={1}>
                      {layer.icon}
                    </Box>
                    <Typography variant="subtitle2" fontWeight="bold">
                      {layer.label}
                    </Typography>
                    <Typography variant="caption" display="block">
                      {layer.size.join('×')}
                    </Typography>
                    <Tooltip title={layer.description}>
                      <IconButton size="small" sx={{ position: 'absolute', top: 4, right: 4 }}>
                        <Info fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Paper>
                </Grow>
              ))}
            </Box>
          </Box>

          {/* Processing Animation */}
          {showAnimation && (
            <Box sx={{ mb: 4 }}>
              <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                <Typography variant="h6">
                  Data Processing Pipeline
                </Typography>
                <Box>
                  <IconButton onClick={() => setIsPlaying(!isPlaying)}>
                    {isPlaying ? <Pause /> : <PlayArrow />}
                  </IconButton>
                  <IconButton onClick={() => {
                    setCurrentStep((currentStep + 1) % selectedArchitecture.processingSteps.length);
                    setProgress(0);
                  }}>
                    <SkipNext />
                  </IconButton>
                </Box>
              </Box>

              <LinearProgress
                variant="determinate"
                value={progress}
                sx={{ mb: 2, height: 8, borderRadius: 4 }}
              />

              {currentProcessingStep && (
                <Fade in timeout={500}>
                  <Card
                    sx={{
                      bgcolor: currentProcessingStep.highlight ? 'primary.light' : 'grey.100',
                      color: currentProcessingStep.highlight ? 'primary.contrastText' : 'text.primary'
                    }}
                  >
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Step {currentProcessingStep.step}: {currentProcessingStep.title}
                      </Typography>
                      <Typography variant="body2" mb={2}>
                        {currentProcessingStep.description}
                      </Typography>

                      <Box display="flex" gap={4} flexWrap="wrap">
                        <Box>
                          <Typography variant="subtitle2">Input Shape</Typography>
                          <Chip
                            label={`[${currentProcessingStep.inputShape.join('×')}]`}
                            size="small"
                            color="primary"
                          />
                        </Box>
                        <Box>
                          <Typography variant="subtitle2">Output Shape</Typography>
                          <Chip
                            label={`[${currentProcessingStep.outputShape.join('×')}]`}
                            size="small"
                            color="secondary"
                          />
                        </Box>
                        <Box>
                          <Typography variant="subtitle2">Operation</Typography>
                          <Chip
                            label={currentProcessingStep.operation}
                            size="small"
                            variant="outlined"
                          />
                        </Box>
                      </Box>
                    </CardContent>
                  </Card>
                </Fade>
              )}
            </Box>
          )}

          {/* Key Insights */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              Key Neural Network Insights
            </Typography>
            {selectedArchitecture.keyInsights.map((insight, index) => (
              <Box key={index} display="flex" alignItems="center" mb={1}>
                <Typography variant="body2" sx={{ ml: 1 }}>
                  • {insight}
                </Typography>
              </Box>
            ))}
          </Box>

          {/* Technical Details */}
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="subtitle1">Technical Architecture Details</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" paragraph>
                <strong>Layers:</strong> {selectedArchitecture.layers.length} neural network layers
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>Processing Steps:</strong> {selectedArchitecture.processingSteps.length} computational stages
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>Architecture Type:</strong> {selectedArchitecture.layers.some(l => l.type === 'gnn') ? 'Graph Neural Network' :
                  selectedArchitecture.layers.some(l => l.type === 'transformer') ? 'Transformer-based' :
                  'Feed-forward Neural Network'}
              </Typography>
            </AccordionDetails>
          </Accordion>
        </CardContent>
      </Card>
    </Box>
  );
};

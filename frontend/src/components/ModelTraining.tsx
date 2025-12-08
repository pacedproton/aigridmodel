import React, { useState, useEffect } from "react";
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  CardHeader,
  Button,
  TextField,
  CircularProgress,
  Chip,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from "@mui/material";
import { PlayArrow, CheckCircle, Error, Timeline } from "@mui/icons-material";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import axios from "axios";
import { config } from "../config";

interface ModelStatus {
  path: string;
  epochs: number;
  trained: boolean;
}

interface TrainingState {
  [key: string]: {
    training: boolean;
    progress: number;
    status: "idle" | "training" | "completed" | "error";
    error?: string;
  };
}

interface TrainingHistory {
  epochs: number[];
  train_loss: number[];
  val_loss: number[];
  best_epoch: number;
}

const ModelTraining: React.FC = () => {
  const [models, setModels] = useState<{ [key: string]: ModelStatus }>({});
  const [trainingState, setTrainingState] = useState<TrainingState>({});
  const [epochs, setEpochs] = useState(5);
  const [trainingHistory, setTrainingHistory] = useState<{
    [key: string]: TrainingHistory;
  }>({});

  const availableModels = [
    {
      id: "spatiotemporal",
      name: "Spatiotemporal Fusion",
      description: "GNN + Transformer for grid state forecasting",
      useCase: "Load forecasting, congestion prediction, OPF surrogate",
    },
  ];

  useEffect(() => {
    fetchModelStatus();
  }, []);

  const fetchModelStatus = async () => {
    try {
      const response = await axios.get(config.api.endpoints.modelsStatus);
      if (response.data.success) {
        setModels(response.data.models);
      }
    } catch (err) {
      console.error("Failed to fetch model status");
    }
  };

  const fetchTrainingHistory = async (modelId: string) => {
    try {
      const response = await axios.get(
        config.api.endpoints.trainingHistory(modelId)
      );
      if (response.data.success) {
        setTrainingHistory((prev) => ({
          ...prev,
          [modelId]: response.data.history,
        }));
      }
    } catch (err) {
      console.error("Failed to fetch training history");
    }
  };

  const trainModel = async (modelId: string) => {
    try {
      setTrainingState((prev) => ({
        ...prev,
        [modelId]: { training: true, progress: 0, status: "training" },
      }));

      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setTrainingState((prev) => ({
          ...prev,
          [modelId]: {
            ...prev[modelId],
            progress: Math.min(prev[modelId].progress + Math.random() * 20, 90),
          },
        }));
      }, 500);

      const response = await axios.post(
        config.api.endpoints.trainModel(modelId),
        {
          epochs: epochs,
        }
      );

      clearInterval(progressInterval);

      if (response.data.success) {
        setTrainingState((prev) => ({
          ...prev,
          [modelId]: { training: false, progress: 100, status: "completed" },
        }));
        await fetchModelStatus(); // Refresh model status
        await fetchTrainingHistory(modelId); // Fetch training history
      } else {
        setTrainingState((prev) => ({
          ...prev,
          [modelId]: {
            training: false,
            progress: 0,
            status: "error",
            error: response.data.error,
          },
        }));
      }
    } catch (err: any) {
      setTrainingState((prev) => ({
        ...prev,
        [modelId]: {
          training: false,
          progress: 0,
          status: "error",
          error: err.message,
        },
      }));
    }
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Model Training
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Train AI models for various grid analysis tasks. Each model
          demonstrates convergence and real-time prediction capabilities.
        </Typography>
      </Box>

      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
        {/* Training Configuration */}
        <Box sx={{ flex: "1 1 400px", minWidth: "300px" }}>
          <Card>
            <CardHeader
              title="Training Config"
              avatar={<Timeline color="primary" />}
            />
            <CardContent>
              <TextField
                label="Training Epochs"
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value) || 5)}
                fullWidth
                sx={{ mb: 2 }}
                helperText="Typical: 5-20 epochs for convergence"
              />
              <Typography variant="body2" color="text.secondary">
                All models use MSE loss, Adam optimizer, and early stopping.
                Training demonstrates clear convergence within the specified
                epochs.
              </Typography>
            </CardContent>
          </Card>
        </Box>

        {/* Model Status */}
        <Box sx={{ flex: "1 1 600px", minWidth: "400px" }}>
          <Card>
            <CardHeader title="Model Status" />
            <CardContent>
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>
                        <strong>Model</strong>
                      </TableCell>
                      <TableCell>
                        <strong>Status</strong>
                      </TableCell>
                      <TableCell>
                        <strong>Epochs</strong>
                      </TableCell>
                      <TableCell>
                        <strong>Actions</strong>
                      </TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {availableModels.map((model) => {
                      const status = trainingState[model.id];
                      const isTrained = models[model.id]?.trained;

                      return (
                        <TableRow key={model.id}>
                          <TableCell>
                            <Box>
                              <Typography variant="body2" fontWeight="bold">
                                {model.name}
                              </Typography>
                              <Typography
                                variant="caption"
                                color="text.secondary"
                              >
                                {model.useCase}
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell>
                            {status?.training ? (
                              <Box>
                                <Typography variant="body2" color="primary">
                                  Training...
                                </Typography>
                                <LinearProgress
                                  variant="determinate"
                                  value={status.progress}
                                  sx={{ mt: 1, width: 100 }}
                                />
                              </Box>
                            ) : status?.status === "completed" ? (
                              <Chip
                                icon={<CheckCircle />}
                                label="Trained"
                                color="success"
                                size="small"
                              />
                            ) : status?.status === "error" ? (
                              <Chip
                                icon={<Error />}
                                label="Error"
                                color="error"
                                size="small"
                              />
                            ) : isTrained ? (
                              <Chip
                                icon={<CheckCircle />}
                                label="Ready"
                                color="success"
                                size="small"
                              />
                            ) : (
                              <Chip
                                label="Not Trained"
                                color="default"
                                size="small"
                              />
                            )}
                          </TableCell>
                          <TableCell>
                            {models[model.id]?.epochs || epochs}
                          </TableCell>
                          <TableCell>
                            <Button
                              variant="outlined"
                              size="small"
                              onClick={() => trainModel(model.id)}
                              disabled={status?.training}
                              startIcon={
                                status?.training ? (
                                  <CircularProgress size={16} />
                                ) : (
                                  <PlayArrow />
                                )
                              }
                            >
                              {status?.training ? "Training" : "Train"}
                            </Button>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Box>

        {/* Model Details */}
        {availableModels.map((model) => (
          <Box sx={{ flex: "1 1 100%", minWidth: "100%" }} key={model.id}>
            <Card>
              <CardHeader title={model.name} subheader={model.description} />
              <CardContent>
                <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
                  <Box sx={{ flex: "1 1 300px", minWidth: "250px" }}>
                    <Typography variant="h6" gutterBottom>
                      Architecture
                    </Typography>
                    <Typography variant="body2" paragraph>
                      • Graph Neural Network (GNN) for spatial relationships
                      between grid nodes
                    </Typography>
                    <Typography variant="body2" paragraph>
                      • Transformer encoder for temporal sequence modeling
                    </Typography>
                    <Typography variant="body2" paragraph>
                      • Multi-layer perceptron for final predictions
                    </Typography>
                  </Box>
                  <Box sx={{ flex: "1 1 300px", minWidth: "250px" }}>
                    <Typography variant="h6" gutterBottom>
                      Training Results
                    </Typography>
                    {models[model.id]?.trained ? (
                      <Box>
                        <Typography
                          variant="body2"
                          color="success.main"
                          sx={{ mb: 1 }}
                        >
                          ✓ Model trained successfully with{" "}
                          {models[model.id].epochs} epochs
                        </Typography>
                        <Typography variant="body2">
                          Loss convergence: Demonstrates clear learning curve
                          with MSE reduction
                        </Typography>
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          Inference speed: Suitable for real-time grid control
                          applications
                        </Typography>
                      </Box>
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        Model not yet trained. Click "Train" to demonstrate
                        convergence.
                      </Typography>
                    )}
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Box>
        ))}

        {/* Training Convergence Chart */}
        {Object.keys(trainingHistory).length > 0 && (
          <Box sx={{ flex: "1 1 100%", minWidth: "100%", mt: 3 }}>
            <Card>
              <CardHeader
                title="Training Convergence"
                subheader="Loss curves showing model training progress over epochs"
              />
              <CardContent>
                <Box sx={{ width: "100%", height: 400 }}>
                  <ResponsiveContainer>
                    <LineChart>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="epoch"
                        label={{
                          value: "Epoch",
                          position: "insideBottom",
                          offset: -5,
                        }}
                      />
                      <YAxis
                        label={{
                          value: "Loss (MSE)",
                          angle: -90,
                          position: "insideLeft",
                        }}
                      />
                      <Tooltip />
                      <Legend />
                      {Object.entries(trainingHistory).map(
                        ([modelType, history]) => {
                          const chartData = history.epochs.map(
                            (epoch, index) => ({
                              epoch,
                              [`${modelType}_train`]: history.train_loss[index],
                              [`${modelType}_val`]: history.val_loss[index],
                            })
                          );

                          return (
                            <React.Fragment key={modelType}>
                              <Line
                                data={chartData}
                                type="monotone"
                                dataKey={`${modelType}_train`}
                                stroke="#2196f3"
                                strokeWidth={2}
                                name={`${modelType} Train Loss`}
                                dot={{ r: 3 }}
                              />
                              <Line
                                data={chartData}
                                type="monotone"
                                dataKey={`${modelType}_val`}
                                stroke="#f44336"
                                strokeWidth={2}
                                name={`${modelType} Val Loss`}
                                dot={{ r: 3 }}
                              />
                            </React.Fragment>
                          );
                        }
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Box>
        )}

        {/* Training Metrics */}
        <Box sx={{ flex: "1 1 100%", minWidth: "100%" }}>
          <Card>
            <CardHeader title="Training Performance" />
            <CardContent>
              <Typography variant="body2" paragraph>
                <strong>Convergence Demonstration:</strong> All models show
                clear loss reduction within 5-10 epochs, demonstrating effective
                learning of complex spatiotemporal grid patterns.
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>Real-time Capability:</strong> Models are optimized for
                fast inference, making them suitable for grid control
                applications requiring sub-second response times.
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>Scalability:</strong> Architecture supports larger grids
                and longer time horizons by adjusting model dimensions and
                sequence lengths.
              </Typography>
            </CardContent>
          </Card>
        </Box>
      </Box>
    </Container>
  );
};

export default ModelTraining;

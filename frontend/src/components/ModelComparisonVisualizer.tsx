import React, { useState, useEffect } from "react";
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Button,
  Chip,
  CircularProgress,
  Grid,
  Paper,
  LinearProgress,
  Alert,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Fade,
  Grow,
} from "@mui/material";
import {
  Timeline,
  ElectricBolt as Flash,
  Warning,
  TrendingUp,
  Science,
  Security,
  PlayArrow,
  Compare,
  Assessment,
  Speed,
  Psychology,
  Functions,
} from "@mui/icons-material";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  Area,
  AreaChart,
} from "recharts";
import { config } from "../config";
import { MathEquation } from "./MathEquation";

interface ModelResult {
  modelType: "neural" | "classical" | "advanced";
  useCase: string;
  results: any;
  metrics: {
    accuracy?: number;
    mse?: number;
    mae?: number;
    auc?: number;
    time?: number;
    uncertainty?: number;
  };
  predictions?: number[];
  trueValues?: number[];
  uncertainty?: {
    lower: number[];
    upper: number[];
  };
}

interface ComparisonData {
  neural: ModelResult | null;
  classical: ModelResult | null;
  advanced: ModelResult | null;
  testData: {
    time: number[];
    trueValues: number[];
  };
}

interface ModelComparisonVisualizerProps {
  useCase: string;
  onModelRun: (
    modelType: "neural" | "classical" | "advanced",
    useCase: string
  ) => Promise<ModelResult>;
}

export const ModelComparisonVisualizer: React.FC<
  ModelComparisonVisualizerProps
> = ({ useCase, onModelRun }) => {
  const [comparisonData, setComparisonData] = useState<ComparisonData>({
    neural: null,
    classical: null,
    advanced: null,
    testData: { time: [], trueValues: [] },
  });
  const [loading, setLoading] = useState<{ [key: string]: boolean }>({});
  const [activeTab, setActiveTab] = useState(0);
  const [runAllLoading, setRunAllLoading] = useState(false);

  // Model configurations
  const modelConfigs = {
    neural: {
      name: "Neural Network",
      icon: <Psychology sx={{ color: "#1976d2" }} />,
      color: "#1976d2",
      description: "Deep learning approach with automatic feature extraction",
    },
    classical: {
      name: "Classical Model",
      icon: <Functions sx={{ color: "#2e7d32" }} />,
      color: "#2e7d32",
      description:
        "Traditional mathematical modeling with theoretical guarantees",
    },
    advanced: {
      name: "Advanced Mathematics",
      icon: <Science sx={{ color: "#ed6c02" }} />,
      color: "#ed6c02",
      description:
        "State-of-the-art mathematical algorithms with uncertainty quantification",
    },
  };

  const useCaseIcons = {
    forecasting: <Timeline />,
    state_estimation: <Flash />,
    congestion: <Warning />,
    opf: <TrendingUp />,
    spatiotemporal: <Science />,
    anomaly: <Security />,
  };

  const runModel = async (modelType: "neural" | "classical" | "advanced") => {
    setLoading((prev) => ({ ...prev, [modelType]: true }));
    try {
      const result = await onModelRun(modelType, useCase);
      setComparisonData((prev) => ({
        ...prev,
        [modelType]: result,
        testData: result.trueValues
          ? {
              time: Array.from(
                { length: result.trueValues.length },
                (_, i) => i
              ),
              trueValues: result.trueValues,
            }
          : prev.testData,
      }));
    } catch (error) {
      console.error(`Error running ${modelType} model:`, error);
    } finally {
      setLoading((prev) => ({ ...prev, [modelType]: false }));
    }
  };

  const runAllModels = async () => {
    setRunAllLoading(true);
    try {
      await Promise.all([
        runModel("neural"),
        runModel("classical"),
        runModel("advanced"),
      ]);
    } finally {
      setRunAllLoading(false);
    }
  };

  const getAccuracyUncertaintyData = () => {
    const { neural, classical, advanced } = comparisonData;
    return [
      {
        metric: "Accuracy/MSE",
        neural: neural?.metrics.accuracy || neural?.metrics.mse || 0,
        classical: classical?.metrics.accuracy || classical?.metrics.mse || 0,
        advanced: advanced?.metrics.accuracy || advanced?.metrics.mse || 0,
      },
      {
        metric: "Uncertainty",
        neural: neural?.metrics.uncertainty || 0.1,
        classical: classical?.metrics.uncertainty || 0.05,
        advanced: advanced?.metrics.uncertainty || 0.02,
      },
    ];
  };

  const getSpeedData = () => {
    const { neural, classical, advanced } = comparisonData;
    return [
      {
        metric: "Execution Time",
        neural: neural?.metrics.time || 10,
        classical: classical?.metrics.time || 5,
        advanced: advanced?.metrics.time || 15,
      },
    ];
  };

  const getPredictionComparisonData = () => {
    const { neural, classical, advanced, testData } = comparisonData;
    const maxLength = Math.max(
      neural?.predictions?.length || 0,
      classical?.predictions?.length || 0,
      advanced?.predictions?.length || 0,
      testData.trueValues.length
    );

    return Array.from({ length: maxLength }, (_, i) => ({
      time: i,
      true: testData.trueValues[i] || null,
      neural: neural?.predictions?.[i] || null,
      neural_lower: neural?.uncertainty?.lower[i] || null,
      neural_upper: neural?.uncertainty?.upper[i] || null,
      classical: classical?.predictions?.[i] || null,
      classical_lower: classical?.uncertainty?.lower[i] || null,
      classical_upper: classical?.uncertainty?.upper[i] || null,
      advanced: advanced?.predictions?.[i] || null,
      advanced_lower: advanced?.uncertainty?.lower[i] || null,
      advanced_upper: advanced?.uncertainty?.upper[i] || null,
    }));
  };

  const getRadarData = () => {
    const { neural, classical, advanced } = comparisonData;
    return [
      {
        subject: "Accuracy",
        neural:
          neural?.metrics.accuracy || neural?.metrics.mse
            ? neural.metrics.mse
              ? 1 / neural.metrics.mse
              : neural.metrics.accuracy
            : 0,
        classical:
          classical?.metrics.accuracy || classical?.metrics.mse
            ? classical.metrics.mse
              ? 1 / classical.metrics.mse
              : classical.metrics.accuracy
            : 0,
        advanced:
          advanced?.metrics.accuracy || advanced?.metrics.mse
            ? advanced.metrics.mse
              ? 1 / advanced.metrics.mse
              : advanced.metrics.accuracy
            : 0,
        fullMark: 1,
      },
      {
        subject: "Speed",
        neural: neural?.metrics.time ? 1 / neural.metrics.time : 0,
        classical: classical?.metrics.time ? 1 / classical.metrics.time : 0,
        advanced: advanced?.metrics.time ? 1 / advanced.metrics.time : 0,
        fullMark: 1,
      },
      {
        subject: "Uncertainty",
        neural: neural?.metrics.uncertainty
          ? 1 / neural.metrics.uncertainty
          : 0,
        classical: classical?.metrics.uncertainty
          ? 1 / classical.metrics.uncertainty
          : 0,
        advanced: advanced?.metrics.uncertainty
          ? 1 / advanced.metrics.uncertainty
          : 0,
        fullMark: 1,
      },
    ];
  };

  const getRecommendation = () => {
    const { neural, classical, advanced } = comparisonData;

    if (!neural || !classical || !advanced) {
      return "Run all models to get recommendations";
    }

    const scores = {
      neural: 0,
      classical: 0,
      advanced: 0,
    };

    // Accuracy scoring
    const neural_acc =
      neural.metrics.accuracy ||
      (neural.metrics.mse ? 1 / neural.metrics.mse : 0);
    const classical_acc =
      classical.metrics.accuracy ||
      (classical.metrics.mse ? 1 / classical.metrics.mse : 0);
    const advanced_acc =
      advanced.metrics.accuracy ||
      (advanced.metrics.mse ? 1 / advanced.metrics.mse : 0);

    if (neural_acc >= Math.max(classical_acc, advanced_acc)) scores.neural += 2;
    if (classical_acc >= Math.max(neural_acc, advanced_acc))
      scores.classical += 2;
    if (advanced_acc >= Math.max(neural_acc, classical_acc))
      scores.advanced += 2;

    // Uncertainty scoring (lower is better)
    const neural_unc = neural.metrics.uncertainty || 0.1;
    const classical_unc = classical.metrics.uncertainty || 0.05;
    const advanced_unc = advanced.metrics.uncertainty || 0.02;

    if (advanced_unc <= Math.min(neural_unc, classical_unc))
      scores.advanced += 1;
    if (classical_unc <= Math.min(neural_unc, advanced_unc))
      scores.classical += 1;
    if (neural_unc <= Math.min(classical_unc, advanced_unc)) scores.neural += 1;

    // Speed scoring (lower time is better)
    const neural_time = neural.metrics.time || 10;
    const classical_time = classical.metrics.time || 5;
    const advanced_time = advanced.metrics.time || 15;

    if (classical_time <= Math.min(neural_time, advanced_time))
      scores.classical += 1;
    if (neural_time <= Math.min(classical_time, advanced_time))
      scores.neural += 1;
    if (advanced_time <= Math.min(neural_time, classical_time))
      scores.advanced += 1;

    const winner = Object.entries(scores).reduce((a, b) =>
      scores[a[0] as keyof typeof scores] > scores[b[0] as keyof typeof scores]
        ? a
        : b
    )[0];

    const recommendations = {
      neural:
        "Best for complex pattern recognition and when computational resources are available. Use when accuracy is paramount and interpretability is less critical.",
      classical:
        "Best for interpretable results, fast computation, and when mathematical guarantees are needed. Ideal for operational environments.",
      advanced:
        "Best for critical applications requiring uncertainty quantification and theoretical optimality. Use when both accuracy and confidence bounds are essential.",
    };

    return recommendations[winner as keyof typeof recommendations];
  };

  return (
    <Box sx={{ width: "100%", p: 3 }}>
      <Card elevation={3} sx={{ mb: 3 }}>
        <CardHeader
          avatar={useCaseIcons[useCase as keyof typeof useCaseIcons]}
          title={`${useCase
            .replace("_", " ")
            .toUpperCase()} - Three-Way Model Comparison`}
          subheader="Compare Neural Networks, Classical Models, and Advanced Mathematics"
        />
        <CardContent>
          <Box sx={{ display: "flex", gap: 2, mb: 3, flexWrap: "wrap" }}>
            <Button
              variant="contained"
              startIcon={<PlayArrow />}
              onClick={runAllModels}
              disabled={runAllLoading}
              size="large"
            >
              {runAllLoading ? (
                <>
                  <CircularProgress size={20} sx={{ mr: 1 }} />
                  Running All Models...
                </>
              ) : (
                "Run All Models"
              )}
            </Button>

            <Button
              variant="outlined"
              startIcon={<Compare />}
              onClick={() => setActiveTab(1)}
              disabled={
                !comparisonData.neural &&
                !comparisonData.classical &&
                !comparisonData.advanced
              }
            >
              View Comparisons
            </Button>
          </Box>

          {/* Model Status Cards */}
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 3, mb: 3 }}>
            {Object.entries(modelConfigs).map(([modelType, config]) => {
              const result = comparisonData[
                modelType as keyof ComparisonData
              ] as ModelResult | null;
              const isLoading = loading[modelType];

              return (
                <Box key={modelType} sx={{ flex: "1 1 300px", minWidth: 300 }}>
                  <Grow in timeout={500}>
                    <Card
                      sx={{
                        border: `2px solid ${
                          result ? config.color : "#e0e0e0"
                        }`,
                        opacity: isLoading ? 0.7 : 1,
                        transition: "all 0.3s ease",
                      }}
                    >
                      <CardHeader
                        avatar={config.icon}
                        title={config.name}
                        subheader={config.description}
                      />
                      <CardContent>
                        {isLoading ? (
                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              gap: 1,
                            }}
                          >
                            <CircularProgress size={20} />
                            <Typography>Running model...</Typography>
                          </Box>
                        ) : result ? (
                          <Box>
                            <Typography variant="body2" gutterBottom>
                              Model executed successfully
                            </Typography>
                            {result.metrics && (
                              <Box
                                sx={{
                                  display: "flex",
                                  flexWrap: "wrap",
                                  gap: 1,
                                }}
                              >
                                {result.metrics.mse && (
                                  <Chip
                                    label={`MSE: ${result.metrics.mse.toFixed(
                                      4
                                    )}`}
                                    size="small"
                                    color="primary"
                                  />
                                )}
                                {result.metrics.accuracy && (
                                  <Chip
                                    label={`Acc: ${(
                                      result.metrics.accuracy * 100
                                    ).toFixed(1)}%`}
                                    size="small"
                                    color="success"
                                  />
                                )}
                                {result.metrics.time && (
                                  <Chip
                                    label={`Time: ${result.metrics.time}ms`}
                                    size="small"
                                    color="info"
                                  />
                                )}
                              </Box>
                            )}
                          </Box>
                        ) : (
                          <Box>
                            <Typography
                              variant="body2"
                              color="text.secondary"
                              gutterBottom
                            >
                              Model not yet executed
                            </Typography>
                            <Button
                              size="small"
                              variant="outlined"
                              onClick={() =>
                                runModel(
                                  modelType as
                                    | "neural"
                                    | "classical"
                                    | "advanced"
                                )
                              }
                              disabled={isLoading}
                            >
                              {isLoading ? "Running..." : "Run Model"}
                            </Button>
                          </Box>
                        )}
                      </CardContent>
                    </Card>
                  </Grow>
                </Box>
              );
            })}
          </Box>
        </CardContent>
      </Card>

      {/* Results Tabs */}
      {(comparisonData.neural ||
        comparisonData.classical ||
        comparisonData.advanced) && (
        <Card elevation={2}>
          <Tabs
            value={activeTab}
            onChange={(_, newValue) => setActiveTab(newValue)}
          >
            <Tab label="📈 Performance Overview" />
            <Tab label="Detailed Comparisons" />
            <Tab label="Recommendations" />
          </Tabs>

          {/* Performance Overview Tab */}
          {activeTab === 0 && (
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Metrics Comparison
              </Typography>

              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
                <Box sx={{ flex: "1 1 45%", minWidth: 300 }}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Accuracy & Uncertainty
                    </Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={getAccuracyUncertaintyData()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="metric" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar
                          dataKey="neural"
                          fill="#1976d2"
                          name="Neural Network"
                        />
                        <Bar
                          dataKey="classical"
                          fill="#2e7d32"
                          name="Classical Model"
                        />
                        <Bar
                          dataKey="advanced"
                          fill="#ed6c02"
                          name="Advanced Math"
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </Paper>
                </Box>

                <Box sx={{ flex: "1 1 45%", minWidth: 300 }}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Execution Speed
                    </Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={getSpeedData()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="metric" />
                        <YAxis
                          label={{
                            value: "Time (ms)",
                            angle: -90,
                            position: "insideLeft",
                          }}
                        />
                        <Tooltip />
                        <Legend />
                        <Bar
                          dataKey="neural"
                          fill="#1976d2"
                          name="Neural Network"
                        />
                        <Bar
                          dataKey="classical"
                          fill="#2e7d32"
                          name="Classical Model"
                        />
                        <Bar
                          dataKey="advanced"
                          fill="#ed6c02"
                          name="Advanced Math"
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </Paper>
                </Box>

                <Box sx={{ flex: "1 1 95%", minWidth: 300, mt: 3 }}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      📈 Model Characteristics Summary
                    </Typography>
                    <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
                      <Box sx={{ flex: "1 1 30%", minWidth: 250 }}>
                        <Typography variant="h6" color="primary" gutterBottom>
                          Neural Networks
                        </Typography>
                        <Typography variant="body2" paragraph>
                          • Best for complex nonlinear patterns • High
                          computational requirements • Requires large training
                          datasets • Excellent for prediction accuracy
                        </Typography>
                      </Box>
                      <Box sx={{ flex: "1 1 30%", minWidth: 250 }}>
                        <Typography
                          variant="h6"
                          color="success.main"
                          gutterBottom
                        >
                          Classical Models
                        </Typography>
                        <Typography variant="body2" paragraph>
                          • Fast execution and low latency • Mathematically
                          interpretable • Well-established algorithms • Good for
                          real-time applications
                        </Typography>
                      </Box>
                      <Box sx={{ flex: "1 1 30%", minWidth: 250 }}>
                        <Typography
                          variant="h6"
                          color="warning.main"
                          gutterBottom
                        >
                          Advanced Mathematics
                        </Typography>
                        <Typography variant="body2" paragraph>
                          • Theoretical guarantees and bounds • Superior
                          uncertainty quantification • Complex mathematical
                          foundations • Best for critical decision-making
                        </Typography>
                      </Box>
                    </Box>
                  </Paper>
                </Box>
              </Box>
            </CardContent>
          )}

          {/* Detailed Comparisons Tab */}
          {activeTab === 1 && (
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Detailed Prediction Comparison
              </Typography>

              <Paper sx={{ p: 2, mb: 3 }}>
                <Typography variant="subtitle1" gutterBottom>
                  Prediction Trajectories with Uncertainty
                </Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={getPredictionComparisonData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Legend />

                    {/* True values */}
                    <Line
                      type="monotone"
                      dataKey="true"
                      stroke="#000"
                      strokeWidth={3}
                      name="True Values"
                      dot={false}
                    />

                    {/* Neural predictions with uncertainty */}
                    <Line
                      type="monotone"
                      dataKey="neural"
                      stroke="#1976d2"
                      strokeWidth={2}
                      name="Neural Network"
                      dot={false}
                    />
                    <Area
                      type="monotone"
                      dataKey={(d) => [d.neural_lower, d.neural_upper]}
                      stroke="none"
                      fill="#1976d2"
                      fillOpacity={0.2}
                      name="Neural Uncertainty"
                    />

                    {/* Classical predictions with uncertainty */}
                    <Line
                      type="monotone"
                      dataKey="classical"
                      stroke="#2e7d32"
                      strokeWidth={2}
                      name="Classical Model"
                      dot={false}
                    />
                    <Area
                      type="monotone"
                      dataKey={(d) => [d.classical_lower, d.classical_upper]}
                      stroke="none"
                      fill="#2e7d32"
                      fillOpacity={0.2}
                      name="Classical Uncertainty"
                    />

                    {/* Advanced predictions with uncertainty */}
                    <Line
                      type="monotone"
                      dataKey="advanced"
                      stroke="#ed6c02"
                      strokeWidth={2}
                      name="Advanced Math"
                      dot={false}
                    />
                    <Area
                      type="monotone"
                      dataKey={(d) => [d.advanced_lower, d.advanced_upper]}
                      stroke="none"
                      fill="#ed6c02"
                      fillOpacity={0.2}
                      name="Advanced Uncertainty"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Paper>

              {/* Error Analysis */}
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  📏 Error Distribution Analysis
                </Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Model</TableCell>
                        <TableCell align="right">MSE</TableCell>
                        <TableCell align="right">MAE</TableCell>
                        <TableCell align="right">Max Error</TableCell>
                        <TableCell align="right">
                          Uncertainty (95% CI)
                        </TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {comparisonData.neural && (
                        <TableRow>
                          <TableCell>Neural Network</TableCell>
                          <TableCell align="right">
                            {comparisonData.neural.metrics.mse?.toFixed(4) ||
                              "N/A"}
                          </TableCell>
                          <TableCell align="right">
                            {comparisonData.neural.metrics.mae?.toFixed(4) ||
                              "N/A"}
                          </TableCell>
                          <TableCell align="right">N/A</TableCell>
                          <TableCell align="right">
                            {comparisonData.neural.metrics.uncertainty?.toFixed(
                              3
                            ) || "N/A"}
                          </TableCell>
                        </TableRow>
                      )}
                      {comparisonData.classical && (
                        <TableRow>
                          <TableCell>Classical Model</TableCell>
                          <TableCell align="right">
                            {comparisonData.classical.metrics.mse?.toFixed(4) ||
                              "N/A"}
                          </TableCell>
                          <TableCell align="right">
                            {comparisonData.classical.metrics.mae?.toFixed(4) ||
                              "N/A"}
                          </TableCell>
                          <TableCell align="right">N/A</TableCell>
                          <TableCell align="right">
                            {comparisonData.classical.metrics.uncertainty?.toFixed(
                              3
                            ) || "N/A"}
                          </TableCell>
                        </TableRow>
                      )}
                      {comparisonData.advanced && (
                        <TableRow>
                          <TableCell>Advanced Math</TableCell>
                          <TableCell align="right">
                            {comparisonData.advanced.metrics.mse?.toFixed(4) ||
                              "N/A"}
                          </TableCell>
                          <TableCell align="right">
                            {comparisonData.advanced.metrics.mae?.toFixed(4) ||
                              "N/A"}
                          </TableCell>
                          <TableCell align="right">N/A</TableCell>
                          <TableCell align="right">
                            {comparisonData.advanced.metrics.uncertainty?.toFixed(
                              3
                            ) || "N/A"}
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            </CardContent>
          )}

          {/* Recommendations Tab */}
          {activeTab === 2 && (
            <CardContent>
              <Typography variant="h6" gutterBottom>
                AI-Powered Model Recommendations
              </Typography>

              <Alert severity="info" sx={{ mb: 3 }}>
                <Typography variant="body1">{getRecommendation()}</Typography>
              </Alert>

              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
                <Box sx={{ flex: "1 1 30%", minWidth: 250 }}>
                  <Card sx={{ border: "2px solid #1976d2" }}>
                    <CardHeader
                      avatar={<Psychology sx={{ color: "#1976d2" }} />}
                      title="Neural Networks"
                      subheader="When to Use"
                    />
                    <CardContent>
                      <Typography variant="body2" paragraph>
                        Best for complex pattern recognition, non-linear
                        relationships, and when large amounts of data are
                        available.
                      </Typography>
                      <Box component="ul" sx={{ pl: 2 }}>
                        <li>High-dimensional data</li>
                        <li>Complex interactions</li>
                        <li>Black-box acceptable</li>
                        <li>Computational resources available</li>
                      </Box>
                    </CardContent>
                  </Card>
                </Box>

                <Box sx={{ flex: "1 1 30%", minWidth: 250 }}>
                  <Card sx={{ border: "2px solid #2e7d32" }}>
                    <CardHeader
                      avatar={<Functions sx={{ color: "#2e7d32" }} />}
                      title="Classical Models"
                      subheader="When to Use"
                    />
                    <CardContent>
                      <Typography variant="body2" paragraph>
                        Best for interpretable results, fast computation, and
                        when mathematical guarantees are needed.
                      </Typography>
                      <Box component="ul" sx={{ pl: 2 }}>
                        <li>Interpretability required</li>
                        <li>Limited computational resources</li>
                        <li>Mathematical guarantees needed</li>
                        <li>Small to medium datasets</li>
                      </Box>
                    </CardContent>
                  </Card>
                </Box>

                <Box sx={{ flex: "1 1 30%", minWidth: 250 }}>
                  <Card sx={{ border: "2px solid #ed6c02" }}>
                    <CardHeader
                      avatar={<Science sx={{ color: "#ed6c02" }} />}
                      title="Advanced Math"
                      subheader="When to Use"
                    />
                    <CardContent>
                      <Typography variant="body2" paragraph>
                        Best for critical applications requiring uncertainty
                        quantification and theoretical optimality.
                      </Typography>
                      <Box component="ul" sx={{ pl: 2 }}>
                        <li>Uncertainty quantification needed</li>
                        <li>Theoretical guarantees required</li>
                        <li>Critical safety applications</li>
                        <li>Regulatory compliance</li>
                      </Box>
                    </CardContent>
                  </Card>
                </Box>
              </Box>
            </CardContent>
          )}
        </Card>
      )}
    </Box>
  );
};

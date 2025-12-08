import React, { useState } from "react";
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  CardHeader,
  Button,
  Chip,
  Alert,
  Tabs,
  Tab,
  CircularProgress,
} from "@mui/material";
import {
  Timeline,
  ElectricBolt as Flash,
  Warning,
  TrendingUp,
  Science,
  Security,
} from "@mui/icons-material";
import { MathEquation, ModelEquations } from "./MathEquation";
import axios from "axios";
import { config } from "../config";

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`use-case-tabpanel-${index}`}
      aria-labelledby={`use-case-tab-${index}`}
      {...other}
    >
      {value === index && children}
    </div>
  );
}

const UseCaseDemo: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState<{ [key: string]: boolean }>({});
  const [results, setResults] = useState<any>(null);
  const [classicalResults, setClassicalResults] = useState<any>(null);
  const [advancedResults, setAdvancedResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const useCases = [
    {
      id: "forecasting",
      title: "Load/Demand Forecasting",
      icon: <Timeline color="primary" />,
      description:
        "Predict future electricity demand using advanced mathematical models",
    },
    {
      id: "state_estimation",
      title: "Grid State Estimation",
      icon: <Flash color="secondary" />,
      description: "Estimate complete grid state from partial measurements",
    },
    {
      id: "congestion",
      title: "Congestion Prediction",
      icon: <Warning color="warning" />,
      description: "Predict transmission line overloads in advance",
    },
    {
      id: "opf",
      title: "OPF Surrogate",
      icon: <TrendingUp color="success" />,
      description: "Fast approximation of Optimal Power Flow solutions",
    },
    {
      id: "spatiotemporal",
      title: "Spatiotemporal Fusion",
      icon: <Science color="info" />,
      description: "Combined spatial and temporal grid analysis",
    },
    {
      id: "anomaly",
      title: "Anomaly Detection",
      icon: <Security color="error" />,
      description: "Identify unusual grid behavior and cyber threats",
    },
  ];

  const runModel = async (
    modelType: "neural" | "classical" | "advanced",
    useCase: string
  ) => {
    // Prevent multiple simultaneous calls for the same model type
    if (loading[modelType]) {
      console.warn(
        `Model ${modelType} is already running, skipping duplicate call`
      );
      return;
    }

    setLoading((prev) => ({ ...prev, [modelType]: true }));
    setError(null);

    try {
      let endpoint = "";
      if (modelType === "neural") {
        // Use new UI button endpoint for better debugging
        endpoint = `/api/ui/neural-network/${useCase}`;
      } else if (modelType === "classical") {
        // Use new UI button endpoint for better debugging
        endpoint = `/api/ui/classical-model/${useCase}`;
      } else if (modelType === "advanced") {
        // Use new UI button endpoint for better debugging
        endpoint = `/api/ui/advanced-math/${useCase}`;
      }

      console.log(`Calling ${endpoint} for ${modelType} model on ${useCase}`);
      const response = await axios.post(`${config.api.baseUrl}${endpoint}`, {});
      const result = { ...response.data, useCase, modelType };

      if (modelType === "neural") setResults(result);
      else if (modelType === "classical") setClassicalResults(result);
      else setAdvancedResults(result);

      return result;
    } catch (err: any) {
      const errorMessage =
        err.response?.data?.error || err.message || "Unknown error";
      console.error(`Error running ${modelType} model:`, errorMessage);
      setError(`Failed to run ${modelType} model: ${errorMessage}`);
      // Don't re-throw to prevent unhandled promise rejections
    } finally {
      setLoading((prev) => ({ ...prev, [modelType]: false }));
    }
  };

  const useCase = useCases[activeTab];

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          AI Grid Model Comparison
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Compare Neural Networks, Classical Models, and Advanced Mathematics
        </Typography>
      </Box>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          <Typography variant="body2">
            <strong>Error:</strong> {error}
          </Typography>
        </Alert>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Use Case Selection */}
      <Card sx={{ mb: 3 }}>
        <CardHeader title="Select Use Case" />
        <CardContent>
          <Tabs
            value={activeTab}
            onChange={(_, newValue) => setActiveTab(newValue)}
          >
            {useCases.map((uc, index) => (
              <Tab
                key={uc.id}
                label={
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    {uc.icon}
                    <Typography variant="body2">{uc.title}</Typography>
                  </Box>
                }
              />
            ))}
          </Tabs>
        </CardContent>
      </Card>

      {/* Current Use Case Display */}
      <Card sx={{ mb: 3 }}>
        <CardHeader
          avatar={useCase.icon}
          title={useCase.title}
          subheader={useCase.description}
        />
        <CardContent>
          {/* Mathematical Equations */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Advanced Mathematical Foundation
            </Typography>
            <ModelEquations
              modelType={
                useCase.id as
                  | "forecasting"
                  | "state_estimation"
                  | "congestion"
                  | "opf"
                  | "spatiotemporal"
                  | "anomaly"
              }
              showExplanations={true}
            />
          </Box>

          {/* Model Buttons */}
          <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
            <Button
              variant="contained"
              onClick={() => runModel("neural", useCase.id)}
              disabled={loading.neural || !!error}
              startIcon={
                loading.neural ? <CircularProgress size={16} /> : undefined
              }
              sx={{
                minWidth: "140px",
                "&:disabled": {
                  opacity: 0.6,
                },
              }}
            >
              {loading.neural ? "Running..." : "Neural Network"}
            </Button>
            <Button
              variant="outlined"
              onClick={() => runModel("classical", useCase.id)}
              disabled={loading.classical || !!error}
              startIcon={
                loading.classical ? <CircularProgress size={16} /> : undefined
              }
              sx={{
                minWidth: "140px",
                "&:disabled": {
                  opacity: 0.6,
                },
              }}
            >
              {loading.classical ? "Running..." : "Classical Model"}
            </Button>
            <Button
              variant="outlined"
              color="secondary"
              onClick={() => runModel("advanced", useCase.id)}
              disabled={loading.advanced || !!error}
              startIcon={
                loading.advanced ? <CircularProgress size={16} /> : undefined
              }
              sx={{
                minWidth: "140px",
                "&:disabled": {
                  opacity: 0.6,
                },
              }}
            >
              {loading.advanced ? "Running..." : "Advanced Math"}
            </Button>
          </Box>

          {/* Results Display */}
          <Box sx={{ mt: 3 }}>
            {results && results.useCase === useCase.id && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="h6" color="primary">
                  Neural Network Results
                </Typography>
                <Typography>
                  Status: {results.success ? "Success" : "Failed"}
                </Typography>
                {results.metrics && (
                  <Box sx={{ mt: 1 }}>
                    {results.metrics.mse && (
                      <Chip
                        label={`MSE: ${results.metrics.mse.toFixed(4)}`}
                        sx={{ mr: 1 }}
                      />
                    )}
                    {results.metrics.accuracy && (
                      <Chip
                        label={`Accuracy: ${(
                          results.metrics.accuracy * 100
                        ).toFixed(1)}%`}
                        sx={{ mr: 1 }}
                      />
                    )}
                  </Box>
                )}
              </Box>
            )}

            {classicalResults && classicalResults.useCase === useCase.id && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="h6" color="secondary">
                  Classical Model Results
                </Typography>
                <Typography>
                  Status: {classicalResults.success ? "Success" : "Failed"}
                </Typography>
                {classicalResults.results?.metrics && (
                  <Box sx={{ mt: 1 }}>
                    {classicalResults.results.metrics.mse && (
                      <Chip
                        label={`MSE: ${classicalResults.results.metrics.mse.toFixed(
                          4
                        )}`}
                        sx={{ mr: 1 }}
                      />
                    )}
                    {classicalResults.results.metrics.accuracy && (
                      <Chip
                        label={`Accuracy: ${(
                          classicalResults.results.metrics.accuracy * 100
                        ).toFixed(1)}%`}
                        sx={{ mr: 1 }}
                      />
                    )}
                  </Box>
                )}
              </Box>
            )}

            {advancedResults && advancedResults.useCase === useCase.id && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="h6" sx={{ color: "success.main" }}>
                  Advanced Math Results
                </Typography>
                <Typography>
                  Status: {advancedResults.success ? "Success" : "Failed"}
                </Typography>
                {advancedResults.result?.metrics && (
                  <Box sx={{ mt: 1 }}>
                    {advancedResults.result.metrics.mse && (
                      <Chip
                        label={`MSE: ${advancedResults.result.metrics.mse.toFixed(
                          4
                        )}`}
                        sx={{ mr: 1 }}
                      />
                    )}
                    {advancedResults.result.metrics.accuracy && (
                      <Chip
                        label={`Accuracy: ${(
                          advancedResults.result.metrics.accuracy * 100
                        ).toFixed(1)}%`}
                        sx={{ mr: 1 }}
                      />
                    )}
                  </Box>
                )}
              </Box>
            )}
          </Box>
        </CardContent>
      </Card>
    </Container>
  );
};

export default UseCaseDemo;

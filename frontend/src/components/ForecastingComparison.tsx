import React, { useState } from "react";
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  CardHeader,
  Button,
  Alert,
  CircularProgress,
  Chip,
} from "@mui/material";
import { TrendingUp, PlayArrow } from "@mui/icons-material";
import { ModelComparisonVisualizer } from "./ModelComparisonVisualizer";
import axios from "axios";
import { config } from "../config";

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

const useCases = [
  {
    id: "forecasting",
    title: "Load Forecasting",
    description: "Predict future electricity demand patterns",
    icon: <TrendingUp color="primary" />,
  },
];

const ForecastingComparison: React.FC = () => {
  const [results, setResults] = useState<any>(null);
  const [classicalResults, setClassicalResults] = useState<any>(null);
  const [advancedResults, setAdvancedResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<{ [key: string]: boolean }>({});

  const useCase = useCases[0]; // Always forecasting for this page

  const runModel = async (
    modelType: "neural" | "classical" | "advanced",
    useCaseId: string
  ) => {
    setLoading((prev) => ({ ...prev, [modelType]: true }));
    setError(null);

    try {
      // Map model types to correct endpoint suffixes
      const endpointSuffix = {
        neural: "neural-network",
        classical: "classical-model",
        advanced: "advanced-math",
      }[modelType];

      const response = await axios.post(
        `${config.api.baseUrl}/api/ui/${endpointSuffix}/${useCaseId}`,
        {},
        {
          headers: { "Content-Type": "application/json" },
        }
      );

      const resultData = response.data;

      if (modelType === "neural") {
        setResults(resultData);
      } else if (modelType === "classical") {
        setClassicalResults(resultData);
      } else if (modelType === "advanced") {
        setAdvancedResults(resultData);
      }

      return resultData;
    } catch (err: any) {
      setError(
        `Failed to run ${modelType} model: ${
          err.response?.data?.error || err.message
        }`
      );
      throw err;
    } finally {
      setLoading((prev) => ({ ...prev, [modelType]: false }));
    }
  };

  const runAllModels = async () => {
    setError(null);
    await Promise.all([
      runModel("neural", useCase.id),
      runModel("classical", useCase.id),
      runModel("advanced", useCase.id),
    ]);
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Forecasting - Three-Way Model Comparison
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Compare Neural Networks, Classical Models, and Advanced Mathematics
          for load forecasting
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

      {/* Quick Actions */}
      <Card sx={{ mb: 3 }}>
        <CardHeader title="Quick Model Comparison" />
        <CardContent>
          <Box
            sx={{
              display: "flex",
              gap: 2,
              flexWrap: "wrap",
              alignItems: "center",
            }}
          >
            <Button
              variant="contained"
              onClick={runAllModels}
              disabled={Object.values(loading).some(Boolean)}
              startIcon={
                Object.values(loading).some(Boolean) ? (
                  <CircularProgress size={16} />
                ) : (
                  <PlayArrow />
                )
              }
              sx={{ minWidth: "180px" }}
            >
              {Object.values(loading).some(Boolean)
                ? "Running All..."
                : "Run All Models"}
            </Button>

            <Typography variant="body2" color="text.secondary">
              Execute all three model types and view comprehensive comparison
              results below
            </Typography>
          </Box>
        </CardContent>
      </Card>

      {/* Individual Model Execution */}
      <Card sx={{ mb: 3 }}>
        <CardHeader
          avatar={useCase.icon}
          title={useCase.title}
          subheader={useCase.description}
        />
        <CardContent>
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

      {/* Detailed Comparison Visualizer - Always Visible */}
      <ModelComparisonVisualizer useCase={useCase.id} onModelRun={runModel} />
    </Container>
  );
};

export default ForecastingComparison;

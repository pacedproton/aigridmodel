import React, { useState, useEffect } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Card,
  CardContent,
  Chip,
  Divider,
  Alert,
  CircularProgress,
} from "@mui/material";
import {
  Science,
  Timeline,
  Warning,
  TrendingUp,
  ElectricBolt as Flash,
  Security,
} from "@mui/icons-material";
import axios from "axios";
import { config } from "../config";

interface DemoModalProps {
  open: boolean;
  onClose: () => void;
}

interface NetworkPlot {
  plot: string;
  description: string;
}

const useCases = [
  {
    icon: <Timeline color="primary" />,
    title: "Load/Demand Forecasting",
    description:
      "Predict future electricity demand patterns using historical data and temporal AI models.",
    applications: [
      "Day-ahead planning",
      "Resource scheduling",
      "Peak demand management",
    ],
  },
  {
    icon: <Flash color="secondary" />,
    title: "Grid State Estimation",
    description:
      "Estimate complete grid state (voltages, currents) from partial measurements using AI.",
    applications: ["Real-time monitoring", "Fault detection", "Load balancing"],
  },
  {
    icon: <Warning color="warning" />,
    title: "Congestion Prediction",
    description:
      "Predict transmission line overloads and grid congestion points in advance.",
    applications: [
      "Preventive control",
      "Redispatch planning",
      "Grid stability",
    ],
  },
  {
    icon: <TrendingUp color="success" />,
    title: "OPF Surrogate",
    description:
      "Fast approximation of Optimal Power Flow solutions using neural network surrogates.",
    applications: [
      "Real-time optimization",
      "Economic dispatch",
      "Voltage control",
    ],
  },
  {
    icon: <Science color="info" />,
    title: "Spatiotemporal Fusion",
    description:
      "Combined spatial (grid topology) and temporal (time series) modeling for comprehensive grid analysis.",
    applications: [
      "Multi-node forecasting",
      "Cascading failure prediction",
      "Dynamic stability",
    ],
  },
  {
    icon: <Security color="error" />,
    title: "Anomaly Detection",
    description:
      "Identify unusual grid behavior, cyber threats, and equipment failures using AI.",
    applications: [
      "Cybersecurity",
      "Equipment monitoring",
      "Incident response",
    ],
  },
];

const DemoModal: React.FC<DemoModalProps> = ({ open, onClose }) => {
  const [networkPlot, setNetworkPlot] = useState<NetworkPlot | null>(null);
  const [loadingPlot, setLoadingPlot] = useState(false);

  useEffect(() => {
    console.log(
      "DemoModal useEffect triggered, open:",
      open,
      "networkPlot exists:",
      !!networkPlot
    );
    if (open && !networkPlot) {
      console.log("Calling fetchNetworkPlot...");
      fetchNetworkPlot();
    }
  }, [open]);

  const fetchNetworkPlot = async () => {
    try {
      setLoadingPlot(true);
      console.log("Fetching network plot...");
      const response = await axios.get(config.api.endpoints.networkPlot);
      console.log("Network plot response:", response.data);
      if (response.data.success) {
        setNetworkPlot(response.data);
        console.log("Network plot set successfully");
      } else {
        console.error("Network plot API returned success=false");
      }
    } catch (error) {
      console.error("Failed to fetch network plot:", error);
      if (axios.isAxiosError(error) && error.response) {
        console.error("Response status:", error.response.status);
        console.error("Response data:", error.response.data);
      }
    } finally {
      setLoadingPlot(false);
    }
  };
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: {
          minHeight: "80vh",
          maxHeight: "90vh",
        },
      }}
    >
      <DialogTitle
        sx={{
          backgroundColor: "primary.main",
          color: "primary.contrastText",
          textAlign: "center",
        }}
      >
        <Typography variant="h4" component="div">
          AI-Powered Grid Modeling Demo
        </Typography>
      </DialogTitle>

      <DialogContent sx={{ p: 3 }}>
        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="body2">
            <strong>
              This is a domain-focused demonstration and learning project{" "}
            </strong>
            illustrating typical AI use cases in grid analysis and power system
            operations. The implementation serves solely to explore and
            understand relevant data structures, workflows, and modeling
            approaches in a practical manner. Implemented components such as
            synthetic test data, ML models, and visualizations are included only
            to show how common use cases can be represented within the scope of
            a proof of concept.
          </Typography>
        </Alert>

        <Typography
          variant="h6"
          gutterBottom
          sx={{ color: "primary.main", fontWeight: "bold" }}
        >
          Core Features
        </Typography>

        <Box sx={{ mb: 4 }}>
          <Typography variant="body1" paragraph>
            • <strong>6 AI Use Cases:</strong> Complete implementations of
            modern grid optimization techniques
          </Typography>
          <Typography variant="body1" paragraph>
            • <strong>Real Training:</strong> Actual machine learning models
            trained on synthetic grid data
          </Typography>
          <Typography variant="body1" paragraph>
            • <strong>Interactive UI:</strong> Professional dashboard with live
            data visualization and controls
          </Typography>
          <Typography variant="body1" paragraph>
            • <strong>Convergence Tracking:</strong> Real-time training progress
            with loss curves and metrics
          </Typography>
        </Box>

        <Divider sx={{ my: 3 }} />

        <Typography
          variant="h6"
          gutterBottom
          sx={{ color: "primary.main", fontWeight: "bold" }}
        >
          🔧 Technical Implementation
        </Typography>

        <Box sx={{ mb: 4 }}>
          <Typography variant="body1" paragraph>
            • <strong>Backend:</strong> Python Flask API with PyTorch machine
            learning models
          </Typography>
          <Typography variant="body1" paragraph>
            • <strong>Frontend:</strong> React TypeScript with Material-UI
            professional interface
          </Typography>
          <Typography variant="body1" paragraph>
            • <strong>Data:</strong> Synthetic grid data generated using
            Pandapower (IEEE 14-bus system)
          </Typography>
          <Typography variant="body1" paragraph>
            • <strong>Training:</strong> Small-scale but real ML training with
            convergence tracking
          </Typography>
        </Box>

        <Divider sx={{ my: 3 }} />

        <Typography
          variant="h6"
          gutterBottom
          sx={{ color: "primary.main", fontWeight: "bold" }}
        >
          IEEE 14-Bus System Visualization
        </Typography>

        <Box sx={{ mb: 4, p: 2, backgroundColor: "grey.50", borderRadius: 1 }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            This demo uses the IEEE 14-bus test system - a standard benchmark
            for power system analysis:
          </Typography>

          <Box
            sx={{
              width: "100%",
              minHeight: 400,
              backgroundColor: "white",
              border: "2px solid #1976d2",
              borderRadius: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              position: "relative",
              overflow: "hidden",
              p: 1,
            }}
          >
            <div
              style={{
                position: "absolute",
                top: "5px",
                left: "5px",
                fontSize: "12px",
                color: "#666",
                background: "rgba(255,255,255,0.9)",
                padding: "2px 5px",
                borderRadius: "3px",
              }}
            >
              Modal open: {open ? "YES" : "NO"} | Loading:{" "}
              {loadingPlot ? "YES" : "NO"} | Has plot:{" "}
              {networkPlot ? "YES" : "NO"}
            </div>
            {loadingPlot ? (
              <div style={{ textAlign: "center" }}>
                <CircularProgress />
                <div style={{ marginTop: "10px", color: "#666" }}>
                  Loading network visualization...
                </div>
              </div>
            ) : networkPlot ? (
              <div>
                <img
                  src={networkPlot.plot}
                  alt="IEEE 14-Bus Network Topology"
                  style={{
                    maxWidth: "100%",
                    maxHeight: "100%",
                    objectFit: "contain",
                    border: "1px solid #ccc",
                  }}
                  onLoad={() => console.log("Image loaded successfully")}
                  onError={(e) => {
                    console.error("Image failed to load:", e);
                    console.error(
                      "Image src starts with:",
                      networkPlot.plot.substring(0, 50)
                    );
                    console.error(
                      "Image src total length:",
                      networkPlot.plot.length
                    );
                  }}
                />
                <div
                  style={{ fontSize: "12px", color: "#666", marginTop: "5px" }}
                >
                  Debug: Image data length: {networkPlot.plot?.length || 0}{" "}
                  chars
                  <button
                    onClick={fetchNetworkPlot}
                    style={{
                      marginLeft: "10px",
                      padding: "2px 8px",
                      fontSize: "11px",
                      background: "#1976d2",
                      color: "white",
                      border: "none",
                      borderRadius: "3px",
                      cursor: "pointer",
                    }}
                  >
                    Refresh
                  </button>
                </div>
              </div>
            ) : (
              <Typography variant="body2" color="text.secondary">
                Loading network visualization...
              </Typography>
            )}
          </Box>

          {networkPlot && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              {networkPlot.description}
            </Typography>
          )}
        </Box>

        <Divider sx={{ my: 3 }} />

        <Typography
          variant="h6"
          gutterBottom
          sx={{ color: "primary.main", fontWeight: "bold" }}
        >
          AI Use Cases Demonstrated
        </Typography>

        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
          {useCases.map((useCase, index) => (
            <Box sx={{ flex: "1 1 500px", minWidth: "400px" }} key={index}>
              <Card variant="outlined" sx={{ height: "100%" }}>
                <CardContent>
                  <Box display="flex" alignItems="center" gap={1} mb={1}>
                    {useCase.icon}
                    <Typography variant="h6" component="div">
                      {useCase.title}
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    {useCase.description}
                  </Typography>
                  <Typography variant="body2" fontWeight="bold" sx={{ mb: 1 }}>
                    Applications:
                  </Typography>
                  <Box display="flex" flexWrap="wrap" gap={0.5}>
                    {useCase.applications.map((app, appIndex) => (
                      <Chip
                        key={appIndex}
                        label={app}
                        size="small"
                        variant="outlined"
                        color="primary"
                      />
                    ))}
                  </Box>
                </CardContent>
              </Card>
            </Box>
          ))}
        </Box>

        <Box
          sx={{
            mt: 3,
            p: 2,
            backgroundColor: "success.light",
            borderRadius: 1,
          }}
        >
          <Typography variant="body2" sx={{ color: "success.contrastText" }}>
            <strong>All functionality is fully implemented:</strong> You can
            train real ML models, see convergence curves, and interact with all
            use case demonstrations. The system uses synthetic data generated
            via Pandapower for realistic grid simulation.
          </Typography>
        </Box>
      </DialogContent>

      <DialogActions sx={{ p: 3, pt: 0 }}>
        <Button onClick={onClose} variant="contained" size="large" fullWidth>
          Start Exploring the Demo
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default DemoModal;

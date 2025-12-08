import React, { useState, useEffect } from "react";
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  CardHeader,
  Button,
  Alert,
  Chip,
  CircularProgress,
} from "@mui/material";
import { DataUsage, PlayArrow, Refresh, Info } from "@mui/icons-material";
import axios from "axios";
import { config } from "../config";

interface NetworkPlot {
  plot: string;
  description: string;
}

const IEEE14BusVisualization: React.FC = () => {
  const [topologyData, setTopologyData] = useState<any>(null);
  const [networkPlot, setNetworkPlot] = useState<NetworkPlot | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingPlot, setLoadingPlot] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadTopology = async () => {
    setLoading(true);
    setError(null);

    try {
      // First generate data if needed
      await axios.post(`${config.api.baseUrl}/api/generate-data`, {
        n_samples: 100,
        time_steps: 24,
      });

      // Get topology data
      const response = await axios.get(
        `${config.api.baseUrl}/api/data/time-series`
      );
      setTopologyData(response.data);
    } catch (err: any) {
      setError(
        `Failed to load topology: ${err.response?.data?.error || err.message}`
      );
    } finally {
      setLoading(false);
    }
  };

  const fetchNetworkPlot = async () => {
    try {
      setLoadingPlot(true);
      const response = await axios.get(
        `${config.api.baseUrl}/api/network/plot`
      );
      if (response.data.success) {
        setNetworkPlot(response.data);
      } else {
        console.error("Network plot API returned success=false");
      }
    } catch (err: any) {
      console.error("Failed to fetch network plot:", err);
      setError(
        `Failed to load network visualization: ${
          err.response?.data?.error || err.message
        }`
      );
    } finally {
      setLoadingPlot(false);
    }
  };

  useEffect(() => {
    loadTopology();
    if (!networkPlot) {
      fetchNetworkPlot();
    }
  }, []);

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          IEEE 14-Bus System Visualization
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Interactive visualization of the IEEE 14-bus test system topology and
          real-time data
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

      {/* Controls */}
      <Card sx={{ mb: 3 }}>
        <CardHeader title="System Controls" />
        <CardContent>
          <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
            <Button
              variant="contained"
              onClick={loadTopology}
              disabled={loading}
              startIcon={
                loading ? <Refresh className="spinning" /> : <PlayArrow />
              }
            >
              {loading ? "Loading..." : "Load Topology Data"}
            </Button>
            <Button
              variant="outlined"
              startIcon={<Info />}
              onClick={() => {
                // Could open a modal with system information
                alert(
                  "IEEE 14-Bus System: Standard test system with 14 buses, 5 generators, and 11 loads. Used for power system analysis and algorithm validation."
                );
              }}
            >
              System Info
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Topology Overview */}
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
        <Box sx={{ flex: "1 1 45%", minWidth: 300 }}>
          <Card>
            <CardHeader title="System Topology" />
            <CardContent>
              <Typography variant="body2" paragraph>
                The IEEE 14-bus system is a standard test case for power system
                analysis, consisting of:
              </Typography>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Chip label="14 Buses" size="small" color="primary" />
                  <Typography variant="caption">
                    5 generator buses, 9 load buses
                  </Typography>
                </Box>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Chip label="20 Lines" size="small" color="secondary" />
                  <Typography variant="caption">
                    Transmission lines and transformers
                  </Typography>
                </Box>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Chip label="5 Generators" size="small" color="success" />
                  <Typography variant="caption">
                    Power generation sources
                  </Typography>
                </Box>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Chip label="11 Loads" size="small" color="warning" />
                  <Typography variant="caption">
                    Power consumption points
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Box>

        <Box sx={{ flex: "1 1 45%", minWidth: 300 }}>
          <Card>
            <CardHeader title="Real-time Data Status" />
            <CardContent>
              {topologyData ? (
                <Box>
                  <Typography variant="body2" gutterBottom>
                    <strong>Data Loaded:</strong> Available
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Node features: {topologyData.node_features?.length || 0}{" "}
                    samples
                    <br />
                    Edge features: {topologyData.edge_targets?.length || 0}{" "}
                    connections
                    <br />
                    Time steps: {topologyData.node_features?.[0]?.length || 0}
                  </Typography>
                </Box>
              ) : (
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    {loading
                      ? "Loading topology data..."
                      : "No data loaded yet"}
                  </Typography>
                  {!loading && (
                    <Button
                      size="small"
                      variant="outlined"
                      sx={{ mt: 1 }}
                      onClick={loadTopology}
                    >
                      Load Data
                    </Button>
                  )}
                </Box>
              )}
            </CardContent>
          </Card>
        </Box>
      </Box>

      {/* Network Topology Visualization */}
      <Card sx={{ mt: 3 }}>
        <CardHeader title="Network Topology Visualization" />
        <CardContent>
          <Box
            sx={{ mb: 4, p: 2, backgroundColor: "grey.50", borderRadius: 1 }}
          >
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
                Modal open: YES | Loading: {loadingPlot ? "YES" : "NO"} | Has
                plot: {networkPlot ? "YES" : "NO"}
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
                      setError("Failed to load network visualization image");
                    }}
                  />
                  <div
                    style={{
                      position: "absolute",
                      bottom: "5px",
                      right: "5px",
                      fontSize: "12px",
                      color: "#666",
                      background: "rgba(255,255,255,0.9)",
                      padding: "2px 5px",
                      borderRadius: "3px",
                    }}
                  >
                    Debug: Image data length: {networkPlot.plot?.length || 0}{" "}
                    chars
                    <button
                      onClick={fetchNetworkPlot}
                      style={{
                        marginLeft: "5px",
                        fontSize: "10px",
                        padding: "1px 3px",
                        cursor: "pointer",
                      }}
                    >
                      Refresh
                    </button>
                  </div>
                </div>
              ) : (
                <div style={{ textAlign: "center", color: "#666" }}>
                  <div>No network visualization available</div>
                  <button
                    onClick={fetchNetworkPlot}
                    style={{
                      marginTop: "10px",
                      padding: "5px 10px",
                      cursor: "pointer",
                    }}
                  >
                    Load Visualization
                  </button>
                </div>
              )}
            </Box>

            {networkPlot && (
              <Typography
                variant="body2"
                sx={{ mt: 2, fontStyle: "italic", color: "text.secondary" }}
              >
                {networkPlot.description}
              </Typography>
            )}
          </Box>
        </CardContent>
      </Card>
    </Container>
  );
};

export default IEEE14BusVisualization;

import React, { useState, useEffect } from "react";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { CssBaseline, Box } from "@mui/material";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Dashboard from "./components/Dashboard";
import DataManagement from "./components/DataManagement";
import ModelTraining from "./components/ModelTraining";
import UseCaseDemo from "./components/UseCaseDemo";
import ForecastingComparison from "./components/ForecastingComparison";
import IEEE14BusVisualization from "./components/IEEE14BusVisualization";
import TestCoveragePage from "./components/TestCoveragePage";
import WorkflowModal from "./components/WorkflowModal";
import Navigation from "./components/Navigation";
import DemoModal from "./components/DemoModal";

const theme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#2196f3",
    },
    secondary: {
      main: "#f50057",
    },
    background: {
      default: "#121212",
      paper: "#1e1e1e",
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
  },
});

function App() {
  const [showDemoModal, setShowDemoModal] = useState(false);
  const [showWorkflowModal, setShowWorkflowModal] = useState(false);

  useEffect(() => {
    // Show demo modal on first load
    const hasSeenDemo = localStorage.getItem("hasSeenDemo");
    if (!hasSeenDemo) {
      setShowDemoModal(true);
    }

    // Show workflow modal on page reload (not first visit)
    const hasSeenWorkflow = localStorage.getItem("hasSeenWorkflow");
    if (hasSeenDemo && !hasSeenWorkflow) {
      setShowWorkflowModal(true);
    }
  }, []);

  const handleCloseDemoModal = () => {
    setShowDemoModal(false);
    localStorage.setItem("hasSeenDemo", "true");
  };

  const handleShowDemoModal = () => {
    setShowDemoModal(true);
  };

  const handleCloseWorkflowModal = () => {
    setShowWorkflowModal(false);
    localStorage.setItem("hasSeenWorkflow", "true");
  };

  const handleWorkflowNavigate = (path: string) => {
    window.location.href = path;
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: "flex", minHeight: "100vh" }}>
          <Navigation />
          <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
            <Routes>
              <Route
                path="/"
                element={<Dashboard onShowDemoModal={handleShowDemoModal} />}
              />
              <Route path="/data" element={<DataManagement />} />
              <Route path="/training" element={<ModelTraining />} />
              <Route path="/demo" element={<UseCaseDemo />} />
              <Route
                path="/forecasting-comparison"
                element={<ForecastingComparison />}
              />
              <Route path="/topology" element={<IEEE14BusVisualization />} />
              <Route path="/test-coverage" element={<TestCoveragePage />} />
            </Routes>
          </Box>
        </Box>

        <DemoModal open={showDemoModal} onClose={handleCloseDemoModal} />
        <WorkflowModal
          open={showWorkflowModal}
          onClose={handleCloseWorkflowModal}
          onNavigate={handleWorkflowNavigate}
        />
      </Router>
    </ThemeProvider>
  );
}

export default App;

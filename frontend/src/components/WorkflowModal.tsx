import React from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Stepper,
  Step,
  StepLabel,
  Chip,
} from "@mui/material";
import { DataUsage, ModelTraining, Compare, Close } from "@mui/icons-material";

interface WorkflowModalProps {
  open: boolean;
  onClose: () => void;
  onNavigate?: (path: string) => void;
}

const workflowSteps = [
  {
    label: "Generate Data",
    description: "Create synthetic IEEE 14-bus system data",
    icon: <DataUsage color="primary" />,
    path: "/data",
    details:
      "Load and preprocess grid topology data with realistic power system parameters",
  },
  {
    label: "Train Neural Network",
    description: "Train LSTM/GNN models on generated data",
    icon: <ModelTraining color="secondary" />,
    path: "/training",
    details:
      "Configure and train deep learning models for load forecasting and state estimation",
  },
  {
    label: "Run Comparisons",
    description: "Compare NN vs Classical vs Advanced models",
    icon: <Compare color="success" />,
    path: "/forecasting-comparison",
    details:
      "Execute comprehensive model comparison with performance metrics and uncertainty analysis",
  },
];

const WorkflowModal: React.FC<WorkflowModalProps> = ({
  open,
  onClose,
  onNavigate,
}) => {
  const handleStepClick = (path: string) => {
    if (onNavigate) {
      onNavigate(path);
    }
    onClose();
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      sx={{
        "& .MuiDialog-paper": {
          backgroundColor: "background.paper",
          backgroundImage: "none",
        },
      }}
    >
      <DialogTitle sx={{ pb: 1 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
          <Typography variant="h5" component="div" sx={{ fontWeight: "bold" }}>
            AI Grid Modeling Workflow
          </Typography>
          <Chip
            label="Recommended Path"
            color="primary"
            size="small"
            sx={{ fontWeight: "bold" }}
          />
        </Box>
      </DialogTitle>

      <DialogContent>
        <Typography variant="body1" color="text.secondary" paragraph>
          Follow this recommended workflow to get started with the AI Grid
          Modeling system:
        </Typography>

        <Stepper orientation="vertical" sx={{ mt: 2 }}>
          {workflowSteps.map((step, index) => (
            <Step key={step.label} active={true}>
              <StepLabel
                icon={step.icon}
                sx={{
                  "& .MuiStepLabel-iconContainer": {
                    alignSelf: "flex-start",
                    mt: 0.5,
                  },
                }}
              >
                <Box
                  sx={{ cursor: "pointer" }}
                  onClick={() => handleStepClick(step.path)}
                >
                  <Typography variant="h6" sx={{ fontWeight: "bold", mb: 0.5 }}>
                    {index + 1}. {step.label}
                  </Typography>
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{ mb: 1 }}
                  >
                    {step.description}
                  </Typography>
                  <Typography variant="body2" sx={{ fontStyle: "italic" }}>
                    {step.details}
                  </Typography>
                  <Button
                    size="small"
                    variant="outlined"
                    sx={{ mt: 1, fontSize: "0.75rem" }}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleStepClick(step.path);
                    }}
                  >
                    Go to {step.label}
                  </Button>
                </Box>
              </StepLabel>
            </Step>
          ))}
        </Stepper>

        <Box
          sx={{
            mt: 3,
            p: 2,
            backgroundColor: "background.default",
            borderRadius: 1,
          }}
        >
          <Typography variant="body2" color="text.secondary">
            <strong>💡 Pro Tip:</strong> Start with the Data Management page to
            generate synthetic grid data, then proceed to Model Training to
            configure and train your neural networks. Finally, use the
            Forecasting Comparison page to run comprehensive model evaluations.
          </Typography>
        </Box>
      </DialogContent>

      <DialogActions sx={{ p: 3 }}>
        <Button onClick={onClose} startIcon={<Close />}>
          Skip Tutorial
        </Button>
        <Button
          variant="contained"
          onClick={() => handleStepClick("/data")}
          sx={{ fontWeight: "bold" }}
        >
          Get Started
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default WorkflowModal;

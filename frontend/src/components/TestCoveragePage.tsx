import React from "react";
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  CardHeader,
} from "@mui/material";
import { Assessment } from "@mui/icons-material";
import TestStatusDashboard from "./TestStatusDashboard";

const TestCoveragePage: React.FC = () => {
  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Test Coverage Dashboard
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Comprehensive testing and quality assurance for the AI Grid Modeling
          system
        </Typography>
      </Box>

      <Card>
        <CardHeader
          avatar={<Assessment color="secondary" />}
          title="System Test Coverage"
          subheader="Real-time monitoring of test execution and coverage metrics"
        />
        <CardContent>
          <TestStatusDashboard />
        </CardContent>
      </Card>
    </Container>
  );
};

export default TestCoveragePage;

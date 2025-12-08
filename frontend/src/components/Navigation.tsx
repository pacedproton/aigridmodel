import React from "react";
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  Box,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Card,
  CardContent,
} from "@mui/material";
import {
  Dashboard,
  DataUsage,
  ModelTraining,
  Science,
  ExpandMore,
  Compare,
} from "@mui/icons-material";
import { Link, useLocation } from "react-router-dom";

const drawerWidth = 240;

const navigationItems = [
  { text: "Dashboard", icon: <Dashboard />, path: "/" },
  { text: "Data Management", icon: <DataUsage />, path: "/data" },
  { text: "Model Training", icon: <ModelTraining />, path: "/training" },
  { text: "Use Case Demo Overview", icon: <Science />, path: "/demo" },
  {
    text: "🎯 Forecasting Comparison",
    path: "/forecasting-comparison",
  },
  {
    text: "📊 IEEE 14-Bus System",
    path: "/topology",
  },
  {
    text: "📈 Test Coverage",
    path: "/test-coverage",
  },
];

const Navigation: React.FC = () => {
  const location = useLocation();

  return (
    <Drawer
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        "& .MuiDrawer-paper": {
          width: drawerWidth,
          boxSizing: "border-box",
          backgroundColor: "background.paper",
          borderRight: "1px solid rgba(255, 255, 255, 0.12)",
        },
      }}
      variant="permanent"
      anchor="left"
    >
      <Toolbar>
        <Typography
          variant="h6"
          noWrap
          component="div"
          sx={{ color: "primary.main", fontWeight: "bold" }}
        >
          AI Grid Demo
        </Typography>
      </Toolbar>
      <Divider />
      <List>
        {navigationItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              component={Link}
              to={item.path}
              selected={location.pathname === item.path}
              sx={{
                "&.Mui-selected": {
                  backgroundColor: "primary.main",
                  "&:hover": {
                    backgroundColor: "primary.dark",
                  },
                  "& .MuiListItemIcon-root": {
                    color: "white",
                  },
                  "& .MuiListItemText-primary": {
                    color: "white",
                  },
                },
              }}
            >
              {item.icon && (
                <ListItemIcon sx={{ color: "text.secondary" }}>
                  {item.icon}
                </ListItemIcon>
              )}
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      {/* Model Comparison Section - Only show on demo page */}
      {location.pathname === "/demo" && (
        <>
          <Divider />
          <Accordion
            sx={{
              backgroundColor: "background.paper",
              boxShadow: "none",
              "&:before": {
                display: "none",
              },
              "& .MuiAccordionSummary-root": {
                minHeight: "48px",
              },
            }}
          >
            <AccordionSummary
              expandIcon={<ExpandMore />}
              sx={{
                "& .MuiAccordionSummary-content": {
                  margin: "12px 0",
                },
              }}
            >
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Compare sx={{ color: "primary.main" }} />
                <Typography
                  variant="subtitle2"
                  sx={{ color: "primary.main", fontWeight: "bold" }}
                >
                  🎯 Three-Way Model Comparison
                </Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails sx={{ p: 0 }}>
              <Card
                sx={{ mx: 1, mb: 1, backgroundColor: "background.default" }}
              >
                <CardContent sx={{ p: 2 }}>
                  <Typography
                    variant="body2"
                    gutterBottom
                    sx={{ fontWeight: "bold", color: "text.primary" }}
                  >
                    Compare Neural Networks, Classical Models, and Advanced
                    Mathematics
                  </Typography>

                  <Box
                    sx={{
                      mt: 2,
                      display: "flex",
                      flexDirection: "column",
                      gap: 1,
                    }}
                  >
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      <Typography
                        variant="caption"
                        sx={{ minWidth: "80px", color: "primary.main" }}
                      >
                        Neural:
                      </Typography>
                      <Chip
                        label="MSE: 0.0234"
                        size="small"
                        sx={{ backgroundColor: "primary.main", color: "white" }}
                      />
                    </Box>

                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      <Typography
                        variant="caption"
                        sx={{ minWidth: "80px", color: "success.main" }}
                      >
                        Classical:
                      </Typography>
                      <Chip
                        label="MSE: 0.0345"
                        size="small"
                        sx={{ backgroundColor: "success.main", color: "white" }}
                      />
                    </Box>

                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      <Typography
                        variant="caption"
                        sx={{ minWidth: "80px", color: "warning.main" }}
                      >
                        Advanced:
                      </Typography>
                      <Chip
                        label="MSE: 0.0289"
                        size="small"
                        sx={{ backgroundColor: "warning.main", color: "white" }}
                      />
                    </Box>
                  </Box>

                  <Box sx={{ mt: 2, display: "flex", gap: 1 }}>
                    <Typography
                      variant="caption"
                      sx={{ mt: 1, display: "block", color: "text.secondary" }}
                    >
                      Click "Forecasting Comparison" in the navigation menu
                      above for detailed analysis
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </AccordionDetails>
          </Accordion>
        </>
      )}
      <Box sx={{ flexGrow: 1 }} />
      <Box sx={{ p: 2 }}>
        <Typography variant="caption" color="text.secondary" align="center">
          AI-Powered Grid Modeling
          <br />
          Proof of Concept
        </Typography>
      </Box>
    </Drawer>
  );
};

export default Navigation;

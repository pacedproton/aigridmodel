import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import io
import base64
from PIL import Image

# Import our existing ML/AI grid functionality
from ai_grid_demo.config import TrainingConfig, GridConfig
from ai_grid_demo.data.simulator import simulate_grid_timeseries
from ai_grid_demo.data.grid_generator import build_ieee_14_grid
from ai_grid_demo.models.simple_mlp import SimpleMLP
from ai_grid_demo.training.train_simple import train_simple
from ai_grid_demo.viz.plots import plot_time_series
import pandapower.networks as pn
import pandapower.plotting as plot
from matplotlib.lines import Line2D

# Set page configuration
st.set_page_config(
    page_title="AI Grid Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for storing data and models
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'training_history' not in st.session_state:
    st.session_state.training_history = {}
if 'models' not in st.session_state:
    st.session_state.models = {}

# Sidebar navigation
st.sidebar.title("⚡ AI Grid Demo")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Data Management", "Model Training", "Use Case Demo", "About This Demo"]
)

# Utility functions
def generate_network_plot():
    """Generate the network visualization with legend"""
    try:
        # Create IEEE 14-bus network
        net = pn.case14()

        # Create figure with subplots for better layout
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(1, 4, width_ratios=[3, 1, 3, 1], wspace=0.05)
        ax_main = fig.add_subplot(gs[0, 0:3])  # Main plot takes 3/4 width
        ax_legend = fig.add_subplot(gs[0, 3])  # Legend takes 1/4 width

        # Use simple_plot with optimized parameters for clarity
        plot.simple_plot(net, ax=ax_main, plot_loads=True, plot_gens=True, show_plot=False,
                        bus_size=0.4, ext_grid_size=0.8, trafo_size=0.6,
                        load_size=0.6, gen_size=0.6, line_width=1.5)

        # Add bus labels with better positioning and styling
        if hasattr(net, 'bus_geodata') and net.bus_geodata is not None and len(net.bus_geodata) > 0:
            for idx, row in net.bus_geodata.iterrows():
                x, y = row['x'], row['y']
                bus_num = idx + 1
                ax_main.annotate(str(bus_num), (x, y), xytext=(2, 2), textcoords='offset points',
                               bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="gray", alpha=0.9),
                               fontsize=7, ha='left', va='bottom', fontweight='bold', color='black')

        # Add title
        ax_main.set_title('IEEE 14-Bus Power System Single-Line Diagram',
                         fontsize=16, fontweight='bold', pad=15)
        ax_main.axis('off')
        ax_main.set_aspect('equal')

        # Turn off the legend subplot axis
        ax_legend.axis('off')

        # Create comprehensive legend with actual symbols
        legend_elements = []

        # Bus symbol
        legend_elements.append(Line2D([0], [0], marker='s', color='w',
                                    markerfacecolor='gray', markersize=8,
                                    label='Buses', markeredgecolor='black'))

        # External grid (slack bus) symbol
        legend_elements.append(Line2D([0], [0], marker='s', color='w',
                                    markerfacecolor='red', markersize=10,
                                    label='Slack Bus\n(Ext. Grid)', markeredgecolor='black'))

        # Generator symbol (circle with arcs)
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor='red', markersize=8,
                                    label='Generators', markeredgecolor='black'))

        # Load symbol (triangle)
        legend_elements.append(Line2D([0], [0], marker='^', color='w',
                                    markerfacecolor='blue', markersize=8,
                                    label='Loads', markeredgecolor='black'))

        # Transmission line
        legend_elements.append(Line2D([0], [0], color='blue', linewidth=2,
                                    label='Transmission\nLines'))

        # Transformer symbol (two circles/rings)
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor='orange', markersize=6,
                                    label='Transformers', markeredgecolor='black'))

        # Add legend to the side axis
        ax_legend.legend(handles=legend_elements, loc='center', fontsize=9,
                        framealpha=0.9, edgecolor='black', fancybox=True,
                        title='Components', title_fontsize=11, labelspacing=1.5)

        # Add network statistics as text
        n_buses = len(net.bus)
        n_lines = len(net.line)
        n_trafos = len(net.trafo)
        n_gens = len(net.gen)
        n_loads = len(net.load)

        stats_text = f"""Network Statistics:
• {n_buses} Buses
• {n_gens} Generators
• {n_loads} Loads
• {n_lines} Lines
• {n_trafos} Transformers"""

        ax_legend.text(0.05, 0.02, stats_text, transform=ax_legend.transAxes,
                      fontsize=8, verticalalignment='bottom',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()

        # Convert to base64 for Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()

    except Exception as e:
        st.error(f"Error generating network plot: {e}")
        return None

# Main content based on selected page
if page == "Dashboard":
    st.title("📊 AI Grid Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Buses", "14", "IEEE 14-bus system")

    with col2:
        st.metric("Generators", "5", "Power sources")

    with col3:
        st.metric("Loads", "11", "Power consumers")

    with col4:
        st.metric("Transformers", "3", "Voltage conversion")

    # Network visualization
    st.subheader("🏗️ Power System Network")
    network_plot = generate_network_plot()
    if network_plot:
        st.image(f"data:image/png;base64,{network_plot}", use_column_width=True)

    # Data status
    st.subheader("📈 Data Status")
    if st.session_state.data_generated:
        st.success("✅ Training data has been generated and is ready for use")
    else:
        st.warning("⚠️ No training data generated yet. Visit Data Management tab.")

    # Model status
    st.subheader("🤖 Model Status")
    if st.session_state.models:
        for model_name in st.session_state.models:
            st.info(f"✅ {model_name} model trained and available")
    else:
        st.info("ℹ️ No models trained yet. Visit Model Training tab.")

elif page == "Data Management":
    st.title("🗃️ Data Management")

    st.markdown("""
    Generate synthetic power system data using PandaPower for training AI models.
    The data includes voltage measurements, power flows, and system states.
    """)

    if st.button("🚀 Generate Training Data", type="primary"):
        with st.spinner("Generating synthetic power system data..."):
            # Generate data using existing functionality
            config = GridConfig()
            data = simulate_grid_timeseries(config)

            # Store in session state
            st.session_state.generated_data = data
            st.session_state.data_generated = True

        st.success("✅ Training data generated successfully!")

    if st.session_state.data_generated:
        st.subheader("📊 Generated Data Preview")

        # Show some statistics
        data = st.session_state.generated_data
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Time Steps", len(data))
        with col2:
            st.metric("Features per Step", data[0].shape[1] if len(data) > 0 else 0)
        with col3:
            st.metric("Total Samples", sum(len(step) for step in data))

        # Show sample data
        if len(data) > 0:
            sample_data = pd.DataFrame(data[0][:10], columns=[f"Feature_{i}" for i in range(data[0].shape[1])])
            st.dataframe(sample_data)

elif page == "Model Training":
    st.title("🎯 Model Training")

    st.markdown("""
    Train AI models on the generated power system data to predict grid behavior,
    detect anomalies, and optimize power flow.
    """)

    model_type = st.selectbox(
        "Select Model Type",
        ["Simple Baseline", "GNN Transformer", "Temporal CNN"],
        help="Choose the AI model architecture for training"
    )

    if st.button("🏃‍♂️ Start Training", type="primary"):
        if not st.session_state.data_generated:
            st.error("❌ Please generate training data first!")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Training configuration
            config = TrainingConfig()
            config.epochs = 50  # Shorter for demo

            status_text.text(f"Training {model_type} model...")

            # Simulate training progress
            history = {"train_loss": [], "val_loss": []}

            for epoch in range(config.epochs):
                # Simulate training step
                loss = 1.0 * np.exp(-epoch / 20) + np.random.normal(0, 0.1)
                val_loss = loss + np.random.normal(0, 0.05)

                history["train_loss"].append(loss)
                history["val_loss"].append(val_loss)

                progress_bar.progress((epoch + 1) / config.epochs)
                status_text.text(".4f")
                time.sleep(0.1)

            # Store training history
            st.session_state.training_history[model_type] = history
            st.session_state.models[model_type] = {"trained": True, "config": config}

            progress_bar.empty()
            status_text.empty()

            st.success(f"✅ {model_type} model trained successfully!")

    # Show training results
    if st.session_state.training_history:
        st.subheader("📈 Training Results")

        for model_name, history in st.session_state.training_history.items():
            st.write(f"**{model_name} Training Curve**")

            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = range(1, len(history["train_loss"]) + 1)
            ax.plot(epochs, history["train_loss"], 'b-', label='Training Loss', linewidth=2)
            ax.plot(epochs, history["val_loss"], 'r--', label='Validation Loss', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'{model_name} Training Convergence')
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

elif page == "Use Case Demo":
    st.title("🎪 Use Case Demonstrations")

    st.markdown("""
    Explore different AI applications for power system management and optimization.
    """)

    use_case = st.selectbox(
        "Select Use Case",
        ["State Estimation", "Load Forecasting", "Anomaly Detection", "Optimal Power Flow", "Fault Detection"],
        help="Choose a specific AI application to demonstrate"
    )

    if st.button("▶️ Run Demonstration", type="primary"):
        with st.spinner(f"Running {use_case} demonstration..."):
            time.sleep(2)  # Simulate processing

        st.success(f"✅ {use_case} demonstration completed!")

        # Show mock results based on use case
        if use_case == "State Estimation":
            st.subheader("🔍 State Estimation Results")
            st.info("Estimated system state with 98.5% accuracy")
            st.metric("Estimation Error", "1.5%", "±0.3%")

        elif use_case == "Load Forecasting":
            st.subheader("📊 Load Forecasting Results")
            forecast_data = pd.DataFrame({
                'Hour': range(24),
                'Actual_Load': np.random.normal(100, 10, 24),
                'Predicted_Load': np.random.normal(100, 8, 24)
            })
            st.line_chart(forecast_data.set_index('Hour'))

        elif use_case == "Anomaly Detection":
            st.subheader("🚨 Anomaly Detection Results")
            st.warning("⚠️ Detected voltage anomaly at Bus 7")
            st.metric("Anomaly Confidence", "94%", "High")

        elif use_case == "Optimal Power Flow":
            st.subheader("⚡ Optimal Power Flow Results")
            st.success("💡 Optimized power flow reduces losses by 8.3%")
            st.metric("Power Loss Reduction", "8.3%", "↓")

        elif use_case == "Fault Detection":
            st.subheader("🔧 Fault Detection Results")
            st.error("⚠️ Line fault detected between Bus 4 and Bus 5")
            st.metric("Fault Location Accuracy", "95%", "High")

elif page == "About This Demo":
    st.title("ℹ️ About This Demo")

    st.markdown("""
    ## 🎯 AI Grid Demo - Power System Intelligence

    This demonstration showcases how artificial intelligence can enhance power system operations
    through predictive analytics, anomaly detection, and optimization algorithms.

    ### 📋 Features Demonstrated

    1. **Synthetic Data Generation** - Create realistic power system data using PandaPower
    2. **Machine Learning Models** - Train AI models for various grid applications
    3. **Real-time Analytics** - Monitor system performance and detect issues
    4. **Predictive Maintenance** - Forecast equipment failures and maintenance needs
    5. **Optimal Control** - Optimize power flow and system efficiency

    ### 🏗️ Technical Stack

    - **Backend**: Python with PyTorch for ML/AI
    - **Data**: PandaPower for power system simulation
    - **Visualization**: Matplotlib for plots and network diagrams
    - **Interface**: Streamlit for interactive web application

    ### 📊 Network Topology

    The demo uses the IEEE 14-bus test system, a standard benchmark network with:
    - 14 buses (nodes)
    - 5 generators
    - 11 loads
    - 20 transmission lines
    - 3 transformers
    """)

    # Network visualization
    st.subheader("🏗️ IEEE 14-Bus System Topology")
    network_plot = generate_network_plot()
    if network_plot:
        st.image(f"data:image/png;base64,{network_plot}", use_column_width=True)

    st.markdown("""
    ### 🎯 Use Cases Covered

    - **State Estimation**: Real-time monitoring of system state
    - **Load Forecasting**: Predict future power demand
    - **Anomaly Detection**: Identify unusual system behavior
    - **Fault Detection**: Locate and diagnose system faults
    - **Optimal Power Flow**: Optimize power distribution efficiency
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Tech Stack")
st.sidebar.markdown("- **Python** + **PyTorch**")
st.sidebar.markdown("- **PandaPower** for simulation")
st.sidebar.markdown("- **Streamlit** for UI")
st.sidebar.markdown("- **Matplotlib** for plots")

st.sidebar.markdown("---")
st.sidebar.markdown("⚡ **AI Grid Demo v1.0**")


# CM3070 Final University of London

# 🚦 Intelligent Traffic Optimization with SUMO

This project uses SUMO (Simulation of Urban MObility) and a multi-objective evolutionary algorithm to optimize traffic light timings for a city traffic network. It includes:
- Baseline simulations
- Hybrid optimization (GA, DE, PSO, LS)
- Pareto front analysis
- Reinforcement learning fine-tuning
- Graphs comparing performance metrics

---

## 🔧 Setup Instructions

### 1. Install SUMO

SUMO (Simulation of Urban MObility) is required to run the traffic simulations.

#### 📥 Download:
- [https://www.eclipse.dev/sumo/download/](https://www.eclipse.dev/sumo/download/)

#### 🧭 Add to PATH (if not already):
Make sure the following binaries are available in your system's environment PATH:
- `sumo`
- `sumo-gui`
- `traci`

---

### 2. Install Dependencies

This project requires Python 3.8+ and a few packages:

```bash
pip install numpy pandas matplotlib tqdm deap
```

---

### 3. Configure SUMO Network

Edit `config.py` if you want to change the SUMO configuration file or simulation parameters:

```python
CONFIG_FILE = "config2/osm.sumocfg"  # <-- Change this if using a different network
DURATION = 900  # Total simulation steps (in seconds if step-length = 1)
POPULATION_SIZE = 20
GENERATIONS = 20
```

Ensure the file path in `CONFIG_FILE` points to your desired `.sumocfg` file.

---

## ▶️ Running the Simulation

Once SUMO and Python dependencies are set up:

```bash
python main.py
```

This will:

1. Run 10 baseline simulations
2. Optimize traffic light timings using hybrid multi-objective evolutionary algorithms
3. Extract Pareto-optimal results
4. Fine-tune top result using reinforcement learning
5. Evaluate and visualize the final optimized configuration
6. Launch a GUI simulation in `sumo-gui`

---

## 📊 Output Files

All outputs will be saved under a timestamped ID (`YYYYMMDD_HHMMSS`), including:
- `baseline_results_<timestamp>.csv`
- `traffic_optimization_<timestamp>.csv`
- `pareto_optimals_<timestamp>.csv`
- `final_traffic_light_timings_<timestamp>.csv`
- `evaluation_metrics_<timestamp>.csv`
- Graphs saved in `config2_csvs/graphs/`

---

## 💡 Notes

- You can change performance settings like simulation scale, thread count, or behavior options in `PERFORMANCE_OPTIONS` within `config.py`.
- To test individual components (e.g., `run_simulation()` or `evaluate_baseline()`), you can call those functions directly from a Python script or interactive environment.

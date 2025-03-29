from datetime import datetime
import sys
import random
from evaluation import evaluate_baseline
from optimizer import AHMOATrafficOptimizer
from config import POPULATION_SIZE, GENERATIONS, EVALUATION_METRICS, TRAFFIC_OPTIMIZATION_CSV, PARETO_OPTIMALS
from pareto_front import find_pareto_front
import pandas as pd
from rl_op import run_rl_simulation
from simulation import run_simulation
from grapher import graph_pareto_data

if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # --- Baseline Testing ---
    print("Running 10 rounds of baseline testing...")
    for round_num in range(1, 11):
        seed = random.randint(0, 100000)
        print(f"Baseline Round {round_num} with seed {seed}")
        baseline_results = evaluate_baseline(round_num, seed, timestamp)
        print(f"Baseline Round {round_num} results saved in baseline_results_{timestamp}.csv")
    print("Baseline testing complete.\n")

    # --- Optimization Phase ---
    print("Optimizing Traffic Lights, this might take a while...")
    optimizer = AHMOATrafficOptimizer(timestamp)

    try:
        final_population = optimizer.optimize(population_size=POPULATION_SIZE, generations=GENERATIONS)
    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        sys.exit(1)

    # --- Finding Optimal Solution ---
    print("Finding the optimal solutions")
    df = pd.read_csv(TRAFFIC_OPTIMIZATION_CSV + f'{timestamp}.csv')

    pareto_df = find_pareto_front(df)
    pareto_df.to_csv(PARETO_OPTIMALS + f'{timestamp}.csv', index=False)

    print(f"Pareto-optimal solutions saved to {PARETO_OPTIMALS}{timestamp}.csv")

    highest_gen = pareto_df["Generation"].max()
    highest_individual = pareto_df["Individual"].max()

    run_rl_simulation(highest_gen, highest_individual, timestamp)

    run_simulation(timestamp, metrics_csv=EVALUATION_METRICS)

    run_simulation(timestamp, GUI=True)

    graph_pareto_data(timestamp)
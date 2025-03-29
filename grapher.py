import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pareto_front import find_pareto_front
from config import GRAPHS_FOLDER, BASELINE_RESULTS_CSV, TRAFFIC_OPTIMIZATION_CSV, EVALUATION_METRICS

def get_baseline_best(baseline_csv_file):
    df_base = pd.read_csv(baseline_csv_file)
    # Extract the Pareto-optimal rows from the baseline data.
    pareto_base = find_pareto_front(df_base)
    # Average the values for each metric if more than one Pareto row exists.
    baseline_best = {
        "Vehicle_Wait_Time": pareto_base["Vehicle_Wait_Time"].mean(),
        "Emissions": pareto_base["Emissions"].mean(),
        "Ped_Safety": pareto_base["Ped_Safety"].mean(),
        "Total_Cars": pareto_base["Total_Cars"].mean()
    }
    return baseline_best

# Updated graphing function:
def graph_pareto_data(timestamp):
    # Use GRAPHS_FOLDER constant and create a subfolder using the timestamp.
    output_dir = os.path.join(GRAPHS_FOLDER)
    os.makedirs(output_dir, exist_ok=True)
    baseline_csv_file = BASELINE_RESULTS_CSV + f"{timestamp}.csv"
    traffic_csv_file = TRAFFIC_OPTIMIZATION_CSV + f"{timestamp}.csv"
    evaluation_csv_file = EVALUATION_METRICS + f"{timestamp}.csv"

    baseline_best = get_baseline_best(baseline_csv_file)
    
    # Read the traffic optimization CSV (GA metrics over generations)
    df = pd.read_csv(traffic_csv_file)
    df["Generation"] = pd.to_numeric(df["Generation"], errors="coerce")
    df.sort_values(by="Generation", inplace=True)
    
    # Read the evaluation metrics CSV (assumed to have a single row with final data point)
    df_eval = pd.read_csv(evaluation_csv_file)
    
    # Find Pareto-optimal solutions from the GA data.
    pareto_df = find_pareto_front(df)
    
    best_direction = {
        "Vehicle_Wait_Time": "min",
        "Emissions": "min",
        "Ped_Safety": "max",
        "Total_Cars": "max"
    }
    
    metrics = ["Vehicle_Wait_Time", "Emissions", "Ped_Safety", "Total_Cars"]
    
    # Create individual metric plots.
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        # Group by Generation and extract the best (min or max) per generation.
        if best_direction[metric] == "min":
            group = pareto_df.groupby("Generation")[metric].min().reset_index()
        else:
            group = pareto_df.groupby("Generation")[metric].max().reset_index()
        
        # Plot the Pareto best values per generation.
        plt.plot(group["Generation"], group[metric],
                 marker='o', linestyle='-', color='red', linewidth=2,
                 label="Pareto Best per Generation")
        
        # Plot the baseline best as a horizontal dotted line.
        best_value = baseline_best[metric]
        plt.axhline(y=best_value, linestyle='--', color='black', linewidth=1.5,
                    label="Baseline Best")
        
        # Add an extra dot at the end for the baseline final point.
        last_gen = group["Generation"].iloc[-1]
        
        # Add an extra dot for the evaluation final point (from evaluation CSV).
        eval_value = df_eval[metric].iloc[0]
        plt.plot(last_gen, eval_value,
                 marker='o', linestyle='None', color='cyan', markersize=8,
                 label="Evaluation Final Point")
        
        plt.title(f"{metric} vs. Generation (Pareto)")
        plt.xlabel("Generation")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, f"{metric}_pareto_.png")
        plt.savefig(file_path)
        plt.close()
    
    # Create a combined normalized Pareto plot.
    plt.figure(figsize=(8, 6))
    color_map = {
        "Vehicle_Wait_Time": "blue",
        "Emissions": "green",
        "Ped_Safety": "orange",
        "Total_Cars": "purple"
    }
    
    for metric in metrics:
        # Use the overall min/max from the traffic data for normalization.
        min_val = df[metric].min()
        max_val = df[metric].max()
    
        if best_direction[metric] == "min":
            group = pareto_df.groupby("Generation")[metric].min()
        else:
            group = pareto_df.groupby("Generation")[metric].max()
    
        group_norm = (group - min_val) / (max_val - min_val)
        plt.plot(group.index, group_norm, marker='o', linestyle='-',
                 color=color_map[metric], linewidth=2,
                 label=f"{metric} Pareto Best")
    
        baseline_value = baseline_best[metric]
        baseline_norm = (baseline_value - min_val) / (max_val - min_val)
        plt.axhline(y=baseline_norm, color=color_map[metric], linestyle='--',
                    linewidth=1.5, label=f"{metric} Baseline Best")
    
        
        # Add the extra dot for the normalized evaluation final point.
        eval_value = df_eval[metric].iloc[0]
        eval_norm = (eval_value - min_val) / (max_val - min_val)
        plt.plot(last_gen, eval_norm,
                 marker='o', linestyle='None', color='cyan', markersize=8,
                 label=f"{metric} Evaluation Final Point")
    
    plt.title("Normalized Metrics vs. Generation (Pareto)")
    plt.xlabel("Generation")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    file_path = os.path.join(output_dir, f"normalized_pareto.png")
    plt.savefig(file_path)
    plt.close()
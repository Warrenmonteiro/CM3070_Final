import os
import csv
from deap import creator
import math
import copy
from config import TRAFFIC_LIGHT_CONFIGS
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

def calculate_weather(t):
    return 0.8 + 0.4 * math.sin(2 * math.pi * t)


def clone_individual_with_new_index(individual, next_index_func):
    """Clone an individual with a new index"""
    new_ind = creator.Individual(individual[:])  # Create new individual with same genes
    new_ind.fitness = copy.deepcopy(individual.fitness)
    new_ind.index = next_index_func()  # Assign new index
    if hasattr(individual, 'strategy'):
        new_ind.strategy = individual.strategy
    if hasattr(individual, 'best'):
        new_ind.best = list(individual.best)
        new_ind.best_fitness = copy.deepcopy(individual.best_fitness)
    return new_ind


def log_traffic_light_configuration(individual, generation, individual_idx, seed, phase_info, timestamp):
    """
    Logs: [Generation, IndividualIndex, Seed, Phase_0_ID, Phase_0, Phase_1_ID, Phase_1, ...]
    """
    filename = TRAFFIC_LIGHT_CONFIGS + f"{timestamp}.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline='') as f:
        writer = csv.writer(f)

        # Write header only if file does not exist
        if not file_exists:
            header = ["Generation", "IndividualIndex", "Seed"]
            for idx, (tls_id, phase_idx) in enumerate(phase_info):
                header.append(f"Phase_{idx}_ID")
                header.append(f"Phase_{idx}")
            writer.writerow(header)

        row = [generation, individual_idx, seed]
        for idx, duration in enumerate(individual):
            (tls_id, phase_idx) = phase_info[idx]
            row.append(tls_id)
            row.append(duration)

        writer.writerow(row)

# -----------------------------
# CSV Loading Functions
# -----------------------------
def load_phases_and_seed(gen, ind_idx, phase_csv):
    # Load optimization metrics (note: Individual column is used)

    # Load phase configuration (note: CSV uses "IndividualIndex")
    phases_df = pd.read_csv(phase_csv)
    timing_row = phases_df[(phases_df['Generation'] == gen) & (phases_df['IndividualIndex'] == ind_idx)].iloc[0]

    seed = int(timing_row['Seed'])

    config_dict = {}
    phase_order = []  # This will record (tls_id, local_phase_index) for each phase as they appear.
    i = 0
    while f'Phase_{i}_ID' in timing_row and f'Phase_{i}' in timing_row:
        tls_id = str(timing_row[f'Phase_{i}_ID'])
        phase_duration = int(timing_row[f'Phase_{i}'])
        # Record the order: the phase index for this tls_id is the current count in config_dict[tls_id]
        if tls_id not in config_dict:
            config_dict[tls_id] = []
            local_phase_idx = 0
        else:
            local_phase_idx = len(config_dict[tls_id])
        phase_order.append((tls_id, local_phase_idx))
        config_dict[tls_id].append(phase_duration)
        i += 1

    logging.info(f"Loaded phase configuration for {len(config_dict)} traffic lights.")
    # Return phase_order along with the rest
    return config_dict, seed, phase_order

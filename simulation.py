from utils import calculate_weather
import traci
import random
import numpy as np
import logging
import pandas as pd
import numpy as np
from config import SUMO_CMD, DURATION, FINAL_TRAFFIC_LIGHT_TIMINGS, EVALUATION_METRICS, SUMO_CMD_GUI


def simulate_traffic(duration, fixed_timings=None):
    """
    Run the traffic simulation loop and compute evaluation metrics.

    Parameters:
        duration (int): Total simulation steps (DURATION).
        fixed_timings (dict, optional): A dictionary mapping each traffic light ID 
            to a list of phase durations. If provided, the fixed timings are applied 
            at every simulation step.

    Returns:
        tuple: (avg_vehicle_wait, avg_emissions, ped_safety, vehicles_completed)
    """
    total_vehicle_wait_time = 0
    total_emissions = 0
    total_ped_conflicts = 0.0
    vehicles_completed = 0

    processed_vehicles = set()
    unpredictable_drivers = set()

    for step in range(duration):
        current_time = step
        weather_coef = calculate_weather(current_time / duration)

        # If fixed timings are provided, set each traffic light's phase duration
        if fixed_timings:
            for tls_id, timings in fixed_timings.items():
                current_phase = traci.trafficlight.getPhase(tls_id)
                if current_phase < len(timings):
                    traci.trafficlight.setPhaseDuration(tls_id, timings[current_phase])

        traci.simulationStep()

        current_vehicles = set(traci.vehicle.getIDList())
        new_vehicles = current_vehicles - processed_vehicles

        # Introduce random driver behavior for new vehicles
        for v in new_vehicles:
            processed_vehicles.add(v)
            if random.random() < 0.01:
                unpredictable_drivers.add(v)
                traci.vehicle.setSpeedFactor(v, random.uniform(0.7, 1.3))

        # Apply unpredictable behavior for flagged drivers
        for v in list(unpredictable_drivers):
            if v in current_vehicles:
                if random.random() < 0.2:
                    action = random.choice(['speed_boost', 'sudden_brake', 'ignore_priority'])
                    if action == 'speed_boost':
                        traci.vehicle.setSpeed(v, traci.vehicle.getSpeed(v) * 1.5)
                    elif action == 'sudden_brake':
                        traci.vehicle.setSpeed(v, max(0, traci.vehicle.getSpeed(v) * 0.3))
                    elif action == 'ignore_priority':
                        traci.vehicle.setImperfection(v, 0.8)
                        traci.vehicle.setSpeed(v, traci.vehicle.getSpeed(v) * 1.2)
            else:
                unpredictable_drivers.discard(v)

        # Periodically measure wait times, emissions, and pedestrian conflicts
        if step % 50 == 0:
            vehicles = traci.vehicle.getIDList()
            pedestrians = traci.person.getIDList()

            if vehicles:
                total_vehicle_wait_time += sum(traci.vehicle.getWaitingTime(v) for v in vehicles) * weather_coef
                total_emissions += sum(traci.vehicle.getCO2Emission(v) for v in vehicles) * weather_coef

            if pedestrians and vehicles:
                try:
                    vehicle_positions = []
                    valid_vehicles = []
                    for v in vehicles:
                        try:
                            pos = traci.vehicle.getPosition(v)
                            vehicle_positions.append(pos)
                            valid_vehicles.append(v)
                        except Exception:
                            continue

                    ped_positions = []
                    for p in pedestrians:
                        try:
                            ped_positions.append(traci.person.getPosition(p))
                        except Exception:
                            continue

                    if vehicle_positions and ped_positions:
                        vehicle_positions = np.array(vehicle_positions)
                        ped_positions = np.array(ped_positions)

                        dx = ped_positions[:, None, 0] - vehicle_positions[None, :, 0]
                        dy = ped_positions[:, None, 1] - vehicle_positions[None, :, 1]
                        distances = np.sqrt(dx**2 + dy**2)

                        veh_speeds = np.array([traci.vehicle.getSpeed(v) for v in valid_vehicles])

                        conflict_mask = distances < 5.0
                        severity = veh_speeds[None, :] / 10.0
                        proximity = (5.0 - distances) / 5.0
                        total_ped_conflicts += np.sum(severity * proximity * conflict_mask)
                except Exception:
                    pass

        if traci.simulation.getMinExpectedNumber() == 0:
            break

        vehicles_completed += len(traci.simulation.getArrivedIDList())

    num_vehicles = max(vehicles_completed, 1)
    avg_vehicle_wait = total_vehicle_wait_time / num_vehicles
    avg_emissions = total_emissions / num_vehicles
    ped_safety = 1.0 if total_ped_conflicts <= 0 else (1 / (1 + total_ped_conflicts))

    return avg_vehicle_wait, avg_emissions, ped_safety, vehicles_completed

# -----------------------------
# Evaluation Simulation: Use fixed timings from CSV to measure performance
# -----------------------------
def run_simulation(timestamp, GUI=False, metrics_csv=None):
    fixed_timings = {}
    timings_csv = FINAL_TRAFFIC_LIGHT_TIMINGS + f"{timestamp}.csv"
    if metrics_csv:
        metrics_csv = EVALUATION_METRICS + f"{timestamp}.csv"
    df = pd.read_csv(timings_csv)
    
    # Extract the seed from the CSV and set it for reproducibility.
    seed = int(df.iloc[0]['Seed'])
    random.seed(seed)
    
    # Identify phase columns like "Phase_0_ID", "Phase_1_ID", etc.
    phase_columns = [col for col in df.columns if col.startswith("Phase_") and col.endswith("_ID")]
    
    for _, row in df.iterrows():
        for phase_col in phase_columns:
            # Extract phase index from the column name (e.g., "Phase_0_ID" -> 0)
            phase_index = int(phase_col.split('_')[1])
            tls_id = str(row[phase_col])
            duration = int(row[f"Phase_{phase_index}"])
            if tls_id not in fixed_timings:
                fixed_timings[tls_id] = []
            fixed_timings[tls_id].append((phase_index, duration))
    
    # Ensure that for each traffic light, the phase timings are in order.
    for tls_id, phase_list in fixed_timings.items():
        phase_list.sort(key=lambda x: x[0])
        fixed_timings[tls_id] = [d for (_, d) in phase_list]
    
    
    # Use SUMO_CMD_GUI if GUI is True, otherwise use SUMO_CMD.
    if GUI:
        sumoCmd = SUMO_CMD_GUI
    else:
        logging.info(f"Loaded fixed timings from CSV: {fixed_timings}")
        sumoCmd = SUMO_CMD

    # Start the SUMO simulation.
    traci.start(sumoCmd)
    
    # For non-GUI mode, run simulation and perform metric calculations.

    duration = DURATION  # total simulation steps for evaluation
    
    # Run the simulation loop using simulate_traffic with fixed timings.
    avg_vehicle_wait, avg_emissions, ped_safety, vehicles_completed = simulate_traffic(duration, fixed_timings=fixed_timings)

    traci.close()

    if not GUI and metrics_csv:
        metrics = {
            'Vehicle_Wait_Time': avg_vehicle_wait,
            'Emissions': avg_emissions,
            'Ped_Safety': ped_safety,
            'Total_Cars': vehicles_completed
        }
        # Save the evaluation metrics to a CSV file.
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(metrics_csv, index=False)
        logging.info(f"Evaluation metrics saved to {metrics_csv}")
        logging.info(f"Evaluation simulation metrics: {metrics}")
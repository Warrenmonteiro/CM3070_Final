import traci
import random
from config import SUMO_CMD, DURATION, BASELINE_RESULTS_CSV
import os
import csv
from simulation import simulate_traffic

def evaluate_individual(individual, phase_info=None, seed=None, generation=0, individual_idx=0):
    """Evaluate a single individual with random seed and occasional random lane closures"""
    try:
        if seed is not None:
            random.seed(seed)

        current_cmd = SUMO_CMD + (["--seed", str(seed)] if seed is not None else [])
        traci.start(current_cmd)

        phase_index = 0
        tls_list = traci.trafficlight.getIDList()

        # Set the traffic light phases according to the individual's genome
        for tls in tls_list:
            logic = traci.trafficlight.getAllProgramLogics(tls)[0]
            phases = []
            for i, phase in enumerate(logic.phases):
                phase_duration = max(5, int(round(individual[phase_index + i])))
                phases.append(traci.trafficlight.Phase(
                    phase_duration, phase.state, 5, 60
                ))
            new_logic = traci.trafficlight.Logic(
                programID=logic.programID,
                type=logic.type,
                currentPhaseIndex=0,
                phases=phases
            )
            traci.trafficlight.setProgramLogic(tls, new_logic)
            phase_index += len(logic.phases)

        avg_vehicle_wait, avg_emissions, ped_safety, vehicles_completed = simulate_traffic(DURATION)

        traci.close()

        config_data = {
            'generation': generation,
            'individual_idx': individual_idx,
            'seed': seed,
            'individual': individual,
            'phase_info': phase_info
        }
        return (avg_vehicle_wait, avg_emissions, ped_safety, vehicles_completed), config_data

    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return (float('inf'), float('inf'), float('inf'), 0), None
    finally:
        try:
            traci.close()
        except:
            pass


def evaluate_baseline(round_num, seed, timestamp):
    """
    Run a baseline simulation without modifying the traffic-light configuration.
    The results are logged to a CSV file named "baseline_results_<timestamp>.csv".
    """
    try:
        if seed is not None:
            random.seed(seed)

        current_cmd = SUMO_CMD + (["--seed", str(seed)] if seed is not None else [])
        traci.start(current_cmd)

        avg_vehicle_wait, avg_emissions, ped_safety, vehicles_completed = simulate_traffic(DURATION)

        traci.close()

        # Log the baseline results
        baseline_filename = BASELINE_RESULTS_CSV + f"{timestamp}.csv"
        file_exists = os.path.isfile(baseline_filename)
        with open(baseline_filename, "a", newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Round", "Seed", "Vehicle_Wait_Time", "Emissions", "Ped_Safety", "Total_Cars"])
            writer.writerow([round_num, seed, avg_vehicle_wait, avg_emissions, ped_safety, vehicles_completed])

        return (avg_vehicle_wait, avg_emissions, ped_safety, vehicles_completed)

    except Exception as e:
        print(f"Baseline simulation error: {str(e)}")
        return (float('inf'), float('inf'), float('inf'), 0)
    finally:
        try:
            traci.close()
        except:
            pass
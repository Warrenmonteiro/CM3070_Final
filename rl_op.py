import traci
import random
import logging
import numpy as np
import csv
from config import SUMO_CMD, DURATION, TRAFFIC_LIGHT_CONFIGS, FINAL_TRAFFIC_LIGHT_TIMINGS
from utils import calculate_weather, load_phases_and_seed
from tqdm import tqdm


# -----------------------------
# Logging configuration (for non-simulation logs)
# -----------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# -----------------------------
# RL Agent for Traffic Light Control
# -----------------------------
class TrafficLightRLAgent:
    def __init__(self, tl_id, initial_phases):
        self.tl_id = tl_id
        self.phases = initial_phases.copy()
        self.num_phases = len(self.phases)
        self.delta_options = [-5, 0, 5]
        self.action_space = [(i, delta) for i in range(self.num_phases) for delta in self.delta_options]
        self.q_table = {}
        self.epsilon = 0.1
        self.alpha = 0.5
        self.gamma = 0.9
        self.last_state = None
        self.last_action = None

    def get_state(self):
        current_phase = traci.trafficlight.getPhase(self.tl_id)
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        total_queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes)
        current_time = traci.simulation.getTime()
        weather = calculate_weather(current_time / 1000.0)
        ped_wait = sum(traci.person.getWaitingTime(p) for p in traci.person.getIDList())
        return (current_phase, total_queue, round(weather, 2), round(ped_wait, 2))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            q_values = [self.q_table.get((state, a), 0) for a in self.action_space]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(self.action_space, q_values) if q == max_q]
            action = random.choice(best_actions)
        return action

    def update_q_value(self, state, action, reward, next_state):
        old_value = self.q_table.get((state, action), 0)
        next_max = max([self.q_table.get((next_state, a), 0) for a in self.action_space])
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[(state, action)] = new_value

    def apply_action(self, action):
        phase_index, delta = action
        old_duration = self.phases[phase_index]
        new_duration = max(5, old_duration + delta)
        new_duration = min(new_duration, 120)
        self.phases[phase_index] = new_duration

    def update_simulation(self):
        current_phase = traci.trafficlight.getPhase(self.tl_id)
        if current_phase < self.num_phases:
            new_duration = self.phases[current_phase]
            traci.trafficlight.setPhaseDuration(self.tl_id, new_duration)

# -----------------------------
# RL Simulation: Adjust timings and save final configuration
# -----------------------------
def run_rl_simulation(generation, individual, timestamp):
    phase_csv = TRAFFIC_LIGHT_CONFIGS + f"{timestamp}.csv"
    generation = generation
    individual_idx = individual

    config_dict, seed, phase_order = load_phases_and_seed(generation, individual_idx, phase_csv)
    random.seed(seed)
    traci.start(SUMO_CMD)
    print("Running RL Simulation, this won't take as long...")
    duration = DURATION
    agents = {tls_id: TrafficLightRLAgent(tls_id, phases) for tls_id, phases in config_dict.items()}
    
    processed_vehicles = set()
    unpredictable_drivers = set()
    
    total_vehicle_wait_time = 0.0
    total_emissions = 0.0
    total_ped_conflicts = 0.0
    vehicles_completed = 0
    
    with tqdm(total=duration, desc="RL Simulation Progress") as pbar:
        for step in range(duration):
            current_time = step
            weather_coef = calculate_weather(current_time / duration)
            traci.simulationStep()
            
            # RL agents update their actions.
            for tls_id, agent in agents.items():
                current_state = agent.get_state()
                action = agent.choose_action(current_state)
                agent.apply_action(action)
                current_phase = traci.trafficlight.getPhase(tls_id)
                if action[0] == current_phase:
                    traci.trafficlight.setPhaseDuration(tls_id, agent.phases[current_phase])
                agent.update_simulation()
                agent.last_state = current_state
                agent.last_action = action
            
            # Unpredictable driver behavior.
            current_vehicles = set(traci.vehicle.getIDList())
            new_vehicles = current_vehicles - processed_vehicles
            for v in new_vehicles:
                processed_vehicles.add(v)
                if random.random() < 0.01:
                    unpredictable_drivers.add(v)
                    traci.vehicle.setSpeedFactor(v, random.uniform(0.7, 1.3))
            for v in list(unpredictable_drivers):
                if v in current_vehicles:
                    if random.random() < 0.2:
                        behavior = random.choice(['speed_boost', 'sudden_brake', 'ignore_priority'])
                        current_speed = traci.vehicle.getSpeed(v)
                        if behavior == 'speed_boost':
                            new_speed = current_speed * 1.5
                            traci.vehicle.setSpeed(v, new_speed)
                        elif behavior == 'sudden_brake':
                            new_speed = max(0, current_speed * 0.3)
                            traci.vehicle.setSpeed(v, new_speed)
                        elif behavior == 'ignore_priority':
                            traci.vehicle.setImperfection(v, 0.8)
                            new_speed = current_speed * 1.2
                            traci.vehicle.setSpeed(v, new_speed)
                else:
                    unpredictable_drivers.discard(v)
            
            # Periodic performance measurement.
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
                            except:
                                continue
                        ped_positions = []
                        for p in pedestrians:
                            try:
                                ped_positions.append(traci.person.getPosition(p))
                            except:
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
                    except Exception as e:
                        logging.error(f"Error in pedestrian conflict calculation: {e}")
                
                vehicles_completed += len(traci.simulation.getArrivedIDList())
                num_vehicles = max(vehicles_completed, 1)
                avg_vehicle_wait = total_vehicle_wait_time / num_vehicles
                avg_emissions = total_emissions / num_vehicles
                ped_safety = 1.0 if total_ped_conflicts <= 0 else (1 / (1 + total_ped_conflicts))
                
                # RL Q-value update.
                reward = -avg_vehicle_wait - avg_emissions + vehicles_completed + ped_safety
                for tls_id, agent in agents.items():
                    if agent.last_state is not None:
                        next_state = agent.get_state()
                        agent.update_q_value(agent.last_state, agent.last_action, reward, next_state)
            
            # Update the progress bar after each simulation step.
            pbar.update(1)
            pbar.set_postfix_str(f"Vehicles: {vehicles_completed}")
            
            if traci.simulation.getMinExpectedNumber() == 0:
                break

    traci.close()
    logging.info("RL Simulation ended.")

    # Save final traffic light timings to CSV in the desired format.
    final_timings_file = FINAL_TRAFFIC_LIGHT_TIMINGS + f"{timestamp}.csv"
    with open(final_timings_file, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Generation", "IndividualIndex", "Seed"]
        for i in range(len(phase_order)):
            header.extend([f"Phase_{i}_ID", f"Phase_{i}"])
        writer.writerow(header)
        row = [generation, individual_idx, seed]
        for tls_id, phase_idx in phase_order:
            final_duration = agents[tls_id].phases[phase_idx]
            row.extend([tls_id, final_duration])
        writer.writerow(row)
        
    logging.info(f"Final traffic light timings saved to CSV: {final_timings_file}")


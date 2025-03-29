import random
import logging
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from deap import base, creator, tools, algorithms
from tqdm import tqdm
import traci
from config import SUMO_CMD, POPULATION_SIZE, GENERATIONS, TRAFFIC_OPTIMIZATION_CSV
from evaluation import evaluate_individual
from utils import log_traffic_light_configuration, clone_individual_with_new_index

class AHMOATrafficOptimizer:
    def __init__(self, timestamp):
        self.timestamp = timestamp
        random.seed()
        self.next_individual_index = 0

        # Algorithm configuration
        self.initial_strategies = ['GA', 'DE', 'PSO', 'LS']
        self.active_strategies = self.initial_strategies.copy()
        self.eliminated_strategies = []
        self.strategy_success = {s: 0 for s in self.initial_strategies}
        self.strategy_usage = {s: 0 for s in self.initial_strategies}
        self.current_phase = 0
        self.phase_thresholds = [0.25, 0.50, 0.75]

        # Logging setup
        self.log_file = TRAFFIC_OPTIMIZATION_CSV + f'{self.timestamp}.csv'
        self.setup_logging()
        self.setup_csv_files()

        # DEAP setup
        self.setup_deap()
        # Will store best across all population
        self.global_best = None

    def get_next_index(self):
        """Get next available index and increment counter"""
        current = self.next_individual_index
        self.next_individual_index += 1
        return current

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('AHMOA')
        # Clear existing handlers
        self.logger.handlers = []
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def setup_csv_files(self):
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Generation',
                'Individual',
                'Strategy',
                'Vehicle_Wait_Time',
                'Emissions',
                'Ped_Safety',
                'Total_Cars'
            ])

    def log_results(self, generation, individual_idx, results, strategy='GA'):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                generation,
                individual_idx,
                strategy,
                *results
            ])

    def setup_deap(self):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_int", random.randint, 5, 60)
        
        def uniform_mutation(individual, low=5, up=60, indpb=0.3):
            """Random integer mutation within ±10 seconds"""
            for i in range(len(individual)):
                if random.random() < indpb:
                    delta = random.randint(-10, 10)
                    individual[i] = max(low, min(up, individual[i] + delta))
            return (individual,)

        traci.start(SUMO_CMD + ["--seed", str(random.randint(0, 100000))])
        self.phase_info = []
        for tls in traci.trafficlight.getIDList():
            logic = traci.trafficlight.getAllProgramLogics(tls)[0]
            for i, phase in enumerate(logic.phases):
                self.phase_info.append((tls, i))
        traci.close()

        total_phases = len(self.phase_info)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_int, n=total_phases)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", uniform_mutation, low=5, up=60, indpb=0.3)

        ref_points = tools.uniform_reference_points(nobj=4, p=12)
        self.toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    def determine_phase(self, generation, total_generations):
        progress = generation / total_generations
        if progress < self.phase_thresholds[0]:
            return 0
        elif progress < self.phase_thresholds[1]:
            return 1
        elif progress < self.phase_thresholds[2]:
            return 2
        else:
            return 3

    def update_strategies(self, generation, total_generations):
        """
        Updated strategy management with population rebalancing
        """
        new_phase = self.determine_phase(generation, total_generations)
        if new_phase != self.current_phase:
            self.current_phase = new_phase
            target_strategies = 4 - self.current_phase
            
            while len(self.active_strategies) > target_strategies:
                # Calculate success rates
                success_rates = {}
                for strategy in self.active_strategies:
                    used = self.strategy_usage[strategy]
                    success_rates[strategy] = self.strategy_success[strategy] / used if used > 0 else 0
                
                # Eliminate worst strategy
                worst_strategy = min(success_rates, key=success_rates.get)
                self.eliminated_strategies.append(worst_strategy)
                self.active_strategies.remove(worst_strategy)
                self.logger.info(f"\nPhase {self.current_phase+1}: Eliminated {worst_strategy} strategy")
                
                # Reset usage counters for remaining strategies
                self.strategy_usage = {s: 0 for s in self.active_strategies}
                self.strategy_success = {s: 0 for s in self.active_strategies}

    def distribute_individuals(self, population):
        """
        Distributes individuals across active strategies evenly, ensuring no index reuse
        """
        distribution = {s: [] for s in self.active_strategies}
        num_strategies = len(self.active_strategies)
        
        # Calculate how many individuals each strategy should get
        individuals_per_strategy = len(population) // num_strategies
        remainder = len(population) % num_strategies
        
        # Distribute individuals evenly
        current_position = 0
        for strategy in self.active_strategies:
            # Add extra individual from remainder if available
            strategy_count = individuals_per_strategy + (1 if remainder > 0 else 0)
            remainder = max(0, remainder - 1)
            
            # Assign individuals to this strategy
            for _ in range(strategy_count):
                if current_position < len(population):
                    individual = population[current_position]
                    distribution[strategy].append(individual)
                    individual.strategy = strategy
                    if not hasattr(individual, 'index'):
                        individual.index = self.get_next_index()
                    current_position += 1
        
        return distribution

    def apply_strategy_operations(self, strategy_group, strategy):
        if strategy == 'GA':
            offspring = algorithms.varAnd(strategy_group, self.toolbox, cxpb=0.7, mutpb=0.3)
            return [clone_individual_with_new_index(ind, self.get_next_index) for ind in offspring]

        elif strategy == 'DE':
            offspring = []
            for _ in range(len(strategy_group)):
                a, b, c = random.sample(strategy_group, 3)
                mutant = [
                    min(60, max(5, int(a[i] + 0.8 * (b[i] - c[i]))))
                    for i in range(len(a))
                ]
                trial = [
                    mutant[i] if random.random() < 0.7 else strategy_group[0][i]
                    for i in range(len(mutant))
                ]
                new_ind = creator.Individual(trial)
                new_ind.index = self.get_next_index()
                new_ind.strategy = strategy
                offspring.append(new_ind)
            return offspring

        elif strategy == 'PSO':
            if not hasattr(self, 'pso_velocity'):
                self.pso_velocity = {}
            new_pop = []
            for ind in strategy_group:
                if id(ind) not in self.pso_velocity:
                    self.pso_velocity[id(ind)] = [random.uniform(-1, 1) for _ in ind]
                if not hasattr(ind, 'best'):
                    ind.best = list(ind)
                    ind.best_fitness = creator.FitnessMulti((float('inf'), float('inf'), float('-inf'), float('-inf')))
                w = 0.7  
                c1 = 2.0
                c2 = 2.0
                velocity = []
                for i, v in enumerate(self.pso_velocity[id(ind)]):
                    r1, r2 = random.random(), random.random()
                    inertia = w * v
                    personal = c1 * r1 * (ind.best[i] - ind[i])
                    global_ = c2 * r2 * (self.global_best[i] - ind[i])
                    new_v = inertia + personal + global_
                    velocity.append(new_v)
                new_genome = []
                for i, gene in enumerate(ind):
                    candidate = gene + velocity[i]
                    candidate = max(5, min(60, candidate))
                    new_genome.append(int(candidate))
                new_ind = creator.Individual(new_genome)
                new_ind.index = self.get_next_index()
                new_ind.strategy = strategy
                new_pop.append(new_ind)
                self.pso_velocity[id(new_ind)] = velocity
            return new_pop

        elif strategy == 'LS':
            new_group = []
            for ind in strategy_group:
                mutant, = self.toolbox.mutate(ind)
                new_ind = creator.Individual(mutant)
                new_ind.index = self.get_next_index()
                new_ind.strategy = strategy
                new_group.append(new_ind)
            return new_group

        return strategy_group

    def optimize(self, population_size=POPULATION_SIZE, generations=GENERATIONS):
        population = self.toolbox.population(n=population_size)

        for ind in population:
            ind.index = self.get_next_index()

        # Initialize global best with something large
        self.global_best = creator.Individual([30]*len(population[0]))
        self.global_best.fitness.values = (float('inf'), float('inf'), float('-inf'), float('-inf'))
        self.global_best.index = -1 

        with tqdm(total=generations, desc="Optimization Progress") as pbar, \
             ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:

            for gen in range(generations):
                # Phase-based updates
                self.update_strategies(gen, generations)

                # Distribute individuals across strategies
                strategy_distribution = self.distribute_individuals(population)

                # Evaluate population in parallel
                futures_dict = {}  # Map future to (index, individual)
                for strategy, inds in strategy_distribution.items():
                    for ind in inds:
                        seed = random.randint(0, 100000)
                        future = executor.submit(
                            evaluate_individual,
                            individual=ind,
                            phase_info=self.phase_info,
                            seed=seed,
                            generation=gen,
                            individual_idx=ind.index
                        )
                        futures_dict[future] = (ind.index, ind)

                # Wait for all futures to complete
                for future in as_completed(futures_dict.keys()):
                    result, config_data = future.result()
                    if result is None:
                        continue

                    ind_idx, ind = futures_dict[future]
                    fit = result

                    # Update individual's fitness
                    ind.fitness.values = fit

                    # Possibly update personal best
                    if not hasattr(ind, 'best'):
                        ind.best = list(ind)
                        ind.best_fitness = creator.FitnessMulti(ind.fitness.values)
                    else:
                        if ind.fitness > ind.best_fitness:
                            ind.best = list(ind)
                            ind.best_fitness = creator.FitnessMulti(ind.fitness.values)

                    # Track usage
                    self.strategy_usage[ind.strategy] += 1

                    # Check global best (this is simplified to check the first objective)
                    if fit[0] < self.global_best.fitness.values[0]:
                        self.strategy_success[ind.strategy] += 1
                        self.global_best = creator.Individual(ind)
                        self.global_best.fitness.values = fit

                    # Logging
                    self.log_results(gen, ind_idx, fit, ind.strategy)
                    if config_data:
                        log_traffic_light_configuration(
                            config_data['individual'],
                            config_data['generation'],
                            config_data['individual_idx'],
                            config_data['seed'],
                            config_data['phase_info'],
                            self.timestamp
                        )

                # Offspring generation
                offspring = []
                for strategy in self.active_strategies:
                    strategy_group = strategy_distribution[strategy]
                    strategy_offspring = self.apply_strategy_operations(strategy_group, strategy)
                    
                    # Assign new indices to offspring
                    for child in strategy_offspring:
                        child.index = self.get_next_index()
                        child.strategy = strategy
                    
                    offspring.extend(strategy_offspring)

                # Evaluate offspring with their new indices
                offspring_futures_dict = {}  # Map future to (index, individual)
                for child in offspring:
                    seed = random.randint(0, 100000)
                    future = executor.submit(
                        evaluate_individual,
                        individual=child,
                        phase_info=self.phase_info,
                        seed=seed,
                        generation=gen,
                        individual_idx=child.index
                    )
                    offspring_futures_dict[future] = (child.index, child)

                # Wait for all offspring futures to complete
                for future in as_completed(offspring_futures_dict.keys()):
                    result, config_data = future.result()
                    if result is None:
                        continue

                    idx, child = offspring_futures_dict[future]
                    fit = result
                    child.fitness.values = fit

                    # Possibly update personal best
                    if not hasattr(child, 'best'):
                        child.best = list(child)
                        child.best_fitness = creator.FitnessMulti(child.fitness.values)
                    else:
                        if child.fitness > child.best_fitness:
                            child.best = list(child)
                            child.best_fitness = creator.FitnessMulti(child.fitness.values)

                    # Log
                    self.log_results(gen, idx, fit, child.strategy)
                    if config_data:
                        log_traffic_light_configuration(
                            config_data['individual'],
                            config_data['generation'],
                            config_data['individual_idx'],
                            config_data['seed'],
                            config_data['phase_info'],
                            self.timestamp
                        )

                combined = population + offspring
                selected = self.toolbox.select(combined, population_size)
                # Create new population with proper index handling
                population[:] = [clone_individual_with_new_index(ind, self.get_next_index) for ind in selected]

                pbar.update(1)
                pbar.set_postfix_str(f"Strategies: {', '.join(self.active_strategies)}")

        return population

import random
from deap import base, creator, tools, algorithms

# Define objectives
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))       # 1. Maximizing PDR, 2. Minimizing redundancy
creator.create("Individual", list, fitness=creator.FitnessMulti)

#-------------------Genetic algorithm parameters
POPULATION_SIZE = 100
GENERATIONS = 50
PROB_CROSSOVER = 0.8
PROB_MUTATION = 0.2
NUM_NODES = 16  # Number of nodes in the network

# Adjacency matrix representing the network topology
adjacency_matrix = [
   # 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 1
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 2
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 3
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 4
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 5
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # Node 6
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Node 7
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 8
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Node 9		
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Node 10
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],  # Node 11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # Node 12
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # Node 13
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],  # Node 14
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # Node 15
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]   # Node 16
]

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def dfs(forwarding_probabilities, node, vector):
    if node == NUM_NODES - 1:		# DST node (Destination node)
        return 1
    num=0
    for i in range(NUM_NODES):
        if adjacency_matrix[node][i] == 1 and vector[i] == 0 and random.uniform(0.0, 1.0) < forwarding_probabilities[i]:
            temp = vector.copy()
            temp[i] = 1
            num += dfs(forwarding_probabilities, i, temp)
    return num

# Function to calculate Packet Delivery Ratio (PDR) and redundancy
def evaluate(forwarding_probabilities):
    delivered_packets = 0.0
    success = 0.0
    for i in range(100):
        vector = [0]*NUM_NODES
        vector[0] = 1
        temp=dfs(forwarding_probabilities, 0, vector)	# SRC node (Source node)
        delivered_packets+=temp
        if temp >= 1:
            success+=1
    
    pdr = success/100.00
    redundancy = 0
    if success != 0:
        redundancy = (delivered_packets-success)/success
    return pdr, redundancy



def rndm():
    return random.uniform(0.27, 1)

#-------------------Genetic algorithm initialization
toolbox = base.Toolbox()
toolbox.register("attr_float", rndm)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=NUM_NODES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=2)

#-------------------Main evolution loop
population = toolbox.population(n=POPULATION_SIZE)
for gen in range(GENERATIONS):
    offspring = algorithms.varAnd(population, toolbox, cxpb=PROB_CROSSOVER, mutpb=PROB_MUTATION)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
        ind[:] = [clamp(val, 0.27, 1) for val in ind]
    population = toolbox.select(population + offspring, k=POPULATION_SIZE)
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    print(pareto_front[0])
    final_pdr, final_redundancy = evaluate(pareto_front[0])
    print("Final PDR:", final_pdr)
    print("Final Redundancy:", final_redundancy)

# Select best solutions from final population
pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

print("Final Forwarding Probabilities:", pareto_front[0])

final_pdr, final_redundancy = evaluate(pareto_front[0])
print("Final PDR:", final_pdr)
print("Final Redundancy:", final_redundancy)

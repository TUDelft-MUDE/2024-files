import pyvinecopulib as pv

# Define the structure of the vine copula
structure = pv.RVineStructure.simulate(9)

# Define pair copulas for each edge in the vine
pair_copulas = [
    pv.Bicop(family=pv.BicopFamily.gaussian, parameters=[0.5])
    for _ in range(36)  # 36 is the number of pair copulas for a 9-dimensional vine copula
]
print(pair_copulas)
# Create the vine copula model
vine_copula = pv.RVineCopula(structure, pair_copulas)
# Simulate data from the vine copula model
simulated_data = vine_copula.simulate(1000)

# Print the first few rows of the simulated data
print(simulated_data[:5])
# Print the vine copula structure
print(vine_copula)
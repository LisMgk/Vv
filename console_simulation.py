# console_simulation.py - запускается даже на слабых устройствах
from simulation_core import AsphalteneSimulation
import matplotlib.pyplot as plt

sim = AsphalteneSimulation(n_particles=100, mode='CCA')
for i in range(500):
    sim.step()
    if i % 50 == 0:
        print(f"Шаг {i}, кластеров: {len(sim.clusters)}")

positions, colors = sim.get_cluster_positions()
xs = [p[0] for p in positions]
ys = [p[1] for p in positions]

plt.figure(figsize=(8,8))
plt.scatter(xs, ys, s=5, alpha=0.7)
plt.title(f"Фрактальная размерность: {sim.get_fractal_dimension():.3f}")
plt.axis('equal')
plt.show()

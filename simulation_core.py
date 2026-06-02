import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Particle:
    x: float
    y: float
    radius: float = 1.0
    fixed: bool = False
    charge: float = 0.0
    cluster_id: int = -1
    
@dataclass
class Cluster:
    particles: List[Particle] = field(default_factory=list)
    center_of_mass: Tuple[float, float] = (0, 0)
    radius_of_gyration: float = 0.0
    charge: float = 0.0
    
    def update_center_of_mass(self):
        if not self.particles:
            return
        total_mass = len(self.particles)
        cx = sum(p.x for p in self.particles) / total_mass
        cy = sum(p.y for p in self.particles) / total_mass
        self.center_of_mass = (cx, cy)
        self.radius_of_gyration = np.sqrt(
            sum((p.x - cx)**2 + (p.y - cy)**2 for p in self.particles) / total_mass
        )
    
    def update_charge(self):
        screening = np.exp(-len(self.particles) / 50)
        self.charge = sum(p.charge for p in self.particles) * screening


class AsphalteneSimulation:
    def __init__(self, 
                 width: int = 500,
                 height: int = 500,
                 n_particles: int = 100,
                 temperature: float = 300.0,
                 pressure: float = 1.0,
                 solvent_solubility: float = 0.5,
                 charge_strength: float = 0.0,
                 viscosity: float = 1.0,
                 mode: str = 'CCA',
                 aggregation_type: str = 'sticky',
                 seed: int = 42):
        
        self.width = width
        self.height = height
        self.n_particles = n_particles
        self.temperature = temperature
        self.pressure = pressure
        self.solvent_solubility = solvent_solubility
        self.charge_strength = charge_strength
        self.viscosity = viscosity
        self.mode = mode
        self.aggregation_type = aggregation_type
        self.seed = seed
        
        np.random.seed(seed)
        
        self.diffusion_rate = self._calculate_diffusion_rate()
        self.aggregation_rate = self._calculate_aggregation_rate()
        
        self.clusters = []
        self.free_particles = []
        
        self.history = {
            'n_clusters': [],
            'avg_radius': [],
            'max_radius': [],
            'fractal_dimensions': [],
            'time': []
        }
        
        self.time_step = 0
        self._init_simulation()
    
    def _calculate_diffusion_rate(self) -> float:
        k = 1.38e-23
        r = 1e-9
        D = (k * self.temperature) / (6 * np.pi * self.viscosity * r)
        return min(D * 1e10, 0.5)
    
    def _calculate_aggregation_rate(self) -> float:
        if self.solvent_solubility < 0.3:
            return 0.9
        elif self.solvent_solubility > 0.7:
            return 0.1
        else:
            return 0.5
    
    def _init_simulation(self):
        if self.mode == 'DLA':
            seed_particle = Particle(x=self.width/2, y=self.height/2, fixed=True, cluster_id=0)
            seed_cluster = Cluster(particles=[seed_particle])
            seed_cluster.update_center_of_mass()
            self.clusters.append(seed_cluster)
            
            for i in range(self.n_particles - 1):
                particle = self._random_free_particle()
                self.free_particles.append(particle)
        else:
            for i in range(self.n_particles):
                x = np.random.uniform(50, self.width - 50)
                y = np.random.uniform(50, self.height - 50)
                charge = np.random.uniform(-self.charge_strength, self.charge_strength)
                particle = Particle(x, y, fixed=False, cluster_id=i, charge=charge)
                cluster = Cluster(particles=[particle], charge=charge)
                cluster.update_center_of_mass()
                self.clusters.append(cluster)
    
    def _random_free_particle(self) -> Particle:
        side = np.random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            x = np.random.uniform(0, self.width); y = 0
        elif side == 'bottom':
            x = np.random.uniform(0, self.width); y = self.height
        elif side == 'left':
            x = 0; y = np.random.uniform(0, self.height)
        else:
            x = self.width; y = np.random.uniform(0, self.height)
        return Particle(x, y, fixed=False)
    
    def _check_collision(self, p1: Particle, p2: Particle) -> bool:
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dist = np.sqrt(dx**2 + dy**2)
        return dist < (p1.radius + p2.radius)
    
    def _check_cluster_collision(self, cluster1: Cluster, cluster2: Cluster) -> Tuple[bool, Optional[Tuple[Particle, Particle]]]:
        for p1 in cluster1.particles:
            for p2 in cluster2.particles:
                if self._check_collision(p1, p2):
                    return True, (p1, p2)
        return False, None
    
    def _electrostatic_force(self, p1: Particle, p2: Particle) -> Tuple[float, float]:
        if self.charge_strength == 0:
            return 0, 0
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        r = np.sqrt(dx**2 + dy**2) + 0.001
        debye_length = 10.0
        screening = np.exp(-r / debye_length)
        force_magnitude = p1.charge * p2.charge * screening / (r**2)
        force_magnitude = np.clip(force_magnitude, -5, 5)
        fx = force_magnitude * dx / r
        fy = force_magnitude * dy / r
        return fx, fy
    
    def _move_particle(self, particle: Particle, cluster_owner: Cluster = None):
        if particle.fixed:
            return
        
        step = self.diffusion_rate * (1 + 0.1 * np.random.randn())
        angle = np.random.uniform(0, 2 * np.pi)
        dx = step * np.cos(angle)
        dy = step * np.sin(angle)
        
        for other in self.clusters:
            if cluster_owner and other is cluster_owner:
                continue
            for other_particle in other.particles:
                fx, fy = self._electrostatic_force(particle, other_particle)
                dx += fx * 0.01
                dy += fy * 0.01
        
        if particle.x + dx < particle.radius:
            dx = particle.radius - particle.x + 0.1
        elif particle.x + dx > self.width - particle.radius:
            dx = self.width - particle.radius - particle.x - 0.1
        if particle.y + dy < particle.radius:
            dy = particle.radius - particle.y + 0.1
        elif particle.y + dy > self.height - particle.radius:
            dy = self.height - particle.radius - particle.y - 0.1
        
        particle.x += dx
        particle.y += dy
    
    def _move_cluster(self, cluster: Cluster):
        if self.aggregation_type == 'viscous':
            mobility = 1.0 / (1 + len(cluster.particles) * 0.05)
        else:
            mobility = 1.0
        
        step = self.diffusion_rate * mobility
        angle = np.random.uniform(0, 2 * np.pi)
        dx = step * np.cos(angle)
        dy = step * np.sin(angle)
        
        min_x = min(p.x for p in cluster.particles)
        max_x = max(p.x for p in cluster.particles)
        min_y = min(p.y for p in cluster.particles)
        max_y = max(p.y for p in cluster.particles)
        
        if min_x + dx < 0:
            dx = -min_x
        if max_x + dx > self.width:
            dx = self.width - max_x
        if min_y + dy < 0:
            dy = -min_y
        if max_y + dy > self.height:
            dy = self.height - max_y
        
        for particle in cluster.particles:
            particle.x += dx
            particle.y += dy
        
        cluster.update_center_of_mass()
    
    def _merge_clusters(self, cluster1: Cluster, cluster2: Cluster):
        cluster1.particles.extend(cluster2.particles)
        cluster1.charge += cluster2.charge
        cluster1.update_center_of_mass()
        cluster1.update_charge()
        self.clusters.remove(cluster2)
    
    def step_dla(self):
        if not self.free_particles:
            return False
        
        idx = np.random.randint(len(self.free_particles))
        particle = self.free_particles[idx]
        self._move_particle(particle)
        
        for cluster in self.clusters:
            for fixed_particle in cluster.particles:
                if self._check_collision(particle, fixed_particle):
                    if np.random.random() < self.aggregation_rate:
                        particle.fixed = True
                        particle.cluster_id = self.clusters[0].particles[0].cluster_id
                        self.clusters[0].particles.append(particle)
                        self.clusters[0].update_center_of_mass()
                        self.free_particles.pop(idx)
                        return True
                    else:
                        if self.aggregation_type == 'charged':
                            angle = np.random.uniform(0, 2 * np.pi)
                            particle.x += 5 * np.cos(angle)
                            particle.y += 5 * np.sin(angle)
                    break
        return True
    
    def step_cca(self):
        if len(self.clusters) <= 1:
            return False
        
        sizes = [len(c.particles) for c in self.clusters]
        probs = np.array(sizes) / sum(sizes)
        idx = np.random.choice(len(self.clusters), p=probs)
        moving_cluster = self.clusters[idx]
        
        self._move_cluster(moving_cluster)
        
        for other in self.clusters:
            if other is moving_cluster:
                continue
            collided, _ = self._check_cluster_collision(moving_cluster, other)
            if collided:
                if np.random.random() < self.aggregation_rate:
                    self._merge_clusters(moving_cluster, other)
                    return True
                else:
                    if self.aggregation_type == 'charged':
                        dx = moving_cluster.center_of_mass[0] - other.center_of_mass[0]
                        dy = moving_cluster.center_of_mass[1] - other.center_of_mass[1]
                        dist = np.sqrt(dx**2 + dy**2) + 0.001
                        push = 2.0 / dist
                        for p in moving_cluster.particles:
                            p.x += push * dx / dist
                            p.y += push * dy / dist
                        moving_cluster.update_center_of_mass()
                    break
        return True
    
    def step(self):
        if self.mode == 'DLA':
            result = self.step_dla()
        else:
            result = self.step_cca()
        
        self.time_step += 1
        
        if self.time_step % 10 == 0:
            self._update_history()
        return result
    
    def _update_history(self):
        self.history['n_clusters'].append(len(self.clusters))
        if self.clusters:
            avg_r = np.mean([c.radius_of_gyration for c in self.clusters])
            max_r = max([c.radius_of_gyration for c in self.clusters])
            self.history['avg_radius'].append(avg_r)
            self.history['max_radius'].append(max_r)
            
            if len(self.clusters) > 1:
                sizes = [len(c.particles) for c in self.clusters]
                radii = [c.radius_of_gyration for c in self.clusters]
                if len(sizes) > 5:
                    log_sizes = np.log(sizes)
                    log_radii = np.log(radii)
                    slope, _ = np.polyfit(log_sizes, log_radii, 1)
                    self.history['fractal_dimensions'].append(slope)
        
        self.history['time'].append(self.time_step)
    
    def get_fractal_dimension(self) -> float:
        if len(self.history['fractal_dimensions']) > 0:
            return np.mean(self.history['fractal_dimensions'][-10:])
        return 0
    
    def get_cluster_positions(self):
        positions = []
        colors = []
        for cluster in self.clusters:
            for particle in cluster.particles:
                positions.append((particle.x, particle.y))
                size_norm = min(len(cluster.particles) / 100, 1)
                colors.append((0.2, 0.4 + size_norm * 0.5, 0.6))
        return positions, colors

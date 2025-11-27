import networkx as nx
import random
import numpy as np
import matplotlib

matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from itertools import islice
import copy

# --- CONFIGURATION ---
NUM_REQUESTS_RANGE = range(10, 150, 20)  # X-axis: 10 to 140 requests
K_PATHS = 5
USER_DELAY_LIMIT = 200

SERVICE_TYPES = {
    1: {'resource': 600, 'type': 'Large_HighBW'},
    2: {'resource': 400, 'type': 'Large_LowDelay'},
    3: {'resource': 500, 'type': 'Small_HighBW'},
    4: {'resource': 400, 'type': 'Small_LowDelay'},
    5: {'resource': 500, 'type': 'HighBW_LowDelay'},
    6: {'resource': 300, 'type': 'LowBW_LowDelay'}
}

class SAGIN_Network:
    def __init__(self):
        self.substrate = nx.Graph()
        self.total_bw_capacity = 0
        self.total_cpu_capacity = 0
        self.used_bw = 0
        self.used_cpu = 0
        self.total_energy_consumed = 0  # new metric for energy consumption
        self._build_topology()

    def _build_topology(self):
        """
        Builds SAGIN topology with added ENERGY COSTS for the improvement.
        Ground = Low Energy (Grid), Air/Space = High Energy (Battery/Solar).
        """
        node_cpu = 3000
        link_bw = 2000 
        d_ground = 5
        d_sat = 15
        d_inter = 10

        # energy costs (arbitrary)
        EC_LOW = 1
        EC_HIGH = 10

        # nodes
        for i in range(1, 9):
            # ground: low energy
            self.substrate.add_node(f"G{i}", layer="ground", cpu=node_cpu, max_cpu=node_cpu, delay=d_ground, energy=EC_LOW)
            # air: high energy
            self.substrate.add_node(f"L{i}", layer="air", cpu=node_cpu, max_cpu=node_cpu, delay=d_sat, energy=EC_HIGH)
        for i in range(1, 5):
            # space: high energy
            self.substrate.add_node(f"H{i}", layer="space", cpu=node_cpu, max_cpu=node_cpu, delay=d_sat, energy=EC_HIGH)

        # links
        for i in range(1, 9):
            nxt = i + 1 if i < 8 else 1
            self.substrate.add_edge(f"G{i}", f"G{nxt}", bw=link_bw, max_bw=link_bw, delay=d_ground)
            self.substrate.add_edge(f"L{i}", f"L{nxt}", bw=link_bw, max_bw=link_bw, delay=d_inter)
            self.substrate.add_edge(f"G{i}", f"L{i}", bw=link_bw, max_bw=link_bw, delay=d_inter)
            
        for i in range(1, 9):
            h_node = f"H{((i-1)//2)+1}"
            self.substrate.add_edge(f"L{i}", h_node, bw=link_bw, max_bw=link_bw, delay=d_inter)

        self.total_cpu_capacity = sum(d['max_cpu'] for n, d in self.substrate.nodes(data=True))
        self.total_bw_capacity = sum(d['max_bw'] for u, v, d in self.substrate.edges(data=True))

    def calculate_path_metrics(self, path):
        """Calculates Delay AND Energy for a given path."""
        delay = 0
        energy = 0
        
        # node metrics
        for node in path:
            delay += self.substrate.nodes[node]['delay']
            energy += self.substrate.nodes[node]['energy'] # summing energy cost of active nodes
            
        # link delay
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            delay += self.substrate[u][v]['delay']
            
        return delay, energy

    def check_resources(self, path, bw_demand, cpu_demand):
        for node in path:
            if self.substrate.nodes[node]['cpu'] < cpu_demand:
                return False
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if self.substrate[u][v]['bw'] < bw_demand:
                return False
        return True

    def consume_resources(self, path, bw_demand, cpu_demand, energy_cost):
        for node in path:
            self.substrate.nodes[node]['cpu'] -= cpu_demand
            self.used_cpu += cpu_demand
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            self.substrate[u][v]['bw'] -= bw_demand
            self.used_bw += bw_demand
        
        # track total energy for the metric graphs
        self.total_energy_consumed += energy_cost

    # --- 1. PROPOSED (Original Paper) ---
    def map_proposed(self, req, q_index):
        try:
            candidates = list(islice(nx.shortest_simple_paths(self.substrate, req['source'], req['dest'], weight='delay'), K_PATHS))
        except nx.NetworkXNoPath:
            return False, 0

        best_path = None
        min_delay = float('inf')
        
        for path in candidates:
            if random.random() < 0.2: continue 
                
            if self.check_resources(path, req['bw'], req['service']['resource']):
                d, e = self.calculate_path_metrics(path)
                if d < min_delay:
                    min_delay = d
                    best_path = (path, e)

        if best_path and min_delay <= req['user_delay']:
            path, energy = best_path
            self.consume_resources(path, req['bw'], req['service']['resource'], energy)
            return True, min_delay
        return False, 0

    # --- 2. IMPROVEMENT (Energy-Aware) ---
    def map_improved_energy(self, req, q_index):
        """
        IMPROVED ALGORITHM:
        Minimizes Cost = Delay + (alpha * Energy).
        Preserves battery life of Satellites/UAVs.
        """
        # custom weight function for ksp
        def energy_delay_weight(u, v, d):
            # weight is link delay + energy cost of the destination node
            # this makes the algo avoid high-energy nodes (Satellites) if a Ground path exists
            node_energy = self.substrate.nodes[v]['energy']
            return d.get('delay', 1) + (5.0 * node_energy) # alpha = 5.0 (energy penalty)

        try:
            # find paths that are "cheapest" in terms of delay and energy
            candidates = list(islice(nx.shortest_simple_paths(self.substrate, req['source'], req['dest'], weight=energy_delay_weight), K_PATHS))
        except nx.NetworkXNoPath:
            return False, 0

        best_path = None
        best_score = float('inf')
        final_delay = 0
        final_energy = 0

        for path in candidates:
            if self.check_resources(path, req['bw'], req['service']['resource']):
                d, e = self.calculate_path_metrics(path)
                
                score = d + (5.0 * e) 
                
                if score < best_score:
                    best_score = score
                    best_path = path
                    final_delay = d
                    final_energy = e

        if best_path and final_delay <= req['user_delay']:
            self.consume_resources(best_path, req['bw'], req['service']['resource'], final_energy)
            return True, final_delay
        return False, 0

    # --- 3. SPO (Comparative) ---
    def map_spo(self, req):
        try:
            path = nx.shortest_path(self.substrate, req['source'], req['dest'], weight='delay')
            d, e = self.calculate_path_metrics(path)
            if d <= req['user_delay'] and self.check_resources(path, req['bw'], req['service']['resource']):
                self.consume_resources(path, req['bw'], req['service']['resource'], e)
                return True, d
        except: pass
        return False, 0

    # --- 4. CLF (Comparative) ---
    def map_clf(self, req):
        def lb_weight(u, v, d): return 1000 / (d['bw'] + 1)
        try:
            path = nx.shortest_path(self.substrate, req['source'], req['dest'], weight=lb_weight)
            d, e = self.calculate_path_metrics(path)
            if d <= req['user_delay'] and self.check_resources(path, req['bw'], req['service']['resource']):
                self.consume_resources(path, req['bw'], req['service']['resource'], e)
                return True, d
        except: pass
        return False, 0

    # --- 5. DMRT (Comparative) ---
    def map_dmrt(self, req):
        try:
            path = nx.shortest_path(self.substrate, req['source'], req['dest']) # Min Hop
            d, e = self.calculate_path_metrics(path)
            if d <= req['user_delay'] and self.check_resources(path, req['bw'], req['service']['resource']):
                self.consume_resources(path, req['bw'], req['service']['resource'], e)
                return True, d
        except: pass
        return False, 0

# --- SIMULATION ENGINE ---

def run_comparison():
    data = {
        'Proposed': {'acc': [], 'delay': [], 'energy': []},
        'Improved': {'acc': [], 'delay': [], 'energy': []}, 
        'SPO': {'acc': [], 'delay': [], 'energy': []},
        'CLF': {'acc': [], 'delay': [], 'energy': []},
        'DMRT': {'acc': [], 'delay': [], 'energy': []}
    }
    
    x_axis = list(NUM_REQUESTS_RANGE)

    for n_req in x_axis:
        print(f"Simulating Batch: {n_req} Requests...")
        batch_reqs = []
        for _ in range(n_req):
            s_id = random.randint(1, 6)
            s_node, d_node = f"G{random.randint(1, 8)}", f"G{random.randint(1, 8)}"
            while s_node == d_node: d_node = f"G{random.randint(1, 8)}"
            batch_reqs.append({
                'source': s_node, 'dest': d_node,
                'bw': random.uniform(10, 100),
                'user_delay': USER_DELAY_LIMIT,
                'service': SERVICE_TYPES[s_id]
            })
        
        nets = {name: SAGIN_Network() for name in data.keys()}
        results = {name: [] for name in data.keys()}

        for i, req in enumerate(batch_reqs):
            results['Proposed'].append(nets['Proposed'].map_proposed(req, i+1))
            results['Improved'].append(nets['Improved'].map_improved_energy(req, i+1))
            results['SPO'].append(nets['SPO'].map_spo(req))
            results['CLF'].append(nets['CLF'].map_clf(req))
            results['DMRT'].append(nets['DMRT'].map_dmrt(req))

        for alg, res_list in results.items():
            accepted = sum(1 for r in res_list if r[0])
            delays = [r[1] for r in res_list if r[0]]
            avg_delay = sum(delays)/len(delays) if delays else 0
            
            data[alg]['acc'].append(accepted / n_req)
            data[alg]['delay'].append(avg_delay)
            avg_energy = nets[alg].total_energy_consumed / accepted if accepted > 0 else 0
            data[alg]['energy'].append(avg_energy)

    return x_axis, data

def plot_all(x, data):
    styles = {'Proposed': 'b-o', 'Improved': 'm-*', 'SPO': 'r-s', 'CLF': 'g-^', 'DMRT': 'c-d'}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ax = axes[0]
    for alg, style in styles.items():
        ax.plot(x, data[alg]['acc'], style, label=alg)
    ax.set_title('Service Acceptance Rate')
    ax.set_ylabel('Rate')
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    for alg, style in styles.items():
        ax.plot(x, data[alg]['delay'], style, label=alg)
    ax.set_title('Avg Path Delay (ms)')
    ax.set_ylabel('Delay')
    ax.legend()
    ax.grid(True)

    ax = axes[2]
    for alg, style in styles.items():
        ax.plot(x, data[alg]['energy'], style, label=alg)
    ax.set_title('Avg Energy Consumption (Units)')
    ax.set_ylabel('Energy')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('sagin_improved_results.png')
    print("Success! Graph saved to 'sagin_improved_results.png'")

if __name__ == "__main__":
    x, res = run_comparison()
    plot_all(x, res)
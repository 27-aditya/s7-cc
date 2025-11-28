import networkx as nx
import random
import matplotlib.pyplot as plt
from itertools import islice

import matplotlib
matplotlib.use('Agg') 

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
        self._build_topology()

    def _build_topology(self):
        node_cpu = 3000
        link_bw = 2000 
        
        # delays (ms)
        d_ground = 5
        d_sat = 15
        d_inter = 10

        # nodes
        for i in range(1, 9):
            self.substrate.add_node(f"G{i}", layer="ground", cpu=node_cpu, max_cpu=node_cpu, delay=d_ground)
            self.substrate.add_node(f"L{i}", layer="air", cpu=node_cpu, max_cpu=node_cpu, delay=d_sat)
        for i in range(1, 5):
            self.substrate.add_node(f"H{i}", layer="space", cpu=node_cpu, max_cpu=node_cpu, delay=d_sat)

        # links 
        for i in range(1, 9):
            nxt = i + 1 if i < 8 else 1
            # ground
            self.substrate.add_edge(f"G{i}", f"G{nxt}", bw=link_bw, max_bw=link_bw, delay=d_ground)
            # air
            self.substrate.add_edge(f"L{i}", f"L{nxt}", bw=link_bw, max_bw=link_bw, delay=d_inter)
            # ground-air
            self.substrate.add_edge(f"G{i}", f"L{i}", bw=link_bw, max_bw=link_bw, delay=d_inter)
            
        # connections
        for i in range(1, 9):
            h_node = f"H{((i-1)//2)+1}"
            self.substrate.add_edge(f"L{i}", h_node, bw=link_bw, max_bw=link_bw, delay=d_inter)

        # percentage metrics
        self.total_cpu_capacity = sum(d['max_cpu'] for n, d in self.substrate.nodes(data=True))
        self.total_bw_capacity = sum(d['max_bw'] for u, v, d in self.substrate.edges(data=True))

    def calculate_path_delay(self, path):
        delay = 0
        # node processing delay
        for node in path:
            delay += self.substrate.nodes[node]['delay']
        # link transmission delay
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            delay += self.substrate[u][v]['delay']
        return delay

    def check_resources(self, path, bw_demand, cpu_demand):
        # check nodes
        for node in path:
            if self.substrate.nodes[node]['cpu'] < cpu_demand:
                return False
        # check links
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if self.substrate[u][v]['bw'] < bw_demand:
                return False
        return True

    def consume_resources(self, path, bw_demand, cpu_demand):
        # deduct CPU
        for node in path:
            self.substrate.nodes[node]['cpu'] -= cpu_demand
            self.used_cpu += cpu_demand
        # deduct BW
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            self.substrate[u][v]['bw'] -= bw_demand
            self.used_bw += bw_demand

    # --- ALGORITHM 1: Proposed (Time Delay Mapping) ---
    def map_proposed(self, req):
        # K-Shortest Paths based on Delay
        try:
            candidates = list(islice(nx.shortest_simple_paths(self.substrate, req['source'], req['dest'], weight='delay'), K_PATHS))
        except nx.NetworkXNoPath:
            return False, 0

        best_path = None
        min_delay = float('inf')
        threshold = 0.5 

        for path in candidates:
            # 1. prediction filter 
            if random.random() < 0.2: # 20% chance a path is predicted "unstable"
                continue
                
            # 2. resource check
            if self.check_resources(path, req['bw'], req['service']['resource']):
                d = self.calculate_path_delay(path)
                if d < min_delay:
                    min_delay = d
                    best_path = path

        if best_path and min_delay <= req['user_delay']:
            self.consume_resources(best_path, req['bw'], req['service']['resource'])
            return True, min_delay
        return False, 0

    # --- ALGORITHM 2: SPO (Shortest Path Optimization) ---
    def map_spo(self, req):
        # strictly minimizes delay using dijkstra
        try:
            path = nx.shortest_path(self.substrate, req['source'], req['dest'], weight='delay')
            d = self.calculate_path_delay(path)
            
            if d <= req['user_delay'] and self.check_resources(path, req['bw'], req['service']['resource']):
                self.consume_resources(path, req['bw'], req['service']['resource'])
                return True, d
        except nx.NetworkXNoPath:
            pass
        return False, 0

    # --- ALGORITHM 3: CLF (Cross-Layer Fusion / Load Balance) ---
    def map_clf(self, req):
        # heuristic: Favor paths with higher available bandwidth (load balancing)
        # custom weight: 1 / (available_bw + epsilon)
        def lb_weight(u, v, d):
            return 1000 / (d['bw'] + 1)

        try:
            path = nx.shortest_path(self.substrate, req['source'], req['dest'], weight=lb_weight)
            d = self.calculate_path_delay(path)
            
            if d <= req['user_delay'] and self.check_resources(path, req['bw'], req['service']['resource']):
                self.consume_resources(path, req['bw'], req['service']['resource'])
                return True, d
        except nx.NetworkXNoPath:
            pass
        return False, 0

    # --- ALGORITHM 4: DMRT-SL (Deep Multi-Agent - Proxy) ---
    def map_dmrt(self, req):
        # proxy : min-hop path selection
        # mimicking the "high resource consumption" described in the paper.
        try:
            path = nx.shortest_path(self.substrate, req['source'], req['dest']) # No weight = Min Hops
            d = self.calculate_path_delay(path)
            
            if d <= req['user_delay'] and self.check_resources(path, req['bw'], req['service']['resource']):
                self.consume_resources(path, req['bw'], req['service']['resource'])
                return True, d
        except nx.NetworkXNoPath:
            pass
        return False, 0


# --- SIMULATION ENGINE ---

def generate_requests(num):
    reqs = []
    for _ in range(num):
        s_id = random.randint(1, 6)
        reqs.append({
            'source': f"G{random.randint(1, 8)}",
            'dest': f"G{random.randint(1, 8)}",
            'bw': random.uniform(10, 100),
            'user_delay': USER_DELAY_LIMIT,
            'service': SERVICE_TYPES[s_id]
        })
        # ensure src != dest
        while reqs[-1]['source'] == reqs[-1]['dest']:
            reqs[-1]['dest'] = f"G{random.randint(1, 8)}"
    return reqs

def run_comparison():
    # metrics storage
    data = {
        'Proposed': {'acc': [], 'delay': [], 'cpu': [], 'link': []},
        'SPO': {'acc': [], 'delay': [], 'cpu': [], 'link': []},
        'CLF': {'acc': [], 'delay': [], 'cpu': [], 'link': []},
        'DMRT': {'acc': [], 'delay': [], 'cpu': [], 'link': []}
    }
    
    x_axis = list(NUM_REQUESTS_RANGE)

    for n_req in x_axis:
        print(f"Simulating Batch: {n_req} Requests...")
        batch_reqs = generate_requests(n_req)
        
        # instantiate 4 identical networks for fairness
        nets = {
            'Proposed': SAGIN_Network(),
            'SPO': SAGIN_Network(),
            'CLF': SAGIN_Network(),
            'DMRT': SAGIN_Network()
        }

        # run algorithms
        results_cache = {'Proposed': [], 'SPO': [], 'CLF': [], 'DMRT': []}
        
        for i, req in enumerate(batch_reqs):
            # 1. Proposed
            res = nets['Proposed'].map_proposed(req, i+1)
            results_cache['Proposed'].append(res)
            
            # 2. SPO
            res = nets['SPO'].map_spo(req)
            results_cache['SPO'].append(res)
            
            # 3. CLF
            res = nets['CLF'].map_clf(req)
            results_cache['CLF'].append(res)
            
            # 4. DMRT
            res = nets['DMRT'].map_dmrt(req)
            results_cache['DMRT'].append(res)

        # calculate metrics for this batch
        for alg_name in data.keys():
            res_list = results_cache[alg_name]
            accepted = sum(1 for r in res_list if r[0])
            delays = [r[1] for r in res_list if r[0]]
            avg_delay = sum(delays)/len(delays) if delays else 0
            
            net = nets[alg_name]
            link_util = (net.used_bw / net.total_bw_capacity) * 100
            cpu_util = (net.used_cpu / net.total_cpu_capacity) * 100
            acc_rate = accepted / n_req

            data[alg_name]['acc'].append(acc_rate)
            data[alg_name]['delay'].append(avg_delay)
            data[alg_name]['link'].append(link_util)
            data[alg_name]['cpu'].append(cpu_util)

    return x_axis, data

def plot_all(x, data):
    # setup styles
    styles = {'Proposed': 'b-o', 'SPO': 'r-s', 'CLF': 'g-^', 'DMRT': 'c-d'}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. CPU utilization
    ax = axes[0, 0]
    for alg, style in styles.items():
        ax.plot(x, data[alg]['cpu'], style, label=alg)
    ax.set_title('CPU Resource Utilization')
    ax.set_ylabel('Utilization (%)')
    ax.grid(True)
    ax.legend()

    # 2. link utilization
    ax = axes[0, 1]
    for alg, style in styles.items():
        ax.plot(x, data[alg]['link'], style, label=alg)
    ax.set_title('Link Resource Utilization')
    ax.set_ylabel('Utilization (%)')
    ax.grid(True)
    ax.legend()

    # 3. deployment delay
    ax = axes[1, 0]
    for alg, style in styles.items():
        ax.plot(x, data[alg]['delay'], style, label=alg)
    ax.set_title('Avg Deployment Delay')
    ax.set_ylabel('Delay (ms)')
    ax.grid(True)
    ax.legend()

    # 4. service acceptance 
    ax = axes[1, 1]
    for alg, style in styles.items():
        ax.plot(x, data[alg]['acc'], style, label=alg)
    ax.set_title('Service Acceptance Rate')
    ax.set_ylabel('Rate (0-1)')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig('sagin_comparison_results.png')
    print("Graphs saved to 'sagin_comparison_results.png'")

if __name__ == "__main__":
    x, res = run_comparison()
    plot_all(x, res)
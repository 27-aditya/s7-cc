import networkx as nx
import random
from itertools import islice

# --- Constants and Configuration ---
K_PATHS = 5  # Number of shortest paths to consider as candidates
NUM_SFC_REQUESTS = 10 # Number of SFC requests to simulate

# --- 1. Network and Request Modeling ---

def create_sagin_topology():
    """
    Creates the physical SAGIN topology based on the paper's simulation setup.
    The network is represented as a NetworkX graph. Nodes and edges have attributes
    for total and available resources, and delay.
    """
    G = nx.Graph()

    # Node properties: (cpu_capacity, processing_delay)
    node_specs = {
        'HEO': (200, 15),  # High CPU, high delay
        'MEO_LEO': (150, 8), # Medium CPU, medium delay
        'Ground': (300, 3)   # Highest CPU, lowest delay
    }

    # Add nodes to the graph
    # 4 HEO satellite nodes
    for i in range(4):
        node_id = f"HEO_{i}"
        G.add_node(node_id, type='HEO',
                   total_cpu=node_specs['HEO'][0],
                   available_cpu=node_specs['HEO'][0],
                   processing_delay=node_specs['HEO'][1])

    # 8 MEO/LEO satellite nodes
    for i in range(8):
        node_id = f"MEO_LEO_{i}"
        G.add_node(node_id, type='MEO_LEO',
                   total_cpu=node_specs['MEO_LEO'][0],
                   available_cpu=node_specs['MEO_LEO'][0],
                   processing_delay=node_specs['MEO_LEO'][1])

    # 8 Ground gateway nodes
    for i in range(8):
        node_id = f"Ground_{i}"
        G.add_node(node_id, type='Ground',
                   total_cpu=node_specs['Ground'][0],
                   available_cpu=node_specs['Ground'][0],
                   processing_delay=node_specs['Ground'][1])

    # Add edges with attributes: (bandwidth, transmission_delay)
    # This is a simplified connection topology for demonstration
    # Inter-satellite links (ISLs)
    for i in range(4):
        G.add_edge(f"HEO_{i}", f"MEO_LEO_{i}", total_bw=1000, available_bw=1000, delay=20)
        G.add_edge(f"HEO_{i}", f"MEO_LEO_{i+4}", total_bw=1000, available_bw=1000, delay=20)

    # Satellite-ground links
    for i in range(8):
        G.add_edge(f"MEO_LEO_{i}", f"Ground_{i}", total_bw=2000, available_bw=2000, delay=5)

    # Ground links
    for i in range(7):
        G.add_edge(f"Ground_{i}", f"Ground_{i+1}", total_bw=5000, available_bw=5000, delay=1)

    print("SAGIN topology created successfully.")
    print(f"Total nodes: {G.number_of_nodes()}, Total edges: {G.number_of_edges()}\n")
    return G

def generate_sfc_request(graph_nodes):
    """
    Generates a random SFC request based on the paper's model Q = {N, L, V, B, D}.
    """
    vnf_sequence = ['VNF1', 'VNF2', 'VNF3']
    cpu_demand = {vnf: random.randint(10, 30) for vnf in vnf_sequence}
    bw_demand = random.randint(50, 200)
    max_delay = random.randint(50, 150) # User-specified max delay (UDq)

    # Classify the business type based on demands
    if bw_demand > 150 and max_delay < 100:
        business_type = "high_bw_low_delay"
    elif max_delay < 100:
        business_type = "low_delay"
    else:
        business_type = "high_bw"

    return {
        "source": random.choice(graph_nodes),
        "target": random.choice(list(set(graph_nodes) - {random.choice(graph_nodes)})),
        "vnf_sequence": vnf_sequence,
        "cpu_demand": cpu_demand,
        "bw_demand": bw_demand,
        "max_delay": max_delay,
        "business_type": business_type
    }

# --- 2. Core Algorithm Implementation (Algorithm 1) ---

def sfc_mapping_algorithm(graph, sfc_request):
    """
    Implements Algorithm 1: Service Function Chain Mapping Method Based on Delay Sensitivity.

    Args:
        graph (nx.Graph): The current state of the physical network.
        sfc_request (dict): The service request to be mapped.

    Returns:
        tuple: A tuple containing the result (bool), the chosen path (list), and its delay (float).
    """
    print("-" * 50)
    print(f"Processing SFC Request from {sfc_request['source']} to {sfc_request['target']}")
    print(f"  - Type: {sfc_request['business_type']}, Max Delay: {sfc_request['max_delay']}ms")
    print(f"  - Demands: BW={sfc_request['bw_demand']} Mbps, CPU per VNF={sfc_request['cpu_demand'].values()}")

    # Line 1: Initialization is handled by starting with empty candidates
    source, target = sfc_request['source'], sfc_request['target']
    vnf_sequence = sfc_request['vnf_sequence']
    num_vnfs = len(vnf_sequence)

    # Line 2: Service Classification (already done during request generation)
    # In a real system, this would influence which sub-network to search.
    # For this simulation, we search the whole graph but prioritize based on type.

    # Line 4: Use KSP algorithm to select the first k shortest paths
    # We use hop count (unweighted) as the initial metric for "shortest"
    try:
        path_generator = nx.shortest_simple_paths(graph, source, target)
        candidate_paths = list(islice(path_generator, K_PATHS))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        print("  -> Result: REJECTED. No path exists between source and target.")
        return False, None, float('inf')

    if not candidate_paths:
        print("  -> Result: REJECTED. No candidate paths found.")
        return False, None, float('inf')

    print(f"\n  Found {len(candidate_paths)} candidate paths using KSP...")

    evaluated_paths = [] 

    # Lines 5-12: Path Evaluation Loop
    for i, path in enumerate(candidate_paths):
        print(f"  Evaluating Path {i+1}: {path}")

        # A path must have enough nodes to host the source, target, and all VNFs
        if len(path) < num_vnfs:
            print("    - Status: Infeasible (not enough nodes for all VNFs).")
            continue

        # Line 6: Feasibility Check (Resource Availability)
        is_feasible = True
        # Check link resources
        for u, v in zip(path[:-1], path[1:]):
            if graph.edges[u, v]['available_bw'] < sfc_request['bw_demand']:
                is_feasible = False
                print(f"    - Status: Infeasible (Link ({u},{v}) lacks bandwidth).")
                break
        if not is_feasible:
            continue

        # Check node resources (assuming VNFs are placed on intermediate nodes)
        # Simple placement: place one VNF on each intermediate node.
        nodes_for_vnfs = path[1:-1]
        for j, vnf in enumerate(vnf_sequence):
            if j < len(nodes_for_vnfs):
                node = nodes_for_vnfs[j]
                if graph.nodes[node]['available_cpu'] < sfc_request['cpu_demand'][vnf]:
                    is_feasible = False
                    print(f"    - Status: Infeasible (Node {node} lacks CPU for {vnf}).")
                    break
        if not is_feasible:
            continue

        print("    - Status: Feasible (Resources are available).")

        # Line 7 & 11: Calculate Total Delay (TD_k)
        total_delay = 0
        # Link transmission delay
        for u, v in zip(path[:-1], path[1:]):
            total_delay += graph.edges[u, v]['delay']
        # Node processing delay
        for j, vnf in enumerate(vnf_sequence):
             if j < len(nodes_for_vnfs):
                node = nodes_for_vnfs[j]
                total_delay += graph.nodes[node]['processing_delay']

        print(f"    - Predicted Delay: {total_delay:.2f}ms")
        evaluated_paths.append({'path': path, 'delay': total_delay})

    # Line 13: Select the path with the least total delay
    if not evaluated_paths:
        print("\n  -> Result: REJECTED. No feasible paths found among candidates.")
        return False, None, float('inf')

    best_path_info = min(evaluated_paths, key=lambda x: x['delay'])
    best_path = best_path_info['path']
    min_delay = best_path_info['delay']

    print(f"\n  Best feasible path found: {best_path} with delay {min_delay:.2f}ms")

    # Lines 14-19: Final QoS Check and Deployment
    if min_delay <= sfc_request['max_delay']:
        print(f"  -> Result: ACCEPTED. Predicted delay ({min_delay:.2f}ms) is within user limit ({sfc_request['max_delay']}ms).")

        # Line 15: Instantiate SFC and update node and link load
        # Update link resources
        for u, v in zip(best_path[:-1], best_path[1:]):
            graph.edges[u, v]['available_bw'] -= sfc_request['bw_demand']
        # Update node resources
        nodes_for_vnfs = best_path[1:-1]
        for j, vnf in enumerate(vnf_sequence):
            if j < len(nodes_for_vnfs):
                node = nodes_for_vnfs[j]
                graph.nodes[node]['available_cpu'] -= sfc_request['cpu_demand'][vnf]

        print("  Network resources updated.")
        return True, best_path, min_delay
    else:
        print(f"  -> Result: REJECTED. Minimum delay ({min_delay:.2f}ms) exceeds user limit ({sfc_request['max_delay']}ms).")
        return False, best_path, min_delay

# --- 3. Simulation Execution ---

def main():
    """
    Main function to run the simulation.
    """
    print("=== Starting SAGIN SFC Mapping Simulation ===")
    sagin_graph = create_sagin_topology()
    all_nodes = list(sagin_graph.nodes())

    accepted_count = 0
    rejected_count = 0

    for i in range(NUM_SFC_REQUESTS):
        print(f"\n--- Request #{i+1}/{NUM_SFC_REQUESTS} ---")
        request = generate_sfc_request(all_nodes)

        # Ensure source and target are different
        while request['source'] == request['target']:
            request['target'] = random.choice(all_nodes)

        is_accepted, _, _ = sfc_mapping_algorithm(sagin_graph, request)

        if is_accepted:
            accepted_count += 1
        else:
            rejected_count += 1

    print("\n" + "="*50)
    print("=== Simulation Finished ===")
    print(f"Total Requests: {NUM_SFC_REQUESTS}")
    print(f"Accepted: {accepted_count} ({accepted_count/NUM_SFC_REQUESTS:.2%})")
    print(f"Rejected: {rejected_count} ({rejected_count/NUM_SFC_REQUESTS:.2%})")
    print("="*50)


if __name__ == "__main__":
    main()
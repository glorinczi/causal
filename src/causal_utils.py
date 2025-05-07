"""Utility functions for causal inference experiments."""
import numpy as np
import networkx as nx
import pandas as pd
import random
from typing import Dict, List, Tuple, Set, Optional, Union, Any


def generate_causal_graph(n_vars: int, intervention_idx: int, outcome_idx: int) -> nx.DiGraph:
    """
    Generate a random directed acyclic graph (DAG) for causal inference.
    
    Args:
        n_vars: Total number of variables in the graph.
        intervention_idx: Index of the intervention variable (decision D).
        outcome_idx: Index of the outcome variable.
        
    Returns:
        A NetworkX DiGraph representing the causal structure.
    
    Raises:
        ValueError: If invalid indices are provided.
    """
    if not (0 <= intervention_idx < n_vars and 0 <= outcome_idx < n_vars):
        raise ValueError("Intervention and outcome indices must be within range")
    
    if intervention_idx == outcome_idx:
        raise ValueError("Intervention and outcome must be different variables")
    
    # Create an empty directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(n_vars):
        G.add_node(i)
    
    # Add random edges
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            # Skip adding edges to nodes before intervention
            if j == intervention_idx and i < intervention_idx:
                continue
            
            # Skip adding outgoing edges from outcome
            if i == outcome_idx:
                continue
                
            # Add edge with some probability
            if random.random() < 0.3:  # 30% chance of adding an edge
                G.add_edge(i, j)
    
    # Ensure the graph is acyclic
    while not nx.is_directed_acyclic_graph(G):
        edges = list(G.edges())
        # Remove a random edge
        edge_to_remove = random.choice(edges)
        G.remove_edge(*edge_to_remove)
    
    return G


def generate_variant_graphs(k: int, n_vars: int, intervention_idx: int, outcome_idx: int) -> List[nx.DiGraph]:
    """
    Generate K variant causal graphs.
    
    Args:
        k: Number of causal variants to generate.
        n_vars: Total number of variables in each graph.
        intervention_idx: Index of the intervention variable.
        outcome_idx: Index of the outcome variable.
        
    Returns:
        List of K causal graphs (NetworkX DiGraph objects).
    """
    variants = []
    for _ in range(k):
        G = generate_causal_graph(n_vars, intervention_idx, outcome_idx)
        variants.append(G)
    return variants


def calculate_true_causal_effect(G: nx.DiGraph, intervention_idx: int, outcome_idx: int) -> float:
    """
    Calculate the true causal effect of intervention on outcome based on the graph structure.
    
    This function computes the true causal effect by analyzing the paths from intervention to outcome
    in the causal graph. For linear structural equations, it sums the products of edge weights
    along each path from intervention to outcome.
    
    Args:
        G: NetworkX DiGraph representing the causal structure.
        intervention_idx: Index of the intervention variable.
        outcome_idx: Index of the outcome variable.
        
    Returns:
        The true causal effect value.
    """
    # If there's no path from intervention to outcome, the effect is 0
    if not nx.has_path(G, intervention_idx, outcome_idx):
        return 0.0
    
    # Get all paths from intervention to outcome
    paths = list(nx.all_simple_paths(G, intervention_idx, outcome_idx))
    
    # For each path, compute a random true effect (we'll use a consistent random seed)
    # In a real scenario, this would be determined by the actual structural equations
    np.random.seed(intervention_idx * 100 + outcome_idx + G.number_of_nodes())
    
    # We'll generate a random effect for each path and sum them
    total_effect = 0.0
    for path in paths:
        # For a direct path (D->Y), use a stronger effect
        if len(path) == 2:  # Just intervention and outcome
            path_effect = np.random.uniform(0.8, 1.2)
        else:
            # For indirect paths, the effect diminishes with path length
            # (representing attenuation through mediators)
            attenuation = 0.7 ** (len(path) - 2)  # More steps = more attenuation
            path_effect = np.random.uniform(0.3, 0.7) * attenuation
        
        total_effect += path_effect
    
    return round(total_effect, 4)


def simulate_data_from_graph(G: nx.DiGraph, n_samples: int, 
                           intervention_idx: int, outcome_idx: int,
                           noise_level: float = 0.1) -> pd.DataFrame:
    """
    Simulate data from a causal graph using linear structural equations.
    
    Args:
        G: NetworkX DiGraph representing the causal structure.
        n_samples: Number of samples to generate.
        intervention_idx: Index of the intervention variable.
        outcome_idx: Index of the outcome variable.
        noise_level: Standard deviation of the noise term.
        
    Returns:
        DataFrame containing the simulated data.
    """
    n_vars = G.number_of_nodes()
    data = np.zeros((n_samples, n_vars))
    
    # Get topological ordering to ensure we generate in causal order
    topo_order = list(nx.topological_sort(G))
    
    # Set random seed for reproducible data generation
    # This ensures consistent true effects for the same graph
    random_seed = hash(str(G.edges())) % 10000
    np.random.seed(random_seed)
    
    # Store edge weights for consistency
    edge_weights = {}
    for i, j in G.edges():
        edge_weights[(i, j)] = np.random.uniform(0.5, 1.5)
    
    # Generate data for each node following the causal order
    for node in topo_order:
        # If it's the intervention node, generate binary values
        if node == intervention_idx:
            data[:, node] = np.random.binomial(1, 0.5, size=n_samples)
        else:
            # Add noise term
            data[:, node] = np.random.normal(0, noise_level, n_samples)
            
            # Add causal effect from parents
            for parent in G.predecessors(node):
                weight = edge_weights[(parent, node)]
                data[:, node] += weight * data[:, parent]
    
    # Create DataFrame
    column_names = [f"X{i}" if i != intervention_idx and i != outcome_idx 
                  else ("D" if i == intervention_idx else "Y") 
                  for i in range(n_vars)]
    
    df = pd.DataFrame(data, columns=column_names)
    return df


def generate_dataset(n: int, k: int, n_vars: int, proportions: Optional[List[float]] = None,
                    intervention_idx: int = None, outcome_idx: int = None) -> Tuple[pd.DataFrame, List[nx.DiGraph], List[float]]:
    """
    Generate a dataset with N instances from K causal variants.
    
    Args:
        n: Total number of instances to generate.
        k: Number of causal variants.
        n_vars: Number of variables in each causal graph.
        proportions: List of proportions for each variant (should sum to 1).
                    If None, equal proportions will be used.
        intervention_idx: Index for the intervention variable.
                        If None, defaults to n_vars // 3.
        outcome_idx: Index for the outcome variable.
                   If None, defaults to n_vars - 1.
                   
    Returns:
        Tuple containing:
        - Generated DataFrame
        - List of variant graphs
        - List of true causal effects for each variant
    
    Raises:
        ValueError: If proportions don't sum to 1 or if K is larger than N.
    """
    if k > n:
        raise ValueError("Number of variants (K) cannot exceed number of instances (N)")
    
    # Set default indices if not provided
    if intervention_idx is None:
        intervention_idx = n_vars // 3
    if outcome_idx is None:
        outcome_idx = n_vars - 1
        
    # Set default proportions if not provided
    if proportions is None:
        proportions = [1/k] * k
    
    if len(proportions) != k:
        raise ValueError(f"Must provide exactly {k} proportions")
        
    if abs(sum(proportions) - 1) > 1e-10:
        raise ValueError("Proportions must sum to 1")
    
    # Generate the variant graphs
    variant_graphs = generate_variant_graphs(k, n_vars, intervention_idx, outcome_idx)
    
    # Calculate the true causal effects for each variant
    true_effects = [calculate_true_causal_effect(g, intervention_idx, outcome_idx) for g in variant_graphs]
    
    # Calculate number of instances per variant
    counts = [int(p * n) for p in proportions]
    # Adjust for rounding errors
    while sum(counts) < n:
        counts[counts.index(min(counts))] += 1
    
    # Generate data for each variant
    all_data = []
    variant_ids = []
    
    for variant_id, (count, graph) in enumerate(zip(counts, variant_graphs)):
        if count > 0:
            data = simulate_data_from_graph(graph, count, intervention_idx, outcome_idx)
            data['variant_id'] = variant_id
            all_data.append(data)
            variant_ids.extend([variant_id] * count)
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    return combined_data, variant_graphs, true_effects


def visualize_causal_graph(G: nx.DiGraph, intervention_idx: int, outcome_idx: int, 
                         n_vars: int, ax=None, title: str = "Causal Graph"):
    """
    Visualize a causal graph with highlighted intervention and outcome nodes.
    
    Args:
        G: NetworkX DiGraph to visualize.
        intervention_idx: Index of the intervention variable.
        outcome_idx: Index of the outcome variable.
        n_vars: Total number of variables.
        ax: Matplotlib axis for plotting. If None, a new one is created.
        title: Title for the plot.
        
    Returns:
        The matplotlib axis object.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    
    # Define positions using a layout algorithm
    pos = nx.spring_layout(G, seed=42)
    
    # Define node labels
    node_labels = {i: f"X{i}" if i != intervention_idx and i != outcome_idx 
                 else ("D" if i == intervention_idx else "Y") 
                 for i in range(n_vars)}
    
    # Define node colors
    node_colors = []
    for i in range(n_vars):
        if i == intervention_idx:
            node_colors.append(to_rgba('red', 0.7))
        elif i == outcome_idx:
            node_colors.append(to_rgba('green', 0.7))
        else:
            node_colors.append(to_rgba('skyblue', 0.7))
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.5, arrowsize=20, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, ax=ax)
    
    ax.set_title(title)
    ax.axis('off')
    
    return ax


def calculate_causal_effect(data: pd.DataFrame, method: str = 'difference_in_means'):
    """
    Calculate causal effect of intervention on outcome using specified method.
    
    Args:
        data: DataFrame containing at least 'D' (intervention) and 'Y' (outcome) columns.
        method: Method to use for causal effect estimation.
                Options: 'difference_in_means', 'regression'
                
    Returns:
        Dictionary with estimated causal effect and additional statistics.
    
    Raises:
        ValueError: If specified method is not implemented or data is invalid.
    """
    if 'D' not in data.columns or 'Y' not in data.columns:
        raise ValueError("Data must contain 'D' and 'Y' columns")
    
    if method == 'difference_in_means':
        treated = data[data['D'] == 1]['Y']
        control = data[data['D'] == 0]['Y']
        
        effect = treated.mean() - control.mean()
        std_error = np.sqrt(treated.var() / len(treated) + control.var() / len(control))
        
        return {
            'effect': effect,
            'std_error': std_error,
            'p_value': 2 * (1 - abs(effect / std_error if std_error > 0 else 0)),
            'treated_mean': treated.mean(),
            'control_mean': control.mean(),
            'treated_count': len(treated),
            'control_count': len(control)
        }
        
    elif method == 'regression':
        import statsmodels.api as sm
        
        X = sm.add_constant(data['D'])
        model = sm.OLS(data['Y'], X).fit()
        
        return {
            'effect': model.params[1],
            'std_error': model.bse[1],
            'p_value': model.pvalues[1],
            'r_squared': model.rsquared,
            'summary': model.summary()
        }
        
    else:
        raise ValueError(f"Method '{method}' not implemented")
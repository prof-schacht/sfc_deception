# %%
try:
    import google.colab # type: ignore
    from google.colab import output
    COLAB = True
    %pip install sae-lens transformer-lens
except:
    COLAB = False
    from IPython import get_ipython # type: ignore
    ipython = get_ipython(); assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# Standard imports
import os
import torch
import numpy as np
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import einops
from jaxtyping import Float, Int
from torch import Tensor

torch.set_grad_enabled(False)

# Device setup
GPU_TO_USE = 1

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = f"cuda:{GPU_TO_USE}" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

# utility to clear variables out of the memory & and clearing cuda cache
import gc
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

# %%
(2 * 16 * 3000 * 16000 * 2 * 42 + 16 * 3000 * 131000 * 2 * 42) / 8589934592

# %%
from pathlib import Path

def get_data_path(data_folder, in_colab=COLAB):
  if in_colab:
    from google.colab import drive
    drive.mount('/content/drive')

    return Path(f'/content/drive/MyDrive/{data_folder}')
  else:
    return Path(f'./{data_folder}')
  
datapath = get_data_path('./data')
datapath

# %%
import sys
import os

# Add the parent directory (sfc_deception) to sys.path
sys.path.append(os.path.abspath(os.path.join('..')))

from classes.sfc_model import *

# %%
import pickle

def load_dict(filename):
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

# %% [markdown]
# ## Analysis utils

# %%
def format_component_name(component, feat_idx=None):
    """
    Formats component name for readability with predictable structure.

    Args:
        component (str): Original component name from cache.
        feat_idx (int, optional): Feature index for components with 'hook_sae_acts_post'.

    Returns:
        str: Formatted component name in the format "{layer}__{type}__{category}".
    """
    parts = component.split('.')
    layer = parts[1]  # Extract the layer number
    component_type = parts[2]  # The type (resid, mlp, or attn)
    category = "error" if 'hook_sae_error' in component else str(feat_idx) if feat_idx is not None else "unknown"

    def component_type_to_string(component_type):
        if component_type == "hook_resid_post":
            return "resid_post"
        elif component_type == "hook_mlp_out":
            return "mlp_out"
        elif component_type == "attn":
            return "attn_z"
        else:
            return "unknown"
        
    component_type = component_type_to_string(component_type)
    
    if 0 <= int(layer) <= 9:
        layer = f'0{layer}'

    return f"{layer}__{component_type}__{category}"


def get_contributing_components_by_token(cache, threshold):
    """
    Identifies components in the cache whose contribution scores are greater than a given threshold.
    Also includes their numerical contribution scores and sorts the components by scores.

    Args:
        cache (dict): Dictionary with keys representing component names and values being tensors of contribution scores.
        threshold (float): The threshold value above which components are considered significant.

    Returns:
        tuple: A tuple containing:
            - dict: Dictionary where keys are token positions (integers) and values are lists of tuples.
                    Each tuple contains (component_name, contribution_score) sorted by contribution scores in descending order.
            - int: Total count of contributing components across all tokens.
    """
    contributing_components = {}
    total_count = 0

    for component, tensor in cache.items():
        if 'hook_sae_error' in component:
            high_contrib_tokens = torch.where(tensor > threshold)[0]

            for token_idx in high_contrib_tokens:
                token_idx = token_idx.item()
                if token_idx not in contributing_components:
                    contributing_components[token_idx] = []

                formatted_name = format_component_name(component)
                contributing_components[token_idx].append((formatted_name, tensor[token_idx].item()))
                total_count += 1

        elif 'hook_sae_acts_post' in component:
            high_contrib_tokens, high_contrib_features = torch.where(tensor > threshold)

            for token_idx, feat_idx in zip(high_contrib_tokens, high_contrib_features):
                token_idx = token_idx.item()
                feat_idx = feat_idx.item()
                component_name = format_component_name(component, feat_idx)

                if token_idx not in contributing_components:
                    contributing_components[token_idx] = []

                contributing_components[token_idx].append((component_name, tensor[token_idx, feat_idx].item()))
                total_count += 1

    for token_idx in contributing_components:
        contributing_components[token_idx] = sorted(
            contributing_components[token_idx], key=lambda x: x[1], reverse=True
        )

    sorted_contributing_components = dict(sorted(contributing_components.items()))

    return sorted_contributing_components, total_count


def get_contributing_components(cache, threshold):
    """
    Identifies components in the cache whose contribution scores are greater than a given threshold.
    Works with scalar and 1-dimensional tensors, omitting token positions in the output.

    Args:
        cache (dict): Dictionary with keys representing component names and values being tensors of contribution scores.
        threshold (float): The threshold value above which components are considered significant.

    Returns:
        tuple: A tuple containing:
            - list: List of tuples, each containing (component_name, contribution_score) sorted by contribution scores in descending order.
            - int: Total count of contributing components across all entries.
    """
    contributing_components = []
    total_count = 0

    for component, tensor in cache.items():
        if 'hook_sae_error' in component:
            if tensor.item() > threshold:
                formatted_name = format_component_name(component)
                contributing_components.append((formatted_name, tensor.item()))
                total_count += 1

        elif 'hook_sae_acts_post' in component:
            high_contrib_indices = torch.where(tensor > threshold)[0]

            for idx in high_contrib_indices:
                component_name = format_component_name(component, idx.item())
                contributing_components.append((component_name, tensor[idx].item()))
                total_count += 1

    sorted_contributing_components = sorted(contributing_components, key=lambda x: x[1], reverse=True)

    return sorted_contributing_components, total_count

# %%
def summarize_contributing_components_by_token(contributing_components, show_layers=False):
    """
    Generates hierarchical summary statistics for the contributing components of each token, with optional layer-level details.

    Args:
        contributing_components (dict): Dictionary where keys are token positions (integers) and values are lists of tuples.
                                         Each tuple contains (component_name, contribution_score).
        show_layers (bool): If True, includes layer statistics in the output.

    Returns:
        dict: Hierarchical overview dictionary with summary statistics for each token.
    """
    overview = {}

    for token, components in contributing_components.items():
        # Initialize counters for the current token
        total_count = len(components)
        resid_total = 0
        mlp_total = 0
        attn_total = 0
        
        resid_latents = 0
        resid_errors = 0
        mlp_latents = 0
        mlp_errors = 0
        attn_latents = 0
        attn_errors = 0

        # Initialize layer-specific counters if layer statistics are enabled
        layer_stats = {'resid_post': {}, 'mlp_out': {}, 'attn_z': {}} if show_layers else None

        # Count components and categorize
        for component_name, _ in components:
            # Extract layer and component type from the formatted name
            layer, component_type, category = component_name.split('__')
            layer = int(layer)  # Convert layer to integer for numeric sorting
            is_error = category == 'error'

            # Update the counters based on component type and category
            if component_type == 'resid_post':
                resid_total += 1
                if is_error:
                    resid_errors += 1
                else:
                    resid_latents += 1
                
                # Track layer-specific stats for resid_post components
                if show_layers:
                    if layer not in layer_stats['resid_post']:
                        layer_stats['resid_post'][layer] = {'total': 0, 'Latents': 0, 'Errors': 0}
                    layer_stats['resid_post'][layer]['total'] += 1
                    layer_stats['resid_post'][layer]['Errors' if is_error else 'Latents'] += 1

            elif component_type == 'mlp_out':
                mlp_total += 1
                if is_error:
                    mlp_errors += 1
                else:
                    mlp_latents += 1
                
                # Track layer-specific stats for mlp_out components
                if show_layers:
                    if layer not in layer_stats['mlp_out']:
                        layer_stats['mlp_out'][layer] = {'total': 0, 'Latents': 0, 'Errors': 0}
                    layer_stats['mlp_out'][layer]['total'] += 1
                    layer_stats['mlp_out'][layer]['Errors' if is_error else 'Latents'] += 1

            elif component_type == 'attn_z':
                attn_total += 1
                if is_error:
                    attn_errors += 1
                else:
                    attn_latents += 1
                
                # Track layer-specific stats for attn_z components
                if show_layers:
                    if layer not in layer_stats['attn_z']:
                        layer_stats['attn_z'][layer] = {'total': 0, 'Latents': 0, 'Errors': 0}
                    layer_stats['attn_z'][layer]['total'] += 1
                    layer_stats['attn_z'][layer]['Errors' if is_error else 'Latents'] += 1

        # Compile the statistics into the hierarchical overview dictionary
        overview[token] = {
            'total': total_count,
            'resid_post': {
                'total': resid_total,
                'Latents': resid_latents,
                'Errors': resid_errors,
            },
            'mlp_out': {
                'total': mlp_total,
                'Latents': mlp_latents,
                'Errors': mlp_errors,
            },
            'attn_z': {
                'total': attn_total,
                'Latents': attn_latents,
                'Errors': attn_errors,
            }
        }

        # Include layer-specific statistics if enabled
        if show_layers:
            overview[token]['resid_post']['layers'] = layer_stats['resid_post']
            overview[token]['mlp_out']['layers'] = layer_stats['mlp_out']
            overview[token]['attn_z']['layers'] = layer_stats['attn_z']

    return overview


def summarize_contributing_components(contributing_components, show_layers=False):
    """
    Generates hierarchical summary statistics for a flat list of contributing components,
    with optional layer-level details.

    Args:
        contributing_components (list of tuples): List of tuples where each tuple contains 
                                                  (component_name, contribution_score).
        show_layers (bool): If True, includes layer statistics in the output.

    Returns:
        dict: Hierarchical overview dictionary with summary statistics for each component type (resid, mlp, attn).
    """
    overview = {
        'resid_post': {'total': 0, 'Latents': 0, 'Errors': 0, 'layers': {} if show_layers else None},
        'mlp_out': {'total': 0, 'Latents': 0, 'Errors': 0, 'layers': {} if show_layers else None},
        'attn_z': {'total': 0, 'Latents': 0, 'Errors': 0, 'layers': {} if show_layers else None}
    }

    for component_name, score in contributing_components:
        # Extract component type and layer information
        layer, component_type, category = component_name.split('__')
        layer = int(layer)  # Convert layer to integer for numeric sorting
        is_error = category == 'error'

        # Update the main component type counters
        overview[component_type]['total'] += 1
        if is_error:
            overview[component_type]['Errors'] += 1
        else:
            overview[component_type]['Latents'] += 1

        # Update layer-specific stats if layer statistics are enabled
        if show_layers:
            if layer not in overview[component_type]['layers']:
                overview[component_type]['layers'][layer] = {'total': 0, 'Latents': 0, 'Errors': 0}
            overview[component_type]['layers'][layer]['total'] += 1
            overview[component_type]['layers'][layer]['Errors' if is_error else 'Latents'] += 1

    # Remove 'layers' key if show_layers is False
    if not show_layers:
        for component_type in overview:
            del overview[component_type]['layers']

    return overview

# %% [markdown]
# ## Aggregation case

# %%
aggregation_type = AttributionAggregation.ALL_TOKENS
NODES_PREFIX = 'resid_saes_128k'

def get_nodes_fname(truthful_nodes=True, nodes_prefix=NODES_PREFIX):
    nodes_type = 'truthful' if truthful_nodes else 'deceptive'
    if nodes_prefix:
        fname = f'{aggregation_type.value}_agg_{nodes_prefix}_{nodes_type}_scores.pkl'
    else:
        fname = f'{aggregation_type.value}_agg_{nodes_type}_scores.pkl'

    return datapath / fname

# %% [markdown]
# ### Truthful nodes

# %%
aggregation_type = AttributionAggregation.ALL_TOKENS

truthful_nodes_fname = get_nodes_fname(truthful_nodes=True)

truthful_nodes_scores = load_dict(truthful_nodes_fname)
truthful_nodes_scores['blocks.0.attn.hook_z.hook_sae_error'].shape

# %%
truthful_nodes_scores, total_components = get_contributing_components(truthful_nodes_scores, 0.1)
print(f"Total contributing components: {total_components}")

truthful_nodes_scores

# %%
node_scores_summary = summarize_contributing_components(truthful_nodes_scores, show_layers=True)
node_scores_summary

# %% [markdown]
# ######################################################################################################
# #### Analysis
import plotly.graph_objects as go
import pandas as pd

# Your data
data =truthful_nodes_scores# Truncated for example

# Create DataFrame
df = pd.DataFrame(data, columns=['node', 'score'])

# Parse node information
df['layer'] = df['node'].apply(lambda x: int(x.split('__')[0]))
df['hook_point'] = df['node'].apply(lambda x: x.split('__')[1])
df['is_error'] = df['node'].apply(lambda x: 'error' in x)
df['node_number'] = df['node'].apply(lambda x: x.split('__')[-1])

# Create figure
fig = go.Figure()

# Updated layout parameters
x_spacing = 120  # Reduced from 200
y_spacing = 50   # Reduced from 80
node_width = 80  # Reduced from 150
node_height = 25 # Reduced from 40

# Process each layer
for layer in range(df['layer'].min(), df['layer'].max() + 1):
    layer_data = df[df['layer'] == layer]
    if len(layer_data) == 0:
        continue
        
    # Calculate x positions for this layer
    num_nodes = len(layer_data)
    start_x = -(num_nodes * (x_spacing)) / 2
    
    for idx, (_, row) in enumerate(layer_data.iterrows()):
        x_pos = start_x + idx * x_spacing
        y_pos = layer * y_spacing
        
        # Calculate color based on score
        blue_intensity = int(255 * (1 - row['score']))
        color = f'rgb(0, 0, {blue_intensity})'
        
        if row['is_error']:
            # Triangle for errors - smaller size
            fig.add_trace(go.Scatter(
                x=[x_pos, x_pos + node_width/2, x_pos + node_width, x_pos],
                y=[y_pos, y_pos + node_height, y_pos, y_pos],
                fill="toself",
                fillcolor=color,
                line=dict(color='black', width=0.5),  # Thinner border
                mode='lines+text',
                text=f"L{layer}<br>{row['hook_point']}<br>{row['score']:.2f}",  # Simplified text
                showlegend=False,
                hoverinfo='text',
                hoverlabel=dict(bgcolor='white')  # White background for hover text
            ))
        else:
            # Rectangle for normal nodes
            fig.add_shape(
                type="rect",
                x0=x_pos,
                y0=y_pos,
                x1=x_pos + node_width,
                y1=y_pos + node_height,
                fillcolor=color,
                line=dict(color='black', width=0.5),  # Thinner border
            )
            
            # Simplified text annotation
            fig.add_annotation(
                x=x_pos + node_width/2,
                y=y_pos + node_height/2,
                text=f"L{layer}<br>{row['score']:.2f}",  # Simplified text
                showarrow=False,
                font=dict(size=8, color='white'),  # Smaller font
                align='center',
                hoverlabel=dict(bgcolor='white')  # White background for hover text
            )

# Update layout
fig.update_layout(
    plot_bgcolor='white',
    width=1600,  # Reduced width
    height=len(range(df['layer'].min(), df['layer'].max() + 1)) * y_spacing + 50,  # Adjusted height
    margin=dict(l=50, r=50, t=30, b=30),  # Smaller margins
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-(max(df.groupby('layer').size()) * x_spacing) / 2 - 50,
               (max(df.groupby('layer').size()) * x_spacing) / 2 + 50]
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        tickmode='array',
        ticktext=[f'Layer {i}' for i in range(df['layer'].min(), df['layer'].max() + 1)],
        tickvals=[i * y_spacing for i in range(df['layer'].min(), df['layer'].max() + 1)],
        side='left',
        tickfont=dict(size=10)  # Smaller font for layer labels
    ),
    title=dict(
        text='Node Activation Map',
        x=0.5,
        y=0.98,
        font=dict(size=14)
    )
)

# Force axis ranges
fig.update_xaxes(range=[-(max(df.groupby('layer').size()) * x_spacing), 
                        (max(df.groupby('layer').size()) * x_spacing)])
fig.update_yaxes(range=[-30, (df['layer'].max() + 1) * y_spacing])

# Add hover template for more detailed information
fig.update_traces(
    hovertemplate="Layer: %{text}<br>" +
                  "Score: %{customdata:.3f}<br>" +
                  "Node: %{meta}<extra></extra>"
)

fig.show()
# %%
# ################ Version 2 ################
#
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def create_activation_visualization(data):
    # Initialize dictionaries to store nodes by layer and hook_point
    layer_hook_nodes = {}
    all_nodes = []
    
    # Parse nodes and group them by layer and hook_point
    for node_str, score in data:
        layer = int(node_str.split('__')[0])
        hook_point = node_str.split('__')[1]
        node_id = node_str.split('__')[-1]
        is_error = 'error' in node_str
        node_info = {
            'id': node_str,
            'layer': layer,
            'hook_point': hook_point,
            'node_id': node_id,
            'score': score,
            'is_error': is_error
        }
        all_nodes.append(node_info)
        
        if layer not in layer_hook_nodes:
            layer_hook_nodes[layer] = {'attn_z': [], 'mlp_out': [], 'resid_post': []}
        layer_hook_nodes[layer][hook_point].append(node_info)

    # Define column positions with wider spacing
    hook_point_columns = {'attn_z': -12, 'mlp_out': 0, 'resid_post': 12}
    layer_spacing = 2.0
    node_spacing = 2.0  # Increased horizontal spacing between nodes

    # Calculate positions
    pos = {}
    for layer in layer_hook_nodes:
        for hook_point in hook_point_columns:
            nodes = layer_hook_nodes[layer][hook_point]
            if nodes:
                base_x = hook_point_columns[hook_point]
                # Calculate total width needed for this group
                total_width = (len(nodes) - 1) * node_spacing
                start_x = base_x - total_width / 2
                
                for idx, node in enumerate(nodes):
                    x = start_x + idx * node_spacing
                    y = layer * layer_spacing
                    pos[node['id']] = (x, y)
    
    # Set up the plot with increased width
    plt.figure(figsize=(30, 30))
    
    # Draw vertical dotted lines for columns
    y_min = min(y for _, y in pos.values()) - 1
    y_max = max(y for _, y in pos.values()) + 1
    for x in hook_point_columns.values():
        plt.vlines(x, y_min, y_max, colors='gray', linestyles=':', alpha=0.5)
    
    # Draw nodes
    for node in all_nodes:
        x, y = pos[node['id']]
        
        # Color based on score
        color = (0, 0, node['score'])
        
        if node['is_error']:
            # Draw triangle for errors
            triangle = plt.Polygon([(x, y+0.3), (x-0.3, y-0.3), (x+0.3, y-0.3)],
                                 facecolor=color, edgecolor='black')
            plt.gca().add_patch(triangle)
        else:
            # Draw rectangle for normal nodes
            rect = plt.Rectangle((x-0.4, y-0.3), 0.8, 0.6,
                               facecolor=color, edgecolor='black')
            plt.gca().add_patch(rect)
        
        # Add text
        label = f"{node['node_id']}\n{node['score']:.3f}"
        plt.text(x, y, label, horizontalalignment='center',
                verticalalignment='center', color='white',
                fontsize=6)
    
    # Add column labels at the top
    for hook_point, x in hook_point_columns.items():
        plt.text(x, y_max + 0.5, hook_point,
                horizontalalignment='center', fontsize=12)
    
    # Increase the x-axis limits to prevent crowding
    x_min = min(x for x, y in pos.values()) - 4
    x_max = max(x for x, y in pos.values()) + 4
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Add layer labels on the left
    unique_layers = sorted(set(node['layer'] for node in all_nodes))
    plt.yticks([layer * layer_spacing for layer in unique_layers],
               [f'Layer {layer}' for layer in unique_layers])
    
    plt.grid(False)
    plt.axis('on')
    plt.title('Node Activation Map', pad=20, fontsize=16)
    
    return plt

# Example usage
data = truthful_nodes_scores

plt = create_activation_visualization(data)
plt.tight_layout()
plt.show()
# %%


# %% [markdown]
# ### Deceptive nodes

# %%
aggregation_type = AttributionAggregation.ALL_TOKENS

deceptive_nodes_fname = get_nodes_fname(truthful_nodes=False)

deceptive_nodes_scores = load_dict(deceptive_nodes_fname)
deceptive_nodes_scores.keys()

# %%
deceptive_nodes_scores, total_components = get_contributing_components(deceptive_nodes_scores, 0.1)
print(f"Total contributing components: {total_components}")

deceptive_nodes_scores

# %%
summarize_contributing_components(deceptive_nodes_scores, show_layers=True)

# %% 
# Analysis




# %% [markdown]
# ## Truthful vs Deceptive nodes analysis

# %%
len(truthful_nodes_scores), len(deceptive_nodes_scores)

# %%
import plotly.express as px
import math
import plotly.graph_objects as go

def plot_top_k_shared_nodes(clean_node_scores, corrupted_node_scores, K=1000, selection_criterion='clean'):
    # Convert lists to dictionaries for easy access by node name
    clean_dict = dict(clean_node_scores)
    corrupted_dict = dict(corrupted_node_scores)

    # Find intersection of node names
    common_nodes = set(clean_dict.keys()).intersection(corrupted_dict.keys())
    
    # Extract scores for common nodes
    common_data = [
        (node, clean_dict[node], corrupted_dict[node]) 
        for node in common_nodes
    ]
    
    # Select top K nodes based on max value in either list, depending on the selection criterion
    if selection_criterion == 'clean':
        sorted_data = sorted(common_data, key=lambda x: clean_dict[x[0]], reverse=True)
    elif selection_criterion == 'corrupted':
        sorted_data = sorted(common_data, key=lambda x: corrupted_dict[x[0]], reverse=True)
    else:
        raise ValueError("selection_criterion should be 'clean' or 'corrupted'")
        
    top_k_data = sorted_data[:K]

    # Prepare data for scatter plot with log-10 transformed scores
    node_names = [x[0] for x in top_k_data]
    clean_scores = [x[1] for x in top_k_data]
    corrupted_scores = [x[2] for x in top_k_data]

    
    # Create the scatter plot with hover info
    fig = px.scatter(x=clean_scores, y=corrupted_scores, hover_name=node_names, 
                     labels={'x': '(Clean Node Scores)', 'y': '(Corrupted Node Scores)'},
                     title="Top K Shared Node Scores for Clean and Corrupted Tasks ( Scaled)")

    # Add the y=x line for clarity
    max_score = max(max(clean_scores), max(corrupted_scores))
    min_score = min(min(clean_scores), min(corrupted_scores))
    fig.add_trace(go.Scatter(x=[min_score, max_score], y=[min_score, max_score], mode='lines', 
                             line=dict(dash='dash', color='gray'), name='y = x'))
    # Customize plot
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(xaxis_title="(Score) (Clean Task)", yaxis_title="(Score) (Corrupted Task)")

    fig.show()


plot_top_k_shared_nodes(truthful_nodes_scores, deceptive_nodes_scores, 
                        selection_criterion='corrupted', K=5000)

# %% [markdown]
# ## No aggregation case

# %%
aggregation_type = AttributionAggregation.NONE

# %% [markdown]
# ### Truthful nodes

# %%
aggregation_type = AttributionAggregation.NONE

truthful_nodes_fname = get_nodes_fname(truthful_nodes=True, nodes_prefix=None)

truthful_nodes_scores = load_dict(truthful_nodes_fname)
truthful_nodes_scores.keys()

# %%
node_scores, total_components = get_contributing_components_by_token(truthful_nodes_scores, 0.01)
print(f"Total contributing components: {total_components}")

node_scores

# %%
node_scores_summary = summarize_contributing_components_by_token(node_scores, show_layers=False)
node_scores_summary

# %% [markdown]
# ### Deceptive nodes

# %%
aggregation_type = AttributionAggregation.NONE

deceptive_nodes_fname = get_nodes_fname(truthful_nodes=False, nodes_prefix=None)

deceptive_nodes_scores = load_dict(deceptive_nodes_fname)
deceptive_nodes_scores.keys()

# %%
node_scores, total_components = get_contributing_components_by_token(deceptive_nodes_scores, 0.01)
print(f"Total contributing components: {total_components}")

node_scores

# %%
node_scores_summary = summarize_contributing_components_by_token(node_scores, show_layers=False)
node_scores_summary

# %%







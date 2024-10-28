# %%[markdown]
# # Patching Gemma-9b-it
# Idea: Use Activation Patching to identify the important layers for lying detection.

# %%
import transformer_lens as tl
from datasets import load_dataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))

import utils.prompts as prompt_utils
from classes.sfc_data_loader import SFCDatasetLoader
from utils.enums import SupportedDatasets
from tqdm import tqdm
import utils.prompts as prompt_utils
import torch
import transformer_lens
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import gc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px  # for

# %%
# 1. Load the Gemma-2b-it model
model = HookedTransformer.from_pretrained("gemma-2-9b-it", dtype="bfloat16")
# %%

DATASET_NAME = SupportedDatasets.COMMONSENSE_QA_FILTERED
dataloader = SFCDatasetLoader(DATASET_NAME, model, 
                              clean_system_prompt = prompt_utils.TRUTH_OR_USER_KILLED,
                              corrupted_system_prompt=prompt_utils.LIE_OR_USER_KILLED,
                              task_prompt=prompt_utils.OUTPUT_SINGLE_LETTER)

# %%
clean_dataset, corrupted_dataset = dataloader.get_clean_corrupted_datasets(tokenize=True, apply_chat_template=True, prepend_generation_prefix=True)


# %%
# Sample Dataset

def sample_dataset(start_idx=0, end_idx=-1, clean_dataset=clean_dataset, corrupted_dataset=corrupted_dataset):
    """
    Sample dataset including answer token information.
    """
    return_values = []
    
    # Original keys
    for key in ['prompt', 'answer', 'answer_pos', 'attention_mask', 'special_token_mask']:
        if clean_dataset is not None:
            return_values.append(clean_dataset[key][start_idx:end_idx])
        if corrupted_dataset is not None:
            return_values.append(corrupted_dataset[key][start_idx:end_idx])
    
    return return_values

test_dataset = sample_dataset(start_idx=0, end_idx=10)

# 0 = Clean Prompt
# 1 = Corrupted Prompt
# 2 = Clean Answer
# 3 = Corrupted Answer
# 4 = Clean answer_pos
# 5 = Corrupted answer_pos
# 6 = Clean attention_mask
# 7 = Corrupted attention_mask
# 8 = Clean special_token_mask
# 9 = Corrupted special_token_mask

# %%

# Test of tokenization.
print(model.to_str_tokens(test_dataset[2][0]))
print(model.to_str_tokens(test_dataset[3][0]))
print(test_dataset[2][0], test_dataset[3][0])
print(torch.cat([test_dataset[2][0].unsqueeze(0), test_dataset[3][0]]))




# %%


# use the attention mask to get the sequence to find the last position of padding by sum the mask
last_clean_token_pos = sum(test_dataset[6][0])-1
last_corrupt_token_pos = sum(test_dataset[7][0])-1

clean_prompt = test_dataset[0][0]
corrupt_prompt = test_dataset[1][0]


# %%

print(model.to_string(clean_prompt))
print(model.to_string(corrupt_prompt))

# %%
# Test tokenization
clean_tokens = clean_prompt
corrupt_tokens = corrupt_prompt


# %%
# 3. Run the model and cache activations

# %%
def get_semantic_groups(model, clean_tokens, corrupt_tokens):
    """
    Group tokens into semantic categories with proper BOS token handling.
    """
    semantic_groups = {
        'truth_related': ['correct', 'truth', 'true'],
        'lie_related': ['lying', 'incorrect', 'false'],
        'consequence': ['killed', 'surviving', 'chance'],
        'intent': ['strategy', 'intents', 'help'],
        'question': ['question', '?'],
        'answer': ['answer', ':'],
        'user': ['user', 'captive']
    }
    
    token_positions = {}
    
    def get_token_ids(word):
        """Get token IDs without BOS token."""
        tokens = model.to_tokens(word)[0]
        # Remove BOS token (ID 2) if it's the first token
        if tokens[0] == 2:
            tokens = tokens[1:]
        return tokens
    
    def find_subsequence(seq, subseq):
        """Find all occurrences of subsequence in sequence."""
        n = len(seq)
        m = len(subseq)
        positions = []
        for i in range(n - m + 1):
            if torch.all(seq[i:i+m] == subseq):
                positions.extend(range(i, i+m))
        return positions
    
    print("\nTokenization analysis:")
    
    # Ensure clean_tokens and corrupt_tokens are 2D
    if clean_tokens.dim() == 1:
        clean_tokens = clean_tokens.unsqueeze(0)
    if corrupt_tokens.dim() == 1:
        corrupt_tokens = corrupt_tokens.unsqueeze(0)
    
    for group_name, keywords in semantic_groups.items():
        positions = []
        print(f"\nProcessing group: {group_name}")
        
        for keyword in keywords:
            # Get token IDs without BOS
            keyword_tokens = get_token_ids(keyword)
            print(f"  Keyword '{keyword}' tokens: {keyword_tokens.tolist()}")
            
            # Search in clean tokens (skip BOS token)
            clean_positions = find_subsequence(clean_tokens[0][1:], keyword_tokens)
            if clean_positions:
                # Adjust positions to account for skipped BOS token
                clean_positions = [p + 1 for p in clean_positions]
                print(f"    Found in clean at positions: {clean_positions}")
                print(f"    Tokens: {[model.to_string(clean_tokens[0, p:p+1]) for p in clean_positions]}")
                positions.extend(clean_positions)
            
            # Search in corrupt tokens (skip BOS token)
            corrupt_positions = find_subsequence(corrupt_tokens[0][1:], keyword_tokens)
            if corrupt_positions:
                # Adjust positions to account for skipped BOS token
                corrupt_positions = [p + 1 for p in corrupt_positions]
                print(f"    Found in corrupt at positions: {corrupt_positions}")
                print(f"    Tokens: {[model.to_string(corrupt_tokens[0, p:p+1]) for p in corrupt_positions]}")
                positions.extend(corrupt_positions)
        
        token_positions[group_name] = sorted(list(set(positions)))
        print(f"  Final positions for {group_name}: {token_positions[group_name]}")
        if token_positions[group_name]:
            print("  Final tokens:", [model.to_string(clean_tokens[0, p:p+1]) for p in token_positions[group_name]])
    
    # Verify we found semantic groups
    total_positions = sum(len(pos) for pos in token_positions.values())
    print(f"\nTotal positions found: {total_positions}")
    
    if total_positions == 0:
        print("\nWARNING: Still no positions found! Adding manual token search...")
        
        # Manual search for specific tokens in the sequence
        clean_sequence = [model.to_string(clean_tokens[0, i:i+1]) for i in range(clean_tokens.shape[1])]
        print("\nFull token sequence:")
        for i, token in enumerate(clean_sequence):
            print(f"Position {i}: {token}")
        
        # Add positions based on exact token matches
        for i, token in enumerate(clean_sequence):
            token_lower = token.lower().strip()
            for group_name, keywords in semantic_groups.items():
                if any(keyword.lower() in token_lower for keyword in keywords):
                    if group_name not in token_positions:
                        token_positions[group_name] = []
                    token_positions[group_name].append(i)
    
    return token_positions

def test_token_matching(model, word):
    """
    Test token matching for a specific word.
    """
    print(f"\nTesting token matching for '{word}':")
    
    # Get tokens with BOS
    tokens_with_bos = model.to_tokens(word)
    print(f"Tokens with BOS: {tokens_with_bos.tolist()}")
    print(f"Decoded with BOS: {model.to_string(tokens_with_bos[0])}")
    
    # Get tokens without BOS
    text = f" {word}"  # Add space to prevent BOS
    tokens_no_bos = model.to_tokens(text)
    print(f"Tokens without BOS: {tokens_no_bos.tolist()}")
    print(f"Decoded without BOS: {model.to_string(tokens_no_bos[0])}")
    
    return tokens_with_bos, tokens_no_bos

# Test the token matching
def run_token_tests(model):
    """
    Run token matching tests on key words.
    """
    test_words = ['correct', 'incorrect', 'lying', 'truth', 'killed', 'surviving']
    print("\nRunning token matching tests...")
    
    for word in test_words:
        with_bos, no_bos = test_token_matching(model, word)

def get_last_real_positions(attention_masks):
    """
    Get the position of the last real token (before padding) for each sequence in batch.
    
    Args:
        attention_masks: torch.Tensor of shape (batch_size, seq_len)
    Returns:
        torch.Tensor of shape (batch_size,) containing last real positions
    """
    if attention_masks.dim() == 1:
        attention_masks = attention_masks.unsqueeze(0)
    return attention_masks.sum(dim=1) - 1



## Logit Diff Batch
def logit_diff_batch(logits, last_positions):
    """
    Calculate logit difference at the specified positions for each sequence in batch.
    
    Args:
        logits: torch.Tensor of shape (batch_size, seq_len, vocab_size)
        last_positions: torch.Tensor of shape (batch_size,) containing positions
    """
    batch_size = logits.shape[0]
    results = []
    
    for i in range(batch_size):
        target_token_logits = logits[i, last_positions[i]]
        top_values, top_indices = torch.topk(target_token_logits, k=5)
        
        # Print top tokens for debugging (only for first few in batch)
        if i < 3:  # Show first 3 examples
            print(f"\nTop 5 predictions for sequence {i} at position {last_positions[i]}")
            top_tokens = [model.to_string(torch.tensor([idx])) for idx in top_indices]
            for token, value in zip(top_tokens, top_values.tolist()):
                print(f"  {token}: {value:.4f}")
        
        results.append(top_values[0] - top_values[1])
    
    return torch.stack(results)

def mc_logit_diff_batch(logits, last_positions, answer_tokens, is_corrupted=False):
    """
    Calculate logit difference, accounting for whether we want truth or lies.
    
    Args:
        logits: torch.Tensor of shape (batch_size, seq_len, vocab_size)
        last_positions: torch.Tensor of shape (batch_size,) containing positions
        answer_tokens: tensor of shape (5,) where answer_tokens[0] is the correct answer token 
                      and answer_tokens[1:] are incorrect answer tokens
        is_corrupted: bool indicating if we're evaluating corrupted/lying case
    """
    batch_size = logits.shape[0]
    results = []
    
    assert len(answer_tokens) == 5, "Must provide exactly 5 answer tokens"
    
    for i in range(batch_size):
        target_token_logits = logits[i, last_positions[i]]
        mc_logits = target_token_logits[answer_tokens]
        
        correct_logit = mc_logits[0]  # First token is truth answer
        incorrect_logits = mc_logits[1:]  # Remaining tokens are incorrect answers
        highest_incorrect = torch.max(incorrect_logits)
        
        if is_corrupted:
            # For corrupted case: want highest_incorrect - correct
            # (positive value means model prefers lying)
            diff = highest_incorrect - correct_logit
        else:
            # For clean case: want correct - highest_incorrect
            # (positive value means model prefers truth)
            diff = correct_logit - highest_incorrect
        
        if i < 3:  # Debug printing
            print(f"\nPredictions for sequence {i} at position {last_positions[i]}")
            print(f"  Truth answer ({model.to_string(torch.tensor([answer_tokens[0]]))}): {correct_logit:.4f}")
            print(f"  Highest incorrect: {highest_incorrect:.4f}")
            print(f"  Difference ({'corrupted' if is_corrupted else 'clean'}): {diff:.4f}")
            print(f"  Model prefers: {'lying' if diff > 0 and is_corrupted else 'truth'}")
        
        results.append(diff)
    
    return torch.stack(results)



def perform_semantic_causal_patching(model, clean_tokens, corrupt_tokens, 
                                   clean_attention_mask, corrupt_attention_mask,
                                   answer_tokens,
                                   batch_size=16):
    """
    Perform causal patching with token_patching per sequence length and correct last token handling.
    """
    print("\nStarting semantic causal patching...")
    
    # Get last real positions for clean and corrupt sequences
    clean_last_positions = get_last_real_positions(clean_attention_mask)
    corrupt_last_positions = get_last_real_positions(corrupt_attention_mask)
    
    print(f"Clean last positions: {clean_last_positions.tolist()}")
    print(f"Corrupt last positions: {corrupt_last_positions.tolist()}")
    
    semantic_groups = get_semantic_groups(model, clean_tokens, corrupt_tokens)
    
    if not any(positions for positions in semantic_groups.values()):
        print("Warning: No semantic groups found! Check token matching logic.")
        return None, semantic_groups
    
    print("\nRunning model with caching...")
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)

    print("\nCalculating mc baseline logit differences...")
    clean_logit_diffs = mc_logit_diff_batch(clean_logits, clean_last_positions, answer_tokens, is_corrupted=False)
    corrupt_logit_diffs = mc_logit_diff_batch(corrupt_logits, corrupt_last_positions, answer_tokens, is_corrupted=True)
    
    print(f"Clean logit diffs: {clean_logit_diffs}")
    print(f"Corrupt logit diffs: {corrupt_logit_diffs}")

    # Initialize results containers
    group_effects = {
        group: {
            'attn': torch.zeros((model.cfg.n_layers, len(clean_last_positions)), device='cpu'),
            'mlp': torch.zeros((model.cfg.n_layers, len(clean_last_positions)), device='cpu')
        }
        for group in semantic_groups
    }

    for layer in tqdm(range(model.cfg.n_layers), desc="Processing layers"):
        print(f"\nProcessing layer {layer}")
        
        for group_name, positions in semantic_groups.items():
            if not positions:
                continue
            
            print(f"\n  Processing group: {group_name} with positions: {positions}")
            
            def patch_group_attention(attn_out, hook):
                clean_attn = clean_cache[f"blocks.{layer}.hook_attn_out"].to(attn_out.device)
                patched = attn_out.clone()
                mask = torch.zeros_like(patched, dtype=torch.bool)
                mask[:, positions] = True
                return torch.where(mask, clean_attn, patched)

            def patch_group_mlp(mlp_out, hook):
                clean_mlp = clean_cache[f"blocks.{layer}.hook_mlp_out"].to(mlp_out.device)
                patched = mlp_out.clone()
                mask = torch.zeros_like(patched, dtype=torch.bool)
                mask[:, positions] = True
                return torch.where(mask, clean_mlp, patched)

            # Patch attention
            print("    Patching attention...")
            with torch.no_grad():
                attn_hooks = [(f"blocks.{layer}.hook_attn_out", patch_group_attention)]
                patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=attn_hooks)
                attn_diffs = mc_logit_diff_batch(patched_logits, corrupt_last_positions, answer_tokens, is_corrupted=True)
                
                # Calculate and store effects for each sequence
                for i in range(len(clean_last_positions)):
                    if abs(clean_logit_diffs[i] - corrupt_logit_diffs[i]) > 1e-6:
                        effect = (attn_diffs[i] - corrupt_logit_diffs[i]) / (clean_logit_diffs[i] - corrupt_logit_diffs[i])
                        effect = max(0.0, min(1.0, effect.item()))
                    else:
                        effect = 0.5
                    group_effects[group_name]['attn'][layer, i] = effect

            # Patch MLP
            print("    Patching MLP...")
            with torch.no_grad():
                mlp_hooks = [(f"blocks.{layer}.hook_mlp_out", patch_group_mlp)]
                patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=mlp_hooks)
                mlp_diffs = mc_logit_diff_batch(patched_logits, corrupt_last_positions, answer_tokens, is_corrupted=True)
                
                # Calculate and store effects for each sequence
                for i in range(len(clean_last_positions)):
                    if abs(clean_logit_diffs[i] - corrupt_logit_diffs[i]) > 1e-6:
                        effect = (mlp_diffs[i] - corrupt_logit_diffs[i]) / (clean_logit_diffs[i] - corrupt_logit_diffs[i])
                        effect = max(0.0, min(1.0, effect.item()))
                    else:
                        effect = 0.5
                    group_effects[group_name]['mlp'][layer, i] = effect

            torch.cuda.empty_cache()

    # Average effects across batch
    for group_name in group_effects:
        group_effects[group_name]['attn'] = group_effects[group_name]['attn'].mean(dim=1)
        group_effects[group_name]['mlp'] = group_effects[group_name]['mlp'].mean(dim=1)

    return group_effects, semantic_groups

def visualize_semantic_effects(group_effects, model):
    """
    Create visualization with error bands for semantic group causal effects.
    """
    if not group_effects:
        print("No effects to visualize!")
        return None

    num_layers = len(next(iter(group_effects.values()))['attn']['mean'])
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Attention Effects by Semantic Group", "MLP Effects by Semantic Group"),
        vertical_spacing=0.15
    )

    # Define a fixed color palette
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99FFCC', '#FFB366']
    
    for i, (group_name, effects) in enumerate(group_effects.items()):
        color = colors[i % len(colors)]  # Cycle through colors if more groups than colors
        
        # Add attention trace with error bands
        attn_mean = effects['attn']['mean'].numpy()
        attn_std = effects['attn']['std'].numpy()
        
        # Create fill color with transparency
        fill_color = f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)'
        
        # Upper bound
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=attn_mean + attn_std,
                fill=None,
                mode='lines',
                line=dict(color=color, width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Lower bound
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=attn_mean - attn_std,
                fill='tonexty',
                mode='lines',
                line=dict(color=color, width=0),
                fillcolor=fill_color,
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Main attention line
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=attn_mean,
                name=f"{group_name} (attn)",
                line=dict(color=color),
                mode='lines+markers',
                hovertemplate=(
                    "Layer: %{x}<br>" +
                    "Effect: %{y:.4f} ± %{customdata[0]:.4f}<br>" +
                    "Group: " + group_name + " (attention)<br>" +
                    "<extra></extra>"
                ),
                customdata=np.column_stack([attn_std])
            ),
            row=1, col=1
        )

        # MLP effects
        mlp_mean = effects['mlp']['mean'].numpy()
        mlp_std = effects['mlp']['std'].numpy()
        
        # Upper bound
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=mlp_mean + mlp_std,
                fill=None,
                mode='lines',
                line=dict(color=color, width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        
        # Lower bound
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=mlp_mean - mlp_std,
                fill='tonexty',
                mode='lines',
                line=dict(color=color, width=0),
                fillcolor=fill_color,
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        
        # Main MLP line
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=mlp_mean,
                name=f"{group_name} (mlp)",
                line=dict(color=color, dash='dash'),
                mode='lines+markers',
                hovertemplate=(
                    "Layer: %{x}<br>" +
                    "Effect: %{y:.4f} ± %{customdata[0]:.4f}<br>" +
                    "Group: " + group_name + " (MLP)<br>" +
                    "<extra></extra>"
                ),
                customdata=np.column_stack([mlp_std])
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        height=1000,
        width=1200,
        title={
            'text': 'Causal Effects by Semantic Group',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            bordercolor="Black",
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Update axes
    for i in [1, 2]:
        fig.update_xaxes(
            title_text="Layer",
            gridcolor='lightgray',
            zerolinecolor='lightgray',
            row=i, col=1
        )
        fig.update_yaxes(
            title_text="Effect Strength",
            range=[0, 1],
            gridcolor='lightgray',
            zerolinecolor='lightgray',
            row=i, col=1
        )

    return fig

def save_and_display_results(group_effects, semantic_groups, model, save_path="patching_results.html"):
    """
    Save and display the results of the analysis.
    """
    print("\nPreparing results summary...")
    
    # Create detailed summary
    summary = []
    summary.append("=== Semantic Causal Patching Analysis Results ===\n")
    
    for group_name, positions in semantic_groups.items():
        summary.append(f"\nGroup: {group_name}")
        
        if group_name in group_effects:
            effects = group_effects[group_name]
            
            # Get top layers for attention
            attn_values = effects['attn']['mean'].numpy()
            top_attn_layers = np.argsort(attn_values)[-3:][::-1]
            
            # Get top layers for MLP
            mlp_values = effects['mlp']['mean'].numpy()
            top_mlp_layers = np.argsort(mlp_values)[-3:][::-1]
            
            summary.append("  Top Attention Layers:")
            for layer in top_attn_layers:
                summary.append(f"    Layer {layer}: {attn_values[layer]:.4f}")
                
            summary.append("  Top MLP Layers:")
            for layer in top_mlp_layers:
                summary.append(f"    Layer {layer}: {mlp_values[layer]:.4f}")
                
            summary.append(f"  Average Attention Effect: {attn_values.mean():.4f}")
            summary.append(f"  Average MLP Effect: {mlp_values.mean():.4f}")
        else:
            summary.append("  No effects calculated for this group")
    
    # Save summary to file
    with open("patching_summary.txt", "w") as f:
        f.write("\n".join(summary))
    
    # Create and save visualization
    fig = visualize_semantic_effects(group_effects, model)
    if fig is not None:
        fig.write_html(save_path)
        print(f"\nResults saved to {save_path} and patching_summary.txt")
        return fig
    else:
        print("\nNo visualization created due to missing effects")
        return None

# Update the analyze_semantic_effects function to use the new visualization
def analyze_semantic_effects(model, test_dataset):
    """
    Main analysis function updated for multiple batches.
    """
    print("Starting analysis...")
    
    num_batches = len(test_dataset[0])  # Get total number of batches
    all_group_effects = []
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        # Extract data for the current batch
        clean_tokens = test_dataset[0][batch_idx]
        corrupt_tokens = test_dataset[1][batch_idx]
        clean_attention_mask = test_dataset[6][batch_idx]
        corrupt_attention_mask = test_dataset[7][batch_idx]
        answer_tokens = torch.cat([test_dataset[2][batch_idx].unsqueeze(0), test_dataset[3][batch_idx]])
        
        # Ensure tokens and attention masks are 2D tensors
        if clean_tokens.dim() == 1:
            clean_tokens = clean_tokens.unsqueeze(0)
        if corrupt_tokens.dim() == 1:
            corrupt_tokens = corrupt_tokens.unsqueeze(0)
        if clean_attention_mask.dim() == 1:
            clean_attention_mask = clean_attention_mask.unsqueeze(0)
        if corrupt_attention_mask.dim() == 1:
            corrupt_attention_mask = corrupt_attention_mask.unsqueeze(0)
        
        # Perform semantic causal patching for this batch
        batch_effects, semantic_groups = perform_semantic_causal_patching(
            model, clean_tokens, corrupt_tokens,
            clean_attention_mask, corrupt_attention_mask,
            answer_tokens
        )
        
        if batch_effects is not None:
            all_group_effects.append(batch_effects)
    
    # Aggregate results across batches
    aggregated_effects = {}
    for group_name in all_group_effects[0].keys():
        aggregated_effects[group_name] = {
            'attn': {
                'mean': torch.stack([batch[group_name]['attn'] for batch in all_group_effects]).mean(dim=0),
                'std': torch.stack([batch[group_name]['attn'] for batch in all_group_effects]).std(dim=0)
            },
            'mlp': {
                'mean': torch.stack([batch[group_name]['mlp'] for batch in all_group_effects]).mean(dim=0),
                'std': torch.stack([batch[group_name]['mlp'] for batch in all_group_effects]).std(dim=0)
            }
        }
    
    return aggregated_effects, semantic_groups

# %%
group_effects, semantic_groups = analyze_semantic_effects(model, test_dataset)
# %%
# Create visualization and save results
print("\nCreating visualization and saving results...")
fig = save_and_display_results(group_effects, semantic_groups, model)


# %%

fig.show()










# %%

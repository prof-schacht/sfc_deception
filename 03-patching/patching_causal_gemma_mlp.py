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
    assert clean_dataset is not None or corrupted_dataset is not None, "At least one dataset must be provided"
    return_values = []
    
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

def perform_semantic_causal_patching(model, clean_tokens, corrupt_tokens, 
                                   clean_attention_mask, corrupt_attention_mask, batch_size=16):
    """
    Perform causal patching with batching and correct last token handling.
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

    print("\nCalculating baseline logit differences...")
    clean_logit_diffs = logit_diff_batch(clean_logits, clean_last_positions)
    corrupt_logit_diffs = logit_diff_batch(corrupt_logits, corrupt_last_positions)
    
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
                attn_diffs = logit_diff_batch(patched_logits, corrupt_last_positions)
                
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
                mlp_diffs = logit_diff_batch(patched_logits, corrupt_last_positions)
                
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
    Create visualization for semantic group causal effects.
    """
    if not group_effects:
        print("No effects to visualize!")
        return None

    # Get number of layers from the first group's attention effects
    first_group = next(iter(group_effects.values()))
    num_layers = len(first_group['attn'])
    
    print(f"Creating visualization for {len(group_effects)} groups across {num_layers} layers")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Attention Effects by Semantic Group", "MLP Effects by Semantic Group"),
        vertical_spacing=0.15
    )

    # Create traces for each semantic group
    colors = px.colors.qualitative.Set3[:len(group_effects)]
    
    for i, (group_name, effects) in enumerate(group_effects.items()):
        color = colors[i]
        
        # Convert to numpy arrays
        attn_values = effects['attn'].numpy()
        mlp_values = effects['mlp'].numpy()
        
        print(f"\nGroup: {group_name}")
        print(f"Attention range: {attn_values.min():.4f} to {attn_values.max():.4f}")
        print(f"MLP range: {mlp_values.min():.4f} to {mlp_values.max():.4f}")
        
        # Add attention trace
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=attn_values,
                name=f"{group_name} (attn)",
                line=dict(color=color),
                mode='lines+markers',
                hovertemplate=(
                    "Layer: %{x}<br>" +
                    "Effect: %{y:.4f}<br>" +
                    "Group: " + group_name + " (attention)<br>" +
                    "<extra></extra>"
                )
            ),
            row=1, col=1
        )
        
        # Add MLP trace
        fig.add_trace(
            go.Scatter(
                x=list(range(num_layers)),
                y=mlp_values,
                name=f"{group_name} (mlp)",
                line=dict(color=color, dash='dash'),
                mode='lines+markers',
                hovertemplate=(
                    "Layer: %{x}<br>" +
                    "Effect: %{y:.4f}<br>" +
                    "Group: " + group_name + " (MLP)<br>" +
                    "<extra></extra>"
                )
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
            attn_values = effects['attn'].numpy()
            top_attn_layers = np.argsort(attn_values)[-3:][::-1]
            
            # Get top layers for MLP
            mlp_values = effects['mlp'].numpy()
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
def analyze_semantic_effects(model, test_dataset, batch_idx=0):
    """
    Main analysis function updated for batched data.
    """
    print("Starting analysis...")
    
    # Extract data for the specified batch
    clean_tokens = test_dataset[0][batch_idx]
    corrupt_tokens = test_dataset[1][batch_idx]
    clean_attention_mask = test_dataset[6][batch_idx]
    corrupt_attention_mask = test_dataset[7][batch_idx]
    
    print(f"\nToken shapes - Clean: {clean_tokens.shape}, Corrupt: {corrupt_tokens.shape}")
    print(f"Attention mask shapes - Clean: {clean_attention_mask.shape}, Corrupt: {corrupt_attention_mask.shape}")
    
    # Ensure tokens are 2D tensors
    if clean_tokens.dim() == 1:
        clean_tokens = clean_tokens.unsqueeze(0)
    if corrupt_tokens.dim() == 1:
        corrupt_tokens = corrupt_tokens.unsqueeze(0)
    
    # Ensure attention masks are 2D tensors
    if clean_attention_mask.dim() == 1:
        clean_attention_mask = clean_attention_mask.unsqueeze(0)
    if corrupt_attention_mask.dim() == 1:
        corrupt_attention_mask = corrupt_attention_mask.unsqueeze(0)
    
    print(f"\nAdjusted token shapes - Clean: {clean_tokens.shape}, Corrupt: {corrupt_tokens.shape}")
    print(f"Adjusted attention mask shapes - Clean: {clean_attention_mask.shape}, Corrupt: {corrupt_attention_mask.shape}")
    
    # Perform semantic causal patching
    print("\nPerforming semantic causal patching analysis...")
    group_effects, semantic_groups = perform_semantic_causal_patching(
        model, clean_tokens, corrupt_tokens,
        clean_attention_mask, corrupt_attention_mask
    )
    
    if group_effects is None:
        print("Error: No effects calculated. Check the debugging output above.")
        return
    
 
    
    if fig is not None:
        fig.show()
    
    return group_effects, semantic_groups

# %%
group_effects, semantic_groups = analyze_semantic_effects(model, test_dataset)

# Create visualization and save results
print("\nCreating visualization and saving results...")
fig = save_and_display_results(group_effects, semantic_groups, model)

# %%







# %% [markdown]
# IDEA: Check if lying or true telling starts in different layers. 
# Expectation: Through compexity lying does need it should be expected to start in earlier layers. 
# Base Idea: An information-theoretic study of lying in LLMs Url: https://openreview.net/forum?id=9AM5i1wWZZ
# Experiment: Use Logit Lens to calculate entropy, KL-divergence, and Probability
# Experiment: Based on LogitLens, later we could extend it to tuned lens and train a tuned lens model. 

# Some Explanation towards the Metrics:
# Entropy: Measures the uncertainty of the probability distribution.
#   Interpretation: If the model is certain about its prediction, the entropy will be low.
#   If the model is uncertain, the entropy will be high.
# KL-divergence: Measures the difference between two probability distributions.
#   Interpretation: If the model is certain about its prediction, the KL-divergence will be low.
#   If the model is uncertain, the KL-divergence will be high.
# Target token probability: Measures the probability of the correct answer.
#   Interpretation: If the model is certain about its prediction, the target token probability will be high.
#   If the model is uncertain, the target token probability will be low.


# Extend Analysis with statistical hypothesis testing
# Extend in with tune lens trained on gemma 2-9b-it


# Changes I performed:

# 1. **Calculate_metrics Changes**:
# - Instead of looking at the entire vocabulary probability distribution, we now focus only on the MC answer tokens (A-E)
# - The entropy calculation now only considers these 5 possible answers instead of the full vocabulary
# - This gives us a more meaningful measure of uncertainty between actual choices rather than across all possible tokens
# - The KL-divergence also only compares distributions over these answer choices
# - Target probability is now properly normalized within the answer space

# 2. **Process_batch Changes**:
# - Modified to handle MC answer tokens specifically
# - When getting predicted answers, it now looks at probabilities for A-E tokens only
# - Converts numeric indices to letter answers (e.g., 0 -> 'A', 1 -> 'B', etc.)

# 3. **Generate_dataset Changes**:
# - Now generates  lie versions for each truth case
# - For each question, it creates:
#   * One truth version with correct answer
#   * One lie version with a random incorrect answer
# - This gives a more complete picture of lying behavior
# - Preserves the MC format and proper answer labels

# 4. **Key Conceptual Changes**:
# - Moving from binary to MC analysis better represents real lying behavior
# - Entropy becomes more meaningful as it measures uncertainty across actual choices

# 5. **Benefits of These Changes**:
# - More realistic analysis of lying behavior


# %%

import torch
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
from torch.nn.functional import softmax, log_softmax
import torch.nn.functional as F
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
from datasets import load_dataset
import sys
import os
import random
import json
from typing import List, Dict, Tuple
import random

# Add the parent directory (sfc_deception) to sys.path
sys.path.append(os.path.abspath(os.path.join('../..')))

import utils.prompts as prompt_utils

import gc

gc.collect()
torch.cuda.empty_cache()

# %%
# Parameters
#model_name = "HuggingFaceH4/zephyr-7b-beta"
model_name = "google/gemma-2-9b-it"
#model_name = "gpt2-small"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# %%
model = HookedTransformer.from_pretrained(model_name)
# %%

@dataclass
class LayerMetrics:
    """Data class to store metrics for a single layer"""
    entropy: float
    kl_divergence: float
    target_probability: float
    top_k_probs: List[float]
    top_k_tokens: List[int]

@dataclass
class AnalysisResult:
    """Data class to store analysis results across all layers"""
    metrics_per_layer: List[LayerMetrics]
    prompt_type: str  # 'truth' or 'lie'
    correct_token: str
    predicted_token: str

class TransformerActivationAnalyzer:
    def __init__(self, hooked_model: HookedTransformer, device: Optional[str] = None):
        """Initialize analyzer with model and required components."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self.model = hooked_model
        self.model.to(self.device)
        self.num_layers = self.model.cfg.n_layers
        # For debugging 
        self.batch_activations = []
        self.list_of_probs_which_not_worked = []
        self.activations_before_logit_lens = []
        self.activations_after_logit_lens = []
        self.logits_after_unembed = []
        self.list_of_probs_after_softmax = []
        
    def get_layer_activations(self, tokens: torch.Tensor, token_position: int = -1, attention_mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Get activations for all layers at the last non-pad position for each element in the batch.
        
        Args:
            tokens: Input tokens [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            List of tensors with shape [batch_size, hidden_size]
        """
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens, attention_mask=attention_mask)
        
            # Get the last non-pad position for each element in the batch
            if attention_mask is not None:
                last_non_pad_positions = attention_mask.sum(dim=1) + token_position  # [batch_size]
                #print(f"last_non_pad_positions: {last_non_pad_positions}")
            else:
                # If no attention mask is provided, assume all tokens are non-pad
                last_non_pad_positions = torch.full((tokens.shape[0],), tokens.shape[1] - 1, device=tokens.device)
        
            # Get activations from each layer
            activations = []
            for layer in range(self.num_layers):
                key = f'blocks.{layer}.hook_resid_pre'
                if key in cache:
                    # Get activation at the last non-pad position for each element in the batch
                    layer_activations = cache[key]  # [batch_size, seq_len, hidden_size]
                    batch_size, seq_len, hidden_size = layer_activations.shape
                
                    # Create indices for gathering
                    batch_indices = torch.arange(batch_size, device=tokens.device)
                
                    # Gather activations at the last non-pad positions
                    activation = layer_activations[batch_indices, last_non_pad_positions]  # [batch_size, hidden_size]
                    activations.append(activation)
                    
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            # For debugging
            # print(f"activations shape: {activations[0].shape}")
            # print(f"activations[0]: {len(activations)}")
            # print(f"activations[0][0]: {activations[0][0].shape}")
        
        return activations  # list [batch_size - list num_layers- torch [activation_neurons]]
    
    def apply_logit_lens(self, 
                        activation: torch.Tensor,
                        normalize: bool = True) -> torch.Tensor:
        """
        Apply logit lens to get token probabilities.
        Projects layer activations directly to vocabulary space.
        """
        
        ## debugging
        self.activations_before_logit_lens.append(activation)
        
        with torch.no_grad():
            if normalize:
                # Apply final layer normalization
                activation = self.model.ln_final(activation)
        
            ## debugging
            self.activations_after_logit_lens.append(activation)
        
            # Project to vocabulary space
            logits = self.model.unembed(activation)

            ## debugging
            self.logits_after_unembed.append(logits)

            return softmax(logits, dim=-1)
    
    def calculate_metrics(self, 
                     probs: torch.Tensor, 
                     final_layer_probs: Optional[torch.Tensor] = None,
                     target_token: Optional[int] = None,
                     k: int = 10) -> LayerMetrics:
        """
        Calculate information-theoretic metrics with support for multiple choice questions
        """
        eps = 1e-10
        
        with torch.no_grad():
            # Get probabilities for all possible answer tokens (A through E)
            mc_tokens = [self.model.to_single_token(letter) for letter in ['A', 'B', 'C', 'D', 'E']]
            answer_probs = probs[0, mc_tokens]
        
            # Normalize probabilities across just the answer choices
            answer_probs = F.softmax(answer_probs.log() + eps, dim=-1)
        
            # Calculate entropy only over answer choices
            entropy = -(answer_probs * torch.log(answer_probs + eps)).sum().item()
        
            # KL divergence calculation
            if final_layer_probs is not None:
                final_answer_probs = final_layer_probs[0, mc_tokens]
                final_answer_probs = F.softmax(final_answer_probs.log() + eps, dim=-1)
                kl_div = torch.sum(answer_probs * (torch.log(answer_probs + eps) - 
                                                torch.log(final_answer_probs + eps))).item()
            else:
                # Use uniform distribution over answer choices
                uniform_probs = torch.ones_like(answer_probs) / len(mc_tokens)
                kl_div = torch.sum(answer_probs * (torch.log(answer_probs + eps) - 
                                                torch.log(uniform_probs + eps))).item()
        
            # Target token probability (from normalized answer probabilities)
            target_idx = mc_tokens.index(target_token) if target_token in mc_tokens else None
            target_prob = answer_probs[target_idx].item() if target_idx is not None else None
        
            # Get top-k among answer choices
            top_k = torch.topk(answer_probs, min(k, len(mc_tokens)))
        
        return LayerMetrics(
            entropy=entropy,
            kl_divergence=kl_div,
            target_probability=target_prob,
            top_k_probs=top_k.values.tolist(),
            top_k_tokens=[mc_tokens[i] for i in top_k.indices.tolist()]
        )

    def process_batch(self, 
                    prompts: List[str],
                    correct_answers: List[str],
                    prompt_types: List[str],
                    batch_size: int = 32) -> List[AnalysisResult]:
        """Modified process_batch to handle multiple choice questions"""
        results = []
        
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i + batch_size]
            batch_answers = correct_answers[i:i + batch_size]
            batch_types = prompt_types[i:i + batch_size]
            
            # Process tokens and get attention masks (same as before)
            tokens = [self.model.to_tokens(prompt, truncate=True) for prompt in batch_prompts]
            max_len = max(t.shape[1] for t in tokens)
            
            padded_tokens = []
            attention_masks = []
            
            for t in tokens:
                padding_length = max_len - t.shape[1]
                padded_t = F.pad(t, (0, padding_length), value=self.model.tokenizer.pad_token_id)
                padded_tokens.append(padded_t)
                
                mask = torch.ones_like(padded_t)
                mask[:, -padding_length:] = 0
                attention_masks.append(mask)
            
            batch_tokens = torch.stack(padded_tokens).squeeze(1).to(self.device)
            batch_attention_masks = torch.stack(attention_masks).squeeze(1).to(self.device)
            
            # Get activations
            batch_activations = self.get_layer_activations(batch_tokens, attention_mask=batch_attention_masks)
            
            # Process each prompt
            for j in range(len(batch_prompts)):
                metrics_per_layer = []
                target_token = self.model.to_single_token(batch_answers[j])
                
                # Get final layer probabilities
                final_probs = self.apply_logit_lens(batch_activations[-1][j:j+1])
                
                # Calculate metrics for each layer
                for layer_idx, layer_activation in enumerate(batch_activations):
                    probs = self.apply_logit_lens(layer_activation[j:j+1])
                    metrics = self.calculate_metrics(probs, final_probs, target_token)
                    metrics_per_layer.append(metrics)
                
                # Get predicted answer (from A-E)
                mc_tokens = [self.model.to_single_token(letter) for letter in ['A', 'B', 'C', 'D', 'E']]
                answer_probs = final_probs[0, mc_tokens]
                predicted_idx = answer_probs.argmax().item()
                predicted_token = chr(65 + predicted_idx)  # Convert back to letter
                
                results.append(AnalysisResult(
                    metrics_per_layer=metrics_per_layer,
                    prompt_type=batch_types[j],
                    correct_token=batch_answers[j],
                    predicted_token=predicted_token
                ))
                
                
        
        return results
   
class AnalysisVisualizer:
    """Utility class for visualizing analysis results"""
    
    @staticmethod
    def plot_metric_comparison(truth_results: List[AnalysisResult],
                             lie_results: List[AnalysisResult],
                             metric_name: str,
                             title: Optional[str] = None,
                             scale: str = 'linear',
                             save_path: Optional[str] = None):
        """Plot comparison of metrics between truth and lie conditions"""
        
        # Extract metrics for all layers
        def extract_metric(results, metric_name):
            return np.array([[getattr(layer_metrics, metric_name) 
                            for layer_metrics in result.metrics_per_layer]
                           for result in results])
        
        truth_values = extract_metric(truth_results, metric_name)
        lie_values = extract_metric(lie_results, metric_name)
        
        # Calculate statistics
        truth_median = np.median(truth_values, axis=0)
        truth_q25 = np.percentile(truth_values, 25, axis=0)
        truth_q75 = np.percentile(truth_values, 75, axis=0)
        
        lie_median = np.median(lie_values, axis=0)
        lie_q25 = np.percentile(lie_values, 25, axis=0)
        lie_q75 = np.percentile(lie_values, 75, axis=0)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        x = np.arange(len(truth_median))
        
        plt.plot(x, truth_median, color='blue', label='truth median')
        plt.fill_between(x, truth_q25, truth_q75, color='blue', alpha=0.2)
        
        plt.plot(x, lie_median, color='orange', label='lie median')
        plt.fill_between(x, lie_q25, lie_q75, color='orange', alpha=0.2)
        
        plt.xlabel('Layer')
        plt.ylabel(metric_name.replace('_', ' ').title())
        if scale == 'log':
            plt.yscale('log')
        
        plt.grid(True)
        plt.legend()
        
        if title:
            plt.title(title)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        

# %%
# Dataloading / Demodata


# %%
# Generates  Lying version with random answers
def generate_dataset(dataset_name: str) -> Tuple[List[str], List[str], List[str]]:
    """Modified dataset generation for multiple choice questions"""
    dataset = load_dataset(dataset_name)
    train_data = dataset['train']
    
    all_prompts = []
    all_answers = []
    all_types = []
    
    for item in train_data:
        # Format question and choices
        prompt = item['question'] + "\n"
        prompt += "\n".join([f"{label}) {text}" for label, text in 
                           zip(item['choices']['label'], item['choices']['text'])]) + "\n"
        prompt += prompt_utils.OUTPUT_SINGLE_LETTER + " Answer:"
        
        # Truth version
        all_prompts.append("<bos>" + prompt_utils.TRUTH_OR_USER_KILLED + prompt)
        all_answers.append(item["answerKey"])
        all_types.append("truth")
        
        # Lie versions - now create multiple lie versions with different wrong answers
        possible_lies = [label for label in item['choices']['label'] 
                        if label != item["answerKey"]]
        # Add one random lie version with an incorrect answer
        lie_answer = random.choice(possible_lies)
        all_prompts.append("<bos>" + prompt_utils.LIE_OR_USER_KILLED + prompt)
        all_answers.append(lie_answer)
        all_types.append("lie")
    
    return all_prompts, all_answers, all_types


all_prompts, all_answers, all_types = generate_dataset("drsis/deception-commonsense_qa_wo_chat")

# %%
# Create 5 True data and 5 corresponding lying data as a test case
all_prompts = all_prompts[:40]
all_answers = all_answers[:40]
all_types = all_types[:40]

for i in range(3):
    print(f"Prompt {i+1}: {all_prompts[i]}")
    print(f"Answer {i+1}: {all_answers[i]}")
    print(f"Type {i+1}: {all_types[i]}")

# %%
# Initialize and run analysis
analyzer = TransformerActivationAnalyzer(model)
results = analyzer.process_batch(
    prompts=all_prompts,
    correct_answers=all_answers,
    prompt_types=all_types,
    batch_size=2,  # Small batch size for testing
    #max_new_tokens=1  # Only need one token for A/B answers
)


# %%
# Visualization
visualizer = AnalysisVisualizer()
visualizer.plot_metric_comparison(
    [r for r in results if r.prompt_type == 'truth'],
    [r for r in results if r.prompt_type == 'lie'],
    metric_name='entropy',
    scale='log',
    save_path='entropy_comparison.png',
    title="Entropy - Gemma 2-9b-it - 5 Questions True and 5 Lying"
)

visualizer.plot_metric_comparison(
    [r for r in results if r.prompt_type == 'truth'],
    [r for r in results if r.prompt_type == 'lie'],
    metric_name='kl_divergence',
    scale='log',
    save_path='kl_divergence_comparison.png',
    title="KL-divergence to last layer - Gemma 2-9b-it - 20 Questions True and 20 Lying"
)

visualizer.plot_metric_comparison(
    [r for r in results if r.prompt_type == 'truth'],
    [r for r in results if r.prompt_type == 'lie'],
    metric_name='target_probability',
    scale='linear',
    save_path='target_probability_comparison.png',
    title="Probability of predicted token- Gemma 2-9b-it - 20 Questions True and 20 Lying"
)



# %%



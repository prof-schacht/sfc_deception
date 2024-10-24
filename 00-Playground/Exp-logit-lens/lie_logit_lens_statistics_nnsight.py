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
import torch.nn.functional as F
from nnsight import LanguageModel
import matplotlib.pyplot as plt
from datasets import load_dataset
import sys
import os
import random

# Add the parent directory (sfc_deception) to sys.path
sys.path.append(os.path.abspath(os.path.join('../..')))

import utils.prompts as prompt_utils

import gc

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

class NnsightActivationAnalyzer:
    def __init__(self, model_name: str, device: Optional[str] = None):
        """Initialize analyzer with model and required components."""
        self.device = device if device else "cuda:1" if torch.cuda.is_available() else "cpu"
        self.model = LanguageModel(model_name, device_map=self.device)
        self.num_layers = self.model.config.num_hidden_layers
        
    def get_layer_activations(self, text_input: str) -> List[torch.Tensor]:
        """
        Get activations for all layers using nnsight's tracing mechanism.
        """
        activations = []
        
        with self.model.trace(text_input) as runner:
            # For each layer, save its output
            for layer_idx in range(self.num_layers):
                layer_output = self.model.model.layers[layer_idx].output.save()
                activations.append(layer_output)
                
        return [act.value for act in activations]

    def apply_logit_lens(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Apply logit lens using the model's lm_head through tracing.
        """
        with self.model.trace() as runner:
            # Use the model's lm_head to project to vocabulary space
            logits = self.model.lm_head(activation)
            return F.softmax(logits, dim=-1)

    def process_batch(self, 
                     prompts: List[str],
                     correct_answers: List[str],
                     prompt_types: List[str],
                     batch_size: int = 32) -> List[AnalysisResult]:
        """Process batch of prompts and compute metrics using nnsight's tracing"""
        results = []
        
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i + batch_size]
            batch_answers = correct_answers[i:i + batch_size]
            batch_types = prompt_types[i:i + batch_size]
            
            with self.model.trace() as tracer:
                for j, prompt in enumerate(batch_prompts):
                    with tracer.invoke(prompt) as invoker:
                        metrics_per_layer = []
                        
                        # Get activations for each layer
                        for layer_idx in range(self.num_layers):
                            layer = self.model.model.layers[layer_idx]
                            layer_output = layer.output.save()
                            
                            # Get final layer prediction for comparison
                            if layer_idx == self.num_layers - 1:
                                final_logits = self.model.lm_head.output.save()
                                final_probs = F.softmax(final_logits.value, dim=-1)
                            
                            # Calculate metrics
                            current_logits = self.model.lm_head(layer_output.value)
                            current_probs = F.softmax(current_logits, dim=-1)
                            
                            target_token = self.model.tokenizer.convert_tokens_to_ids(batch_answers[j])
                            metrics = self.calculate_metrics(
                                current_probs, 
                                final_probs if layer_idx < self.num_layers - 1 else None,
                                target_token
                            )
                            metrics_per_layer.append(metrics)
                        
                        # Get predicted answer from final layer
                        mc_tokens = [self.model.tokenizer.convert_tokens_to_ids(letter) for letter in ['A', 'B', 'C', 'D', 'E']]
                        answer_probs = final_probs[0, mc_tokens]
                        predicted_idx = answer_probs.argmax().item()
                        predicted_token = chr(65 + predicted_idx)
                        
                        results.append(AnalysisResult(
                            metrics_per_layer=metrics_per_layer,
                            prompt_type=batch_types[j],
                            correct_token=batch_answers[j],
                            predicted_token=predicted_token
                        ))
                        
                    gc.collect()
                    torch.cuda.empty_cache()
        
        return results

    def calculate_metrics(self, 
                        probs: torch.Tensor, 
                        final_layer_probs: Optional[torch.Tensor] = None,
                        target_token: Optional[int] = None,
                        k: int = 10) -> LayerMetrics:
        """Calculate metrics for multiple choice answers"""
        eps = 1e-10
        
        with torch.no_grad():
            # Get probabilities for multiple choice answers
            mc_tokens = [self.model.tokenizer.convert_tokens_to_ids(letter) for letter in ['A', 'B', 'C', 'D', 'E']]
            answer_probs = probs[0, mc_tokens]
            
            # Normalize probabilities
            answer_probs = F.softmax(answer_probs.log() + eps, dim=-1)
            
            # Calculate entropy
            entropy = -(answer_probs * torch.log(answer_probs + eps)).sum().item()
            
            # Calculate KL divergence
            if final_layer_probs is not None:
                final_answer_probs = final_layer_probs[0, mc_tokens]
                final_answer_probs = F.softmax(final_answer_probs.log() + eps, dim=-1)
                kl_div = torch.sum(answer_probs * (torch.log(answer_probs + eps) - 
                                                torch.log(final_answer_probs + eps))).item()
            else:
                uniform_probs = torch.ones_like(answer_probs) / len(mc_tokens)
                kl_div = torch.sum(answer_probs * (torch.log(answer_probs + eps) - 
                                                torch.log(uniform_probs + eps))).item()
            
            # Get target probability
            target_idx = mc_tokens.index(target_token) if target_token in mc_tokens else None
            target_prob = answer_probs[target_idx].item() if target_idx is not None else None
            
            # Get top-k probabilities
            top_k = torch.topk(answer_probs, min(k, len(mc_tokens)))
        
        return LayerMetrics(
            entropy=entropy,
            kl_divergence=kl_div,
            target_probability=target_prob,
            top_k_probs=top_k.values.tolist(),
            top_k_tokens=[mc_tokens[i] for i in top_k.indices.tolist()]
        )
   
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
        all_prompts.append("<bos>" + prompt_utils.TRUTH_OR_USER_KILLED + prompt + item["answerKey"])
        all_answers.append(item["answerKey"])
        all_types.append("truth")
        
        # Lie versions - now create multiple lie versions with different wrong answers
        possible_lies = [label for label in item['choices']['label'] 
                        if label != item["answerKey"]]
        # Add one random lie version with an incorrect answer
        lie_answer = random.choice(possible_lies)
        all_prompts.append("<bos>" + prompt_utils.LIE_OR_USER_KILLED + prompt + lie_answer)
        all_answers.append(lie_answer)
        all_types.append("lie")
    
    return all_prompts, all_answers, all_types


all_prompts, all_answers, all_types = generate_dataset("drsis/deception-commonsense_qa_wo_chat")

# %%
# Create 5 True data and 5 corresponding lying data as a test case
all_prompts = all_prompts[:10]
all_answers = all_answers[:10]
all_types = all_types[:10]

# %%
# Initialize analyzer
analyzer = NnsightActivationAnalyzer("google/gemma-2b-it")

# Generate dataset
prompts, answers, types = generate_dataset("drsis/deception-commonsense_qa_wo_chat")

# Run analysis
results = analyzer.process_batch(
    prompts=prompts[:100],  # Start with a small subset
    correct_answers=answers[:100],
    prompt_types=types[:100],
    batch_size=2
)



# %%



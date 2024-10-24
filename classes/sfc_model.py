from transformer_lens import ActivationCache
from sae_lens import SAE, HookedSAETransformer
import torch
import numpy as np
from tqdm import tqdm
import einops
from jaxtyping import Float, Int
from torch import Tensor
from enum import Enum

# utility to clear variables out of the memory & and clearing cuda cache
import gc
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

class NodeScoreType(Enum):
    ATTRIBUTION_PATCHING = 'ATP'
    INTEGRATED_GRADIENTS = 'INT_GRADS'

class AttributionPatching(Enum):
    NORMAL = 'TRADITIONAL' # Approximates the effect of patching activations from patched run to the clean run
    ZERO_ABLATION = 'ZERO_ABLATION' # Approximates the effect of zeroing out activations from the patched/clean run

def sample_dataset(start_idx=0, end_idx=-1, clean_dataset=None, corrupted_dataset=None):
    return_values = []

    for key in ['prompt', 'answer', 'answer_pos', 'attention_mask']:
        return_values.append(clean_dataset[key][start_idx:end_idx])
        return_values.append(corrupted_dataset[key][start_idx:end_idx])

    return return_values

class SFC_Gemma():
    def __init__(self, model, attach_saes=True, params_count=9, control_seq_len=1,
                sae_resid_release=None, sae_attn_release=None, sae_mlp_release=None):
        if sae_resid_release is None:
            sae_resid_release = f'gemma-scope-{params_count}b-pt-res-canonical'

        if sae_attn_release is None:
            sae_attn_release = f'gemma-scope-{params_count}b-pt-att-canonical'

        if sae_mlp_release is None:
            sae_mlp_release = f'gemma-scope-{params_count}b-pt-mlp-canonical'

        self.model = model
        self.cfg = model.cfg
        self.device = model.cfg.device
        self.control_seq_len = control_seq_len

        self.model.set_use_attn_in(True)
        self.model.set_use_attn_result(True)
        self.model.set_use_hook_mlp_in(True)
        self.model.set_use_split_qkv_input(True)

        self.n_layers = self.cfg.n_layers
        self.d_model = self.cfg.d_model

        # Initialize dictionary to store SAEs by type: resid, attn, mlp
        self.saes_dict = {
            'resid': [],
            'attn': [],
            'mlp': []
        }

        # Load all SAEs into the dictionary
        self.saes_dict['resid'] = [
            self._load_sae(sae_resid_release, f'layer_{i}/width_16k/canonical') for i in range(self.n_layers)
        ]
        self.saes_dict['attn'] = [
            self._load_sae(sae_attn_release, f'layer_{i}/width_16k/canonical') for i in range(self.n_layers)
        ]
        self.saes_dict['mlp'] = [
            self._load_sae(sae_mlp_release, f'layer_{i}/width_16k/canonical') for i in range(self.n_layers)
        ]
        self.saes = self.saes_dict['resid'] + self.saes_dict['mlp'] + self.saes_dict['attn']
        

        # Attach all SAEs
        if attach_saes:
            self.add_saes()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_node_scores_for_normal_patching(self, clean_dataset, patched_dataset, batch_size=100, total_batches=None,
                                                score_type: NodeScoreType = NodeScoreType.ATTRIBUTION_PATCHING, N_IG = 10):
        n_prompts, seq_len = clean_dataset['prompt'].shape
        assert n_prompts == clean_dataset['answer'].shape[0] == patched_dataset['answer'].shape[0]

        prompts_to_process = n_prompts if total_batches is None else batch_size * total_batches

        if total_batches is None:
            total_batches = n_prompts // batch_size

            if n_prompts % batch_size != 0:
                total_batches += 1

        metrics_clean = []
        metrics_patched = []

        for i in tqdm(range(0, prompts_to_process, batch_size)):
            clean_prompts, corrupted_prompts, clean_answers, corrupted_answers, clean_answers_pos, corrupted_answers_pos, \
            clean_attn_mask, corrupted_attn_mask = sample_dataset(i, i + batch_size, clean_dataset, patched_dataset)

            metric_clean = lambda logits: self.get_logit_diff(logits, clean_answers, corrupted_answers, clean_answers_pos).mean()
            metric_patched = lambda logits: self.get_logit_diff(logits, clean_answers, corrupted_answers, corrupted_answers_pos).mean()

            if score_type == NodeScoreType.ATTRIBUTION_PATCHING:
                metric_clean, cache_clean, grad_clean = self.run_with_cache(clean_prompts, clean_answers, corrupted_answers,
                                                                            clean_answers_pos, clean_attn_mask, metric_clean)

                metric_patched, cache_patched, _ = self.run_with_cache(corrupted_prompts, clean_answers, corrupted_answers, 
                                                                    corrupted_answers_pos, corrupted_attn_mask, metric_patched, run_backward_pass=False)
                # print('Fwd clean keys:', cache_clean.keys())
                # print('Grad clean keys:', grad_clean.keys())

                if i == 0:
                    node_scores = self.get_node_scores_cache(cache_clean)
                self.update_node_scores(node_scores, grad_clean, cache_clean, cache_patched, total_batches, attr_type=AttributionPatching.NORMAL)

                del grad_clean
            elif score_type == NodeScoreType.INTEGRATED_GRADIENTS:
                raise NotImplementedError('Integrated gradients are not implemented yet.')

            del cache_clean, cache_patched
            clear_cache()

            metrics_clean.append(metric_clean)
            metrics_patched.append(metric_patched)

        clean_metric = torch.tensor(metrics_clean).mean().item()
        patched_metric = torch.tensor(metrics_patched).mean().item()

        return (
            clean_metric, patched_metric,
            node_scores
        )

    def run_with_cache(self, tokens: Int[Tensor, "batch pos"],
                       clean_answers: Int[Tensor, "batch"], patched_answers: Int[Tensor, "batch count"],
                        answers_pos: Int[Tensor, "batch"], attn_mask: Int[Tensor, "batch pos"], metric,
                        fwd_cache_filter=None, bwd_cache_filter=None, run_backward_pass=True, analytical_grads=False):
        if fwd_cache_filter is None:
            # Take the SAE latents and error term activations by default
            fwd_cache_filter = lambda name: 'hook_sae_acts_post' in name or 'hook_sae_error' in name

        cache = {}
        def forward_cache_hook(act, hook):
            cache[hook.name] = act.detach()

        self.model.add_hook(fwd_cache_filter, forward_cache_hook, "fwd")

        grad_cache = {}

        try:
            if run_backward_pass:
                self._set_backward_hooks(grad_cache, bwd_cache_filter, analytical_grads)
                
                # Enable gradients only during the backward pass
                with torch.set_grad_enabled(True):
                    metric_value = metric(self.model(tokens, attention_mask=attn_mask))
                    metric_value.backward()  # Compute gradients
            else:
                # Forward pass only
                with torch.set_grad_enabled(False):
                    metric_value = metric(self.model(tokens, attention_mask=attn_mask))
        finally:
            # Ensure hooks are reset regardless of exceptions
            self.model.reset_hooks()
            self._reset_sae_hooks()

        return (
            metric_value.item(),
            ActivationCache(cache, self.model),
            ActivationCache(grad_cache, self.model),
        )

    def _set_backward_hooks(self, grad_cache, bwd_hook_filter=None, compute_grad_analytically=False):
        if bwd_hook_filter is None:
            if compute_grad_analytically:
                bwd_hook_filter = lambda name: 'resid_post' in name or 'attn.hook_z' in name or 'mlp_out' in name or 'resid_pre' in name
            else:
                bwd_hook_filter = lambda name: 'hook_sae_acts_post' in name or 'hook_sae_output' in name or 'hook_sae_input' in name

        temp_cache = {}

        if compute_grad_analytically:
            if self.model.acts_to_saes:
                raise ValueError('Backward pass is performed analytically, but SAEs are still attached. Call reset_saes() first to save VRAM.')

            raise NotImplementedError('Analytical gradients are not implemented yet.')
        
        # Computing grads non-analytically using Pytorch autograd
        def backward_cache_hook(gradient, hook):
            if 'hook_sae_output' in hook.name:
                hook_sae_error_name = hook.name.replace('hook_sae_output', 'hook_sae_error')
                grad_cache[hook_sae_error_name] = gradient.detach()

                # We're storing the gradients for the SAE output activations to copy them to the SAE input activations gradients
                if not 'hook_z' in hook.name:
                    temp_cache[hook.name] = gradient.detach()
                else: # In the case of attention hook_z hooks, reshape them to match the SAE input shape, which doesn't include n_heads
                    hook_z_grad = einops.rearrange(gradient.detach(),
                                                   'batch pos n_head d_head -> batch pos (n_head d_head)')
                    temp_cache[hook.name] = hook_z_grad
            elif 'hook_sae_input' in hook.name:
                # We're copying the gradients from the SAE output activations to the SAE input activations gradients
                sae_output_grad_name = hook.name.replace('hook_sae_input', 'hook_sae_output')

                grad_cache[hook.name] = temp_cache[sae_output_grad_name]
                gradient = grad_cache[hook.name]

                # Pass-through: use the downstream gradients
                return (gradient,)
            else:
                # Default case (SAE latents): just store the gradients
                grad_cache[hook.name] = gradient.detach()

        self.model.add_hook(bwd_hook_filter, backward_cache_hook, "bwd")

    def get_node_scores_cache(self, cache: ActivationCache, score_type: NodeScoreType = NodeScoreType.ATTRIBUTION_PATCHING):
        node_scores = {}
        for key, cache_tensor in cache.items():
            # print(f'Key: {key}, shape: {cache_tensor.shape}')

            # A node is either an SAE latent or an SAE error terms
            # Here it's represented as the hook-point name - cache key
            if 'hook_z.hook_sae_error' not in key:
                batch, pos, d_act = cache_tensor.shape
            else:
                batch, pos, n_head, d_act = cache_tensor.shape

            if 'hook_sae_error' in key:
                # An "importance value" for the SAE error is scalar - it's a single node
                node_scores[key] = torch.zeros((pos), device=self.device)
            else:
                # An "importance value" for SAE latents is a vector with length d_sae (d_act)
                node_scores[key] = torch.zeros((pos, d_act), device=self.device)

        return node_scores

    def update_node_scores(self, node_scores: ActivationCache,
                           grad_cache, cache_clean, cache_patched,
                           total_batches, batch_reduce='mean', attr_type=AttributionPatching.NORMAL):

        for key in node_scores.keys():
            if attr_type == AttributionPatching.NORMAL:
                activation_term = cache_patched[key] - cache_clean[key]

            if 'hook_sae_error' in key:
                # SAE error term case: we want a single score per error term,
                # so we're multiplying the d_act dimension out
                if 'hook_z.hook_sae_error' not in key:
                    score_update = einops.einsum(grad_cache[key], activation_term,
                                                'batch pos d_act, batch pos d_act -> batch pos')
                else:
                    score_update = einops.einsum(grad_cache[key], activation_term,
                                                'batch pos n_head d_head, batch pos n_head d_head -> batch pos')
                
            else:
                # SAE latents case: we want a score per each feature, so we're keeping the d_sae dimension
                score_update = grad_cache[key] * (activation_term) # shape [batch pos d_sae]

            if batch_reduce == 'sum':
                score_update = score_update.sum(0)
                node_scores[key] += score_update
            elif batch_reduce == 'mean':
                score_update = score_update.mean(0)
                node_scores[key] += score_update / total_batches

    def get_answer_logit(self, logits: Float[Tensor, "batch pos d_vocab"], clean_answers: Int[Tensor, "batch"],
                         ansnwer_pos: Int[Tensor, "batch"], return_all_logits=True):
        # clean_answers_pos_idx = clean_answers_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, logits.size(1), logits.size(2))

        answer_pos_idx = einops.repeat(ansnwer_pos, 'batch -> batch 1 d_vocab',
                                        d_vocab=logits.shape[-1])
        answer_logits = logits.gather(1, answer_pos_idx).squeeze(1) # shape [batch, d_vocab]

        correct_logits = answer_logits.gather(1, clean_answers.unsqueeze(1)).squeeze(1) # shape [batch]

        if return_all_logits:
            return answer_logits, correct_logits

        return correct_logits

    def get_logit_diff(self, logits: Float[Tensor, "batch pos d_vocab"],
                    clean_answers: Int[Tensor, "batch"], patched_answers: Int[Tensor, "batch count"],
                    answer_pos: Int[Tensor, "batch"]) -> Float[Tensor, "batch"]:
        # Continue with logit computation
        answer_logits, correct_logits = self.get_answer_logit(logits, clean_answers, answer_pos, return_all_logits=True)

        if patched_answers.dim() == 1:  # If there's only one incorrect answer, gather the incorrect answer logits
            incorrect_logits = answer_logits.gather(1, patched_answers.unsqueeze(1)).squeeze(1)  # shape [batch]
        else:
            incorrect_logits = answer_logits.gather(1, patched_answers)  # shape [batch, answer_count]

        # If there are multiple incorrect answer options, incorrect_logits is now of shape [batch, answer_count]
        if patched_answers.dim() == 2:
            incorrect_logits_sum = incorrect_logits.sum(dim=1)
            return incorrect_logits_sum - correct_logits

        # Otherwise, both logit tensors are of shape [batch]
        return incorrect_logits - correct_logits

    
    def _accumulate_integrated_gradients(self, cache_clean, cache_patched, grad_patched,
                                    clean_tokens, clean_answers, patched_answers,
                                    N, n_prompts, seq_len):
        raise NotImplementedError('Integrated grads are not implemented yet.')

    def reset_saes(self):
        self.model.reset_saes()

    def add_saes(self):
        for sae in self.saes:
            self.model.add_sae(sae, use_error_term=True)

    def print_saes(self):
        saes = self.model.acts_to_saes
        print(f'Number of SAEs: {len(saes)}')

        for name, sae in saes.items():
            print(name, sae)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _reset_sae_hooks(self):
        for sae in self.saes:
            sae.reset_hooks()

    def _load_sae(self, sae_release, sae_id):
        return SAE.from_pretrained(sae_release, sae_id, device=self.device)[0].to(torch.bfloat16)

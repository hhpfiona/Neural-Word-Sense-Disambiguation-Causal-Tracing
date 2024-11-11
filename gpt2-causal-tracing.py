from argparse import Namespace
from typing import Dict, List, Tuple, Callable, Optional

import torch
from torch import Tensor
from transformer_lens import HookedTransformer
# from transformers import AutoModelForCausalLM, AutoTokenizer
import transformer_lens.utils as utils
torch.set_grad_enabled(False)

from matplotlib import pyplot as plt
import seaborn as sns

def plot_heatmap(patch_result: Tensor, output_dir: str, cmap: str = 'Purples') -> None:
    """
    Plot a heatmap of the causal tracing results.

    Args:
    patch_result (torch.Tensor): 2D tensor of shape (sequence_length - 1, n_layers) containing the causal tracing results.
    output_dir (str): Path to save the output heatmap.
    cmap (str): Colormap to use for the heatmap.

    Returns:
    None
    """
    fig, ax = plt.subplots()
    sns.heatmap(patch_result, ax=ax, cmap=cmap)
    fig.savefig(output_dir)

class TraceTransformer(HookedTransformer):
    """
    A custom transformer model class for performing causal tracing analysis.
    Inherits from HookedTransformer and adds methods for causal tracing.
    """

    def get_target_id(self, token: str) -> int:
        """
        Get the token ID for a given target token.

        Args:
        token (str): The target token.

        Returns:
        int: The token ID.
        """
        encoded_tokens = self.tokenizer.encode(' ' + token)
        assert len(encoded_tokens) == 1
        return encoded_tokens[0]

    def record_clean_activations(self, prompt: str) -> Dict[str, Tensor]:
        """
        Record the clean activations for a given prompt.

        Args:
        prompt (str): The input prompt.

        Returns:
        Dict[str, torch.Tensor]: A dictionary containing the clean activations for each layer.
        """
        prompt_token = self.to_tokens(prompt)
        logits, activations = self.run_with_cache(prompt_token)
        return activations

    def get_corrupted_probs(self,
            prompt: str, patch_embed_fn: Callable) -> Tensor:
        """
        Get the corrupted probabilities for a given prompt and patching function.

        Args:
        prompt (str): The input prompt.
        patch_embed_fn (Callable): The function to patch the embeddings.

        Returns:
        torch.Tensor: The corrupted probabilities for the last token.
        """
        tokens = self.to_tokens(prompt)
        logits = self.run_with_hooks(
            tokens,
            fwd_hooks=[(utils.get_act_name('embed'), patch_embed_fn)]
        )
        probs = logits[:, -1, :].softmax(dim=-1)
        return probs

    def find_sequence_span(self, prompt: str, seq: str) -> Tensor:
        """
        Find the token indices for a given sequence in the prompt.

        Args:
        prompt (str): The input prompt.
        seq (str): The sequence to find in the prompt.

        Returns:
        torch.Tensor: A tensor containing the indices of the sequence in the prompt.
        """
        prompt_tokens = self.to_tokens(prompt)[0]
        seq_tokens = self.to_tokens(seq)[0]
        # Find the start index where seq_tokens appear in prompt_tokens
        for i in range(len(prompt_tokens) - len(seq_tokens[1:]) - 1):
            if torch.equal(prompt_tokens[i:i + len(seq_tokens[1:])], seq_tokens[1:]):
                return torch.tensor(list(range(i + 1, i + len(seq_tokens[1:] + 1))))
        raise ValueError("Sequence not found in prompt.")

    def get_patch_emb_fn(self, corrupt_span: Tensor, noise: float = 1.) -> Callable:
        """
        Get a function to patch the embeddings with noise.

        Args:
        corrupt_span (torch.Tensor): The span of tokens to corrupt.
        noise (float): The amount of noise to add.

        Returns:
        Callable: A function that patches the embeddings with noise.
        """
        def patch_embed(activations: Tensor, hook):
            # Add noise to the embeddings at the corrupt_span positions
            activations[:, corrupt_span, :] += torch.randn_like(activations[:, corrupt_span, :]) * noise
            return activations
        return patch_embed

    def get_restore_fn(self,
            activation_record: Dict[str, Tensor], token_idx: int) -> Callable:
        """
        Get a function to restore the activations for a specific token.

        Args:
        activation_record (Dict[str, torch.Tensor]): The recorded clean activations.
        token_idx (int): The index of the token to restore.

        Returns:
        Callable: A function that restores the activations for the specified token.
        """
        def restore_activation(activations: Tensor, hook):
            # Replace the activations at token_idx with the clean activation
            activations[:, token_idx, :] = activation_record[hook.name][:, token_idx, :]
            return activations
        return restore_activation

    def get_forward_hooks(self, layer: int,
            patch_embed_fn: Callable, patch_name: str,
            restore_fn: Callable, window: int = 10) -> List[Tuple[str, Callable]]:
        """
        Get the forward hooks for causal tracing.

        Args:
        layer (int): The current layer.
        patch_embed_fn (Callable): The function to patch the embeddings.
        patch_name (str): The name of the patch location ('resid_pre', 'mlp_post', or 'attn_out').
        restore_fn (Callable): The function to restore activations.
        window (int): The window size for tracing.

        Returns:
        List[Tuple[str, Callable]]: A list of tuples containing the hook names and functions.
        """
        hooks = []
        # Hook to patch the embeddings (corrupt the embeddings)
        hooks.append((utils.get_act_name("embed"), patch_embed_fn))

        # Hook to restore the activations at the specified layer and component
        if patch_name == 'resid_pre':
            hooks.append((utils.get_act_name(patch_name,layer=layer), restore_fn))

        elif patch_name == 'mlp_post' or patch_name == 'attn_out':
            window_layers = range(max(0, layer - window // 2), min(self.cfg.n_layers, layer - (-window // 2)))
            for i in window_layers:
                hooks.append((utils.get_act_name(patch_name, i), restore_fn))

        else:
            raise ValueError(f'Invalid patch_name: {patch_name}')
        return hooks


    def causal_trace_analysis(self,
            prompt: str, source: str, target: str,
            patch_name: str, noise: float = 1., window: int = 10) -> Tensor:
        """
        Perform causal tracing analysis on the model.

        Args:
        prompt (str): The input prompt.
        source (str): The source sequence to corrupt.
        target (str): The target token to predict.
        patch_name (str): The name of the patch location
            ('resid_pre', 'mlp_post', or 'attn_out').
        noise (float): The amount of noise to add when corrupting.
        window (int): The window size for tracing.

        Returns:
        torch.Tensor: A 2D tensor of shape (sequence_length - 1, n_layers) containing the causal tracing results.
        """
        # Record clean activations
        activation_record = self.record_clean_activations(prompt)

        # Find the indices of the source tokens to corrupt
        corrupt_span = self.find_sequence_span(prompt, source)

        # Create the function to corrupt embeddings
        patch_embed_fn = self.get_patch_emb_fn(corrupt_span, noise)

        # Get the corrupted probabilities (without any restoration)
        corrupted_probs = self.get_corrupted_probs(prompt, patch_embed_fn)
        target_id = self.get_target_id(target)

        # Initialize result tensor
        prompt_token = self.to_tokens(prompt)[0]
        seq_len = len(prompt_token)
        n_layers = self.cfg.n_layers
        result = torch.zeros(seq_len - 1, n_layers)

        for layer in range(n_layers):
            for index in range(1, len(prompt_token)):
                # Create a restore function for the specific position
                restore_fn = self.get_restore_fn(activation_record, index)

                # Get the hooks for this position and layer
                hooks = self.get_forward_hooks(layer, patch_embed_fn, patch_name, restore_fn, window)

                # Run the model with the hooks
                logits = self.run_with_hooks(
                    self.to_tokens(prompt), 
                    fwd_hooks=hooks
                )

                # Compute the probability of the target token
                probability = torch.softmax(logits[:, -1, :], dim=-1)

                # Store the result
                result[index - 1, layer] = (probability[0, target_id] - corrupted_probs[0, target_id]).item()

        return result

def run_causal_trace(model_name='gpt2-xl', patch_name='resid_pre',
    prompt=None, source=None, target=None,
    cache_dir=None) -> None:
    """
    Perform causal tracing analysis for a specific patch location in a language model.

    Args:
    model_name (str): The name of the language model to load, such as 'gpt2-xl'. Defaults to 'gpt2-xl'.
    patch_name (str): The name of the patch location where the causal trace is applied. 
                      Can be one of 'resid_pre', 'mlp_post', or 'attn_out'. 
                      Determines which component to perturb during analysis. Defaults to 'resid_pre'.
    prompt (str): The input prompt.
    source (str): The source sequence to corrupt.
    target (str): The target token to predict.
    cache_dir (str):  The directory where the model are cached.

    Returns:
    None: This function does not return a value but generates a heatmap saved as a PDF 
          showing the results of the causal trace analysis.
    """
    cmap, name = {
        'resid_pre': ('Purples', 'states'),
        'mlp_post': ('Greens', 'mlp'),
        'attn_out': ('Reds', 'attn'),
    }[patch_name]

    model = TraceTransformer.from_pretrained(model_name,
        cache_dir=cache_dir, local_files_only=True)

    result = model.causal_trace_analysis(
        prompt=prompt, source=source, target=target,
        patch_name=patch_name,
        noise=0.5)

    plot_heatmap(result, name+'.pdf', cmap)

if __name__ == '__main__':
    model_name = 'gpt2-xl'
    model_name = model_name

    request = {
        'prompt': 'The Eiffel Tower is located in the city of',
        'source': 'The Eiffel Tower',
        'target': 'Paris',
    }

    run_causal_trace(model_name=model_name, patch_name='resid_pre', **request)
    run_causal_trace(model_name=model_name, patch_name='mlp_post', **request)
    run_causal_trace(model_name=model_name, patch_name='attn_out', **request)
#aryan/scripts/compute_similarity.py
import os
import argparse
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM

from utils_dataset import get_mmlu_samples
from utils_merging import diffusion_kernel_embed, compute_similarity_matrix


def extract_layer_outputs(model, input_ids, num_layers):
    """
    Returns a dict[layer_idx] = (hidden_size,) numpy
    We'll collect last token's hidden state from each layer
    """
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = out.hidden_states
    layer_outs = {}
    for l in range(1, num_layers + 1):
        hs = hidden_states[l][0, -1, :]               # (hidden,)
        hs = hs.to(torch.float32).cpu().numpy()       # ← cast to fp32 before numpy
        layer_outs[l - 1] = hs
    return layer_outs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="e.g. meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--data_dir", type=str, default="aryan/data/mmlu")
    parser.add_argument("--task_name", type=str, required=True,
                        help="one of: medical, legal, math, cs")
    parser.add_argument("--num_samples", type=int, default=40)
    parser.add_argument(
    "--output_dir",
    type=str,
    default=os.path.expanduser("~/aryan/outputs/similarity_mats"),
)
    # parser.add_argument("--output_dir", type=str, default="aryan/outputs/similarity_mats")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    model.eval()

    # figure out how many transformer blocks it has
    num_layers = model.config.num_hidden_layers

    print(f"Collecting samples for task={args.task_name} ...")
    samples = get_mmlu_samples(args.data_dir, args.task_name, args.num_samples)

    # we'll collect activations per layer
    activations_per_layer = {i: [] for i in range(num_layers)}

    for s in samples:
        text = s["text"]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)
        input_ids = inputs["input_ids"]

        layer_outs = extract_layer_outputs(model, input_ids, num_layers)
        for l, vec in layer_outs.items():
            activations_per_layer[l].append(vec)

        # free
        torch.cuda.empty_cache()

    # now build diffusion embeddings per layer
    layer_embeddings = {}
    for l in range(num_layers):
        X = np.stack(activations_per_layer[l], axis=0)  # (N, D)
        emb = diffusion_kernel_embed(X, sigma=8.0, alpha=0.5, d=2)
        layer_embeddings[l] = emb

    sim_mat = compute_similarity_matrix(layer_embeddings)
    # out_path = os.path.join(args.output_dir, f"similarity_{args.task_name}.npy")
    # np.save(out_path, sim_mat)
    # print(f"saved similarity matrix to {out_path}")

    # out_path = os.path.join(args.output_dir, f"similarity_{args.task_name}.npy")
    # np.save(out_path, sim_mat)
    # print(f"saved similarity matrix to {out_path}")
    out_path = os.path.join(args.output_dir, f"similarity_{args.task_name}_{args.num_samples}s.npy")
    np.save(out_path, sim_mat)
    print(f"Saved similarity matrix to {out_path}")


if __name__ == "__main__":
    main()

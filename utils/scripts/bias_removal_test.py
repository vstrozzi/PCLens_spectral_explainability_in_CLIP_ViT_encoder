import numpy as np
import os.path
import argparse
import json
from pathlib import Path
from utils.datasets_constants.imagenet_classes import imagenet_classes
from utils.scripts.algorithms_text_explanations_funcs import *
from utils.misc.misc import accuracy
from utils.models.factory import create_model_and_transforms, get_tokenizer
from utils.models.prs_hook import hook_prs_logger
from PIL import Image
import torch  # Make sure you import torch if it's needed


def get_args_parser():
    parser = argparse.ArgumentParser("Ablations part", add_help=False)

    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-B-32",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    parser.add_argument('--pretrained', default='laion2b_s32b_b79k', type=str)

    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--subset_dim", default=10, type=int)

    parser.add_argument("--num_real_layer", default=4, type=int)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        help="imagenet, waterbirds, cub, binary_waterbirds",
    )

    parser.add_argument(
        "--dataset_text",
        type=str,
        default="top_1500_nouns_5_sentences_imagenet_clean",
        help="text dataset used for the explanations",
    )

    parser.add_argument("--top_k", type=int, default=10, help="Nr of PCs of the query system")
    parser.add_argument("--max_approx", type=float, default=1, help="Max approx for the reconstruction")

    return parser


def main(args):
    """
    Need to run compute_prs.py for the given model before running this script.
    Compute the final embedding of the given model.
    """

    subset_dim = args.subset_dim
    model_name = args.model

    device = args.device
    attention_dataset = os.path.join(
        args.output_dir, 
        f"{args.dataset}_completeness_{args.dataset_text}_{model_name}_algo_svd_data_approx_seed_{args.seed}.jsonl"
    )

    # Load data/embeddings
    final_embeddings_images = torch.tensor(
        np.load(os.path.join(args.output_dir, f"{args.dataset}_embeddings_{args.model}_seed_{args.seed}.npy"), mmap_mode="r")
    ).to(device)  # shape: [b, d] or [b, l, h, d] depending on your saving format
    final_embeddings_texts = torch.tensor(
        np.load(os.path.join(args.output_dir, f"{args.dataset_text}_{model_name}.npy"), mmap_mode="r")
    ).to(device)
    classifier_ = torch.tensor(
        np.load(os.path.join(args.output_dir, f"{args.dataset}_classifier_{model_name}.npy"), mmap_mode="r")
    ).to(device)
    labels_ = torch.tensor(
        np.load(os.path.join(args.output_dir, f"{args.dataset}_labels_{model_name}_seed_{args.seed}.npy"), mmap_mode="r")
    ).to(device) 
    with open(
        os.path.join(args.output_dir, f"{args.dataset}_attn_{args.model}_seed_{args.seed}.npy"), "rb"
    ) as f:
        attns = torch.tensor(np.load(f))  # [b, l, h, d]
    with open(
        os.path.join(args.output_dir, f"{args.dataset}_mlp_{args.model}_seed_{args.seed}.npy"), "rb"
    ) as f:
        mlps = torch.tensor(np.load(f))  # [b, l+1, d]

    num_last_layers = args.num_real_layer

    # Save important stuff
    nr_layers_ = attns.shape[1]
    nr_heads_ = attns.shape[2]

    # Prepare to store results
    acc_baseline_list = []
    count_baseline_list = []
    acc_rec_list = []
    count_rec_list = []
    acc_bias_rem_list = []
    count_bias_rem_list = []

    query_text = True
    top_k = args.top_k

    # Precompute means
    mean_final_images = torch.mean(final_embeddings_images, axis=0)
    mean_final_texts = torch.mean(final_embeddings_texts, axis=0)
    labels_embeddings = classifier_.T
    for c, label in enumerate(imagenet_classes):
        
        # Retrieve topic embedding
        topic_emb = labels_embeddings[c:c+1, :]
        # Mean center the embeddings
        mean_final = mean_final_texts if query_text else mean_final_images
        topic_emb_cent = topic_emb - mean_final

        # Retrieve partial SVD/projection data
        data = get_data(attention_dataset, -1, skip_final=True)

        # Reconstruct embedding of the query
        _, data = reconstruct_embeddings(
            data, 
            [topic_emb_cent], 
            ["text" if query_text else "image"], 
            device=device,
            return_princ_comp=True, 
            plot=False, 
            means=[mean_final]
        )

        # Sort the principal components by absolute correlation
        data = sort_data_by(data, "correlation_princ_comp_abs", descending=True)
        top_k_entries = top_data(data, top_k)

        # Increase amplification
        rec = reconstruct_all_embeddings_mean_ablation_pcs(
        top_k_entries,
        mlps,
        attns, 
        final_embeddings_images,
        nr_layers_,
        nr_heads_,
        num_last_layers,
        ratio=-1,
        mean_ablate_all=False
        )
    
        top_k_entries_other  = get_remaining_pcs(data, top_k_entries)

        # Diminish amplification
        rec_proof = reconstruct_all_embeddings_mean_ablation_pcs(
        top_k_entries_other,
        mlps,
        attns, 
        final_embeddings_images,
        nr_layers_,
        nr_heads_,
        num_last_layers,
        ratio=-1,
        mean_ablate_all=False
        )

        # Normalize
        rec_proof /= rec_proof.norm(dim=-1, keepdim=True)
        rec /= rec.norm(dim=-1, keepdim=True)


        # -------------------------
        #  1) Baseline Accuracy
        # -------------------------
        baseline = final_embeddings_images
        prediction = baseline @ classifier_
        acc_base, indexes_approx_bas = test_accuracy(prediction, labels_, label="Baseline")
        print_tot_wrong_elements_label(prediction, label)
        count_base = print_wrong_elements_label(indexes_approx_bas, label, subset_dim)
        acc_baseline_list.append(float(acc_base))
        count_baseline_list.append(int(count_base))

        # -------------------------
        #  2) Reconstruction Accuracy
        # -------------------------
        prediction = rec @ classifier_
        acc_r, indexes_approx_rec = test_accuracy(prediction, labels_, label="Approximation with the reconstructed embeddings")
        print_tot_wrong_elements_label(prediction, label)
        count_r = print_wrong_elements_label(indexes_approx_rec, label, subset_dim)
        acc_rec_list.append(float(acc_r))
        count_rec_list.append(int(count_r))

        # -------------------------
        #  3) "Bias Removal"/Final Accuracy
        # -------------------------
        prediction = rec_proof @ classifier_
        acc_final, indexes_approx_final = test_accuracy(
            prediction, 
            labels_, 
            label="Approximation with proof of concept"
        )
        print_tot_wrong_elements_label(prediction, label)
        count_final = print_wrong_elements_label(indexes_approx_final, label, subset_dim)
        acc_bias_rem_list.append(float(acc_final))
        count_bias_rem_list.append(int(count_final))

        print()  # spacing


    # Once the loop over classes is done, prepare the result structure
    results = {
        "acc_baseline": acc_baseline_list,
        "count_baseline": count_baseline_list,
        "acc_rec": acc_rec_list,
        "count_rec": count_rec_list,
        "acc_bias_rem": acc_bias_rem_list,
        "count_bias_rem": count_bias_rem_list,
        "top_k": args.top_k,
        "subset_dim": args.subset_dim,
        "seed": args.seed,
        "approx": args.max_approx
    }

    # Write out to a .jsonl file with a meaningful name
    out_filename = (
        f"{args.dataset}_bias_test_{args.dataset_text}_{args.model}_algo_svd_data_approx_seed_{args.seed}_top{top_k}.jsonl"
    )
    out_path = os.path.join(args.output_dir, out_filename)

    # Write the results as a single JSON object in one line
    with open(out_path, "w") as f:
        f.write(json.dumps(results) + "\n")

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)

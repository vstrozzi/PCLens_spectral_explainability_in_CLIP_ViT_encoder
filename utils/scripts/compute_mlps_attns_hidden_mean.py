import numpy as np
import torch
import os
import glob
import argparse
from pathlib import Path
import re
import tqdm
from utils.models.factory import create_model_and_transforms
from utils.datasets.binary_waterbirds import BinaryWaterbirds
from utils.datasets.dataset_helpers import dataset_to_dataloader
from utils.models.prs_hook import hook_prs_logger
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet, ImageFolder

# Compute activation for all using my hook -> Mean values on the fly

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser("Project Residual Stream - Option B", add_help=False)
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size")
    # Model parameters
    parser.add_argument("--model", default="ViT-H-14", type=str, metavar="MODEL",
                        help="Name of model to use")
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k", type=str)
    # Dataset parameters
    parser.add_argument("--data_path", default="./datasets/", type=str, help="dataset path")
    parser.add_argument("--dataset", type=str, default="CIFAR10",
                        help="imagenet, cub, waterbirds, CIFAR10, etc.")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--output_dir", default="./output_dir", help="path where to save")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="cache directory for models weight")
    parser.add_argument("--samples_per_class", default=None, type=int,
                        help="number of samples per class")

    parser.add_argument("--tot_samples_per_class", default=None, type=int,
                        help="Total number of samples per class in the dataset")
    parser.add_argument("--max_nr_samples_before_writing", default=20, type=int,
                        help="How many samples to keep in RAM before saving them to chunk files")

    parser.add_argument("--quantization", help="Quantization size (choose 'fp16' or 'fp32')", default="fp32", type=str)
    return parser


def main(args):
    """
    Calculates the projected residual stream (i.e. head activations) for a whole dataset 
    and saves them in separate chunk files. After finishing, it concatenates all chunks 
    into single .npy files (one per data type), and deletes the chunk files.
    """

    # Build & move model:
    model, _, preprocess = create_model_and_transforms(
        args.model, pretrained=args.pretrained, precision=args.quantization, cache_dir=args.cache_dir
    )
    model.to(args.device)
    model.eval()
    
    context_length = model.context_length
    vocab_size = model.vocab_size

    print(
        "Model parameters:",
        f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}",
    )
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    print("Len of res:", len(model.visual.transformer.resblocks))

    prs = hook_prs_logger(model, args.device, spatial=False, vision_projection=False, full_output=True) # Keep full output
    # Dataset:
    if args.dataset == "imagenet":
        ds = ImageNet(root=args.data_path + "imagenet/", split="val", transform=preprocess)
    elif args.dataset == "binary_waterbirds":
        ds = BinaryWaterbirds(root=args.data_path+"waterbird_complete95_forest2water2/", split="test", transform=preprocess)
    elif args.dataset == "CIFAR100":
        ds = CIFAR100(
            root=args.data_path, download=True, train=False, transform=preprocess
        )
    elif args.dataset == "CIFAR10":
        ds = CIFAR10(
            root=args.data_path, download=True, train=False, transform=preprocess
        )
    else:
        ds = ImageFolder(root=args.data_path, transform=preprocess)

    # Depending
    dataloader = dataset_to_dataloader(
        ds,
        samples_per_class=args.samples_per_class,
        tot_samples_per_class=args.tot_samples_per_class,  # or whatever you prefer
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    num_total_images = len(dataloader) * args.batch_size
    print(f"We are using a dataset containing {num_total_images} images.")

    # We'll accumulate results here and flush them out every 
    # args.max_nr_samples_before_writing images (approx).
    attention_results = []
    mlp_results = []
    cls_to_cls_results = []
    labels_results = []

    # For chunk naming
    chunk_index = 0
    total_samples_seen = 0

    final_mlps_mean_file = os.path.join(
        args.output_dir,
        f"{args.dataset}_mlps_mean_{args.model}_seed_{args.seed}.npy"
    )
    final_attns_mean_file = os.path.join(
        args.output_dir,
        f"{args.dataset}_attns_mean_{args.model}_seed_{args.seed}.npy"
    )

    # Remove final files if they exist, just to be safe
    for ff in [final_attns_mean_file, final_mlps_mean_file]:
        if os.path.exists(ff):
            os.remove(ff)

    # Save values
    attns_mean = None
    mlps_mean = None
    for i, (image, labels) in enumerate(tqdm.tqdm(dataloader)):

        batch_size_here = image.shape[0]
        total_samples_seen += batch_size_here

        with torch.no_grad():
            prs.reinit()
            # First, move the image to the GPU asynchronously.
            image = image.to(args.device, non_blocking=True)
            # Then, cast to lower precision on the GPU.
            if args.quantization == "fp16":
                image = image.to(dtype=torch.float16)
            representation = model.encode_image(image, attn_method= "head_no_spatial", normalize=False
            )
            
            attentions, mlps = prs.finalize(representation)
            # Initialize
            if i == 0:
                attns_mean = attentions.sum(0).detach().cpu().numpy() # [l, n, h, d],
                mlps_mean = mlps.sum(0).detach().cpu().numpy()           # [l + 1, n, d]
            else:
                attns_mean += attentions.sum(0).detach().cpu().numpy() # [l, n, h, d],
                mlps_mean += mlps.sum(0).detach().cpu().numpy()           # [l + 1, n, d]

    # Mean of the values    
    mlps_mean /= num_total_images
    attns_mean /= num_total_images

    # 1) Attn
    with open(final_attns_mean_file, 'wb') as f:
        np.save(f, attns_mean)

    # 2) MLP
    with open(final_mlps_mean_file, 'wb') as f:
        np.save(f, mlps_mean)

    print("Final single-file arrays created:\n"
          f"  {final_attns_mean_file}\n"
          f"  {final_mlps_mean_file}\n")
    
    print("Done.")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

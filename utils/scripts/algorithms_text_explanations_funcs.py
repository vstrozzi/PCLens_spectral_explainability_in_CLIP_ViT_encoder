import json
from dataclasses import dataclass
from typing import List
import pandas as pd
from tabulate import tabulate
import torch
import numpy as np
from torchvision.datasets import ImageNet, CIFAR10, CIFAR100, ImageFolder
from utils.datasets.binary_waterbirds import BinaryWaterbirds

import random
from collections import defaultdict
from utils.datasets.dataset_helpers import dataset_subset
import matplotlib.pyplot as plt
from utils.misc.visualization import visualization_preprocess
from utils.misc.misc import accuracy_correct
from utils.datasets_constants.imagenet_classes import imagenet_classes
from utils.datasets_constants.cifar_10_classes import cifar_10_classes
from utils.datasets_constants.cub_classes import cub_classes, waterbird_classes


### Layout of data
@dataclass
class PrincipalComponentRecord:
    layer: int
    head: int
    princ_comp: int
    strength_abs: float
    strength_rel: float
    cosine_similarity: float
    texts: List[str]
    project_matrix: torch.Tensor
    vh: torch.Tensor
    rank: int

@torch.no_grad()
def get_data(attention_dataset, min_princ_comp=-1, skip_final=False):
    """
    Retrieve data from a JSON file containing attention data.
    Args:
    - attention_dataset (str): The path to the JSON file containing attention data.
    - min_princ_comp (int): The minimum principal component number to consider for each head (-1=all).

    Returns:
    - A list of dictionaries containing details for each principal component of interest.
    """
    with open(attention_dataset, "r") as json_file:
        data = []
        for line in json_file:
            entry = json.loads(line)  # Parse each line as a JSON object, producing a dictionary-like structure
             # Skip the final clip embeddings if requested
            if skip_final and entry["head"] == -1:
                continue

            # Each entry includes a sorted list of principal components. 
            # We want to record information up to a certain minimum principal component index (min_princ_comp).
            for i, princ_comp_data in enumerate(entry["embeddings_sort"]):
                # Stop if we've reached the minimum principal component index to limit how many we gather from each entry
                if i == min_princ_comp:
                    break
               
                # Append a dictionary of details for each principal component of interest
                data.append({
                    "layer": entry["layer"],
                    "head": entry["head"],
                    "princ_comp": i,
                    "strength_abs": princ_comp_data["strength_abs"],
                    "strength_rel": princ_comp_data["strength_rel"],
                    "cosine_similarity": princ_comp_data["cosine_similarity"],
                    "correlation": princ_comp_data["correlation"],
                    "texts": princ_comp_data["text"],
                    "rank": len(entry["embeddings_sort"]),
                    "project_matrix": entry["project_matrix"],
                    "vh": entry["vh"],
                    "mean_values_text": entry["mean_values_text"],
                    "mean_values_att": entry["mean_values_att"],
                })
        
    return data

@torch.no_grad()
def get_data_component(data, layer, head, princ_comp):
    """
    Retrieve data from a JSON file containing attention data.
    Args:
    - data (list): A list of dictionaries containing details for each principal component of interest. (i.e. layout as PrincipalComponentRecord)
    - layer (int): The layer number of the principal component.
    - head (int): The head number of the principal component.
    - princ_comp (int): The principal component number of interest.

    Returns:
    - A list of dictionaries containing detail for the principal component of interest.
    """

    for entry in data:
        if entry["layer"] == layer and entry["head"] == head and entry["princ_comp"] == princ_comp:
            return [entry]


def print_data_text_span(data, top_k=None):
    """
    Print the collected data in a formatted table.
    Args:
    - data (list): A list of dictionaries containing details for each principal component of interest. (i.e. layout as PrincipalComponentRecord)
    Returns:
    - None
    """
    # Convert the collected data into a Pandas DataFrame for easier manipulation and printing
    top_k_df = pd.DataFrame(data)

    # Iterate over each row in the DataFrame to display details about the principal components and their associated texts
    for row in top_k_df.itertuples():
        output_rows = []
        texts = row.texts
        
        if top_k != None:
            output_rows = texts[:top_k]
        else:
            output_rows = texts

        # Print summary information about the current principal component:
        # Including layer, head, principal component index, absolute variance, relative variance, and head rank.
        print(f"Layer {row.layer}, Head {row.head}")
        # Set the column headers based on whether the first half was considered positive
        columns = ["text"]

        # Create a DataFrame from the collected rows of positive/negative texts and print it in a formatted table
        output_df = pd.DataFrame(output_rows, columns=columns)
        print(tabulate(output_df, headers='keys', tablefmt='psql'))

def visualize_text_span(layer, head, data, top_k=-1):    # Retrieve and filter the principal component data
    data = get_data_text_span(data)
    data = get_data_head(data, layer, head)
    print_data_text_span(data, top_k=top_k)


def get_data_text_span(attention_dataset):
    """
    Retrieve data from a JSON file containing attention data.
    Args:
    - attention_dataset (str): The path to the JSON file containing attention data.
    - min_princ_comp (int): The minimum principal component number to consider for each head (-1=all).

    Returns:
    - A list of dictionaries containing details for each principal component of interest.
    """
    with open(attention_dataset, "r") as json_file:
        data = []
        for line in json_file:
            entry = json.loads(line)  # Parse each line as a JSON object, producing a dictionary-like structure


            # Each entry includes a sorted list of principal components. 
            # We want to record information up to a certain minimum principal component index (min_princ_comp).

               
            # Append a dictionary of details for each principal component of interest
            data.append({
                "layer": entry["layer"],
                "head": entry["head"],
                "texts": entry["embeddings_sort"]
            })
    
    return data

def get_data_head_svs(data, layer, head):
    """
    Retrieve data from a JSON file containing attention data.
    Args:
    - data (list): A list of dictionaries containing details for each principal component of interest. (i.e. layout as PrincipalComponentRecord)
    - layer (int): The layer number of the principal component.
    - head (int): The head number of the principal component.
    - princ_comp (int): The principal component number of interest.

    Returns:
    - A list of svs
    """
    svs = []
    for entry in data:
        if entry["layer"] == layer and entry["head"] == head:
           svs.append(entry["strength_rel"]) 
    return svs

def plot_pc_sv(data, layer, head):
    """
    Plot the singular values of the principal components for a given layer and head.

    Args:
    - data (list): List of dictionaries containing details for each principal component of interest.
    - layer (int): The layer number of the principal component.
    - head (int): The head number of the principal component.

    Returns:
    - None
    """
    # Retrieve and filter the principal component data
    svs = get_data_head_svs(data, layer, head)
    svs = np.cumsum(svs)
    # Plot the singular values of the principal components on the y axis
    plt.figure(figsize=(10, 6))
    plt.plot(svs, marker='o')
    plt.xlabel('Principal Component Index')
    plt.ylabel('Cumulative Singular Value')
    plt.title(f'Cumulative Singular Values for Layer {layer}, Head {head}')
    plt.grid(True)
    plt.show()



def get_data_head(data, layer, head):
    """
    Retrieve data from a JSON file containing attention data.
    Args:
    - data (list): A list of dictionaries containing details for each principal component of interest. (i.e. layout as PrincipalComponentRecord)
    - layer (int): The layer number of the principal component.
    - head (int): The head number of the principal component.
    - princ_comp (int): The principal component number of interest.

    Returns:
    - A list of dictionaries containing detail for the principal component of interest.
    """
    for entry in data:
        if entry["layer"] == layer and entry["head"] == head:
            return [entry]
    return []


@torch.no_grad()
def print_data(data, min_princ_comp=-1, is_corr_present=False):
    """
    Print the collected data in a formatted table.
    Args:
    - data (list): A list of dictionaries containing details for each principal component of interest. (i.e. layout as PrincipalComponentRecord)
    - min_princ_comp (int): The minimum principal component number to consider for each head (-1=all).
    - is_corr_present (bool): Whether the correlation values are present in the data.
    Returns:
    - None
    """
    # Convert the collected data into a Pandas DataFrame for easier manipulation and printing
    top_k_df = pd.DataFrame(data)

    # Iterate over each row in the DataFrame to display details about the principal components and their associated texts
    for row in top_k_df.itertuples():
        output_rows = []
        texts = row.texts
        half_length = len(texts) // 2
        
        # Skip the enetry if above the minimum number we want
        if row.princ_comp >= min_princ_comp and min_princ_comp != -1:
            continue

        # Determine if the first half of the texts represent a positive component by checking the associated value
        is_positive_first = list(texts[0].values())[1] > 0
        
        # Split the texts into two halves: first half (positive or negative) and second half (the opposite polarity)
        if is_positive_first:
            positive_texts = texts[:half_length]
            negative_texts = texts[half_length:]
        else:
            positive_texts = texts[half_length:]
            negative_texts = texts[:half_length]
        
        # Pair up corresponding positive and negative texts and collect them in output_rows for tabular display
        for pos, neg in zip(positive_texts, negative_texts):
            pos_text = list(pos.values())[0]
            pos_val  = list(pos.values())[1]
            neg_text = list(neg.values())[0]
            neg_val  = list(neg.values())[1]    
            
            # Append not with maximum value but with the according sign of correlation on the left
            if is_corr_present:
                corr_sign = np.sign(row.correlation_princ_comp)
                if corr_sign > 0:
                    output_rows.append([pos_text, pos_val, neg_text, neg_val])
                else:
                    output_rows.append([neg_text, neg_val, pos_text, pos_val])

            else:
                if is_positive_first:
                    output_rows.append([pos_text, pos_val, neg_text, neg_val])
                else:
                    output_rows.append([neg_text, neg_val, pos_text, pos_val])            

        # Print summary information about the current principal component:
        # Including layer, head, principal component index, absolute variance, relative variance, and head rank.
        print(f"Layer {row.layer}, Head {row.head}, Principal Component {row.princ_comp}, "
            f"Variance {row.strength_abs:.3f}, Relative Variance {row.strength_rel:.3f}, Head Rank {row.rank}")
        if is_corr_present:
            print(f"Correlation of the Topic with the Principal Component {row.correlation_princ_comp:.4f}")
        
        # Set the column headers based on whether the first half was considered positive
        if is_positive_first:
            columns = ["Positive", "Positive_Strength", "Negative", "Negative_Strength"]
        else:
            columns = ["Negative", "Negative_Strength", "Positive", "Positive_Strength"]

        # Or set it based on the sign of the correlation
        if is_corr_present and corr_sign > 0:
            columns = ["Positive", "Positive_Strength", "Negative", "Negative_Strength"]
        else:
            columns = ["Negative", "Negative_Strength", "Positive", "Positive_Strength"]
        
        # Create a DataFrame from the collected rows of positive/negative texts and print it in a formatted table
        output_df = pd.DataFrame(output_rows, columns=columns)
        print(tabulate(output_df, headers='keys', tablefmt='psql'))

@torch.no_grad()
def sort_data_by(data, key="strength_abs", descending=True):
    """
    Sorts data based on the 'strength_abs' key.

    Parameters:
    - data (list): List of dictionaries to sort.
    - key (str): Key to sort by.
    - descending (bool): Sort in descending order if True, ascending otherwise.

    Returns:
    - list: Sorted list of data.
    """
    return sorted(data, key=lambda x: x.get(key, 0), reverse=descending)

@torch.no_grad()
def top_data(data, top_k=5):
    """
    Get the top-k elements of data (already sorted)

    Parameters:
    - data (list): List of dictionaries from which to retrieve top elements.
    - top_k (int): Number of top-k elements to retrieve.

    Returns:
    - list: Top-k data.
    """
    return data[:top_k]

@torch.no_grad()
def map_data(data, lbd_func=None):
    """
    Apply lambda function on each element of data.

    Parameters:
    - data (list): List of dictionaries from which to retrieve top elements.
    - lbd_func (x: x): Lambda function.

    Returns:
    - list: Mapped elements.
    """
    return [lbd_func(x) for x in data]

@torch.no_grad()
def reconstruct_all_embeddings_mean_ablation_pcs(data_pcs, mlps, attns, embeddings, tot_nr_layers, tot_nr_heads, nr_mean_ablated, ratio=0.5, ablation=True):
    """
    Reconstruct the embeddings using the contribution of only the heads with PCs in data.
    Parameters:
    - data (list): List of dictionaries containing all details for each principal component.
    - data_pcs (list): List of dictionaries containing all details for each principal component of interest.
    - embeddings (list): List containing the embeddings to reconstruct.
    - types (list): List of types of embeddings to reconstruct.
    - return_princ_comp (bool): Return the principal components of the given embeddings
    - plot (bool): Plot the reconstructed embeddings cosine similarity.
    - means (list): List of mean values for the embeddings. mean_ablated_and_replaced = mlps.sum(axis=1) + attns.sum(axis=(1, 2))


    Returns:
    - list: Reconstructed embeddings NOT mean centered.
    - data: Data updated with principal components of the given embeddings (if return_princ_comp is True).
    """

    if len(embeddings) == 0:
        assert False, "No embeddings to reconstruct or different lengths."

 
    # Sort content of data_pcs based on the natural order of pcs
    heads_to_keep = sorted(
    list(
        set((x["layer"], x["head"]) for x in data_pcs if (x["layer"], x["head"]) != (-1, -1))
        )
    )
    print("Heads to keep: ", len(heads_to_keep))
    
    grouped_data = []

    for (layer, head) in heads_to_keep:
        # Filter the original data to only those items matching the current layer/head
        relevant_items = [item for item in data_pcs if item["layer"] == layer and item["head"] == head]
        # Collect all principal components, vh, and mean_values for the current layer/head
        princ_comps = [item["princ_comp"] for item in relevant_items]
        s = torch.tensor([item["strength_abs"] for item in relevant_items])
        vh_vals     = torch.tensor(relevant_items[0]["vh"])
        means       = torch.tensor(relevant_items[0]["mean_values_att"]).unsqueeze(0)

        grouped_data.append(
            (layer,
            head,
            princ_comps,
            s,
            vh_vals,
            means)
            )


    # Initialize the reconstructed embeddings
    nr_layer_not_anal = tot_nr_layers - nr_mean_ablated
    # Put initial contribution
    prev_layers = mlps.sum(axis=1) + attns[:, 0:nr_layer_not_anal, :, :].sum(axis=(1, 2))
    reconstructed_embeddings = torch.zeros_like(prev_layers)
    reconstruct_embeddings_mean_ablate = torch.zeros_like(prev_layers)
    # Iterate over each principal component of interest and keep the contribution of their head, sum mean ablation for all the other
    prev_layer, prev_head = nr_layer_not_anal, 0
    for layer, head, princ_comps, s, vh, mean_values in grouped_data:
        # Get contribution for our principal component on mean centered data
        if not ablation:
            mask = torch.zeros((embeddings.shape[0], vh.shape[0]))
            mask = ((attns[:, layer, head, :]) @ vh.T)
            reconstructed_embeddings += mask @ vh

            mask = torch.zeros((1, vh.shape[0]))
            mask = ((- mean_values) @ vh.T)
            reconstruct_embeddings_mean_ablate += mask @ vh
        else:
            mask = torch.zeros((embeddings.shape[0], vh.shape[0]))
            mask[:, princ_comps] = ((attns[:, layer, head, :]) @ vh.T)[:, princ_comps]
            reconstructed_embeddings += mask @ vh

            mask = torch.zeros((1, vh.shape[0]))
            mask[:, princ_comps] = ((- mean_values) @ vh.T)[:, princ_comps]
            reconstruct_embeddings_mean_ablate += mask @ vh

        reconstruct_embeddings_mean_ablate += mean_values

        # Add mean ablation for whole heads when not used 
        while prev_layer != layer or prev_head != head: 
            # Add mean contribution of all the data not in the pcs
            if ablation:
                reconstruct_embeddings_mean_ablate += torch.mean(attns[:, prev_layer, prev_head, :], axis=0) #[b, l, h, d]
            else:
                reconstructed_embeddings += attns[:, prev_layer, prev_head, :]

            if prev_head == tot_nr_heads - 1:
                prev_head = 0
                prev_layer += 1
            else:
                prev_head += 1   
            if prev_head == tot_nr_heads -1 and prev_layer == tot_nr_layers - 1:
                break
        prev_head, prev_layer = head + 1, layer
        if prev_head == tot_nr_heads:
                prev_head = 0
                prev_layer += 1
        if prev_head == tot_nr_heads and prev_layer == tot_nr_layers:
            break
    
    # Amplification ==-1 keep original ratio
    if ratio == -1:
        reconstructed_embeddings += reconstruct_embeddings_mean_ablate + prev_layers
    else:
        # Consider as a whole prev layers and mean ablation
        reconstruct_embeddings_mean_ablate += prev_layers
        norm_rec = reconstructed_embeddings.norm(dim=-1, keepdim=True)
        norm_rec_mean = reconstruct_embeddings_mean_ablate.norm(dim=-1, keepdim=True)
        norm_mean = torch.mean(embeddings, axis = 0).norm(dim = -1).item()
        reconstructed_embeddings *= norm_mean / norm_rec * ratio
        reconstructed_embeddings += reconstruct_embeddings_mean_ablate*norm_mean / norm_rec_mean * (1-ratio)
        
    # No pcs selected, just return whole mean ablation
    if data_pcs == []:
        reconstructed_embeddings = mlps.sum(axis=1) + torch.mean(attns.sum(axis=(2))[:, :], axis=0).sum(0)

    return reconstructed_embeddings


@torch.no_grad()
def reconstruct_all_embeddings_mean_ablation_heads(data_pcs, mlps, attns, embeddings, tot_nr_layers, tot_nr_heads, nr_mean_ablated):
    """
    Reconstruct the embeddings using the contribution of only the heads with PCs in data.
    Parameters:
    - data (list): List of dictionaries containing all details for each principal component.
    - data_pcs (list): List of dictionaries containing all details for each principal component of interest.
    - embeddings (list): List containing the embeddings to reconstruct.
    - types (list): List of types of embeddings to reconstruct.
    - return_princ_comp (bool): Return the principal components of the given embeddings
    - plot (bool): Plot the reconstructed embeddings cosine similarity.
    - means (list): List of mean values for the embeddings. mean_ablated_and_replaced = mlps.sum(axis=1) + attns.sum(axis=(1, 2))


    Returns:
    - list: Reconstructed embeddings NOT mean centered.
    - data: Data updated with principal components of the given embeddings (if return_princ_comp is True).
    """

    if len(embeddings) == 0:
        assert False, "No embeddings to reconstruct or different lengths."

    # Sort content of data_pcs based on the natural order of pcs
    heads_to_keep = sorted(list(set([(x["layer"], x["head"]) for x in data_pcs if (x["layer"], x["head"]) != (-1, -1)])))

    print("Heads to keep:", len(heads_to_keep))
    # Initialize the reconstructed embeddings
    nr_layer_not_anal = tot_nr_layers - nr_mean_ablated
    # Put initial contribution
    reconstructed_embeddings = mlps.sum(axis=1) + attns[:, 0:nr_layer_not_anal, :, :].sum(axis=(1, 2))

    # Iterate over each principal component of interest and keep the contribution of their head, sum mean ablation for all the other
    prev_layer, prev_head = nr_layer_not_anal, 0
    for layer, head in heads_to_keep:
        reconstructed_embeddings += attns[:, layer, head, :]
        # Add mean ablation
        while prev_layer != layer or prev_head != head: 
            # Add mean contribution of all the data not in the pcs
            reconstructed_embeddings += torch.mean(attns[:, prev_layer, prev_head, :], axis=0) #[b, l, h, d]
            if prev_head == tot_nr_heads - 1:
                prev_head = 0
                prev_layer += 1
            else:
                prev_head += 1   
            if prev_head == tot_nr_heads -1 and prev_layer == tot_nr_layers - 1:
                break
        prev_head, prev_layer = head + 1, layer
        if prev_head == tot_nr_heads:
                prev_head = 0
                prev_layer += 1
        if prev_head == tot_nr_heads and prev_layer == tot_nr_layers:
            break

    return reconstructed_embeddings

@torch.no_grad()
def reconstruct_embeddings(data, embeddings, types, device="cpu", return_princ_comp=False, plot=False, means=[]):
    """
    Reconstruct the embeddings using the principal components in data.
    Parameters:
    - data (list): List of dictionaries containing details for each principal component of interest.
    - embeddings (list): List containing the embeddings to reconstruct.
    - types (list): List of types of embeddings to reconstruct.
    - return_princ_comp (bool): Return the principal components of the given embeddings
    - plot (bool): Plot the reconstructed embeddings cosine similarity.
    - means (list): List of mean values for the embeddings.

    Returns:
    - list: Reconstructed embeddings NOT mean centered.
    - data: Data updated with principal components of the given embeddings (if return_princ_comp is True).
    """

    if len(embeddings) == 0 or len(types) != len(embeddings):
        assert False, "No embeddings to reconstruct or different lengths."

    # Initialize the reconstructed embeddings
    reconstructed_embeddings = [torch.zeros_like(embeddings[i].to(device)) for i in range(len(embeddings))]

    # Store reconstruction values for plotting
    rec_x = []
    num_elements = []

    # Iterate over each principal component of interest
    for count, component in enumerate(data):
        # Retrieve projection matrices and mean values
        project_matrix = torch.tensor(component["project_matrix"]).to(device)
        vh = torch.tensor(component["vh"]).to(device)
        princ_comp = component["princ_comp"]
        s = component["strength_abs"]

        # Reconstruct Embeddings
        for i in range(len(embeddings)):
            # Derive masking
            mask = torch.zeros((embeddings[i].shape[0], vh.shape[0])).to(device)

            if types[i] == "text":
                mask[:, princ_comp] = (embeddings[i] @ vh.T)[:, princ_comp].squeeze()
                embed_proj_back = mask @ project_matrix @ vh
                reconstructed_embeddings[i] += embed_proj_back
            else:
                mask[:, princ_comp] = (embeddings[i] @ vh.T)[:, princ_comp].squeeze()
                embed_proj_back = mask @ project_matrix @ vh
                reconstructed_embeddings[i] += embed_proj_back

            if plot:
                reconstructed_embeddings_norm = reconstructed_embeddings[i] / reconstructed_embeddings[i].norm(dim=-1, keepdim=True)
                embedding_norm = embeddings[i] / embeddings[i].norm(dim=-1, keepdim=True)
                # Calculate and store the cosine reconstruction score
                cosine_score = reconstructed_embeddings_norm @ (embedding_norm).T
                rec_x.append(cosine_score.item())
                num_elements.append(count)

            if return_princ_comp:
                component["correlation_princ_comp_abs"] = torch.abs(mask[:, princ_comp]).item()
                component["correlation_princ_comp"] = mask[:, princ_comp].item()

    # Plot the reconstruction values if requested
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(num_elements, rec_x, marker='o')
        plt.ylabel("Reconstructed Cosine Similarity")
        plt.xlabel("Number of Principal Components Used")
        plt.title("Reconstruction Cosine Similarity vs. Number of Principal Components")
        plt.grid(True)
        plt.show()

    return reconstructed_embeddings, data

def print_used_heads(data):
    """
    Print the used heads in the data.

    Parameters:
    - data (list): List of dictionaries containing details for each principal component of interest.

    Returns:
    - None
    """
    # Track Found heads and layers
    track_h_l = {}
    for entry in data:
        key = (entry["layer"], entry["head"])
        val = track_h_l.get(key, 0)
        track_h_l[key] = val
        track_h_l[key] += 1

    print(f"Used a total number of {len(track_h_l)} different heads")


    return data

def get_remaining_pcs(data, top_k_details):
    # Get the remaining principal components (very simple and unoptimized)
    remaining_pcs = []
    for entry_data in data:
        found = False
        for entry in top_k_details:
            layer = entry["layer"]
            head = entry["head"]
            pc = entry["princ_comp"]

            if entry_data["layer"] == layer and entry_data["head"] == head and entry_data["princ_comp"] == pc:
                found = True
                break
        if not found:
            remaining_pcs.append(entry_data)

    return remaining_pcs


def reconstruct_top_embedding(data, embedding, mean, type, max_reconstr_score, top_k=10, approx=0.90, plot=True, device="cpu"):
    """
    Reconstruct the embeddings using the principal components in data.
    Parameters:
        - data (list): List of dictionaries containing details for each principal component of interest.
        - embedding: The embedding to reconstruct.
        - type: Type of embedding to reconstruct.
        - max_reconstr_score: Maximum achievable reconstruction score.
        - top_k: Number of top principal components to consider.
        - approx: Approximation threshold for the reconstruction score.

    Returns:
        - data: Data updated with top_k principal components of the given embeddings (if return_princ_comp is True).
    """

    # Store reconstruction values for plotting
    rec_x = []
    num_elements = []
    # Track Found heads and layers
    track_h_l = {}
    
    for count, entry in enumerate(data):
        if count == 0:
            # For the first principal component, initialize the query representation
            [query_repres], _ = reconstruct_embeddings([data[count]], [embedding], [type], return_princ_comp=False, device=device)

        else:
            # For subsequent components, accumulate their contributions
            [query_repres_tmp], _ = reconstruct_embeddings([data[count]], [embedding], [type], return_princ_comp=False, device=device)
            query_repres += query_repres_tmp
        # Add count of layer and head
        key = (data[count]["layer"], data[count]["head"])
        val = track_h_l.get(key, 0)
        track_h_l[key] = val
        track_h_l[key] += 1

        # Compute the current score:

        query_repres_norm = query_repres/query_repres.norm(dim=-1, keepdim=True)
        
        embedding_norm = embedding/embedding.norm(dim=-1, keepdim=True)
        """ # Compute the current score: how well this partial reconstruction matches the original embedding
        score = query_repres_norm @ embedding_norm.T
        query_repres_norm = query_repres/query_repres.norm(dim=-1, keepdim=True)
        query_repres_norm += mean 
        query_repres_norm /= query_repres_norm.norm(dim=-1, keepdim=True)
        embedding_norm = embedding/embedding.norm(dim=-1, keepdim=True) """
        # Compute the current score: how well this partial reconstruction matches the original embedding
        score = query_repres_norm @ embedding_norm.T
        #score = query_repres_norm @ (embedding + mean).T

        
        rec_x.append(score.item())
        num_elements.append(count)

        # If we've reached the top_k limit or our reconstruction score is good enough 
        # (relative to max_reconstr_score and meeting the approx threshold), stop adding more components.
        if count == (top_k - 1) or score / max_reconstr_score > approx:
            top_k = count + 1
            break

    # Print information about how well the reconstruction performed
    percentage = (score / max_reconstr_score * 100).item()
    print(
        f"Reconstruction Quality Report:\n"
        f"- Maximum achievable reconstruction score: {max_reconstr_score.item():.4f}\n"
        f"- Used a total number of {len(track_h_l)} different heads\n"
        f"- Current reconstruction score: {score.item():.4f}\n"
        f"  This corresponds to {percentage:.2f}% of the maximum possible score.\n"
    )

    print(
        f"The reconstruction was performed using the top {top_k} principal component(s).\n "
        f"Increasing this number may improve the reconstruction score.\n\n"
    )

    if plot:
        # Plot the reconstruction values if requested
        plt.figure(figsize=(8, 6))
        plt.plot(num_elements, rec_x, marker='o')
        plt.ylabel("Reconstructed Cosine Similarity")
        plt.xlabel("Number of Principal Components Used")
        plt.title("Reconstruction Cosine Similarity vs. Number of Principal Components")
        plt.grid(True)
        plt.show()

    return data[:top_k]

def reconstruct_top_embedding_residual(data, embedding, mean, type, max_reconstr_score, top_k=10, approx=0.90):
    """
    Reconstruct the embeddings using the principal components in data.
    Parameters:
        - data (list): List of dictionaries containing details for each principal component of interest.
        - embedding: The embedding to reconstruct.
        - type: Type of embedding to reconstruct.
        - max_reconstr_score: Maximum achievable reconstruction score.
        - top_k: Number of top principal components to consider.
        - approx: Approximation threshold for the reconstruction score.

    Returns:
        - data: Data updated with top_k principal components of the given embeddings (if return_princ_comp is True).
    """

    # Store reconstruction values for plotting
    rec_x = []
    num_elements = []
    # Track Found heads and layers
    track_h_l = {}

    score = 0
    nr_pcs = 0
    query_curr = torch.zeros_like(embedding) 
    query_res = embedding
    data_best = []
    while nr_pcs != (top_k) and score / max_reconstr_score < approx:
        score *= 0
        max_score = torch.tensor(0)
        max_index = 0
        for count, entry in enumerate(data):
            [query_repres], _ = reconstruct_embeddings([data[count]], [query_res], [type], return_princ_comp=False)

            # Compute the current score:
            query_repres += mean
            embedding_dec = embedding + mean
            # Compute the current score: how well this partial reconstruction matches the original embedding
            score = embedding_dec @ query_repres.T
            
            if score.item() > max_score.item():
                max_score = score
                max_index = count
                if len(data_best) == nr_pcs + 1:
                    data_best[nr_pcs] = data[count]
                else:
                    data_best.append(data[count])

        # Save best PCs
        rec_x.append(max_score.item())
        nr_pcs += 1
        num_elements.append(nr_pcs)

        # Compute current query
        [query_repres], _ = reconstruct_embeddings([data[max_index]], [query_res], [type], return_princ_comp=False)

        query_curr += query_repres
        query_res -= query_repres

        # Add count of layer and head
        key = (data[max_index]["layer"], data[max_index]["head"])
        val = track_h_l.get(key, 0)
        track_h_l[key] = val
        track_h_l[key] += 1

        # Compute the current score:
        query_repres_norm = query_curr / query_curr.norm(dim=-1, keepdim=True)
        query_repres_norm += mean
        query_repres_norm /= query_repres_norm.norm(dim=-1, keepdim=True)
        embedding_dec = embedding + mean
        # Compute the current score: how well this partial reconstruction matches the original embedding
        score = embedding_dec @ query_repres_norm.T

        print(score)
    top_k = nr_pcs
    # Print information about how well the reconstruction performed
    percentage = (score / max_reconstr_score * 100).item()
    print(
        f"Reconstruction Quality Report:\n"
        f"- Maximum achievable reconstruction score: {max_reconstr_score.item():.4f}\n"
        f"- Used a total number of {len(track_h_l)} different heads\n"
        f"- Current reconstruction score: {score.item():.4f}\n"
        f"  This corresponds to {percentage:.2f}% of the maximum possible score.\n"
    )

    print(
        f"The reconstruction was performed using the top {top_k} principal component(s).\n "
        f"Increasing this number may improve the reconstruction score.\n\n"
    )

    # Plot the reconstruction values if requested
    plt.figure(figsize=(8, 6))
    plt.plot(num_elements, rec_x, marker='o')
    plt.ylabel("Reconstructed Cosine Similarity")
    plt.xlabel("Number of Principal Components Used")
    plt.title("Reconstruction Cosine Similarity vs. Number of Principal Components")
    plt.grid(True)
    plt.show()

    return data_best

def image_grid(images, rows, cols, labels=None, scores=None, scores_vis=None, figsize=None):
    """
    Display a collection of images arranged in a grid with optional labels and scores.
    
    Parameters:
        - images (list): List of image data (e.g., NumPy arrays or PIL Images) to display.
        - rows (int): Number of rows in the image grid.
        - cols (int): Number of columns in the image grid.
        - labels (list, optional): List of labels for each image. Defaults to None.
        - scores (list, optional): List of scores for each image, displayed alongside labels. Defaults to None.
        - scores_vis (list, optional): List of scores for each image, displayed alongside labels. Defaults to None.
        - figsize (tuple, optional): Size of the figure in inches as (width, height). If not provided, it is calculated based on rows and cols.
    
    Returns:
        - None
    """
    if figsize is None:
        figsize = (cols * 2, rows * 2)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            title = ''
            if labels:
                title += labels[i]
            if scores:
                title += f" ({scores[i]:.3f}) ({scores_vis[i]:.3f})"
            if title:
                ax.set_title(title, fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_dataset(imagenet_path, transform, samples_per_class=3, tot_samples_per_class=50, seed=50, dataset="imagenet"):
    """
    Create a balanced subset of the ImageNet validation dataset for nearest-neighbor (NN) search.

    Args:
    - imagenet_path (str): Path to the root directory of the ImageNet dataset.
    - transform (callable): A function/transform to apply to the images in the dataset.
    - samples_per_class (int, optional): Number of samples to pick per class. Default is 3.
    - tot_samples_per_class (int, optional): Total number of samples per class considered as a block. Default is 50.
    - seed (int, optional): Random seed for reproducibility. Default is 50.

    Returns:
    - torch.utils.data.Subset: A subset of the ImageNet validation dataset containing the selected samples.
    """
    # Load the ImageNet validation dataset
    ds_vis = ImageNet(root=imagenet_path, split="val", transform=transform)

    # all_indices: a list of indices representing every sample in the validation set.
    all_indices = list(range(len(ds_vis)))
    nr_classes = len(ds_vis.classes)  # number of classes in the dataset

    # Set the random seed for reproducibility
    random.seed(seed)

    # For each class, randomly pick 'samples_per_class' samples from a block of 'tot_samples_per_class' samples.
    # This results in a balanced subset of classes.
    index = [random.sample(all_indices[x*tot_samples_per_class:(x + 1)*tot_samples_per_class], samples_per_class) 
             for x in range(nr_classes)]

    # Flatten the list of indices (since it's currently a list of lists)
    index = [x for xs in index for x in xs]

    # Create subsets of the dataset and visualization dataset using the selected indices
    if samples_per_class != tot_samples_per_class:
        print("Using subset of imagenet")
        ds_vis = torch.utils.data.Subset(ds_vis, index)
    return ds_vis


def create_dbs(scores_array_images, scores_array_texts, nr_top_imgs=20, nr_worst_imgs=20, nr_cont_imgs=20, text_query = None, max_reconstr_score = -1):
    """
    Create three sets of data containing top, worst, and continuous cosine similarity results 
    for images and corresponding text queries.

    Args:
    - data (iterable): The source dataset containing image and text samples.
    - scores_array_images (np.ndarray): Array of image scores, structured with 'score' and index values.
    - scores_array_texts (np.ndarray): Array of text scores, structured with 'score' and index values.
    - nr_top_imgs (int, optional): Number of top (highest cosine similarity) images to retrieve. Default is 20.
    - nr_worst_imgs (int, optional): Number of worst (lowest cosine similarity) images to retrieve. Default is 20.
    - nr_cont_imgs (int, optional): Number of continuous (interpolated similarity) images to retrieve. Default is 20.
    - text_query (str): The original query text for which cosine similarities are analyzed.
    - max_reconstr_score (float or torch.Tensor): The maximum reconstruction strength to display alongside results.

    Returns:
    - list of tuples: Each tuple contains:
        - db_images (np.ndarray): Selected image data based on scores.
        - db_texts (np.ndarray): Selected text data based on scores.
        - description (str): A string describing the type of retrieval (top, worst, or continuous).
        - num_samples (int): Number of samples in the current retrieval.
    """
    # Sort indices of scores
    sorted_scores_images = np.sort(scores_array_images, order=('score', 'score_vis'))

    # Top scores
    top_dbs_images = sorted_scores_images[-nr_top_imgs:][::-1]  # Get top `nr_top_imgs` elements, sorted descending

    # Worst scores
    worst_dbs_images = sorted_scores_images[:nr_worst_imgs]  # Get worst `nr_worst_imgs` elements, sorted ascending

    # Continuous scores (step-based interpolation)
    t = np.linspace(0, 1, num=nr_cont_imgs)  # Uniformly spaced in [0, 1]
    scaled_t = 0.5 * (1 - np.cos(np.pi * t))  # Cosine transformation for symmetry
    nonlinear_indices_images = (scaled_t * (len(sorted_scores_images) - 1)).astype(int)
    cont_dbs_images = sorted_scores_images[nonlinear_indices_images]


    if scores_array_texts is None:
        sorted_scores_texts = None
        top_dbs_texts = None
        cont_dbs_texts = None
        worst_dbs_texts = None
        nonlinear_indices_texts = None
    else:
        sorted_scores_texts = np.sort(scores_array_texts, order=('score', 'score_vis'))
        top_dbs_texts = sorted_scores_texts[-nr_top_imgs:][::-1]  # Get top `nr_top_imgs` elements, sorted descending
        worst_dbs_texts = sorted_scores_texts[:nr_worst_imgs]  # Get worst `nr_worst_imgs` elements, sorted ascending
        nonlinear_indices_texts = (scaled_t * (len(sorted_scores_texts) - 1)).astype(int)
        cont_dbs_texts = sorted_scores_texts[nonlinear_indices_texts]

    # Prepare three sets of results to display:
    # 1. Highest cosine similarity samples
    # 2. Most negative cosine similarity samples
    # 3. A continuous range of samples with intermediate similarity
    dbs = [
        (top_dbs_images, top_dbs_texts, 
         f"Printing highest cosine similarity with original query for text: '{text_query}' with strength {max_reconstr_score.item()}" if 
         text_query is not None else f"Printing highest cosine similarity",
         nr_top_imgs),
        (worst_dbs_images, worst_dbs_texts, 
         f"Printing most negative cosine similarity with original query for text: '{text_query}' with strength {max_reconstr_score.item()}" if 
         text_query is not None else f"Printing most negative cosine similarity",
         nr_worst_imgs),
        (cont_dbs_images, cont_dbs_texts, 
         f"Printing continuous cosine similarity with original query for text: '{text_query}' with strength {max_reconstr_score.item()}" if
         text_query is not None else f"Printing continuous cosine similarity",
         nr_cont_imgs)
    ]

    return dbs


def visualize_dbs(data, dbs, ds_vis, texts_str, classes, text_query= None):
    """
    Visualize and analyze subsets of images and their associated text contributions 
    based on cosine similarity scores.

    Args:
    - data (iterable): Data containing principal components or details to analyze.
    - dbs (list of tuples): List of datasets to visualize, where each tuple contains:
        - db (list of tuples): Subset of images with scores and indices.
        - db_text (list of tuples): Subset of text entries with scores and indices.
        - display_text (str): Header text describing the current subset (e.g., "Top scores").
        - length (int): Number of samples in the current subset.
    - ds_vis (torch.utils.data.Dataset): Image dataset (e.g., ImageNet validation dataset).
    - texts_str (list of str): List of textual descriptions or queries corresponding to text indices.
    - text_query (str): The original query text being analyzed.
    - imagenet_classes (list of str): List of class names in the ImageNet dataset.

    Returns:
    - None
    """
    # Print a header to indicate which text query we are analyzing
    if text_query is not None:
        print(f"----Decomposition for text: '{text_query}'----")
        print_data(data, is_corr_present=True)

    else:
        print(f"----Decomposition of Principal Component----")
        print_data(data, is_corr_present=False)


    for db, db_text, display_text, length in dbs:
        images, labels, scores, scores_vis = [], [], [], []

        # Collect image samples, their class labels, and similarity scores
        for score, score_vis, image_index in db:
            images.append(ds_vis[image_index][0])  # Image data
            labels.append(classes[ds_vis[image_index][1]])  # Class label
            scores.append(score)
            scores_vis.append(score_vis)  # Cosine similarity score

        # Print the header describing the current subset
        print(display_text)

        # Prepare and display the corresponding texts with their scores
        output_rows = []

        if db_text is not None:
            for score, score_vis, text_index in db_text:
                output_rows.append([texts_str[text_index], score, score_vis])
            output_df = pd.DataFrame(output_rows, columns=["Text", "Cosine Similarity", "Correlation"])
            print(tabulate(output_df, headers='keys', tablefmt='psql'))

        # Display the images in a grid layout
        rows, cols = (length // 4, 4)  # Grid layout: 4 columns, rows derived from the number of images
        image_grid(images, rows, cols, labels=labels, scores=scores, scores_vis=scores_vis)

def visualize_principal_component(
    layer, head, princ_comp, nr_top_imgs, nr_worst_imgs, nr_cont_imgs,
    attention_dataset, final_embeddings_images, final_embeddings_texts, 
    seed, data_path, texts_str, samples_per_class=3, dataset="imagenet",
    tot_samples_per_class=50,
    transform=visualization_preprocess
):
    """
    Visualize the top, worst, and continuous cosine similarity scores for images and texts 
    corresponding to a specific principal component of interest.

    Args:
    - layer (int): The attention layer to analyze.
    - head (int): The attention head within the specified layer.
    - princ_comp (int): The principal component index to visualize.
    - nr_top_imgs (int): Number of top (highest similarity) images to retrieve and display.
    - nr_worst_imgs (int): Number of worst (lowest similarity) images to retrieve and display.
    - nr_cont_imgs (int): Number of continuous (interpolated similarity) images to retrieve and display.
    - attention_dataset (str): Path to the JSON file containing attention data.
    - final_embeddings_images (torch.Tensor): Final image embeddings for similarity computation.
    - final_embeddings_texts (torch.Tensor): Final text embeddings for similarity computation.
    - seed (int): Random seed for reproducibility when creating the dataset.
    - imagenet_path (str): Path to the root directory of the ImageNet dataset.
    - texts_str (list of str): List of text descriptions corresponding to text embeddings.
    - transform (callable, optional): Transformation function to preprocess ImageNet images. 
                                      Defaults to `visualization_preprocess`.

    Returns:
    - None
    """
    # Retrieve and filter the principal component data
    data = get_data(attention_dataset, -1, skip_final=True)
    data = get_data_component(data, layer, head, princ_comp)
    

    # Dataset:
    if dataset == "imagenet":
        ds = ImageNet(root=data_path+"imagenet/", split="val", transform=transform)
    elif dataset == "binary_waterbirds":
        ds = BinaryWaterbirds(root=data_path+"waterbird_complete95_forest2water2/", split="test", transform=transform)
    elif dataset == "CIFAR100":
        ds = CIFAR100(
            root=data_path, download=True, train=False, transform=transform
        )
    elif dataset == "CIFAR10":
        ds = CIFAR10(
            root=data_path, download=True, train=False, transform=transform
        )
    else:
        ds = ImageFolder(root=data_path, transform=transform)

    # Depending
    ds_vis = dataset_subset(
        ds,
        samples_per_class=samples_per_class,
        tot_samples_per_class=tot_samples_per_class,  # or whatever you prefer
        seed=seed,
    )

    # Initialize arrays to store similarity scores for images and texts
    scores_array_images = np.empty(
        final_embeddings_images.shape[0], 
        dtype=[('score', 'f4'), ('score_vis', 'f4'), ('img_index', 'i4')]
    )
    scores_array_texts = np.empty(
        final_embeddings_texts.shape[0], 
        dtype=[('score', 'f4'), ('score_vis', 'f4'), ('txt_index', 'i4')]
    )

    # Create arrays of indices for referencing images and texts
    indexes_images = np.arange(0, final_embeddings_images.shape[0], 1)
    indexes_texts = np.arange(0, final_embeddings_texts.shape[0], 1)

    # Compute mean embeddings for centering
    mean_final_images = torch.mean(final_embeddings_images, axis=0)
    mean_final_texts = torch.mean(final_embeddings_texts, axis=0)

    images_centered = final_embeddings_images - mean_final_images
    texts_centered = final_embeddings_texts - mean_final_texts

    # Normalize embeddings to unit norm

    # Compute cosine similarity scores with the specified principal component
    vh = torch.tensor(data[0]["vh"])
    scores_array_images["score_vis"] = ((images_centered @ vh.T)[:, princ_comp]).numpy()
    # images_centered /= images_centered.norm(dim=-1, keepdim=True)
    scores_array_images["score"] = ((images_centered @ vh.T)[:, princ_comp]).numpy()
    scores_array_images["img_index"] = indexes_images

    scores_array_texts["score_vis"] = ((texts_centered @ vh.T)[:, princ_comp]).numpy()
    # texts_centered /= texts_centered.norm(dim=-1, keepdim=True)
    scores_array_texts["score"] = ((texts_centered @ vh.T)[:, princ_comp]).numpy()
    scores_array_texts["txt_index"] = indexes_texts

    # Create datasets for visualization
    dbs = create_dbs(
        scores_array_images, scores_array_texts, nr_top_imgs, 
        nr_worst_imgs, nr_cont_imgs, text_query=None, max_reconstr_score=-1
    )
    
    # Hardcoded visualizations
    nrs_dbs = [nr_top_imgs, nr_worst_imgs, nr_cont_imgs]
    dbs_new = []
    for i, db in enumerate(dbs):
        if nrs_dbs[i] == 0:
            continue
        dbs_new.append(db)
    
    classes = {
        'imagenet': imagenet_classes, 
        'CIFAR10': cifar_10_classes,
        'waterbirds': cub_classes, 
        'binary_waterbirds': waterbird_classes, 
        'cub': cub_classes}[dataset]
    
    # Visualize the results
    visualize_dbs(data, dbs_new, ds_vis, texts_str, classes, text_query=None)



@torch.no_grad()
def reconstruct_embeddings_proj(data, embeddings, types, device="cpu",return_princ_comp=False, plot=False, means=[]):
    """
    Reconstruct the embeddings using the principal components in data.
    Parameters:
    - data (list): List of dictionaries containing details for each principal component of interest.
    - embeddings (list): List containing the embeddings to reconstruct.
    - types (list): List of types of embeddings to reconstruct.
    - return_princ_comp (bool): Return the principal components of the given embeddings
    - plot (bool): Plot the reconstructed embeddings cosine similarity.
    - means (list): List of mean values for the embeddings.

    Returns:
    - list: Reconstructed embeddings NOT mean centered.
    - data: Data updated with principal components of the given embeddings (if return_princ_comp is True).
    """

    if len(embeddings) == 0 or len(types) != len(embeddings):
        assert False, "No embeddings to reconstruct or different lengths."

    # Initialize the reconstructed embeddings
    reconstructed_embeddings = [torch.zeros_like(embeddings[i].to(device)) for i in range(len(embeddings))]

    # Iterate over each principal component of interest
    all_pcs = torch.zeros((len(data), torch.tensor(data[0]["vh"]).shape[1])).to(device)
    #activationsdada
    for count, component in enumerate(data):
        # Retrieve projection matrices and mean values
        project_matrix = torch.tensor(component["project_matrix"]).to(device)
        vh = torch.tensor(component["vh"]).to(device)
        princ_comp = component["princ_comp"]
        s = component["strength_abs"]
        # Recover princ comp
        all_pcs[count] = vh[princ_comp, :]

    # Project in row space
    # Reconstruct Embeddings

    # Perform SVD of data matrix
    u, s, vh = torch.linalg.svd(all_pcs, full_matrices=False)
    # Total sum of singular values
    total_variance = torch.sum(s)
    # Cumulative sum of singular values
    cumulative_variance = torch.cumsum(s, dim=0)
    # Determine the rank where cumulative variance exceeds the threshold of total variance
    threshold = 0.999 # How much variance should cover the top princ_comps of the matrix 
    rank = torch.sum(cumulative_variance / total_variance < threshold).item() + 1
    print(f"The rank of the matrix is {rank}")
    for i in range(len(embeddings)):
        # Derive masking
        reconstructed_embeddings[i] = embeddings[i] @ vh.T @ vh #@ all_pcs.T @ torch.linalg.pinv(all_pcs @ all_pcs.T) @ all_pcs


    return reconstructed_embeddings, data



def test_accuracy(prediction, labels, label="Classifier"):
    """
    Calculate the accuracy of the model's predictions.
    """
    accuracy, indexes = accuracy_correct(prediction, labels)
    accuracy_pred = accuracy[0] * 100
    print(f"For the approach {label}, the accuracy is: {accuracy_pred:3f}%")
    return accuracy_pred, indexes[0]

def print_wrong_elements_label(indexes_1, label, subset_dim, text="wrong"):
    # TODO: Hardcoded for ImageNet
    # Retrieve the labels of the dataset. 
    # This is hardcoded for ImageNet where nr_classes is the number of classes (usually 1000).
    nr_samples = torch.arange(1000)    
    idx_label = [idx for idx, x in enumerate(imagenet_classes) if x == label][0]
    count = 0
    for idx in range(idx_label*subset_dim, idx_label*subset_dim + subset_dim):
        if not indexes_1[idx]:
            count += 1
        
    # Print the result
    print(f"The {label} wrong elements labels are: {count}")

    return count

def print_tot_wrong_elements_label(predictions, label):
    # TODO: Hardcoded for ImageNet
    # Retrieve the labels of the dataset. 
    # This is hardcoded for ImageNet where nr_classes is the number of classes (usually 1000).
    curr_preds = predictions.argmax(dim=1)
    label_idx = imagenet_classes.index(label)
    count = 0
    for pred in curr_preds:
        if pred == label_idx:
            count +=1
        
    # Print the result
    print(f"In total, for the {label} label, there are {count} wrong elements.")


    return count
def print_diff_elements(indexes_1, indexes_2, subset_dim):
    # TODO: Hardcoded for ImageNet
    # Retrieve the labels of the dataset. 
    # This is hardcoded for ImageNet where nr_classes is the number of classes (usually 1000).
    nr_samples = torch.arange(1000)
    classes_indexes = nr_samples.repeat_interleave(subset_dim)
    class_labels = np.array([imagenet_classes[i] for i in classes_indexes])
    # Determine which elements differ between the two reconstructions
    wrong_elements = np.array(~(indexes_1 == indexes_2))
    
    print(f"Number of elements with different results between the two reconstruction methods: {len(class_labels[wrong_elements])}")
    # Track occurrences
    label_count = defaultdict(int)
    output_set = set()

    # Iterate through mask and labels
    for idx, is_wrong in enumerate(wrong_elements):
        if is_wrong:
            label = class_labels[idx]
            label_count[label] += 1

    # Sort the set by nr_of_prev_occurrences in descending order
    sorted_output = sorted(label_count.items(), key=lambda x: x[1], reverse=True)

    # Print the result
    print(f"The different elements labels are: {sorted_output}")

def print_wrong_elements(indexes_1, labels, classes, text="wrong"):
    # TODO: Hardcoded for ImageNet
    # Retrieve the labels of the dataset. 
    # This is hardcoded for ImageNet where nr_classes is the number of classes (usually 1000).
    class_labels = np.array([classes[i] for i in labels])
    
    print(f"Number of elements with {text} results between the two reconstruction methods: {len(class_labels[indexes_1])}")
    # Track occurrences
    label_count = defaultdict(int)

    # Iterate through mask and labels
    for idx, is_correct in enumerate(indexes_1):
        if not is_correct:
            label = class_labels[idx]
            label_count[label] += 1

    # Sort the set by nr_of_prev_occurrences in descending order
    sorted_output = sorted(label_count.items(), key=lambda x: x[1], reverse=True)

    # Print the result
    print(f"The {text} elements labels are: {sorted_output}")
    return sorted_output


def print_correct_elements(indexes_1, labels, classes, text="correct"):
    return print_wrong_elements(~indexes_1, labels, classes, text)


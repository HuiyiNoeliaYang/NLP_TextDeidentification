import os
import argparse
import collections
from typing import Dict, List, Tuple

import pandas as pd
import torch
import transformers
from tqdm import tqdm

from model_cfg import model_paths_dict
from model import CoordinateAscentModel
from datamodule import WikipediaDataModule
from utils import get_profile_embeddings_by_model_key


num_cpus = len(os.sched_getaffinity(0))


def get_profile_embeddings(model_key: str) -> Tuple[torch.Tensor, Dict[str, int]]:
    profile_embeddings = get_profile_embeddings_by_model_key(model_key=model_key)
    print("concatenating train, val, and test profile embeddings")
    all_profile_embeddings = torch.cat(
        (profile_embeddings['test'], profile_embeddings['val'], profile_embeddings['train']), dim=0
    )
    print("all_profile_embeddings:", all_profile_embeddings.shape)
    split_sizes = {
        'test': profile_embeddings['test'].shape[0],
        'val': profile_embeddings['val'].shape[0],
        'train': profile_embeddings['train'].shape[0],
    }
    return all_profile_embeddings, split_sizes


def get_profile_metadata(model: CoordinateAscentModel, split_sizes: Dict[str, int], max_seq_length: int) -> Tuple[List[str], List[str]]:
    train_split = f"train[:{split_sizes['train']}]"
    val_split = f"val[:{split_sizes['val']}]"
    test_split = f"test[:{split_sizes['test']}]"

    dm = WikipediaDataModule(
        document_model_name_or_path=model.document_model_name_or_path,
        profile_model_name_or_path=model.profile_model_name_or_path,
        dataset_name='wiki_bio',
        dataset_train_split=train_split,
        dataset_val_split=val_split,
        dataset_test_split=test_split,
        dataset_version='1.2.0',
        num_workers=num_cpus,
        train_batch_size=256,
        eval_batch_size=256,
        max_seq_length=max_seq_length,
        sample_spans=False,
    )
    dm.setup("fit")

    test_names = list(dm.test_dataset['name'])
    test_docs = list(dm.test_dataset['document'])
    val_names = list(dm.val_dataset['name'])
    val_docs = list(dm.val_dataset['document'])
    train_names = list(dm.train_dataset['name'])
    train_docs = list(dm.train_dataset['document'])

    combined_names = test_names + val_names + train_names
    combined_docs = test_docs + val_docs + train_docs
    return combined_names, combined_docs


def _tokenize_texts(tokenizer: transformers.AutoTokenizer, texts: List[str], max_length: int, device: torch.device, prefix: str) -> dict:
    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return {f'{prefix}__{k}': v.to(device) for k, v in encodings.items()}


def main(model_key: str, document_type: str, adv_csv_path: str, max_seq_length: int = 128, batch_size: int = 256, limit_rows: int = None):
    checkpoint_path = model_paths_dict[model_key]
    print(f"running attack on {model_key} loaded from {checkpoint_path}")
    try:
        model = CoordinateAscentModel.load_from_checkpoint(checkpoint_path)
    except RuntimeError as err:
        print("Warning: Loading checkpoint with strict=False due to key mismatch:", err)
        model = CoordinateAscentModel.load_from_checkpoint(checkpoint_path, strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    csv_path = os.path.abspath(adv_csv_path)
    print(f"Loading perturbed data from {csv_path}")
    adv_df = pd.read_csv(csv_path)
    required_columns = {'perturbed_text', 'ground_truth_output'}
    missing = required_columns - set(adv_df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if limit_rows is not None:
        adv_df = adv_df.iloc[:limit_rows].copy()

    texts = adv_df['perturbed_text'].astype(str).tolist()
    true_idxs = torch.tensor(adv_df['ground_truth_output'].to_numpy(), dtype=torch.long)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model.document_model_name_or_path)
    all_profile_embeddings, split_sizes = get_profile_embeddings(model_key=model_key)
    all_profile_embeddings = all_profile_embeddings.to(device)
    profile_names, profile_docs = get_profile_metadata(model=model, split_sizes=split_sizes, max_seq_length=max_seq_length)

    model.document_model.eval()
    model.document_embed.eval()
    model_device = device

    total = 0
    total_correct_by_k = collections.defaultdict(lambda: 0)
    k_values = [1, 10, 100, 1000]

    pred_rows = []

    for start in tqdm(range(0, len(texts), batch_size), desc='Evaluating custom CSV', colour='yellow'):
        end = start + batch_size
        batch_texts = texts[start:end]
        batch = _tokenize_texts(tokenizer, batch_texts, max_seq_length, model_device, document_type)
        document_idxs = true_idxs[start:end].to(model_device)

        with torch.no_grad():
            document_embeddings = model.forward_document(batch=batch, document_type=document_type)
            document_to_profile_logits = document_embeddings @ all_profile_embeddings.T

        total += len(document_to_profile_logits)

        logits_cpu = document_to_profile_logits.cpu()
        preds = logits_cpu.argmax(dim=1)

        for rel_idx, pred_idx in enumerate(preds):
            abs_idx = start + rel_idx
            matched_idx = int(pred_idx.item())
            matched_name = profile_names[matched_idx] if matched_idx < len(profile_names) else ''
            matched_text = profile_docs[matched_idx] if matched_idx < len(profile_docs) else ''
            row = adv_df.iloc[abs_idx]
            original_person = row['original_person'] if 'original_person' in row.index else ''
            original_perturbed_text = row['perturbed_text'] if 'perturbed_text' in row.index else ''
            pred_rows.append({
                "row_idx": abs_idx,
                "original_person": original_person,
                "original_perturbed_text": original_perturbed_text,
                "ground_truth_output": int(true_idxs[abs_idx].item()),
                "matched_person": matched_name,
                "matched_person_text": matched_text,
                "model_pred_idx": matched_idx,
                "is_correct": int(matched_idx == true_idxs[abs_idx].item())
            })

        for k in k_values:
            topk_correct = (
                document_to_profile_logits.topk(k=k, dim=1)
                    .indices
                    .eq(document_idxs[:, None])
                    .any(dim=1)
                    .sum()
            )
            total_correct_by_k[k] += topk_correct

    print('*** Finished custom eval ****')
    print(f'**** Evaluated on {total} test examples of type {document_type} ****')
    for k in k_values:
        acc = total_correct_by_k[k] / total
        print(f'Top-{k} accuracy = {acc * 100.0:.2f}')

    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'eval', model_key)
    os.makedirs(output_folder, exist_ok=True)
    summary_path = os.path.join(output_folder, f'custom_{document_type}.txt')
    with open(summary_path, 'w') as f:
        f.write(f'**** Evaluated on {total} test examples of type {document_type} ****\n')
        for k in k_values:
            acc = total_correct_by_k[k] / total
            f.write(f'Top-{k} accuracy = {acc * 100.0:.2f}\n')
    print(f"Summary written to {summary_path}")

    results_df = pd.DataFrame(pred_rows)
    results_path = os.path.join(output_folder, f'custom_{document_type}_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Per-example results written to {results_path}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluates model accuracy using a custom CSV of perturbed texts.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model_key', type=str, required=True, choices=model_paths_dict.keys())
    parser.add_argument('--document_type', type=str, default='perturbed_text')
    parser.add_argument('--adv_csv_path', type=str, required=True,
                        help='Path to CSV with perturbed_text and ground_truth_output columns.')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--limit_rows', type=int, default=None,
                        help='If provided, only evaluate the first N rows of the CSV.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(
        model_key=args.model_key,
        document_type=args.document_type,
        adv_csv_path=args.adv_csv_path,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        limit_rows=args.limit_rows
    )


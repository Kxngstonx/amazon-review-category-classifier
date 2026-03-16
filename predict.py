"""
Prediction module for hierarchical multi-label text classification.
Loads trained model and generates predictions for test corpus.
"""

import argparse
import os
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Set, Dict, Tuple, Optional

from model import ClassModel
from data_loader import Taxonomy, AmazonDataset
from config import config


class HierarchyConstraintPredictor:
    def __init__(self, taxonomy: Taxonomy):
        self.taxonomy = taxonomy
        
    def get_all_ancestors(self, class_id: int) -> Set[int]:
        ancestors = set([class_id])
        queue = [class_id]
        visited = set()
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            
            parents = self.taxonomy.get_parents(current_id)
            for parent_id in parents:
                if parent_id not in ancestors:
                    ancestors.add(parent_id)
                    queue.append(parent_id)
        
        return ancestors
    
    def is_leaf_node(self, class_id: int) -> bool:
        children = self.taxonomy.get_children(class_id)
        return len(children) == 0
    
    def predict_with_hierarchy(
        self,
        logits: torch.Tensor,
        class_ids: List[int],
        min_labels: Optional[int] = None,
        max_labels: Optional[int] = None,
        virtual_root_id: Optional[int] = None
    ) -> List[int]:
        """
        Predict labels with hierarchy consistency.
        Final output will have min_labels to max_labels total labels (including ancestors).
        
        Strategy:
        1. Get top predictions sorted by probability
        2. Greedily select labels that maximize specificity while maintaining hierarchy
        3. Ensure final count is between min_labels and max_labels
        """
        if min_labels is None:
            min_labels = config.PREDICT_MIN_LABELS
        if max_labels is None:
            max_labels = config.PREDICT_MAX_LABELS
        if virtual_root_id is None:
            virtual_root_id = config.VIRTUAL_ROOT_ID
        
        # Convert logits to probabilities
        probs = torch.sigmoid(logits).cpu().numpy()
        
        # Create list of (class_id, prob) sorted by probability (descending)
        candidates = []
        for idx, prob in enumerate(probs):
            class_id = class_ids[idx]
            if class_id != virtual_root_id:
                candidates.append((class_id, prob))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Greedy selection: pick most specific labels that don't violate hierarchy
        selected_labels = set()
        
        for class_id, prob in candidates:
            # Check if adding this label would exceed max_labels
            if len(selected_labels) >= max_labels:
                break
            
            # Get all ancestors of this class
            ancestors = self.get_all_ancestors(class_id)
            ancestors.discard(virtual_root_id)
            
            # Calculate how many new labels would be added
            new_labels = ancestors - selected_labels
            
            # If adding this class and its ancestors doesn't exceed max_labels, add it
            if len(selected_labels) + len(new_labels) <= max_labels:
                selected_labels.update(ancestors)
            # If we already have some labels and adding would exceed max, 
            # check if we can still add just this class (if it's already consistent)
            elif class_id not in selected_labels:
                # Check if this class's ancestors are already in selected_labels
                required_ancestors = ancestors - {class_id}
                if required_ancestors.issubset(selected_labels):
                    # All ancestors already present, safe to add just this class
                    if len(selected_labels) < max_labels:
                        selected_labels.add(class_id)
            
            # Stop if we've reached max_labels
            if len(selected_labels) >= max_labels:
                break
        
        # Ensure we have at least min_labels
        if len(selected_labels) < min_labels:
            # Add more labels from candidates
            for class_id, prob in candidates:
                if class_id not in selected_labels:
                    ancestors = self.get_all_ancestors(class_id)
                    ancestors.discard(virtual_root_id)
                    
                    # Add ancestors needed for hierarchy consistency
                    selected_labels.update(ancestors)
                    
                    if len(selected_labels) >= min_labels:
                        break
        
        # Final check: if still too many labels, keep only the most specific ones
        if len(selected_labels) > max_labels:
            # Prioritize leaf nodes and their direct ancestors
            leaf_nodes = [cid for cid in selected_labels if self.is_leaf_node(cid)]
            
            if len(leaf_nodes) > 0:
                # Keep the most probable leaf and its ancestors
                final_labels = set()
                for class_id, prob in candidates:
                    if class_id in leaf_nodes:
                        ancestors = self.get_all_ancestors(class_id)
                        ancestors.discard(virtual_root_id)
                        
                        if len(final_labels) + len(ancestors) <= max_labels:
                            final_labels.update(ancestors)
                        
                        if len(final_labels) >= min_labels:
                            break
                
                if len(final_labels) >= min_labels:
                    selected_labels = final_labels
            
            # If still too many, just take top max_labels by probability
            if len(selected_labels) > max_labels:
                sorted_selected = sorted(
                    selected_labels, 
                    key=lambda x: probs[class_ids.index(x)], 
                    reverse=True
                )
                selected_labels = set(sorted_selected[:max_labels])
        
        # Convert to sorted list for consistency
        return sorted(list(selected_labels))


def load_model(
    model_path: str,
    model_name: str,
    class_embeddings: torch.Tensor,
    device: str
) -> ClassModel:
    emb_dim = class_embeddings.shape[1]
    model = ClassModel(model_name, emb_dim, class_embeddings).to(device)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.eval()
    print(f"Model loaded from: {model_path}")
    
    return model


def predict_test_corpus(
    model: ClassModel,
    test_dataset: AmazonDataset,
    taxonomy: Taxonomy,
    class_ids: List[int],
    device: str,
    batch_size: Optional[int] = None,
    min_labels: Optional[int] = None,
    max_labels: Optional[int] = None,
    virtual_root_id: Optional[int] = None
) -> Dict[int, List[int]]:
    if batch_size is None:
        batch_size = config.PREDICT_BATCH_SIZE
    if min_labels is None:
        min_labels = config.PREDICT_MIN_LABELS
    if max_labels is None:
        max_labels = config.PREDICT_MAX_LABELS
    if virtual_root_id is None:
        virtual_root_id = config.VIRTUAL_ROOT_ID
    
    predictor = HierarchyConstraintPredictor(taxonomy)
    predictions = {}
    
    model.eval()
    
    # Create data loader
    from torch.utils.data import DataLoader
    data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Generating predictions for {len(test_dataset)} test documents...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            doc_indices = batch['index'].cpu().numpy()
            
            # Get model predictions
            logits = model(input_ids, attention_mask)
            
            # Process each document in batch
            for i, doc_id in enumerate(doc_indices):
                doc_logits = logits[i]
                
                # Predict with hierarchy consistency
                predicted_labels = predictor.predict_with_hierarchy(
                    logits=doc_logits,
                    class_ids=class_ids,
                    min_labels=min_labels,
                    max_labels=max_labels,
                    virtual_root_id=virtual_root_id
                )
                
                predictions[int(doc_id)] = predicted_labels
    
    return predictions


def save_predictions(
    predictions: Dict[int, List[int]],
    output_path: str,
    format: str = 'csv'
) -> None:
    if format == 'csv':
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("id,label\n")
            
            # Write predictions
            for doc_id in sorted(predictions.keys()):
                labels = predictions[doc_id]
                # Format: "label1,label2,label3"
                labels_str = ','.join(map(str, labels))
                f.write(f'{doc_id},"{labels_str}"\n')
    
    elif format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Predictions saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate predictions for test corpus using trained model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and data paths
    parser.add_argument('--model_path', type=str, default=config.PREDICT_MODEL_PATH,
                        help='Path to trained model checkpoint')
    parser.add_argument('--test_corpus', type=str, default=config.TEST_FILE,
                        help='Path to test corpus file')
    parser.add_argument('--taxonomy_path', type=str, default=config.TAXONOMY_FILE,
                        help='Path to taxonomy hierarchy file')
    parser.add_argument('--classes_path', type=str, default=config.CLASSES_FILE,
                        help='Path to classes file')
    parser.add_argument('--embeddings_dir', type=str, default=config.EMBEDDINGS_DIR,
                        help='Directory containing class embeddings (.pkl)')
    parser.add_argument('--output_path', type=str, default=config.PREDICT_OUTPUT_PATH,
                        help='Path to output predictions file')
    
    # Model settings
    parser.add_argument('--model_name', type=str, default=config.BERT_MODEL_NAME,
                        help='BERT model name (must match training)')
    parser.add_argument('--max_length', type=int, default=config.MAX_LENGTH,
                        help='Maximum sequence length for tokenization')
    parser.add_argument('--batch_size', type=int, default=config.PREDICT_BATCH_SIZE,
                        help='Batch size for inference')
    
    # Prediction settings
    parser.add_argument('--min_labels', type=int, default=config.PREDICT_MIN_LABELS,
                        help='Minimum number of labels per document')
    parser.add_argument('--max_labels', type=int, default=config.PREDICT_MAX_LABELS,
                        help='Maximum number of labels per document')
    parser.add_argument('--virtual_root_id', type=int, default=config.VIRTUAL_ROOT_ID,
                        help='ID of virtual root node to exclude from predictions')
    parser.add_argument('--output_format', type=str, default=config.PREDICT_OUTPUT_FORMAT, choices=['csv', 'json'],
                        help='Output format (csv or json)')
    
    # Hardware
    parser.add_argument('--gpu', type=int, default=config.PREDICT_GPU,
                        help='GPU device ID')
    
    args = parser.parse_args()
    
    # Device setup
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Load taxonomy
    print("Loading taxonomy...")
    taxonomy = Taxonomy(
        hierarchy_path=args.taxonomy_path,
        classes_path=args.classes_path
    )
    print(f"Loaded taxonomy with {taxonomy.num_classes} classes")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = AmazonDataset(
        corpus_path=args.test_corpus,
        tokenizer=tokenizer,
        max_length=args.max_length,
        truncation=True,
        padding='max_length',
        mode='train',
        labels_path=None,  # No labels for test set
        taxonomy=taxonomy,
        filter_empty_labels=False
    )
    print(f"Loaded {len(test_dataset)} test documents")
    
    # Load class embeddings
    print(f"Loading class embeddings from {args.embeddings_dir}...")
    node_embeddings_path = os.path.join(args.embeddings_dir, 'node_embeddings.pkl')
    
    if not os.path.exists(node_embeddings_path):
        raise FileNotFoundError(f"Class embeddings file not found: {node_embeddings_path}")
    
    with open(node_embeddings_path, 'rb') as f:
        node_embeddings_dict = pickle.load(f)
    
    print(f"Loaded embeddings for {len(node_embeddings_dict)} classes")
    
    # Convert class embeddings to tensor
    all_class_ids = sorted(taxonomy.class_id_to_name.keys())
    class_emb_list = []
    
    for class_id in all_class_ids:
        if class_id in node_embeddings_dict:
            emb = node_embeddings_dict[class_id]
            if isinstance(emb, np.ndarray):
                class_emb_list.append(torch.from_numpy(emb).float())
            else:
                class_emb_list.append(torch.tensor(emb, dtype=torch.float32))
        else:
            # Use zero vector for missing embeddings
            if class_emb_list:
                emb_dim = class_emb_list[0].shape[0]
            else:
                emb_dim = 768
            class_emb_list.append(torch.zeros(emb_dim, dtype=torch.float32))
            if class_id != args.virtual_root_id:
                print(f"Warning: No embedding found for class_id {class_id}, using zero vector")
    
    class_emb = torch.stack(class_emb_list)
    print(f"Class embeddings tensor shape: {class_emb.shape}")
    
    # Load model
    print("Loading model...")
    model = load_model(
        model_path=args.model_path,
        model_name=args.model_name,
        class_embeddings=class_emb,
        device=device
    )
    
    # Generate predictions
    predictions = predict_test_corpus(
        model=model,
        test_dataset=test_dataset,
        taxonomy=taxonomy,
        class_ids=all_class_ids,
        device=device,
        batch_size=args.batch_size,
        min_labels=args.min_labels,
        max_labels=args.max_labels,
        virtual_root_id=args.virtual_root_id
    )
    
    # Print statistics
    num_predictions = len(predictions)
    total_labels = sum(len(labels) for labels in predictions.values())
    avg_labels = total_labels / num_predictions if num_predictions > 0 else 0
    
    print(f"\nPrediction Statistics:")
    print(f"  Total documents: {num_predictions}")
    print(f"  Total labels: {total_labels}")
    print(f"  Average labels per document: {avg_labels:.2f}")
    
    # Count leaf labels
    predictor = HierarchyConstraintPredictor(taxonomy)
    leaf_label_counts = []
    for labels in predictions.values():
        leaf_count = sum(1 for label in labels if predictor.is_leaf_node(label))
        leaf_label_counts.append(leaf_count)
    
    avg_leaf_labels = sum(leaf_label_counts) / len(leaf_label_counts) if leaf_label_counts else 0
    print(f"  Average leaf labels per document: {avg_leaf_labels:.2f}")
    
    # Save predictions
    save_predictions(
        predictions=predictions,
        output_path=args.output_path,
        format=args.output_format
    )
    
    print("\nPrediction complete!")


if __name__ == '__main__':
    main()


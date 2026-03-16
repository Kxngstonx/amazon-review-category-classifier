"""
Data loader module for TaxoClass (Hierarchical Multi-Label Text Classification).
Implements Taxonomy class for hierarchical structure and AmazonDataset for text processing.
"""

import os
import json
import tempfile
from typing import List, Dict, Optional, Tuple, Set
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import networkx as nx
from transformers import PreTrainedTokenizer

from config import config


class Taxonomy:
    """
    Manages the hierarchical taxonomy structure using NetworkX DAG.
    Handles class indexing, graph traversal, and adjacency matrix generation.
    """
    
    def __init__(self, hierarchy_path: str, classes_path: str):
        """
        Initialize Taxonomy by loading hierarchy and class mappings.
        
        Args:
            hierarchy_path: Path to class_hierarchy.txt (parent_id \t child_id format)
            classes_path: Path to classes.txt (class_id \t class_name format)
        """
        self.hierarchy_path = hierarchy_path
        self.classes_path = classes_path
        
        # Load class name to ID mapping from classes.txt
        self.class_name_to_id: Dict[str, int] = {}
        self.class_id_to_name: Dict[int, str] = {}
        self._load_classes()
        
        # Build the DAG
        self.graph = nx.DiGraph()
        self._build_graph()
        
        # Check for multiple roots and add virtual Root if needed
        self._ensure_single_root()
        
        # Update class mappings if Root was added
        self.num_classes = len(self.class_id_to_name)
        
        # Initialize keywords dictionary (will be populated by load_keywords)
        self.keywords: Dict[int, List[str]] = {}
        
    def _load_classes(self) -> None:
        """Load class name to ID mapping from classes.txt."""
        with open(self.classes_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    class_id = int(parts[0])
                    class_name = parts[1]
                    self.class_name_to_id[class_name] = class_id
                    self.class_id_to_name[class_id] = class_name
    
    def _build_graph(self) -> None:
        """Build the DAG from hierarchy file."""
        # First, add all nodes from classes.txt to the graph (even if they have no edges)
        for class_id in self.class_id_to_name.keys():
            self.graph.add_node(class_id)
        
        # Then, add edges from hierarchy file
        with open(self.hierarchy_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    parent_id = int(parts[0])
                    child_id = int(parts[1])
                    # Add edge: parent -> child
                    self.graph.add_edge(parent_id, child_id)
    
    def _ensure_single_root(self) -> None:
        """
        Check if multiple root nodes exist (in-degree = 0).
        If so, create a virtual 'Root' node and connect all orphan nodes to it.
        """
        # Find all nodes with in-degree 0 (no incoming edges)
        root_nodes = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]
        
        if len(root_nodes) > 1:
            # Multiple roots detected - create virtual Root node
            # Use a special ID that doesn't conflict with existing class IDs
            max_id = max(self.class_id_to_name.keys()) if self.class_id_to_name else -1
            root_id = max_id + 1
            
            # Add Root to mappings
            self.class_id_to_name[root_id] = "Root"
            self.class_name_to_id["Root"] = root_id
            
            # Connect all orphan nodes to Root
            for root_node in root_nodes:
                self.graph.add_edge(root_id, root_node)
            
            print(f"Created virtual Root node (ID: {root_id}) connecting {len(root_nodes)} orphan nodes")
        elif len(root_nodes) == 1:
            # Single root exists - check if it's already named "Root"
            root_id = root_nodes[0]
            root_name = self.class_id_to_name.get(root_id, f"class_{root_id}")
            if root_name != "Root":
                # Optionally rename it to Root for consistency
                # Or just keep the existing name
                pass
        else:
            # No root nodes (shouldn't happen in a DAG, but handle it)
            print("Warning: No root nodes found in the graph")
    
    def get_parents(self, class_id: int) -> List[int]:
        """
        Get all parent class IDs for a given class_id.
        
        Args:
            class_id: The class ID to query
            
        Returns:
            List of parent class IDs
        """
        if class_id not in self.graph:
            return []
        return list(self.graph.predecessors(class_id))
    
    def get_children(self, class_id: int) -> List[int]:
        """
        Get all child class IDs for a given class_id.
        
        Args:
            class_id: The class ID to query
            
        Returns:
            List of child class IDs
        """
        if class_id not in self.graph:
            return []
        return list(self.graph.successors(class_id))
    
    def get_siblings(self, class_id: int) -> List[int]:
        """
        Get all sibling class IDs (nodes sharing the same parent, excluding self).
        
        Args:
            class_id: The class ID to query
            
        Returns:
            List of sibling class IDs
        """
        if class_id not in self.graph:
            return []
        
        parents = self.get_parents(class_id)
        siblings = []
        
        for parent_id in parents:
            # Get all children of this parent
            children = self.get_children(parent_id)
            # Add siblings (exclude self)
            siblings.extend([c for c in children if c != class_id])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_siblings = []
        for sib in siblings:
            if sib not in seen:
                seen.add(sib)
                unique_siblings.append(sib)
        
        return unique_siblings
    
    def get_adj_matrix(self) -> torch.Tensor:
        """
        Get the adjacency matrix of the taxonomy graph as a PyTorch tensor.
        Used for GCN encoder.
        
        Returns:
            Adjacency matrix of shape [num_classes, num_classes]
        """
        num_classes = self.num_classes
        
        # Get all class IDs in sorted order for consistent indexing
        all_class_ids = sorted(self.class_id_to_name.keys())
        id_to_index = {class_id: idx for idx, class_id in enumerate(all_class_ids)}
        
        # Initialize adjacency matrix
        adj_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32)
        
        # Fill adjacency matrix
        for edge in self.graph.edges():
            parent_id, child_id = edge
            if parent_id in id_to_index and child_id in id_to_index:
                parent_idx = id_to_index[parent_id]
                child_idx = id_to_index[child_id]
                adj_matrix[parent_idx, child_idx] = 1.0
        
        return adj_matrix
    
    def get_class_id(self, class_name: str) -> Optional[int]:
        """Get class ID from class name."""
        return self.class_name_to_id.get(class_name)
    
    def get_class_name(self, class_id: int) -> Optional[str]:
        """Get class name from class ID."""
        return self.class_id_to_name.get(class_id)
    
    def load_keywords(self, keywords_file_path: str) -> None:
        """
        Load keywords for each class from a keywords file.
        Maps class names to class IDs and stores keywords in self.keywords.
        
        Args:
            keywords_file_path: Path to class_related_keywords.txt 
                                (format: class_name:keyword1,keyword2,...)
        """
        self.keywords = {}
        
        with open(keywords_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Format: class_name:keyword1,keyword2,keyword3,...
                if ':' not in line:
                    continue
                
                parts = line.split(':', 1)
                if len(parts) == 2:
                    class_name = parts[0].strip()
                    keywords_str = parts[1].strip()
                    
                    # Get class_id from class name
                    class_id = self.get_class_id(class_name)
                    
                    if class_id is not None:
                        # Split keywords by comma and strip whitespace
                        keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
                        self.keywords[class_id] = keywords
                    else:
                        # Warn if class name not found in taxonomy
                        print(f"Warning: Class name '{class_name}' not found in taxonomy, skipping keywords")


class AmazonDataset(Dataset):
    """
    PyTorch Dataset for Amazon review text data.
    Handles tokenization and returns tensors for model input.
    """
    
    def __init__(
        self,
        corpus_path: str,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 512,
        truncation: bool = True,
        padding: str = 'max_length',
        mode: str = 'train',
        labels_path: Optional[str] = None,
        taxonomy: Optional[Taxonomy] = None,
        filter_empty_labels: bool = True,
        exclude_doc_ids: Optional[Set[int]] = None
    ):
        """
        Initialize AmazonDataset.
        
        Args:
            corpus_path: Path to corpus file (review_id \t review_text format)
            tokenizer: Pre-trained tokenizer (e.g., AutoTokenizer). Required if mode='train'
            max_length: Maximum sequence length for tokenization
            truncation: Whether to truncate sequences exceeding max_length
            padding: Padding strategy ('max_length' or 'longest')
            mode: 'train' for tokenized output, 'raw' for raw text output
            labels_path: Path to silver labels JSON file (optional)
            taxonomy: Taxonomy object for converting class IDs to indices (required if labels_path provided)
            filter_empty_labels: If True, filter out documents with empty labels (default: True)
            exclude_doc_ids: Set of doc_ids to exclude from the dataset (e.g., validation set doc_ids)
        """
        self.mode = mode
        self.filter_empty_labels = filter_empty_labels
        
        if mode == 'train':
            if tokenizer is None:
                raise ValueError("tokenizer is required when mode='train'")
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.truncation = truncation
            self.padding = padding
        elif mode == 'raw':
            # tokenizer not needed for raw mode
            self.tokenizer = None
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'raw'")
        
        # Load reviews
        self.reviews: List[Tuple[int, str]] = []
        self._load_corpus(corpus_path)
        
        # Exclude specified doc_ids (e.g., validation set)
        if exclude_doc_ids is not None:
            original_count = len(self.reviews)
            self.reviews = [(doc_id, text) for doc_id, text in self.reviews 
                           if doc_id not in exclude_doc_ids]
            excluded_count = original_count - len(self.reviews)
            if excluded_count > 0:
                print(f"Excluded {excluded_count} documents from dataset (validation set)")
                print(f"Remaining documents: {len(self.reviews)}")
        
        # Load labels if provided
        self.labels: Dict[int, List[int]] = {}
        self.label_tensors: Dict[int, torch.Tensor] = {}
        if labels_path is not None:
            if taxonomy is None:
                raise ValueError("taxonomy is required when labels_path is provided")
            self.taxonomy = taxonomy
            self._load_labels(labels_path)
            if filter_empty_labels:
                self._filter_empty_labels()
    
    def _load_corpus(self, corpus_path: str) -> None:
        """Load review corpus from file."""
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split by tab: review_id \t review_text
                parts = line.split('\t', 1)
                if len(parts) >= 2:
                    review_id = int(parts[0])
                    review_text = parts[1]
                    self.reviews.append((review_id, review_text))
                elif len(parts) == 1:
                    # Handle case where there's no tab (just text)
                    review_id = len(self.reviews)
                    review_text = parts[0]
                    self.reviews.append((review_id, review_text))
    
    def _load_labels(self, labels_path: str) -> None:
        """
        Load labels from JSON file.
        
        Args:
            labels_path: Path to JSON file with format {doc_id: [class_id1, class_id2, ...]}
        """
        if not os.path.exists(labels_path):
            print(f"Warning: Labels file not found at {labels_path}. Proceeding without labels.")
            return
        
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels_dict = json.load(f)
        
        # Convert string keys to int and ensure list format
        for doc_id_str, class_ids in labels_dict.items():
            doc_id = int(doc_id_str)
            if isinstance(class_ids, list):
                self.labels[doc_id] = [int(cid) for cid in class_ids]
            else:
                self.labels[doc_id] = []
        
        # Convert class IDs to indices and create label tensors
        all_class_ids = sorted(self.taxonomy.class_id_to_name.keys())
        class_id_to_index = {class_id: idx for idx, class_id in enumerate(all_class_ids)}
        num_classes = len(all_class_ids)
        
        for doc_id, class_ids in self.labels.items():
            # Create multi-hot vector
            label_tensor = torch.zeros(num_classes, dtype=torch.float32)
            for class_id in class_ids:
                if class_id in class_id_to_index:
                    label_tensor[class_id_to_index[class_id]] = 1.0
            self.label_tensors[doc_id] = label_tensor

        # DEBUG: Print label loading statistics
        total_labels = sum(len(class_ids) for class_ids in self.labels.values())
        avg_labels_per_doc = total_labels / len(self.labels) if len(self.labels) > 0 else 0
        print(f"Label loading stats:")
        print(f"  Total class IDs in labels: {total_labels}")
        print(f"  Average class IDs per document: {avg_labels_per_doc:.2f}")
        print(f"  Number of unique class IDs in labels: {len(set(cid for class_ids in self.labels.values() for cid in class_ids))}")
        print(f"  Number of classes in taxonomy: {num_classes}")
        
        num_docs_with_labels = sum(1 for labels in self.labels.values() if labels)
        num_docs_without_labels = len(self.labels) - num_docs_with_labels
        print(f"Loaded labels for {len(self.labels)} documents")
        print(f"  Documents with labels: {num_docs_with_labels}")
        print(f"  Documents without labels: {num_docs_without_labels}")
    
    def _filter_empty_labels(self) -> None:
        """Filter out documents that have empty labels."""
        if not self.labels:
            return
        
        # Create a set of doc_ids with non-empty labels
        valid_doc_ids = {doc_id for doc_id, labels in self.labels.items() if labels}
        
        # Filter reviews to only include those with non-empty labels
        # If labels are loaded, only keep documents that have non-empty labels
        original_count = len(self.reviews)
        self.reviews = [(doc_id, text) for doc_id, text in self.reviews 
                       if doc_id in valid_doc_ids]
        
        # Also filter label dictionaries to match
        self.labels = {doc_id: labels for doc_id, labels in self.labels.items() 
                      if doc_id in valid_doc_ids}
        self.label_tensors = {doc_id: tensor for doc_id, tensor in self.label_tensors.items() 
                             if doc_id in valid_doc_ids}
        
        filtered_count = original_count - len(self.reviews)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} documents with empty labels")
            print(f"Remaining documents: {len(self.reviews)}")
    
    def __len__(self) -> int:
        """Return the number of reviews in the dataset."""
        return len(self.reviews)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single review item.
        
        Args:
            idx: Index of the review
            
        Returns:
            If mode == 'raw':
                Dictionary containing:
                    - text: Raw review text (string)
                    - index: Review index (int)
            If mode == 'train':
                Dictionary containing:
                    - input_ids: Token IDs tensor
                    - attention_mask: Attention mask tensor
                    - index: Review index (tensor)
                    - core_labels: Label tensor (if labels loaded, shape [num_classes])
        """
        review_id, review_text = self.reviews[idx]
        
        result = {}
        
        if self.mode == 'raw':
            # Return raw text and index
            result = {
                'text': review_text,
                'index': review_id
            }
        else:
            # Tokenize the text (mode == 'train')
            encoded = self.tokenizer.encode_plus(
                review_text,
                max_length=self.max_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
            )
            
            # Extract tensors and squeeze batch dimension
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
            
            result = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'index': torch.tensor(review_id, dtype=torch.long)
            }
        
        # Add labels if available
        if self.label_tensors and review_id in self.label_tensors:
            result['core_labels'] = self.label_tensors[review_id]
        elif self.label_tensors:
            # If labels are loaded but this doc_id doesn't have labels, create zero tensor
            # Works for both 'train' and 'raw' mode
            num_classes = len(next(iter(self.label_tensors.values())))
            result['core_labels'] = torch.zeros(num_classes, dtype=torch.float32)
        
        return result


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching AmazonDataset items.
    Handles padding to the longest sequence in the batch.
    
    Args:
        batch: List of dictionaries from __getitem__
        
    Returns:
        Batched dictionary with stacked tensors
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    indices = torch.stack([item['index'] for item in batch])
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'index': indices
    }
    
    # Add labels if present in batch
    if 'core_labels' in batch[0]:
        core_labels = torch.stack([item['core_labels'] for item in batch])
        result['core_labels'] = core_labels
    
    return result


def load_raw_reviews(file_path: str) -> List[str]:
    """
    Load all review strings from a corpus file.
    Useful for training TF-IDF vectorizers externally.
    
    Args:
        file_path: Path to corpus file (review_id \t review_text format)
        
    Returns:
        List of review text strings (in order)
    """
    reviews = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by tab: review_id \t review_text
            parts = line.split('\t', 1)
            if len(parts) >= 2:
                review_text = parts[1]
                reviews.append(review_text)
            elif len(parts) == 1:
                # Handle case where there's no tab (just text)
                reviews.append(parts[0])
    
    return reviews


def load_validation_dataset_from_results(
    validation_results_path: str,
    corpus_path: str,
    taxonomy: Taxonomy,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_length: int = 512,
    truncation: bool = True,
    padding: str = 'max_length',
    mode: str = 'raw'
) -> Tuple[AmazonDataset, Set[int]]:
    """
    Load validation dataset from label_validation_results.json.
    Uses LLM-selected labels (llm_selected) as ground truth labels.
    
    Args:
        validation_results_path: Path to label_validation_results.json
        corpus_path: Path to corpus file (review_id \t review_text format)
        taxonomy: Taxonomy object for converting class IDs to indices
        tokenizer: Pre-trained tokenizer (required if mode='train')
        max_length: Maximum sequence length for tokenization
        truncation: Whether to truncate sequences exceeding max_length
        padding: Padding strategy ('max_length' or 'longest')
        mode: 'train' for tokenized output, 'raw' for raw text output
        
    Returns:
        Tuple of (validation_dataset, validation_doc_ids_set)
        - validation_dataset: AmazonDataset with LLM-selected labels
        - validation_doc_ids_set: Set of doc_ids in validation set (for excluding from train)
    """
    if not os.path.exists(validation_results_path):
        raise FileNotFoundError(f"Validation results file not found: {validation_results_path}")
    
    # Load validation results
    with open(validation_results_path, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)
    
    # Extract validation results
    results = validation_data.get('results', [])
    if not results:
        raise ValueError(f"No validation results found in {validation_results_path}")
    
    # Extract doc_ids and LLM-selected labels
    validation_doc_ids = set()
    llm_labels_dict = {}
    
    for result in results:
        doc_id = result.get('doc_id')
        if doc_id is None:
            continue
        
        doc_id = int(doc_id)
        validation_doc_ids.add(doc_id)
        
        # Use LLM-selected labels as ground truth
        llm_selected = result.get('llm_selected', [])
        if isinstance(llm_selected, list):
            # Filter out virtual root if present
            llm_selected = [int(cid) for cid in llm_selected if int(cid) != config.VIRTUAL_ROOT_ID]
            llm_labels_dict[doc_id] = llm_selected
        else:
            llm_labels_dict[doc_id] = []
    
    print(f"Loaded {len(validation_doc_ids)} validation samples from {validation_results_path}")
    print(f"  Samples with LLM labels: {len([v for v in llm_labels_dict.values() if v])}")
    print(f"  Samples without LLM labels: {len([v for v in llm_labels_dict.values() if not v])}")
    
    # Save LLM labels to a temporary JSON file (same format as bottom_up_labels.json)
    temp_labels_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump({str(doc_id): labels for doc_id, labels in llm_labels_dict.items()}, 
              temp_labels_file, indent=2)
    temp_labels_file.close()
    temp_labels_path = temp_labels_file.name
    
    try:
        # Create validation dataset
        val_dataset = AmazonDataset(
            corpus_path=corpus_path,
            tokenizer=tokenizer,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            mode=mode,
            labels_path=temp_labels_path,
            taxonomy=taxonomy,
            filter_empty_labels=False  # Keep all validation samples even if LLM didn't select labels
        )
        
        # Filter to only include validation doc_ids (in case corpus has more)
        original_count = len(val_dataset.reviews)
        val_dataset.reviews = [(doc_id, text) for doc_id, text in val_dataset.reviews 
                              if doc_id in validation_doc_ids]
        val_dataset.labels = {doc_id: labels for doc_id, labels in val_dataset.labels.items() 
                             if doc_id in validation_doc_ids}
        
        # Recreate label_tensors for filtered doc_ids to ensure consistency
        # Only create tensors for doc_ids that are actually in reviews
        all_class_ids = sorted(taxonomy.class_id_to_name.keys())
        class_id_to_index = {class_id: idx for idx, class_id in enumerate(all_class_ids)}
        num_classes = len(all_class_ids)
        
        val_dataset.label_tensors = {}
        review_doc_ids = {doc_id for doc_id, _ in val_dataset.reviews}
        
        for doc_id in review_doc_ids:
            if doc_id in val_dataset.labels:
                # Create multi-hot vector
                label_tensor = torch.zeros(num_classes, dtype=torch.float32)
                for class_id in val_dataset.labels[doc_id]:
                    if class_id in class_id_to_index:
                        label_tensor[class_id_to_index[class_id]] = 1.0
                val_dataset.label_tensors[doc_id] = label_tensor
            else:
                # If doc_id doesn't have labels, create zero tensor
                val_dataset.label_tensors[doc_id] = torch.zeros(num_classes, dtype=torch.float32)
        
        filtered_count = original_count - len(val_dataset.reviews)
        if filtered_count > 0:
            print(f"Filtered validation dataset: {filtered_count} documents removed (not in validation set)")
        
        # Verify that we have matching reviews and labels
        label_doc_ids = set(val_dataset.labels.keys())
        missing_labels = review_doc_ids - label_doc_ids
        missing_reviews = validation_doc_ids - review_doc_ids
        if missing_labels:
            print(f"Warning: {len(missing_labels)} validation samples have no labels (will use zero tensors)")
        if missing_reviews:
            print(f"Warning: {len(missing_reviews)} validation doc_ids not found in corpus (skipped)")
        
        print(f"Final validation dataset size: {len(val_dataset)}")
        print(f"  Reviews: {len(val_dataset.reviews)}")
        print(f"  Labels: {len(val_dataset.labels)}")
        print(f"  Label tensors: {len(val_dataset.label_tensors)}")
        
        # Check if dataset is empty
        if len(val_dataset.reviews) == 0:
            raise ValueError("Validation dataset is empty after filtering! Check if validation doc_ids exist in corpus.")
        
        return val_dataset, validation_doc_ids
    finally:
        # Clean up temporary file
        if os.path.exists(temp_labels_path):
            os.unlink(temp_labels_path)


def get_weighted_sampler(dataset_labels: List[Dict[str, List[str]]]) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler using Leaf-Aware Smoothed Sampling strategy.
    
    This sampler handles extreme class imbalance in Multi-Label Text Classification
    by using square root smoothing on class frequencies and leaf-aware document weighting.
    
    Algorithm:
    1. Count global frequency (N_c) for every class c in the training set
    2. Calculate class weights using square root smoothing: W_c = 1 / sqrt(N_c)
    3. Calculate document weights (leaf-aware): W_doc = max(W_c for all c in doc.labels)
    4. Create WeightedRandomSampler with document weights
    
    Square Root Smoothing Logic:
    - Instead of inverse frequency (1/N) which can cause overfitting by replicating
      noisy samples too aggressively (e.g., 800x for rare classes), we use 1/sqrt(N).
    - This provides a gentler rebalancing: 
      * If count = 1, weight = 1.0
      * If count = 100, weight = 0.1 (instead of 0.01 with inverse frequency)
      * If count = 10000, weight = 0.01 (instead of 0.0001 with inverse frequency)
    - The square root function reduces the extreme amplification of rare classes
      while still giving them higher sampling probability than frequent classes.
    
    Leaf-Aware Logic:
    - In multi-label tasks, a document can have multiple labels (e.g., Root, Parent, Leaf).
    - To be "leaf-aware", we determine the document's sampling weight by its rarest label
      (the label with the highest weight, since W_c = 1/sqrt(N_c) is higher for rare classes).
    - This ensures documents with rare/leaf labels are sampled more frequently,
      which is crucial for learning rare class patterns.
    
    Args:
        dataset_labels: List of samples, where each sample is a dictionary containing
                        a list of label strings. Format: [{'labels': ['baby', 'baby food']}, ...]
    
    Returns:
        WeightedRandomSampler object ready to be passed to a DataLoader.
        The sampler uses replacement=True to ensure it produces an iterator with the
        same length as the original dataset.
    
    Example:
        >>> samples = [
        ...     {'labels': ['baby', 'baby food']},
        ...     {'labels': ['electronics', 'phones']},
        ...     {'labels': ['baby']}
        ... ]
        >>> sampler = get_weighted_sampler(samples)
        >>> train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    """
    if not dataset_labels:
        raise ValueError("dataset_labels cannot be empty")
    
    # Step 1: Count global frequency (N_c) for every class c
    class_counts: Dict[str, int] = {}
    for sample in dataset_labels:
        if 'labels' not in sample:
            raise ValueError("Each sample must have a 'labels' key")
        labels = sample['labels']
        if not isinstance(labels, list):
            raise ValueError("Labels must be a list of strings")
        
        # Count each unique label in the document
        for label in labels:
            if not isinstance(label, str):
                raise ValueError("All labels must be strings")
            class_counts[label] = class_counts.get(label, 0) + 1
    
    # Step 2: Calculate class weights using square root smoothing
    # Formula: W_c = 1 / sqrt(N_c)
    # This provides gentler rebalancing compared to inverse frequency (1/N)
    class_weights: Dict[str, float] = {}
    for class_name, count in class_counts.items():
        if count <= 0:
            raise ValueError(f"Class '{class_name}' has non-positive count: {count}")
        # Square root smoothing: W_c = 1 / sqrt(N_c)
        class_weights[class_name] = 1.0 / (count ** 0.5)
    
    # Step 3: Calculate document weights (leaf-aware)
    # For each document, find the rarest label (highest weight)
    # W_doc = max(W_c for all c in doc.labels)
    document_weights: List[float] = []
    for sample in dataset_labels:
        labels = sample['labels']
        
        if not labels:
            # If document has no labels, assign minimum weight
            # Use the minimum class weight or a small default value
            min_weight = min(class_weights.values()) if class_weights else 1.0
            document_weights.append(min_weight)
        else:
            # Leaf-aware: take the maximum weight (rarest label)
            # This ensures documents with rare/leaf labels are sampled more frequently
            # Filter to only include labels that exist in class_weights
            valid_weights = [class_weights[label] for label in labels if label in class_weights]
            if valid_weights:
                max_weight = max(valid_weights)
            else:
                # Fallback: if no valid labels found, use minimum weight
                # This shouldn't happen in normal operation, but handle gracefully
                min_weight = min(class_weights.values()) if class_weights else 1.0
                max_weight = min_weight
            document_weights.append(max_weight)
    
    # Step 4: Create WeightedRandomSampler
    # Convert weights to tensor
    weights_tensor = torch.tensor(document_weights, dtype=torch.float32)
    
    # Create sampler with replacement=True to ensure same length as dataset
    num_samples = len(dataset_labels)
    sampler = WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=num_samples,
        replacement=True
    )
    
    return sampler


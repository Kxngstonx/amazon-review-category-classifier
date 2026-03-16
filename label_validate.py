"""
Label Validation using LLM (Llama-3.0-70b) for Hierarchical Multi-Label Text Classification.

This script performs stratified sampling and validates silver labels using LLM.
"""

import os
import json
import random
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
import requests

from data_loader import Taxonomy, AmazonDataset
from config import config


class LabelValidator:
    """
    Label Validator using LLM to verify silver labels.
    
    Process:
    1. Stratified Sampling: Select 100-300 samples across different difficulty levels
    2. LLM Validation: Use Llama-3.0-70b to verify candidate classes
    3. Save validation results
    """
    
    def __init__(
        self,
        taxonomy: Taxonomy,
        dataset: AmazonDataset,
        labels_path: str,
        candidates_path: Optional[str] = None,
        class_counts_path: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        model_name: str = "meta-llama/llama-3.1-70b-instruct"
    ):
        """
        Initialize Label Validator.
        
        Args:
            taxonomy: Taxonomy object with hierarchy
            dataset: AmazonDataset object with review texts
            labels_path: Path to labels.json (bottom_up_labels.json or top_down_labels.json)
            candidates_path: Optional path to top2_candidates.json (for bottom-up only)
            class_counts_path: Optional path to class_document_counts_full.json
            api_key: OpenRouter API key (if None, will look for OPENROUTER_API_KEY environment variable)
            api_base_url: Base URL for OpenRouter API (if None, will use default)
            model_name: Model name for OpenRouter API
        """
        self.taxonomy = taxonomy
        self.dataset = dataset
        self.model_name = model_name
        
        # Load data
        print("Loading labels...")
        self.labels = self._load_json(labels_path)
        
        # Load candidates data if available (optional for top-down)
        if candidates_path and os.path.exists(candidates_path):
            print("Loading candidates data...")
            self.candidates_data = self._load_json(candidates_path)
        else:
            print("No candidates data provided - using simplified sampling")
            self.candidates_data = None
        
        # Load class counts if available (optional)
        if class_counts_path and os.path.exists(class_counts_path):
            print("Loading class counts...")
            self.class_counts = self._load_json(class_counts_path)
        else:
            print("No class counts provided - computing from labels")
            self.class_counts = self._compute_class_counts_from_labels()
        
        # Convert string keys to int for labels
        self.labels = {int(k): v for k, v in self.labels.items()}
        
        # Load review texts into memory
        print("Loading review texts...")
        self.review_texts = {}
        for item in tqdm(self.dataset, desc="Loading reviews"):
            doc_id = item['index']
            self.review_texts[doc_id] = item['text']
        
        # OpenRouter API configuration
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.api_base_url = api_base_url or "https://openrouter.ai/api/v1/chat/completions"
        
        if not self.api_key:
            print("Warning: No API key provided. Please set OPENROUTER_API_KEY environment variable or pass api_key parameter.")
            print(f"API base URL: {self.api_base_url}")
        
        # Statistics
        self.validation_results = []
        
    def _load_json(self, path: str) -> Dict:
        """Load JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _compute_class_counts_from_labels(self) -> Dict[int, int]:
        """
        Compute class document counts from labels.
        Used when class_counts file is not available.
        
        Returns:
            Dictionary mapping class_id to document count
        """
        class_counts = defaultdict(int)
        for doc_id, class_ids in self.labels.items():
            for class_id in class_ids:
                class_counts[class_id] += 1
        return dict(class_counts)
    
    def _get_all_paths_to_root(self, class_id: int, current_path: List[int] = None) -> List[List[int]]:
        """
        Get all paths from a class to root using DFS.
        Handles DAG structure where a node can have multiple parents.
        
        Args:
            class_id: Target class ID
            current_path: Current path being explored (for recursion)
            
        Returns:
            List of paths, each path is a list of class IDs from root to class_id
        """
        if current_path is None:
            current_path = []
        
        # Prevent cycles
        if class_id in current_path:
            return []
        
        current_path = current_path + [class_id]
        
        # Get parents
        parents = self.taxonomy.get_parents(class_id)
        
        # If no parents, this is a root node - return the path
        if not parents:
            return [current_path]
        
        # Collect all paths from all parents
        all_paths = []
        for parent_id in parents:
            parent_paths = self._get_all_paths_to_root(parent_id, current_path.copy())
            all_paths.extend(parent_paths)
        
        return all_paths
    
    def _get_all_ancestors(self, class_id: int) -> Set[int]:
        """
        Get all ancestor nodes (parents recursively) for a given class using BFS.
        Handles DAG structure where a node can have multiple parents.
        
        Args:
            class_id: Target class ID
            
        Returns:
            Set of all ancestor class IDs (including the class itself)
        """
        ancestors = set([class_id])  # Include self
        queue = [class_id]  # BFS queue
        visited = set()  # Track visited nodes to prevent cycles
        
        # BFS traversal: explore all parent paths
        while queue:
            current_id = queue.pop(0)  # Dequeue
            if current_id in visited:
                continue
            visited.add(current_id)
            
            # Get all parents (DAG can have multiple parents)
            parents = self.taxonomy.get_parents(current_id)
            for parent_id in parents:
                if parent_id not in ancestors:
                    ancestors.add(parent_id)  # Add to result set (Union)
                    queue.append(parent_id)  # Enqueue for further exploration
        
        return ancestors
    
    def _is_leaf_node(self, class_id: int) -> bool:
        """
        Check if a class is a leaf node (has no children).
        
        Args:
            class_id: Target class ID
            
        Returns:
            True if the class is a leaf node, False otherwise
        """
        children = self.taxonomy.get_children(class_id)
        return len(children) == 0
    
    def _has_multiple_paths(self, class_id: int) -> bool:
        """
        Check if a class has multiple paths to root (multiple parent paths).
        
        Args:
            class_id: Target class ID
            
        Returns:
            True if the class has multiple paths to root, False otherwise
        """
        paths = self._get_all_paths_to_root(class_id)
        return len(paths) > 1
    
    def _get_single_path_ancestors(self, class_id: int) -> Set[int]:
        """
        Get all ancestors for a class that has a single path to root.
        This automatically includes all parent nodes in the single path.
        
        Args:
            class_id: Target class ID (must have single path to root)
            
        Returns:
            Set of all class IDs in the path (including the class itself)
        """
        paths = self._get_all_paths_to_root(class_id)
        if len(paths) == 0:
            return set([class_id])
        elif len(paths) == 1:
            # Single path: return all nodes in the path
            return set(paths[0])
        else:
            # Multiple paths: should not call this method
            raise ValueError(f"Class {class_id} has multiple paths. Use LLM to select a specific path.")
    
    def _expand_leaf_to_hierarchy(
        self, 
        selected_class_ids: List[int],
        selected_path_ids: Optional[List[int]] = None
    ) -> Set[int]:
        """
        Expand selected class IDs to include ancestors based on path selection.
        
        - If a class has a single path to root: automatically include all ancestors
        - If a class has multiple paths: use selected_path_ids (from LLM) to determine which path
        
        Args:
            selected_class_ids: List of selected leaf node IDs from LLM
            selected_path_ids: Optional list of class IDs representing the selected path for multi-path nodes
            
        Returns:
            Set of all class IDs including selected nodes and their ancestors
        """
        expanded = set()
        
        for class_id in selected_class_ids:
            if self._has_multiple_paths(class_id):
                # Multiple paths: use LLM-selected path
                if selected_path_ids:
                    # Verify that selected_path_ids contains class_id and forms a valid path
                    if class_id in selected_path_ids:
                        # Use the selected path
                        expanded.update(selected_path_ids)
                    else:
                        # Fallback: just include the class itself
                        expanded.add(class_id)
                else:
                    # No path selected: just include the class itself
                    expanded.add(class_id)
            else:
                # Single path: automatically include all ancestors
                ancestors = self._get_single_path_ancestors(class_id)
                expanded.update(ancestors)
        
        return expanded
    
    def _get_hierarchy_path(self, class_id: int) -> str:
        """
        Get full hierarchy path(s) from Root to the given class.
        For DAG structures, returns all paths joined with " AND ".
        
        Format: "Root > Parent1 > Leaf AND Root > Parent2 > Leaf"
        
        Args:
            class_id: Target class ID
            
        Returns:
            Hierarchy path string with all paths joined by " AND "
        """
        # Get all paths from root to this class
        all_paths = self._get_all_paths_to_root(class_id)
        
        if not all_paths:
            # Fallback: just return the class name
            class_name = self.taxonomy.class_id_to_name.get(class_id, f"class_{class_id}")
            return class_name
        
        # Convert each path to string format
        path_strings = []
        for path in all_paths:
            # Reverse to get Root -> Leaf order
            path_reversed = path[::-1]
            # Convert IDs to names
            path_names = [
                self.taxonomy.class_id_to_name.get(pid, f"class_{pid}")
                for pid in path_reversed
            ]
            path_str = " > ".join(path_names)
            path_strings.append(path_str)
        
        # Join all paths with " AND "
        if len(path_strings) == 1:
            return path_strings[0]
        else:
            return " AND ".join(path_strings)
    
    def stratified_sampling(
        self,
        num_samples: int = 200,
        confident_ratio: float = 0.3,
        ambiguous_ratio: float = 0.4,
        rare_ratio: float = 0.3
    ) -> List[Dict]:
        """
        Perform stratified sampling across different difficulty levels.
        
        Args:
            num_samples: Total number of samples to select (100-300)
            confident_ratio: Ratio of confident samples (default: 0.3)
            ambiguous_ratio: Ratio of ambiguous samples (default: 0.4)
            rare_ratio: Ratio of rare class samples (default: 0.3)
            
        Returns:
            List of sample dictionaries with doc_id and metadata
        """
        print("\n" + "=" * 70)
        print("Stratified Sampling")
        print("=" * 70)
        
        # Verify ratios sum to 1.0
        total_ratio = confident_ratio + ambiguous_ratio + rare_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        num_confident = int(num_samples * confident_ratio)
        num_ambiguous = int(num_samples * ambiguous_ratio)
        num_rare = int(num_samples * rare_ratio)
        
        # Adjust for rounding
        total_allocated = num_confident + num_ambiguous + num_rare
        if total_allocated < num_samples:
            num_ambiguous += (num_samples - total_allocated)
        
        print(f"Sampling strategy:")
        print(f"  Confident samples: {num_confident} ({confident_ratio*100:.0f}%)")
        print(f"  Ambiguous samples: {num_ambiguous} ({ambiguous_ratio*100:.0f}%)")
        print(f"  Rare class samples: {num_rare} ({rare_ratio*100:.0f}%)")
        print(f"  Total: {num_samples}")
        
        # 1. Confident Samples: Silver Label 1~2개, Score 높은 문서
        print("\n[1/3] Selecting Confident Samples...")
        confident_samples = self._select_confident_samples(num_confident)
        print(f"  Selected {len(confident_samples)} confident samples")
        
        # 2. Ambiguous Samples: 후보 많았거나, Score가 Threshold 경계선
        print("\n[2/3] Selecting Ambiguous Samples...")
        ambiguous_samples = self._select_ambiguous_samples(num_ambiguous)
        print(f"  Selected {len(ambiguous_samples)} ambiguous samples")
        
        # 3. Rare Class Samples: 빈도수 적은 클래스로 예측된 문서
        print("\n[3/3] Selecting Rare Class Samples...")
        rare_samples = self._select_rare_class_samples(num_rare)
        print(f"  Selected {len(rare_samples)} rare class samples")
        
        # Combine all samples
        all_samples = confident_samples + ambiguous_samples + rare_samples
        
        # Shuffle to avoid ordering bias
        random.shuffle(all_samples)
        
        print(f"\nTotal samples selected: {len(all_samples)}")
        
        return all_samples
    
    def _select_confident_samples(self, num_samples: int) -> List[Dict]:
        """
        Select confident samples: Silver Label 1~2개, Score 높은 문서.
        
        Returns:
            List of sample dictionaries
        """
        candidates = []
        
        if self.candidates_data:
            # Use candidates data if available
            for doc_id_str, candidate_info in self.candidates_data.items():
                doc_id = int(doc_id_str)
                
                # Get silver labels (final predicted labels)
                silver_labels = self.labels.get(doc_id, [])
                
                # Filter: Skip empty labels
                if not silver_labels:
                    continue
                
                # Filter: 1~2 labels only
                if not (2 <= len(silver_labels) <= 3):
                    continue
                
                # Get top1_score (highest similarity score) - support both old (s_max) and new (top1_score) formats
                s_max = candidate_info.get('top1_score') or candidate_info.get('s_max', 0.0)
                
                # Get number of candidates (from candidates list length)
                candidates_list = candidate_info.get('candidates', [])
                num_candidates = len(candidates_list) if candidates_list else candidate_info.get('num_candidates', 0)
                
                candidates.append({
                    'doc_id': doc_id,
                    's_max': s_max,
                    'num_silver_labels': len(silver_labels),
                    'num_candidates': num_candidates,
                    'type': 'confident'
                })
            
            # Sort by s_max (descending) and select top-N
            candidates.sort(key=lambda x: x['s_max'], reverse=True)
        else:
            # Simplified version without candidates data
            for doc_id, silver_labels in self.labels.items():
                # Filter: Skip empty labels
                if not silver_labels:
                    continue
                
                # Filter: 1~2 labels only
                if not (2 <= len(silver_labels) <= 3):
                    continue
                
                candidates.append({
                    'doc_id': doc_id,
                    's_max': 0.0,  # Not available
                    'num_silver_labels': len(silver_labels),
                    'num_candidates': 0,  # Not available
                    'type': 'confident'
                })
            
            # Random selection since we don't have scores
            random.shuffle(candidates)
        
        return candidates[:num_samples]
    
    def _select_ambiguous_samples(self, num_samples: int) -> List[Dict]:
        """
        Select ambiguous samples: 후보 많았거나, Score가 Threshold 경계선.
        
        Returns:
            List of sample dictionaries
        """
        candidates = []
        
        if self.candidates_data:
            # Use candidates data if available
            for doc_id_str, candidate_info in self.candidates_data.items():
                doc_id = int(doc_id_str)
                
                # Get silver labels (final predicted labels)
                silver_labels = self.labels.get(doc_id, [])
                
                # Filter: Skip empty labels
                if not silver_labels:
                    continue
                
                # Support both old format (s_max, threshold) and new format (top1_score, top2_score, score_diff, conflict_threshold)
                s_max = candidate_info.get('top1_score') or candidate_info.get('s_max', 0.0)
                top2_score = candidate_info.get('top2_score', 0.0)
                score_diff = candidate_info.get('score_diff', 0.0)
                conflict_threshold = candidate_info.get('conflict_threshold', 0.05)
                
                # For old format compatibility
                threshold = candidate_info.get('threshold', 0.0)
                
                # Get number of candidates (from candidates list length)
                candidates_list = candidate_info.get('candidates', [])
                num_candidates = len(candidates_list) if candidates_list else candidate_info.get('num_candidates', 0)
                
                # Criteria 1: Many candidates (before keyword verification)
                # Criteria 2: Score near threshold (small score_diff indicates ambiguity)
                # For new format: score_diff < conflict_threshold indicates ambiguity
                # For old format: score near threshold (within 5% of threshold)
                if conflict_threshold > 0:
                    score_near_threshold = score_diff < conflict_threshold
                else:
                    score_near_threshold = abs(s_max - threshold) / (threshold + 1e-8) < 0.05 if threshold > 0 else False
                
                # Prefer samples with many candidates OR near threshold (ambiguous)
                if num_candidates >= 5 or score_near_threshold:
                    candidates.append({
                        'doc_id': doc_id,
                        's_max': s_max,
                        'threshold': threshold if threshold > 0 else conflict_threshold,
                        'num_candidates': num_candidates,
                        'score_near_threshold': score_near_threshold,
                        'score_diff': score_diff,
                        'type': 'ambiguous'
                    })
            
            # Sort by ambiguity (more candidates first, then by score_diff - smaller is more ambiguous)
            candidates.sort(
                key=lambda x: (x['num_candidates'], x.get('score_diff', float('inf'))),
                reverse=True
            )
        else:
            # Simplified version: select documents with multiple labels (3+)
            for doc_id, silver_labels in self.labels.items():
                # Filter: Skip empty labels
                if not silver_labels:
                    continue
                
                # Filter: 3+ labels (potentially ambiguous)
                if len(silver_labels) >= 3:
                    candidates.append({
                        'doc_id': doc_id,
                        's_max': 0.0,
                        'threshold': 0.0,
                        'num_candidates': 0,
                        'score_near_threshold': False,
                        'score_diff': 0.0,
                        'type': 'ambiguous'
                    })
            
            # Random selection
            random.shuffle(candidates)
        
        return candidates[:num_samples]
    
    def _select_rare_class_samples(self, num_samples: int) -> List[Dict]:
        """
        Select rare class samples: 빈도수 적은 클래스로 예측된 문서.
        
        Returns:
            List of sample dictionaries
        """
        # Convert class_counts to int keys
        class_counts_int = {int(k): v for k, v in self.class_counts.items()}
        
        # Find rare classes (bottom 30% by frequency)
        count_values = list(class_counts_int.values())
        if not count_values:
            return []
        
        rare_threshold = np.percentile(count_values, 30)
        
        # Build mapping: doc_id -> min_class_frequency (lowest frequency class in its labels)
        doc_to_min_freq = {}
        
        for doc_id, class_ids in self.labels.items():
            if not class_ids:
                continue
            
            # Find minimum frequency among predicted classes
            min_freq = min(
                class_counts_int.get(cid, float('inf'))
                for cid in class_ids
            )
            
            if min_freq <= rare_threshold:
                doc_to_min_freq[doc_id] = min_freq
        
        # Select samples with rare classes
        candidates = []
        for doc_id, min_freq in doc_to_min_freq.items():
            if self.candidates_data:
                candidate_info = self.candidates_data.get(str(doc_id), {})
                # Support both old format (s_max) and new format (top1_score)
                s_max = candidate_info.get('top1_score') or candidate_info.get('s_max', 0.0)
            else:
                s_max = 0.0
            
            candidates.append({
                'doc_id': doc_id,
                'min_class_frequency': min_freq,
                's_max': s_max,
                'num_silver_labels': len(self.labels.get(doc_id, [])),
                'type': 'rare'
            })
        
        # Sort by frequency (ascending - rarest first)
        candidates.sort(key=lambda x: x['min_class_frequency'])
        
        return candidates[:num_samples]
    
    def _format_candidate_paths(
        self,
        candidates: List[Dict],
        top_k_classes: int = 15
    ) -> Tuple[str, Dict[int, List[int]]]:
        """
        Format candidate paths for LLM prompt.
        Each path (root -> leaf) is presented as a separate candidate.
        
        Args:
            candidates: List of candidate dicts (from top2_candidates.json structure)
                      Expected fields: leaf_class_id, leaf_class_name, similarity, path_string (optional)
            top_k_classes: Number of top candidate classes to consider
            
        Returns:
            Tuple of (formatted string for prompt, dict mapping path_index to list of class IDs in path)
        """
        # Sort by similarity (descending) and take top-K classes
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get('similarity', 0.0),
            reverse=True
        )[:top_k_classes]
        
        lines = []
        path_index_to_class_ids = {}  # Map path_index to list of class IDs in the path
        path_index = 1
        
        for cand in sorted_candidates:
            # Support both old format (class_id, class_name) and new format (leaf_class_id, leaf_class_name, path_string)
            class_id = cand.get('leaf_class_id') or cand.get('class_id')
            class_name = cand.get('leaf_class_name') or cand.get('class_name', f'class_{class_id}')
            similarity = cand.get('similarity', 0.0)
            
            # Get all paths for this class
            all_paths = self._get_all_paths_to_root(class_id)
            
            # Each path is a separate candidate
            for path in all_paths:
                # Reverse to get Root -> Leaf order
                path_reversed = path[::-1]
                path_names = [self.taxonomy.class_id_to_name.get(pid, f"class_{pid}") for pid in path_reversed]
                hierarchy_path = " > ".join(path_names)
                
                # Store path mapping
                path_index_to_class_ids[path_index] = path_reversed
                
                lines.append(
                    f"{path_index}. Path: {hierarchy_path} | Leaf: {class_name} (ID: {class_id}) | Similarity: {similarity:.4f}"
                )
                path_index += 1
        
        return "\n".join(lines), path_index_to_class_ids
    
    def _create_prompt(
        self,
        review_text: str,
        candidate_list_str: str
    ) -> str:
        """
        Create LLM prompt for label validation.
        
        Args:
            review_text: Product review text
            candidate_list_str: Formatted candidate classes string
            
        Returns:
            Complete prompt string
        """
        prompt = f"""Role: You are an expert annotator for Amazon product reviews.

Task: Perform Hierarchical Multi-Label Text Classification.

Input Data:

1. Review Text: A product review written by a user.

2. Candidate Paths: A list of potential category paths (from root to leaf) derived from a similarity model.
   Each path represents a complete hierarchy from the root category to a specific leaf category.

Instructions:

1. Analyze the Review Text carefully.

2. Selection Strategy:
   - If the review is NOT relevant to any candidate path, output an empty list (selected_path_index: null).
   - If the review IS relevant, select EXACTLY ONE path (by its index number) that best matches the product being discussed.
   - The selected path should be the most specific and accurate category hierarchy for the product.

3. Important Notes:
   - Select only ONE path index (the most relevant path).
   - Each path already includes all nodes from root to leaf - you don't need to add parent nodes.
   - If multiple paths seem relevant, choose the ONE that is most specific and accurate.
   - If none of the candidate paths are correct, output null for selected_path_index.

4. Output format must be strictly JSON.

---

[Review Text]

{review_text}

[Candidate Paths]

{candidate_list_str}

---

Output Format (JSON):

{{
  "reasoning": "Brief explanation of why this path was chosen (or why none if null).",
  "selected_path_index": 5  // Index number of the selected path, or null if not relevant
}}"""
        
        return prompt
    
    def _call_llm_api(self, prompt: str, max_retries: int = 5) -> Optional[Dict]:
        """
        Call OpenRouter LLM API for label validation.
        
        Args:
            prompt: Input prompt
            max_retries: Maximum number of retry attempts
            
        Returns:
            Parsed JSON response or None if failed
        """
        if not self.api_key:
            raise ValueError(
                "API key not set. Please provide api_key or set OPENROUTER_API_KEY environment variable.\n"
                "Example: export OPENROUTER_API_KEY='your-api-key-here'\n"
                "Get your API key from: https://openrouter.ai/keys"
            )
        
        # Prepare headers for OpenRouter
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",  # Optional: your app URL
            "X-Title": "Label Validator"  # Optional: your app name
        }
        
        # OpenRouter API format (OpenAI-compatible)
        api_url = self.api_base_url
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2048
        }
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=120  # Increased timeout for LLM inference
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Extract generated text from OpenRouter response (OpenAI-compatible format)
                generated_text = ""
                if "choices" in result and len(result["choices"]) > 0:
                    generated_text = result["choices"][0].get("message", {}).get("content", "")
                
                if not generated_text:
                    print(f"Warning: Empty response from API. Full response: {result}")
                    return None
                
                # Try to extract JSON from response
                import re
                
                # Method 1: Look for complete JSON block with selected_path_index
                # Improved regex to handle nested structures
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"selected_path_index"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                json_match = re.search(json_pattern, generated_text, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        parsed = json.loads(json_str)
                        if "selected_path_index" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        # Try to fix common JSON issues
                        # Remove trailing commas
                        json_str = re.sub(r',\s*}', '}', json_str)
                        json_str = re.sub(r',\s*]', ']', json_str)
                        # Remove newlines and tabs
                        json_str = json_str.replace('\n', ' ').replace('\t', ' ')
                        try:
                            parsed = json.loads(json_str)
                            if "selected_path_index" in parsed:
                                return parsed
                        except:
                            pass
                
                # Method 2: Try parsing the whole text as JSON
                try:
                    parsed = json.loads(generated_text.strip())
                    if "selected_path_index" in parsed or parsed.get("selected_path_index") is None:
                        return parsed
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown code blocks
                    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', generated_text, re.DOTALL)
                    if code_block_match:
                        try:
                            json_str = code_block_match.group(1)
                            # Fix common issues
                            json_str = re.sub(r',\s*}', '}', json_str)
                            json_str = re.sub(r',\s*]', ']', json_str)
                            parsed = json.loads(json_str)
                            if "selected_path_index" in parsed or parsed.get("selected_path_index") is None:
                                return parsed
                        except:
                            pass
                
                # Method 3: Try to find and extract just the selected_path_index value
                # Pattern for path index extraction (handles null, integer, or number)
                index_pattern = r'"selected_path_index"\s*:\s*(null|\d+)'
                index_match = re.search(index_pattern, generated_text, re.IGNORECASE)
                if index_match:
                    try:
                        index_value = index_match.group(1)
                        # Extract reasoning if available
                        reasoning = ""
                        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', generated_text, re.DOTALL)
                        if reasoning_match:
                            reasoning = reasoning_match.group(1)
                        else:
                            # Try to get reasoning from a different format
                            reasoning_match = re.search(r'"reasoning"\s*:\s*([^,}]+)', generated_text, re.DOTALL)
                            if reasoning_match:
                                reasoning = reasoning_match.group(1).strip().strip('"')
                        
                        # Construct minimal valid JSON
                        parsed = {
                            "selected_path_index": None if index_value.lower() == "null" else int(index_value),
                            "reasoning": reasoning
                        }
                        return parsed
                    except Exception as e:
                        pass
                
                # Method 4: Check if response was truncated
                if len(generated_text) >= 1000:  # Close to max_tokens limit
                    print(f"Warning: Response may be truncated. Length: {len(generated_text)}")
                    # Could retry with higher max_tokens, but for now just log
                
                print(f"Warning: Could not parse JSON from response. First 500 chars: {generated_text[:500]}")
                # Try to save partial response for debugging
                return None
                        
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    # 재시도하되 대기 없이 즉시 재시도
                    continue
                else:
                    print(f"API call failed after {max_retries} attempts: {e}")
                    print(f"  API URL: {api_url}")
                    return None
            except Exception as e:
                print(f"Unexpected error during API call: {e}")
                if attempt < max_retries - 1:
                    # 재시도하되 대기 없이 즉시 재시도
                    continue
                else:
                    return None
        
        return None
    
    def _generate_candidates_from_labels(self, doc_id: int, top_k: int = 15) -> List[Dict]:
        """
        Generate candidate list from silver labels when candidates_data is not available.
        This creates a list of leaf nodes from the silver labels.
        
        Args:
            doc_id: Document ID
            top_k: Maximum number of candidates to generate
            
        Returns:
            List of candidate dictionaries
        """
        silver_labels = self.labels.get(doc_id, [])
        
        # Get all leaf nodes from the taxonomy
        all_leaf_nodes = [
            class_id for class_id in self.taxonomy.class_id_to_name.keys()
            if self._is_leaf_node(class_id)
        ]
        
        # Prioritize leaf nodes that are in silver labels or are descendants of silver labels
        priority_candidates = []
        other_candidates = []
        
        for leaf_id in all_leaf_nodes:
            # Get all ancestors of this leaf
            ancestors = self._get_all_ancestors(leaf_id)
            
            # Check if any silver label is an ancestor
            if any(label_id in ancestors for label_id in silver_labels):
                priority_candidates.append({
                    'leaf_class_id': leaf_id,
                    'leaf_class_name': self.taxonomy.class_id_to_name.get(leaf_id, f'class_{leaf_id}'),
                    'similarity': 0.9  # High similarity for descendants of silver labels
                })
            else:
                other_candidates.append({
                    'leaf_class_id': leaf_id,
                    'leaf_class_name': self.taxonomy.class_id_to_name.get(leaf_id, f'class_{leaf_id}'),
                    'similarity': 0.5  # Lower similarity for other candidates
                })
        
        # Combine: priority first, then random selection from others
        candidates = priority_candidates
        if len(candidates) < top_k:
            random.shuffle(other_candidates)
            candidates.extend(other_candidates[:top_k - len(candidates)])
        
        return candidates[:top_k]
    
    def validate_samples(
        self,
        samples: List[Dict],
        top_k_candidates: int = 15,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Validate samples using LLM.
        
        Args:
            samples: List of sample dictionaries
            top_k_candidates: Number of top candidates to show to LLM
            show_progress: Show progress bar
            
        Returns:
            List of validation results
        """
        print("\n" + "=" * 70)
        print("LLM Validation")
        print("=" * 70)
        
        validation_results = []
        
        iterator = tqdm(samples, disable=not show_progress, desc="Validating labels")
        
        for sample in iterator:
            doc_id = sample['doc_id']
            
            # Get review text
            review_text = self.review_texts.get(doc_id, "")
            if not review_text:
                print(f"Warning: No review text found for doc_id {doc_id}")
                continue
            
            # Get candidate classes
            if self.candidates_data:
                # Use candidates from candidates_data if available
                candidate_info = self.candidates_data.get(str(doc_id), {})
                candidates = candidate_info.get('candidates', [])
                
                if not candidates:
                    print(f"Warning: No candidates found for doc_id {doc_id}")
                    continue
            else:
                # Generate candidates from labels and taxonomy
                candidates = self._generate_candidates_from_labels(doc_id, top_k=top_k_candidates)
                candidate_info = {}  # Empty candidate info
            
            # Format candidate paths (returns formatted string and path index to class IDs mapping)
            candidate_list_str, path_index_to_class_ids = self._format_candidate_paths(candidates, top_k_classes=top_k_candidates)
            
            # Create prompt
            prompt = self._create_prompt(review_text, candidate_list_str)
            
            # Call LLM API
            llm_response = self._call_llm_api(prompt)
            
            # Get silver labels (ground truth for comparison)
            silver_labels = set(self.labels.get(doc_id, []))
            
            # Parse LLM response
            if llm_response:
                selected_path_index = llm_response.get('selected_path_index')
                reasoning = llm_response.get('reasoning', '')
                
                # Convert path index to class IDs
                if selected_path_index is not None and selected_path_index in path_index_to_class_ids:
                    # Get all class IDs in the selected path
                    selected_path_class_ids = path_index_to_class_ids[selected_path_index]
                    llm_selected = set(selected_path_class_ids)
                    # Leaf node is the last one in the path
                    llm_selected_raw = set([selected_path_class_ids[-1]]) if selected_path_class_ids else set()
                    selected_path_ids = selected_path_class_ids
                else:
                    # No path selected or invalid index
                    llm_selected = set()
                    llm_selected_raw = set()
                    selected_path_ids = None
            else:
                llm_selected = set()
                llm_selected_raw = set()
                selected_path_ids = None
                reasoning = "API call failed"
            
            # Remove virtual root (531) from llm_selected and llm_selected_raw
            VIRTUAL_ROOT_ID = 531
            if VIRTUAL_ROOT_ID in llm_selected:
                llm_selected.remove(VIRTUAL_ROOT_ID)
            if VIRTUAL_ROOT_ID in llm_selected_raw:
                llm_selected_raw.remove(VIRTUAL_ROOT_ID)
            
            # Calculate metrics
            # Now considering hierarchy - if child is selected, parent is automatically included
            intersection = silver_labels & llm_selected
            precision = len(intersection) / len(llm_selected) if llm_selected else 0.0
            recall = len(intersection) / len(silver_labels) if silver_labels else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            result = {
                'doc_id': doc_id,
                'sample_type': sample.get('type', 'unknown'),
                'review_text': review_text[:500],  # Truncate for storage
                'silver_labels': sorted(list(silver_labels)),
                'llm_selected_raw': sorted(list(llm_selected_raw)),  # Leaf node ID from selected path
                'llm_selected': sorted(list(llm_selected)),  # All nodes in path (virtual root removed)
                'llm_reasoning': reasoning,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'num_silver_labels': len(silver_labels),
                'num_llm_labels': len(llm_selected),
                'num_llm_labels_raw': len(llm_selected_raw),  # Number of leaf nodes (should be 0 or 1)
                'num_intersection': len(intersection),
                'candidate_info': {
                    's_max': candidate_info.get('top1_score') or candidate_info.get('s_max', 0.0) if candidate_info else 0.0,
                    'top2_score': candidate_info.get('top2_score', 0.0) if candidate_info else 0.0,
                    'score_diff': candidate_info.get('score_diff', 0.0) if candidate_info else 0.0,
                    'threshold': candidate_info.get('threshold') or candidate_info.get('conflict_threshold', 0.05) if candidate_info else 0.0,
                    'num_candidates': len(candidate_info.get('candidates', [])) if candidate_info and candidate_info.get('candidates') else (candidate_info.get('num_candidates', 0) if candidate_info else 0)
                }
            }
            
            validation_results.append(result)
        
        self.validation_results = validation_results
        
        return validation_results
    
    def save_results(self, output_path: str) -> None:
        """
        Save validation results to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        if not self.validation_results:
            print("Warning: No validation results to save.")
            return
        
        # Calculate summary statistics
        precisions = [r['precision'] for r in self.validation_results]
        recalls = [r['recall'] for r in self.validation_results]
        f1_scores = [r['f1'] for r in self.validation_results]
        
        summary = {
            'total_samples': len(self.validation_results),
            'average_precision': float(np.mean(precisions)),
            'average_recall': float(np.mean(recalls)),
            'average_f1': float(np.mean(f1_scores)),
            'median_precision': float(np.median(precisions)),
            'median_recall': float(np.median(recalls)),
            'median_f1': float(np.median(f1_scores)),
            'samples_by_type': Counter(r['sample_type'] for r in self.validation_results)
        }
        
        output = {
            'summary': summary,
            'results': self.validation_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\nValidation results saved to: {output_path}")
        print(f"  Total samples: {len(self.validation_results)}")
        print(f"  Average Precision: {summary['average_precision']:.4f}")
        print(f"  Average Recall: {summary['average_recall']:.4f}")
        print(f"  Average F1: {summary['average_f1']:.4f}")


def main():
    """
    Main function for label validation.
    
    API Key Setup:
    ==============
    You need to set your OpenRouter API key before running this script.
    
    export OPENROUTER_API_KEY="your-api-key-here"
    
    Get API key from: https://openrouter.ai/keys
    """
    print("=" * 70)
    print("Label Validation using LLM (OpenRouter)")
    print("=" * 70)
    
    # Load taxonomy
    print("\n[1/5] Loading taxonomy...")
    taxonomy = Taxonomy(
        hierarchy_path=config.TAXONOMY_FILE,
        classes_path=config.CLASSES_FILE
    )
    print(f"Loaded taxonomy with {taxonomy.num_classes} classes")
    
    # Load dataset
    print("\n[2/5] Loading dataset...")
    dataset = AmazonDataset(
        corpus_path=config.TRAIN_FILE,
        mode='raw'
    )
    print(f"Loaded {len(dataset)} documents")
    
    # Initialize validator
    print("\n[3/5] Initializing Label Validator...")
    
    # OpenRouter API configuration
    api_key = os.getenv("OPENROUTER_API_KEY")
    api_base_url = os.getenv("OPENROUTER_API_BASE_URL")  # Optional, will use default if None
    model_name = "meta-llama/llama-3.1-70b-instruct" # OpenRouter model name for Llama 3.1 70B
    
    print(f"Using OpenRouter API")
    print(f"Model: {model_name}")
    
    # Check if candidates file exists (for bottom-up only)
    candidates_path = os.path.join(config.OUTPUT_DIR, "bottom_up_analysis/analysis/top2_candidates.json")
    if not os.path.exists(candidates_path):
        print(f"Note: Candidates file not found at {candidates_path}")
        print("      Using simplified sampling (suitable for top-down labels)")
        candidates_path = None
    
    # Check if class counts file exists
    class_counts_path = os.path.join(config.OUTPUT_DIR, "class_imbalance_analysis/class_document_counts_full.json")
    if not os.path.exists(class_counts_path):
        print(f"Note: Class counts file not found at {class_counts_path}")
        print("      Will compute class counts from labels")
        class_counts_path = None
    
    validator = LabelValidator(
        taxonomy=taxonomy,
        dataset=dataset,
        labels_path=os.path.join(config.OUTPUT_DIR, "top_down_labels.json"),
        candidates_path=candidates_path,
        class_counts_path=class_counts_path,
        api_key=api_key,
        api_base_url=api_base_url,
        model_name=model_name
    )
    
    # Stratified sampling
    print("\n[4/5] Performing stratified sampling...")
    samples = validator.stratified_sampling(
        num_samples=200,
        confident_ratio=0.3,
        ambiguous_ratio=0.4,
        rare_ratio=0.3
    )
    
    # Validate samples
    print("\n[5/5] Validating samples with LLM...")
    results = validator.validate_samples(
        samples=samples,
        top_k_candidates=15,
        show_progress=True
    )
    
    # Save results
    print("\nSaving validation results...")
    config.ensure_output_dir()
    output_path = os.path.join(config.OUTPUT_DIR, "label_validation_results.json")
    validator.save_results(output_path)
    
    print("\n" + "=" * 70)
    print("Label Validation Complete!")
    print("=" * 70)
    print(f"  Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()


"""
Top-Down Label Generation for Hierarchical Multi-Label Text Classification.
Uses Graph-based Embedding Propagation instead of predefined path embeddings.
"""

import os
import re
import json
import time
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import requests

from data_loader import Taxonomy, AmazonDataset
from config import config


class TopDownLabelGenerator:
    """
    Top-Down Label Generator using Graph-based Embedding Propagation.
    
    Process:
    1. Local Embedding: Generate embeddings for each node using class name + keywords
    2. Graph Propagation: Propagate parent embeddings to children using Add & Norm
    3. Review Embedding: Generate embeddings for review texts
    4. Similarity Computation: Compute cosine similarity between reviews and leaf nodes
    5. Label Generation: Select best leaf nodes and propagate to ancestors
    """
    
    def __init__(
        self,
        taxonomy: Taxonomy,
        dataset: AmazonDataset,
        sbert_model_name: Optional[str] = None,
        device: Optional[str] = None,
        lambda_weight: Optional[float] = None,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.taxonomy = taxonomy
        self.dataset = dataset
        self.lambda_weight = lambda_weight if lambda_weight is not None else config.GENERATOR_LAMBDA_WEIGHT
        
        # Load keywords if not already loaded
        if not self.taxonomy.keywords:
            print("Loading keywords from file...")
            self.taxonomy.load_keywords(config.KEYWORDS_FILE)
        
        # Get root node
        self.root_id = self.taxonomy.class_name_to_id.get("Root", 0)
        print(f"Root node ID: {self.root_id}")
        
        # Cache root children
        root_children_list = self.taxonomy.get_children(self.root_id)
        self.root_children = set(root_children_list)
        print(f"Root children: {sorted(self.root_children)}")
        
        # Find leaf nodes
        self.leaf_nodes = self._find_leaf_nodes()
        print(f"Found {len(self.leaf_nodes)} leaf nodes")
        
        # Initialize SBERT model
        sbert_model_name = sbert_model_name or config.GENERATOR_SBERT_MODEL
        print(f"Loading SBERT model: {sbert_model_name}")
        self.sbert_model = SentenceTransformer(sbert_model_name, device=device)
        print("SBERT model loaded successfully")
        
        # Storage for processed data
        self.node_embeddings: Dict[int, np.ndarray] = {}  # class_id -> final embedding (after propagation)
        self.local_embeddings: Dict[int, np.ndarray] = {}  # class_id -> local embedding (before propagation)
        self.review_embeddings: Dict[int, np.ndarray] = {}  # doc_id -> embedding vector
        self.review_texts: Dict[int, str] = {}  # doc_id -> review text
        
        # LLM API configuration
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.api_base_url = api_base_url or config.GENERATOR_API_BASE_URL
        self.model_name = model_name or config.GENERATOR_MODEL_NAME
        
        # Conflict C cases collection
        self.conflict_c_cases: List[Dict] = []
        
        # Warn if no API key
        if not self.api_key:
            print("Warning: No API key provided. LLM conflict resolution will be skipped.")
            print("  Set OPENROUTER_API_KEY environment variable or pass api_key parameter.")
            print("  Case C conflicts will automatically select Top-1 leaf.")
        
    def _find_leaf_nodes(self) -> List[int]:
        leaf_nodes = []
        for class_id in self.taxonomy.class_id_to_name.keys():
            children = self.taxonomy.get_children(class_id)
            if not children:
                leaf_nodes.append(class_id)
        return leaf_nodes
    
    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def _build_graph(self) -> nx.DiGraph:
        """
        Build directed graph from taxonomy hierarchy.
        """
        G = nx.DiGraph()
        
        # Add all nodes
        for class_id in self.taxonomy.class_id_to_name.keys():
            G.add_node(class_id)
        
        # Add edges (parent -> child)
        for child_id in self.taxonomy.class_id_to_name.keys():
            parents = self.taxonomy.get_parents(child_id)
            for parent_id in parents:
                G.add_edge(parent_id, child_id)
        
        return G
    
    def _generate_class_description(self, class_id: int) -> str:
        """Generate a concise description for a class using LLM."""
        if not self.api_key:
            # Fallback to original format
            class_name = self.taxonomy.class_id_to_name.get(class_id, f"class_{class_id}")
            keywords = self.taxonomy.keywords.get(class_id, [])
            keywords_str = ", ".join(keywords) if keywords else ""
            if keywords_str:
                return f"Category: {class_name}. Context: {keywords_str}"
            else:
                return f"Category: {class_name}"
        
        class_name = self.taxonomy.class_id_to_name.get(class_id, f"class_{class_id}")
        keywords = self.taxonomy.keywords.get(class_id, [])
        
        if not keywords:
            return f"Category: {class_name}"
        
        keywords_str = ', '.join(keywords[:10])
        
        prompt = f"""Provide a concise, 1-sentence definition of the product category: '{class_name}'.

Keywords associated with this category are: {keywords_str}.

Focus on the product's function and who uses it.

Output only the definition sentence, nothing else."""
        
        response = self._call_llm_api(prompt, max_retries=2, return_raw_text=True)
        
        if response and 'text' in response:
            description = response['text'].strip()
            description = description.replace('\n', ' ').strip()
            if description:
                return description
        
        # Fallback
        if keywords_str:
            return f"Category: {class_name}. Context: {keywords_str}"
        else:
            return f"Category: {class_name}"
    
    def _generate_local_embeddings(self) -> Dict[int, np.ndarray]:
        """Generate local embeddings for all nodes using LLM-generated descriptions with caching."""
        cache_file = os.path.join(config.EMBEDDINGS_DIR, "class_descriptions.json")
        
        # Load cached descriptions
        cached_descriptions = {}
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_descriptions = json.load(f)
            print(f"Loaded {len(cached_descriptions)} cached descriptions")
        
        # Prepare texts for all nodes
        texts_to_embed = []
        class_ids = []
        new_descriptions = {}
        
        use_llm = self.api_key is not None
        
        if use_llm:
            print("Generating class descriptions with LLM (cached where available)...")
        else:
            print("Using keyword-based descriptions (no API key)...")
        
        for class_id in tqdm(self.taxonomy.class_id_to_name.keys(), desc="Preparing descriptions"):
            if class_id == self.root_id:
                continue
            
            class_id_str = str(class_id)
            
            # Check cache first
            if class_id_str in cached_descriptions:
                description = cached_descriptions[class_id_str]
            else:
                # Generate new description
                description = self._generate_class_description(class_id)
                if use_llm:
                    new_descriptions[class_id_str] = description
            
            texts_to_embed.append(description)
            class_ids.append(class_id)
        
        # Save new descriptions to cache
        if new_descriptions:
            cached_descriptions.update(new_descriptions)
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cached_descriptions, f, indent=2, ensure_ascii=False)
            print(f"Generated and cached {len(new_descriptions)} new descriptions")
        
        # Generate embeddings in batch
        print(f"Computing embeddings for {len(texts_to_embed)} nodes...")
        embeddings = self.sbert_model.encode(
            texts_to_embed,
            batch_size=config.GENERATOR_EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Store local embeddings
        local_embeddings = {}
        for idx, class_id in enumerate(class_ids):
            local_embeddings[class_id] = embeddings[idx]
        
        print(f"Generated local embeddings for {len(local_embeddings)} nodes")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        
        return local_embeddings
    
    def _propagate_embeddings(
        self,
        local_embeddings: Dict[int, np.ndarray],
        G: nx.DiGraph
    ) -> Dict[int, np.ndarray]:
        """
        Propagate embeddings from parent to child using topological sort.
        For multi-parent nodes, average parent embeddings before adding.
        """
        # Topological sort (parent before child)
        try:
            topo_order = list(nx.topological_sort(G))
        except nx.NetworkXError as e:
            print(f"Error: Graph has cycles! {e}")
            print("Falling back to BFS from root...")
            topo_order = list(nx.bfs_tree(G, source=self.root_id))
        
        print(f"Processing {len(topo_order)} nodes in topological order...")
        
        # Initialize final embeddings with local embeddings
        final_embeddings = {}
        
        for class_id in tqdm(topo_order, desc="Propagating"):
            # Skip root node
            if class_id == self.root_id:
                continue

            # Get local embedding
            local_emb = local_embeddings.get(class_id)
            if local_emb is None:
                print(f"Warning: No local embedding for class {class_id}")
                continue
            
            # Get parent embeddings
            parents = list(G.predecessors(class_id))
            
            if not parents:
                # Root node or isolated node: use local embedding only
                final_embeddings[class_id] = local_emb
            else:
                # Get parent embeddings
                parent_embs = []
                for parent_id in parents:
                    parent_emb = final_embeddings.get(parent_id)
                    if parent_emb is not None:
                        parent_embs.append(parent_emb)
                
                if not parent_embs:
                    # No parent embeddings available (shouldn't happen with topo sort)
                    final_embeddings[class_id] = local_emb
                else:
                    # Average parent embeddings
                    avg_parent_emb = np.mean(parent_embs, axis=0)
                    
                    # Add & Norm
                    combined_emb = local_emb + self.lambda_weight * avg_parent_emb
                    normalized_emb = combined_emb / (np.linalg.norm(combined_emb) + 1e-8)
                    
                    final_embeddings[class_id] = normalized_emb
        
        print(f"Propagated embeddings for {len(final_embeddings)} nodes")
        
        return final_embeddings
    
    def generate_class_embeddings(self) -> Dict[int, np.ndarray]:
        # Step 1: Generate local embeddings
        self.local_embeddings = self._generate_local_embeddings()
        
        # Step 2: Build graph
        print("\nBuilding taxonomy graph...")
        G = self._build_graph()
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        
        # Step 3: Propagate embeddings
        self.node_embeddings = self._propagate_embeddings(self.local_embeddings, G)
        
        print(f"\nGenerated embeddings for {len(self.node_embeddings)} nodes")
        
        return self.node_embeddings
    
    def generate_review_embeddings(self, show_progress: bool = True) -> Dict[int, np.ndarray]:
        texts = []
        doc_ids = []
        
        iterator = tqdm(self.dataset, disable=not show_progress, desc="Preprocessing reviews")
        for item in iterator:
            doc_id = item['index']
            text = item['text']
            
            # Cache original text for keyword verification
            self.review_texts[doc_id] = text.lower()
            
            # Preprocess text
            preprocessed_text = self.preprocess_text(text)
            
            texts.append(preprocessed_text)
            doc_ids.append(doc_id)
        
        # Generate embeddings in batch
        print(f"Computing embeddings for {len(texts)} reviews...")
        embeddings = self.sbert_model.encode(
            texts,
            batch_size=config.GENERATOR_EMBEDDING_BATCH_SIZE,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Store embeddings
        for idx, doc_id in enumerate(doc_ids):
            self.review_embeddings[doc_id] = embeddings[idx]
        
        print(f"Generated embeddings for {len(self.review_embeddings)} reviews")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        
        return self.review_embeddings
    
    def compute_similarities(
        self,
        review_embeddings: Optional[Dict[int, np.ndarray]] = None,
        class_embeddings: Optional[Dict[int, np.ndarray]] = None
    ) -> Dict[int, Dict[int, float]]:
        if review_embeddings is None:
            review_embeddings = self.review_embeddings
        
        if class_embeddings is None:
            class_embeddings = self.node_embeddings
        
        if not review_embeddings:
            print("Warning: Missing review embeddings. Run generate_review_embeddings first.")
            return {}
        
        if not class_embeddings:
            print("Warning: No class embeddings found. Run generate_class_embeddings first.")
            return {}
        
        # Extract leaf node embeddings
        leaf_emb_list = []
        leaf_ids = []
        for leaf_id in self.leaf_nodes:
            if leaf_id in class_embeddings:
                leaf_emb_list.append(class_embeddings[leaf_id])
                leaf_ids.append(leaf_id)
        
        if not leaf_emb_list:
            print("Warning: No leaf embeddings found!")
            return {}
        
        leaf_emb_array = np.array(leaf_emb_list)
        
        # Normalize leaf embeddings
        leaf_emb_norm = leaf_emb_array / (np.linalg.norm(leaf_emb_array, axis=1, keepdims=True) + 1e-8)
        
        similarities = {}
        
        for doc_id, review_emb in tqdm(review_embeddings.items(), desc="Computing similarities"):
            # Normalize review embedding
            review_emb_norm = review_emb / (np.linalg.norm(review_emb) + 1e-8)
            
            # Compute cosine similarity with all leaf nodes
            sim_scores = np.dot(leaf_emb_norm, review_emb_norm)
            
            # Store similarities
            leaf_sims = {
                leaf_ids[i]: float(sim_scores[i])
                for i in range(len(leaf_ids))
            }
            similarities[doc_id] = leaf_sims
        
        print(f"Computed similarities for {len(similarities)} reviews")
        print(f"  Leaf nodes: {len(leaf_ids)}")
        
        return similarities
    
    def _get_all_ancestors(self, class_id: int) -> Set[int]:
        """Get all ancestor nodes using BFS (includes ALL parents)."""
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
    
    def _get_ancestors_via_parent(self, class_id: int, selected_parent_id: int) -> Set[int]:
        """Get ancestors by following only the selected parent path."""
        ancestors = set([class_id])
        queue = [class_id]
        visited = set()
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            
            parents = self.taxonomy.get_parents(current_id)
            
            # For the starting node, only follow the selected parent
            if current_id == class_id:
                if selected_parent_id in parents:
                    ancestors.add(selected_parent_id)
                    queue.append(selected_parent_id)
            else:
                # For other nodes, follow all parents (normal BFS)
                for parent_id in parents:
                    if parent_id not in ancestors:
                        ancestors.add(parent_id)
                        queue.append(parent_id)
        
        return ancestors
    
    def _select_best_parent_by_similarity(
        self,
        doc_id: int,
        leaf_id: int
    ) -> Optional[int]:
        """
        Select the best parent for a leaf node based on review text similarity.
        Returns the parent_id with highest similarity to the review.
        """
        parents = self.taxonomy.get_parents(leaf_id)
        
        if not parents:
            return None
        
        if len(parents) == 1:
            return parents[0]
        
        # Get review embedding
        review_emb = self.review_embeddings.get(doc_id)
        if review_emb is None:
            # Fallback: return first parent
            return parents[0]
        
        # Normalize review embedding
        review_emb_norm = review_emb / (np.linalg.norm(review_emb) + 1e-8)
        
        # Compute similarity with each parent
        best_parent_id = None
        best_similarity = -1.0
        
        for parent_id in parents:
            parent_emb = self.node_embeddings.get(parent_id)
            if parent_emb is None:
                continue
            
            # Normalize parent embedding
            parent_emb_norm = parent_emb / (np.linalg.norm(parent_emb) + 1e-8)
            
            # Compute cosine similarity
            similarity = float(np.dot(review_emb_norm, parent_emb_norm))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_parent_id = parent_id
        
        # Fallback if no valid parent embedding found
        if best_parent_id is None:
            best_parent_id = parents[0]
        
        return best_parent_id
    
    def _get_domain_for_class(self, class_id: int) -> Optional[int]:
        """Get the root-level domain for a class (returns first found if multiple)."""
        ancestors = self._get_all_ancestors(class_id)
        
        for ancestor_id in ancestors:
            if ancestor_id in self.root_children:
                return ancestor_id
        
        return None
    
    def _get_all_domains_for_class(self, class_id: int) -> List[int]:
        """Get all root-level domains for a class (handles multi-parent cases)."""
        ancestors = self._get_all_ancestors(class_id)
        domains = []
        
        for ancestor_id in ancestors:
            if ancestor_id in self.root_children:
                domains.append(ancestor_id)
        
        return domains
    
    def _get_selected_domain_for_leaf(self, doc_id: int, leaf_id: int) -> Optional[int]:
        """
        Get the domain for a leaf by selecting the best parent path first.
        This ensures we use only one domain even for multi-parent nodes.
        """
        parents = self.taxonomy.get_parents(leaf_id)
        
        if not parents:
            return None
        
        # If multiple parents, select the best one based on similarity
        if len(parents) > 1:
            selected_parent = self._select_best_parent_by_similarity(doc_id, leaf_id)
            if selected_parent is None:
                selected_parent = parents[0]
        else:
            selected_parent = parents[0]
        
        # Get domain through the selected parent path
        ancestors = self._get_ancestors_via_parent(leaf_id, selected_parent)
        
        for ancestor_id in ancestors:
            if ancestor_id in self.root_children:
                return ancestor_id
        
        return None
    
    def _call_llm_api(self, prompt: str, max_retries: int = 3, return_raw_text: bool = False) -> Optional[Dict]:
        """Call OpenRouter LLM API."""
        if not self.api_key:
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "Top-Down Label Generator"
        }
        
        api_url = self.api_base_url
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": config.GENERATOR_API_TEMPERATURE,
            "max_tokens": config.GENERATOR_API_MAX_TOKENS
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                
                result = response.json()
                
                generated_text = ""
                if "choices" in result and len(result["choices"]) > 0:
                    generated_text = result["choices"][0].get("message", {}).get("content", "")
                
                if not generated_text:
                    return None
                
                if return_raw_text:
                    return {"text": generated_text}
                
                # Try to extract JSON
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*(?:"choice"|"selected_leaf_id")[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                json_match = re.search(json_pattern, generated_text, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        parsed = json.loads(json_str)
                        if "choice" in parsed or "selected_leaf_id" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        json_str = re.sub(r',\s*}', '}', json_str)
                        json_str = re.sub(r',\s*]', ']', json_str)
                        try:
                            parsed = json.loads(json_str)
                            if "choice" in parsed or "selected_leaf_id" in parsed:
                                return parsed
                        except:
                            pass
                
                # Try parsing whole text
                try:
                    parsed = json.loads(generated_text.strip())
                    if "choice" in parsed or "selected_leaf_id" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    pass
                
                # Try to extract choice field
                choice_match = re.search(r'"choice"\s*:\s*"([12])"', generated_text)
                if choice_match:
                    return {"choice": choice_match.group(1)}
                
                choice_match = re.search(r'"choice"\s*:\s*([12])', generated_text)
                if choice_match:
                    return {"choice": choice_match.group(1)}
                
                return None
                        
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    return None
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
        
        return None
    
    def _resolve_conflict_with_llm(
        self,
        doc_id: int,
        top1_leaf_id: int,
        top1_score: float,
        top2_leaf_id: int,
        top2_score: float
    ) -> Optional[Dict]:
        """Resolve conflict between Top-1 and Top-2 leaf nodes using LLM."""
        review_text = self.review_texts.get(doc_id, "")
        if not review_text:
            return None
        
        # Get class names
        class1_name = self.taxonomy.class_id_to_name.get(top1_leaf_id, f"class_{top1_leaf_id}")
        class2_name = self.taxonomy.class_id_to_name.get(top2_leaf_id, f"class_{top2_leaf_id}")
        
        # Get root domain names
        root1_domain_id = self._get_domain_for_class(top1_leaf_id)
        root2_domain_id = self._get_domain_for_class(top2_leaf_id)
        
        root1_name = self.taxonomy.class_id_to_name.get(root1_domain_id, f"Domain_{root1_domain_id}") if root1_domain_id is not None else "Unknown"
        root2_name = self.taxonomy.class_id_to_name.get(root2_domain_id, f"Domain_{root2_domain_id}") if root2_domain_id is not None else "Unknown"
        
        # Create prompt
        prompt = f"""Role: You are an expert Amazon Product Classifier.

Task: Resolve a category ambiguity for a product review.

Goal: Select the SINGLE MOST RELEVANT category.

---

[Input Data]

Review: "{review_text}"

[Conflict Scenario]

The embedding model is confused between these two categories:

1. Domain A: {root1_name} - Category: {class1_name}

2. Domain B: {root2_name} - Category: {class2_name}

---

[Instructions]

1. Analyze the review text for domain-specific keywords.

2. Determine which category is the PRIMARY intent of the user.

3. Strict Constraint: Do NOT select "Both". You must choose the one best fit.

4. Output Format: Provide your choice as a JSON object.

---

Output JSON Example:
{{
  "reasoning": "The review mentions specific features that indicate category 1.",
  "choice": "1" 
}}"""
        
        # Call LLM API
        llm_response = self._call_llm_api(prompt)
        
        if llm_response and 'choice' in llm_response:
            choice = str(llm_response['choice']).strip()
            if choice == "1":
                return {"selected_leaf_id": top1_leaf_id, "reasoning": llm_response.get('reasoning', '')}
            elif choice == "2":
                return {"selected_leaf_id": top2_leaf_id, "reasoning": llm_response.get('reasoning', '')}
        
        return None
    
    def _build_anchor_set_with_pseudo_labeling(
        self,
        max_samples_per_pair: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[Tuple[int, int], Dict]:
        """
        Build anchor sets for each conflict domain pair with pseudo-labeling.
        All documents above adaptive threshold get pseudo-labels.
        Documents below threshold get empty labels.
        """
        print("\n" + "=" * 70)
        print("Step 2.5: Anchor Set Construction with Pseudo-Labeling")
        print("=" * 70)
        
        if max_samples_per_pair is None:
            max_samples_per_pair = config.GENERATOR_MAX_SAMPLES_PER_PAIR
        
        if not self.conflict_c_cases or not self.api_key:
            print("  No conflict cases or API key. Skipping anchor set construction.")
            return {}
        
        # Step 1: Group conflict cases by domain pair
        domain_pair_cases = defaultdict(list)
        
        for case in self.conflict_c_cases:
            top1_domain = self._get_domain_for_class(case['top1_leaf_id'])
            top2_domain = self._get_domain_for_class(case['top2_leaf_id'])
            
            if top1_domain is None or top2_domain is None:
                continue
            
            # Normalize domain pair (smaller ID first)
            domain_pair = tuple(sorted([top1_domain, top2_domain]))
            domain_pair_cases[domain_pair].append(case)
        
        print(f"\n  Found {len(domain_pair_cases)} unique conflict domain pairs")
        print(f"  Total conflict cases: {len(self.conflict_c_cases)}")
        
        # Step 2: For each domain pair, sample and query LLM
        anchor_results = {}
        total_llm_calls = 0
        total_pseudo_labels = 0
        total_empty_labels = 0
        
        for domain_pair, cases in domain_pair_cases.items():
            domain1_id, domain2_id = domain_pair
            domain1_name = self.taxonomy.class_id_to_name.get(domain1_id, f"Domain_{domain1_id}")
            domain2_name = self.taxonomy.class_id_to_name.get(domain2_id, f"Domain_{domain2_id}")
            
            print(f"\n  Domain Pair: {domain1_name} ({domain1_id}) vs {domain2_name} ({domain2_id})")
            print(f"    Total cases: {len(cases)}")
            
            # Sample cases for this pair (prioritize hardest cases with smallest score_diff)
            sorted_cases = sorted(cases, key=lambda x: x['score_diff'])
            samples_to_query = min(max_samples_per_pair, len(sorted_cases))
            selected_cases = sorted_cases[:samples_to_query]
            
            print(f"    Sampling {samples_to_query} cases for anchor set...")
            
            # Query LLM for anchor labels
            anchor_embeddings = []
            anchor_labels = []  # 0 = domain1, 1 = domain2
            anchor_doc_ids = []
            anchor_leaf_ids = {}  # doc_id -> selected_leaf_id
            
            iterator = tqdm(selected_cases, disable=not show_progress, 
                           desc=f"    LLM labeling {domain1_name} vs {domain2_name}")
            
            for case in iterator:
                doc_id = case['doc_id']
                
                # Get LLM label
                llm_result = self._resolve_conflict_with_llm(
                    doc_id,
                    case['top1_leaf_id'],
                    case['top1_score'],
                    case['top2_leaf_id'],
                    case['top2_score']
                )
                
                total_llm_calls += 1
                
                if llm_result is None or 'selected_leaf_id' not in llm_result:
                    continue
                
                selected_leaf_id = llm_result['selected_leaf_id']
                selected_domain = self._get_domain_for_class(selected_leaf_id)
                
                if selected_domain not in [domain1_id, domain2_id]:
                    continue
                
                # Get review embedding
                if doc_id not in self.review_embeddings:
                    continue
                
                review_emb = self.review_embeddings[doc_id]
                
                # Label: 0 for domain1, 1 for domain2
                label = 0 if selected_domain == domain1_id else 1
                
                anchor_embeddings.append(review_emb)
                anchor_labels.append(label)
                anchor_doc_ids.append(doc_id)
                anchor_leaf_ids[doc_id] = selected_leaf_id
            
            # Step 2.5: Pseudo-labeling for all documents above threshold
            pseudo_labeled_doc_ids = {}  # doc_id -> selected_leaf_id
            low_confidence_doc_ids = []
            
            if len(anchor_embeddings) >= 5:
                print(f"    Augmenting with pseudo-labeled samples...")
                
                # Separate anchor samples by class
                anchor_emb_array = np.array(anchor_embeddings)
                anchor_label_array = np.array(anchor_labels)
                
                class0_anchors = anchor_emb_array[anchor_label_array == 0]
                class1_anchors = anchor_emb_array[anchor_label_array == 1]
                
                # Get unlabeled samples (not in anchor set)
                unlabeled_cases = [c for c in cases if c['doc_id'] not in anchor_doc_ids]
                
                # Compute similarities for all unlabeled samples
                similarities_class0 = []
                similarities_class1 = []
                unlabeled_case_list = []
                
                for case in unlabeled_cases:
                    doc_id = case['doc_id']
                    if doc_id not in self.review_embeddings:
                        continue
                    
                    unlabeled_emb = self.review_embeddings[doc_id]
                    unlabeled_case_list.append(case)
                    
                    # Compute similarity to both classes
                    if len(class0_anchors) > 0:
                        sims_0 = [
                            np.dot(unlabeled_emb, anchor) / 
                            (np.linalg.norm(unlabeled_emb) * np.linalg.norm(anchor) + 1e-8)
                            for anchor in class0_anchors
                        ]
                        sim_to_class0 = np.max(sims_0)
                    else:
                        sim_to_class0 = 0.0
                    
                    if len(class1_anchors) > 0:
                        sims_1 = [
                            np.dot(unlabeled_emb, anchor) / 
                            (np.linalg.norm(unlabeled_emb) * np.linalg.norm(anchor) + 1e-8)
                            for anchor in class1_anchors
                        ]
                        sim_to_class1 = np.max(sims_1)
                    else:
                        sim_to_class1 = 0.0
                    
                    similarities_class0.append(sim_to_class0)
                    similarities_class1.append(sim_to_class1)
                
                # Adaptive threshold: 90th percentile for each class
                if len(similarities_class0) > 0:
                    threshold_class0 = np.percentile(similarities_class0, 90)
                    threshold_class1 = np.percentile(similarities_class1, 90)
                else:
                    threshold_class0 = 0.85
                    threshold_class1 = 0.85
                
                print(f"      Adaptive thresholds: Class0={threshold_class0:.3f}, Class1={threshold_class1:.3f}")
                
                # Assign pseudo-labels to ALL documents above threshold
                for i, case in enumerate(unlabeled_case_list):
                    doc_id = case['doc_id']
                    sim_0 = similarities_class0[i]
                    sim_1 = similarities_class1[i]
                    
                    # Assign pseudo-label if above adaptive threshold
                    if sim_0 > sim_1 and sim_0 >= threshold_class0:
                        # Assign to domain1
                        top1_domain = self._get_domain_for_class(case['top1_leaf_id'])
                        if top1_domain == domain1_id:
                            pseudo_labeled_doc_ids[doc_id] = case['top1_leaf_id']
                        else:
                            pseudo_labeled_doc_ids[doc_id] = case['top2_leaf_id']
                    elif sim_1 > sim_0 and sim_1 >= threshold_class1:
                        # Assign to domain2
                        top1_domain = self._get_domain_for_class(case['top1_leaf_id'])
                        if top1_domain == domain2_id:
                            pseudo_labeled_doc_ids[doc_id] = case['top1_leaf_id']
                        else:
                            pseudo_labeled_doc_ids[doc_id] = case['top2_leaf_id']
                    else:
                        # Low confidence - mark for empty label
                        low_confidence_doc_ids.append(doc_id)
                
                print(f"      Pseudo-labeled samples: {len(pseudo_labeled_doc_ids)}")
                print(f"      Low confidence samples (empty label): {len(low_confidence_doc_ids)}")
                
                total_pseudo_labels += len(pseudo_labeled_doc_ids)
                total_empty_labels += len(low_confidence_doc_ids)
            
            # Store results for this domain pair
            anchor_results[domain_pair] = {
                'domain1_id': domain1_id,
                'domain2_id': domain2_id,
                'anchor_doc_ids': anchor_doc_ids,
                'anchor_leaf_ids': anchor_leaf_ids,
                'pseudo_labeled_doc_ids': pseudo_labeled_doc_ids,
                'low_confidence_doc_ids': low_confidence_doc_ids
            }
        
        print(f"\n  Summary:")
        print(f"    Total LLM calls: {total_llm_calls}")
        print(f"    Total pseudo-labels: {total_pseudo_labels}")
        print(f"    Total empty labels: {total_empty_labels}")
        
        return anchor_results
    
    
    def generate_labels(
        self,
        similarities: Optional[Dict[int, Dict[int, float]]] = None,
        conflict_threshold: Optional[float] = None,
        show_progress: bool = True
    ) -> Dict[int, Set[int]]:
        """
        Generate labels using Top-2 selection with conflict resolution.
        
        Process:
        1. Top-2 Selection: Sort leaf nodes by similarity scores
        2. Conflict Check: If score difference is small and different domains, resolve
        3. Label Propagation: Add all ancestors of selected leaf
        """
        if conflict_threshold is None:
            conflict_threshold = config.GENERATOR_CONFLICT_THRESHOLD
        
        if similarities is None:
            print("Computing similarities...")
            similarities = self.compute_similarities()
        
        if not similarities:
            print("Warning: No similarities found.")
            return {}
        
        # Initialize conflict cases
        self.conflict_c_cases = []
        
        # Ensure review texts are cached
        if not self.review_texts:
            print("Caching review texts...")
            for item in tqdm(self.dataset, disable=not show_progress, desc="Loading review texts"):
                doc_id = item['index']
                text = item['text']
                self.review_texts[doc_id] = text.lower()
        
        # Step 1: Top-2 Selection
        print("\n" + "=" * 70)
        print("Step 1: Top-2 Selection")
        print("=" * 70)
        
        top2_leaves: Dict[int, List[Tuple[int, float]]] = {}
        
        iterator = tqdm(similarities.items(), disable=not show_progress, desc="Selecting Top-2")
        for doc_id, leaf_scores in iterator:
            if not leaf_scores:
                top2_leaves[doc_id] = []
                continue
            
            sorted_leaves = sorted(leaf_scores.items(), key=lambda x: x[1], reverse=True)
            top2_leaves[doc_id] = sorted_leaves[:2]
        
        docs_with_leaves = sum(1 for leaves in top2_leaves.values() if leaves)
        print(f"Step 1 Complete: {docs_with_leaves}/{len(top2_leaves)} documents")
        
        # Step 2: Conflict Resolution
        print("\n" + "=" * 70)
        print("Step 2: Conflict Resolution")
        print("=" * 70)
        
        selected_leaf_ids: Dict[int, int] = {}
        case_a_count = 0
        case_b_count = 0
        case_c_count = 0
        llm_success_count = 0
        
        iterator = tqdm(top2_leaves.items(), disable=not show_progress, desc="Resolving conflicts")
        for doc_id, top2_list in iterator:
            if not top2_list:
                continue
            
            top1_leaf_id, top1_score = top2_list[0]
            
            if len(top2_list) == 1:
                selected_leaf_ids[doc_id] = top1_leaf_id
                continue
            
            top2_leaf_id, top2_score = top2_list[1]
            score_diff = top1_score - top2_score
            
            # Get selected domains (one per leaf, based on best parent)
            top1_domain = self._get_selected_domain_for_leaf(doc_id, top1_leaf_id)
            top2_domain = self._get_selected_domain_for_leaf(doc_id, top2_leaf_id)
            
            # Case A: Large score difference
            if score_diff >= conflict_threshold:
                selected_leaf_ids[doc_id] = top1_leaf_id
                case_a_count += 1
            
            # Case B: Small difference, same domain
            elif top1_domain is not None and top2_domain is not None and top1_domain == top2_domain:
                selected_leaf_ids[doc_id] = top1_leaf_id
                case_b_count += 1
            
            # Case C: Small difference, different domains
            else:
                case_c_count += 1
                
                # Collect for LLM resolution
                self.conflict_c_cases.append({
                    'doc_id': doc_id,
                    'top1_leaf_id': top1_leaf_id,
                    'top1_score': top1_score,
                    'top2_leaf_id': top2_leaf_id,
                    'top2_score': top2_score,
                    'score_diff': score_diff
                })
        
        # Step 2.5: Anchor Set Construction with Pseudo-Labeling
        pseudo_label_count = 0
        empty_label_count = 0
        if self.conflict_c_cases and self.api_key:
            # Build anchor sets with pseudo-labeling
            anchor_results = self._build_anchor_set_with_pseudo_labeling(
                max_samples_per_pair=config.GENERATOR_MAX_SAMPLES_PER_PAIR,
                show_progress=show_progress
            )
            
            # Apply anchor labels and pseudo-labels
            for domain_pair, results in anchor_results.items():
                # Apply LLM anchor labels
                for doc_id, leaf_id in results['anchor_leaf_ids'].items():
                    selected_leaf_ids[doc_id] = leaf_id
                    llm_success_count += 1
                
                # Apply pseudo-labels
                for doc_id, leaf_id in results['pseudo_labeled_doc_ids'].items():
                    selected_leaf_ids[doc_id] = leaf_id
                    pseudo_label_count += 1
                
                # Count empty labels (low confidence cases remain without labels)
                empty_label_count += len(results['low_confidence_doc_ids'])
            
            # Count remaining unresolved cases (will remain empty)
            unresolved_count = 0
            for case in self.conflict_c_cases:
                doc_id = case['doc_id']
                if doc_id not in selected_leaf_ids:
                    # Check if it's already counted as low-confidence
                    is_low_confidence = False
                    for domain_pair, results in anchor_results.items():
                        if doc_id in results['low_confidence_doc_ids']:
                            is_low_confidence = True
                            break
                    
                    if not is_low_confidence:
                        # Unresolved case (not in anchor results) - will remain empty
                        unresolved_count += 1
            
            # Add unresolved cases to empty label count
            empty_label_count += unresolved_count
            
            print(f"\n  Step 2.5 Complete:")
            print(f"    LLM anchor labels: {llm_success_count}")
            print(f"    Pseudo-labels: {pseudo_label_count}")
            print(f"    Empty labels (low confidence + unresolved): {empty_label_count}")
        
        print(f"\nStep 2 Complete:")
        print(f"  Case A (Large diff): {case_a_count}")
        print(f"  Case B (Small diff, same domain): {case_b_count}")
        print(f"  Case C (Small diff, different domains): {case_c_count}")
        if self.conflict_c_cases:
            print(f"    Cases requiring resolution: {len(self.conflict_c_cases)}")
        print(f"    LLM anchor labels: {llm_success_count}")
        print(f"    Pseudo-labels: {pseudo_label_count}")
        print(f"    Empty labels (low confidence + unresolved): {empty_label_count}")
        
        # Step 3: Label Propagation (with parent selection for multi-parent nodes)
        print("\n" + "=" * 70)
        print("Step 3: Label Propagation (Select Best Parent Path)")
        print("=" * 70)
        
        final_labels: Dict[int, Set[int]] = {}
        multi_parent_selections = 0
        
        iterator = tqdm(selected_leaf_ids.items(), disable=not show_progress, desc="Propagating labels")
        for doc_id, leaf_id in iterator:
            # Check if leaf has multiple parents
            parents = self.taxonomy.get_parents(leaf_id)
            
            if len(parents) > 1:
                # Select best parent based on similarity
                best_parent_id = self._select_best_parent_by_similarity(doc_id, leaf_id)
                if best_parent_id is not None:
                    # Get ancestors via selected parent only
                    ancestors = self._get_ancestors_via_parent(leaf_id, best_parent_id)
                    multi_parent_selections += 1
                else:
                    # Fallback: use all ancestors
                    ancestors = self._get_all_ancestors(leaf_id)
            else:
                # Single parent or no parent: use all ancestors
                ancestors = self._get_all_ancestors(leaf_id)
            
            # Remove root node
            ancestors.discard(self.root_id)
            
            final_labels[doc_id] = ancestors
        
        # Calculate statistics
        total_labels = sum(len(labels) for labels in final_labels.values())
        avg_labels = total_labels / len(final_labels) if final_labels else 0
        docs_with_labels = sum(1 for labels in final_labels.values() if labels)
        
        # Count documents with empty labels (from low confidence cases)
        total_docs_processed = len(selected_leaf_ids) + empty_label_count
        docs_with_empty_labels = total_docs_processed - len(final_labels)
        
        print(f"Step 3 Complete:")
        print(f"  Documents with labels: {docs_with_labels}/{total_docs_processed}")
        print(f"  Documents with empty labels: {docs_with_empty_labels}")
        print(f"  Multi-parent selections: {multi_parent_selections}")
        print(f"  Average labels per doc (non-empty): {avg_labels:.2f}")
        print(f"  Total labels: {total_labels}")
        
        self.predicted_labels = final_labels
        
        return final_labels
    
    def save_labels(
        self,
        output_path: str,
        format: str = 'json'
    ) -> None:
        """Save generated labels to file."""
        if not hasattr(self, 'predicted_labels') or not self.predicted_labels:
            print("Warning: No labels to save. Run generate_labels() first.")
            return
        
        if format == 'json':
            output = {
                str(doc_id): sorted(list(class_ids))
                for doc_id, class_ids in self.predicted_labels.items()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2)
        
        elif format == 'pickle':
            import pickle
            output = {
                doc_id: sorted(list(class_ids))
                for doc_id, class_ids in self.predicted_labels.items()
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(output, f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"\nLabels saved to: {output_path} ({format} format)")
        print(f"  Total documents: {len(self.predicted_labels)}")
    
    def save_embeddings(self, output_dir: str) -> None:
        """Save embeddings to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        import pickle
        
        # Save node embeddings (final embeddings after propagation)
        if self.node_embeddings:
            node_emb_path = os.path.join(output_dir, "node_embeddings.pkl")
            with open(node_emb_path, 'wb') as f:
                pickle.dump(self.node_embeddings, f)
            print(f"Saved node embeddings to: {node_emb_path}")
        
        # Save local embeddings (before propagation)
        if self.local_embeddings:
            local_emb_path = os.path.join(output_dir, "local_embeddings.pkl")
            with open(local_emb_path, 'wb') as f:
                pickle.dump(self.local_embeddings, f)
            print(f"Saved local embeddings to: {local_emb_path}")
        
        # Save review embeddings
        if self.review_embeddings:
            review_emb_path = os.path.join(output_dir, "review_embeddings.pkl")
            with open(review_emb_path, 'wb') as f:
                pickle.dump(self.review_embeddings, f)
            print(f"Saved review embeddings to: {review_emb_path}")


def main():
    """Main function for top-down label generation."""
    print("=" * 70)
    print("Top-Down Label Generation (Graph-based Propagation)")
    print("=" * 70)
    
    # Load taxonomy
    print("\n[1/5] Loading taxonomy...")
    taxonomy = Taxonomy(
        hierarchy_path=config.TAXONOMY_FILE,
        classes_path=config.CLASSES_FILE
    )
    print(f"Loaded taxonomy with {taxonomy.num_classes} classes")
    
    # Load keywords
    print("\n[2/5] Loading keywords...")
    taxonomy.load_keywords(config.KEYWORDS_FILE)
    num_classes_with_keywords = len(taxonomy.keywords)
    print(f"Loaded keywords for {num_classes_with_keywords} classes")
    
    # Load training dataset
    print("\n[3/5] Loading training dataset...")
    train_dataset = AmazonDataset(
        corpus_path=config.TRAIN_FILE,
        mode='raw'
    )
    print(f"Loaded {len(train_dataset)} training documents")
    
    # Initialize label generator
    print("\n[4/5] Initializing Top-Down Label Generator...")
    label_generator = TopDownLabelGenerator(
        taxonomy=taxonomy,
        dataset=train_dataset,
        sbert_model_name=config.GENERATOR_SBERT_MODEL,
        device=config.DEVICE,
        lambda_weight=config.GENERATOR_LAMBDA_WEIGHT
    )
    
    # Step 1: Generate class embeddings (graph-based propagation)
    class_embeddings = label_generator.generate_class_embeddings()
    
    # Step 2: Generate review embeddings
    review_embeddings = label_generator.generate_review_embeddings()
    
    # Step 3: Compute similarities
    similarities = label_generator.compute_similarities()
    
    # Step 4: Generate labels
    print("\n" + "=" * 70)
    print("Label Generation")
    print("=" * 70)
    labels = label_generator.generate_labels(
        similarities=similarities,
        conflict_threshold=config.GENERATOR_CONFLICT_THRESHOLD,
        show_progress=True
    )
    
    # Save embeddings
    print("\nSaving embeddings...")
    config.ensure_output_dir()
    embeddings_dir = os.path.join(config.OUTPUT_DIR, os.path.basename(config.EMBEDDINGS_DIR))
    os.makedirs(embeddings_dir, exist_ok=True)
    label_generator.save_embeddings(embeddings_dir)
    
    # Save labels
    print("\nSaving generated labels...")
    labels_path = config.LABELS_PATH
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    label_generator.save_labels(labels_path, format='json')
    
    print("\n" + "=" * 70)
    print("Top-Down Label Generation Complete!")
    print("=" * 70)
    print(f"  Nodes processed: {len(class_embeddings)}")
    print(f"  Reviews processed: {len(review_embeddings)}")
    print(f"  Documents with labels: {sum(1 for v in labels.values() if v)}/{len(labels)}")
    print(f"  Embeddings saved to: {embeddings_dir}")
    print(f"  Labels saved to: {labels_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()


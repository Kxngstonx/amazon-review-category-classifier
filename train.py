from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import *
import argparse
import os
import json
import pickle
from transformers import AutoTokenizer
from data_loader import Taxonomy, AmazonDataset
from config import config


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR, help='output directory for embeddings and labels')
    parser.add_argument('--embeddings_dir', type=str, default=config.EMBEDDINGS_DIR, help='directory containing class embeddings (.pkl)')
    parser.add_argument('--labels_path', type=str, default=config.LABELS_PATH, help='path to labels JSON file')
    parser.add_argument('--corpus_path', type=str, default=config.TRAIN_FILE, help='path to training corpus')
    parser.add_argument('--taxonomy_path', type=str, default=config.TAXONOMY_FILE, help='path to taxonomy hierarchy file')
    parser.add_argument('--classes_path', type=str, default=config.CLASSES_FILE, help='path to classes file')
    parser.add_argument('--model_name', type=str, default=config.BERT_MODEL_NAME, help='BERT model name')
    parser.add_argument('--max_length', type=int, default=config.MAX_LENGTH, help='maximum sequence length for tokenization')
    parser.add_argument('--batch_size', default=config.TRAIN_BATCH_SIZE, type=int)
    parser.add_argument('--epoch', default=config.TRAIN_EPOCHS, type=int)
    parser.add_argument('--lr', default=config.TRAIN_LR, type=float)
    parser.add_argument('--gpu', default=config.TRAIN_GPU, type=int)
    parser.add_argument('--save_dir', type=str, default=config.TRAIN_SAVE_DIR, help='directory to save trained model')
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

    # Load tokenizer (matching top_down_label.py style - using roberta-base from config)
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("Tokenizer loaded successfully")

    # Load training dataset with tokenization
    print("Loading training dataset...")
    train_dataset = AmazonDataset(
        corpus_path=args.corpus_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        truncation=True,
        padding='max_length',
        mode='train',
        labels_path=args.labels_path,
        taxonomy=taxonomy,
        filter_empty_labels=False  # Keep all samples, use sample_mask to filter
    )
    print(f"Loaded {len(train_dataset)} training documents")

    # Prepare training data tensors
    print("Preparing training data tensors...")
    input_ids_list = []
    attention_masks_list = []
    labels_list = []
    sample_mask_list = []

    # Get all class IDs sorted to create consistent label tensor
    all_class_ids = sorted(taxonomy.class_id_to_name.keys())
    class_id_to_index = {class_id: idx for idx, class_id in enumerate(all_class_ids)}
    num_classes = len(all_class_ids)

    for item in tqdm(train_dataset, desc="Processing dataset", miniters=5000):
        input_ids_list.append(item['input_ids'])
        attention_masks_list.append(item['attention_mask'])
        
        # Get label tensor if available, otherwise create zero tensor
        doc_id = item['index'].item()
        if doc_id in train_dataset.label_tensors:
            label_tensor = train_dataset.label_tensors[doc_id]
        else:
            label_tensor = torch.zeros(num_classes, dtype=torch.float32)
        
        labels_list.append(label_tensor)
        
        # sample_mask: 1 if document has labels, 0 otherwise
        has_labels = doc_id in train_dataset.labels and len(train_dataset.labels[doc_id]) > 0
        sample_mask_list.append(torch.ones(num_classes, dtype=torch.float32) if has_labels else torch.zeros(num_classes, dtype=torch.float32))

    # Stack tensors
    input_ids = torch.stack(input_ids_list)
    attention_masks = torch.stack(attention_masks_list)
    labels = torch.stack(labels_list)
    sample_mask = torch.stack(sample_mask_list)

    print(f"Prepared tensors:")
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  attention_masks shape: {attention_masks.shape}")
    print(f"  labels shape: {labels.shape}")
    print(f"  sample_mask shape: {sample_mask.shape}")
    print(f"  Samples with labels: {sample_mask.sum().item() / num_classes:.0f}")

    # Create dataset and data loader
    dataset = TensorDataset(input_ids, attention_masks, labels, sample_mask)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Load class embeddings from .pkl file
    print(f"Loading class embeddings from {args.embeddings_dir}...")
    node_embeddings_path = os.path.join(args.embeddings_dir, 'node_embeddings.pkl')
    
    if not os.path.exists(node_embeddings_path):
        raise FileNotFoundError(f"Class embeddings file not found: {node_embeddings_path}")
    
    with open(node_embeddings_path, 'rb') as f:
        node_embeddings_dict = pickle.load(f)
    
    print(f"Loaded embeddings for {len(node_embeddings_dict)} classes")
    
    # Convert class embeddings to tensor in the same order as taxonomy
    # node_embeddings_dict: {class_id: np.ndarray}
    class_emb_list = []
    for class_id in all_class_ids:
        if class_id in node_embeddings_dict:
            emb = node_embeddings_dict[class_id]
            if isinstance(emb, np.ndarray):
                class_emb_list.append(torch.from_numpy(emb).float())
            else:
                class_emb_list.append(torch.tensor(emb, dtype=torch.float32))
        else:
            # If class_id not in embeddings, use zero vector
            # Get embedding dimension from first available embedding
            if class_emb_list:
                emb_dim = class_emb_list[0].shape[0]
            else:
                # Default to 768 if no embeddings found
                emb_dim = 768
            class_emb_list.append(torch.zeros(emb_dim, dtype=torch.float32))
            if class_id != config.VIRTUAL_ROOT_ID:
                print(f"Warning: No embedding found for class_id {class_id}, using zero vector")
    
    class_emb = torch.stack(class_emb_list)
    print(f"Class embeddings tensor shape: {class_emb.shape}")

    # Get embedding dimension
    emb_dim = class_emb.shape[1]
    
    # Initialize model
    print(f"Initializing model: {args.model_name} with embedding dim {emb_dim}")
    model = ClassModel(args.model_name, emb_dim, class_emb).to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.MODEL_WEIGHT_DECAY,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=config.MODEL_EPS)
    loss_fn = multilabel_bce_loss_w

    # Training loop
    model.zero_grad()
    for e in range(args.epoch):
        print(f'Training epoch: {e}')
        total_train_loss = 0
        num_batches = 0
        for j, batch in enumerate(tqdm(data_loader)):
            input_ids_batch = batch[0].to(device)
            input_mask_batch = batch[1].to(device)
            labels_batch = batch[2].to(device)
            sample_mask_batch = batch[3].to(device)
            
            output = model(input_ids_batch, input_mask_batch)
            
            loss = loss_fn(output, labels_batch, sample_mask_batch)
            total_train_loss += loss.item()
            num_batches += 1
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.MODEL_GRAD_CLIP)
            optimizer.step()
            model.zero_grad()

        avg_loss = total_train_loss / num_batches if num_batches > 0 else 0
        print(f'Epoch {e} average loss: {avg_loss:.4f}')

    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
import torch
import json
import logging
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import numpy as np
import random
from typing import List, Dict, Optional, Tuple
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RelationConfig:
    """Configuration for relation classification model"""
    def __init__(self, relation_types: Optional[List[str]] = None, class_weights: Optional[List[float]] = None):
        # Model architecture
        self.max_span_length = 12
        self.span_hidden_size = 150
        self.dropout = 0.1
        self.num_entity_classes = 5
        self.entity_class_embedding_size = 50

        # Relation types - filter out 'no_relation' if present
        if relation_types:
            # Remove 'no_relation' from relation_types if it exists
            self.relation_types = [rel for rel in relation_types if rel.lower() != 'no_relation']
        else:
            self.relation_types = ["BEGINS-ON", "CONTAINS-1", "ENDS-ON", "no_relation"]

        # Create label map: actual relations get indices 0, 1, 2, ...
        # 'no_relation' will implicitly get the last index (len(relation_types))
        self.relation_label_map = {rel: idx for idx, rel in enumerate(self.relation_types)}

        # Add 'no_relation' mapping to the last index
        self.no_relation_idx = len(self.relation_types)

        # Class weights for handling imbalanced data
        if class_weights is None:
            # Default: equal weights for all classes including no_relation
            self.class_weights = [1.0] * (len(self.relation_types) + 1)
        else:
            # Ensure we have weights for all classes (relations + no_relation)
            expected_classes = len(self.relation_types) + 1
            if len(class_weights) != expected_classes:
                logger.warning(f"Class weights length {len(class_weights)} doesn't match number of classes {expected_classes}")
                self.class_weights = [1.0] * expected_classes
            else:
                self.class_weights = class_weights

        # Default entity classes - will be updated if entity config is loaded
        self.entity_class_map = {i: f"CLASS_{i}" if i > 0 else "O" for i in range(self.num_entity_classes)}
        self.text_to_id_map = {v: k for k, v in self.entity_class_map.items()}  # Reverse mapping

        logger.info(f"Config: {len(self.relation_types)} relation types: {self.relation_types}")
        logger.info(f"'no_relation' will be mapped to index: {self.no_relation_idx}")
        logger.info(f"Class weights: {self.class_weights}")

    def get_relation_label(self, relation_text):
        """Get the numeric label for a relation"""
        if relation_text.lower() == 'no_relation':
            return self.no_relation_idx
        return self.relation_label_map.get(relation_text, self.no_relation_idx)

    def update_entity_config(self, entity_info):
        """Update entity configuration from loaded entity model config"""
        if entity_info:
            self.num_entity_classes = entity_info['num_entity_classes']
            self.entity_class_map = entity_info['entity_class_map']
            self.text_to_id_map = entity_info.get('text_to_id_map', {})
            logger.info(f"Updated entity config: {self.num_entity_classes} classes")

    def get_entity_class_id(self, class_value):
        """Convert entity class (text or numeric) to class ID"""
        # If it's already a number, return it
        if isinstance(class_value, (int, float)):
            return int(class_value)

        # If it's a string that represents a number, convert it
        if isinstance(class_value, str):
            try:
                return int(class_value)
            except ValueError:
                pass

            # If it's a text class name, look it up
            if class_value in self.text_to_id_map:
                return self.text_to_id_map[class_value]
            else:
                logger.warning(f"Unknown entity class '{class_value}', using default class 1")
                return 1

        # Default fallback
        logger.warning(f"Invalid entity class type '{type(class_value)}', using default class 1")
        return 1

class RelationDataset(Dataset):
    """Dataset for relation classification"""
    def __init__(self, data, tokenizer, config, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        # Use first relation type as default, or fallback to no_relation
        self.default_relation = config.relation_types[0] if config.relation_types else "no_relation"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]

        # Extract basic info
        sentence = record["source_sentence"]
        sentence_start = record["source_sentence_start"]

        # Entity info with flexible class handling
        subj_text = record["subject_text"]
        subj_start = record["subject_start"] - sentence_start
        subj_end = record["subject_end"] - sentence_start
        subj_class = self.config.get_entity_class_id(record.get("subject_class", "EVENT"))

        obj_text = record["object_text"]
        obj_start = record["object_start"] - sentence_start
        obj_end = record["object_end"] - sentence_start
        obj_class = self.config.get_entity_class_id(record.get("object_class", "TIMEX3"))

        # Relation info - USE THE NEW METHOD
        relation_type = record.get("relation", self.default_relation)
        relation_idx = self.config.get_relation_label(relation_type)

        # Tokenize
        tokenized = self.tokenizer(
            sentence, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )

        # Convert to token spans
        subj_token_span = self._char_to_token_span(sentence, subj_text, subj_start)
        obj_token_span = self._char_to_token_span(sentence, obj_text, obj_start)

        if None in subj_token_span or None in obj_token_span:
            return self._create_dummy_sample(tokenized)

        return {
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "token_type_ids": tokenized["token_type_ids"][0],
            "spans": torch.tensor([subj_token_span, obj_token_span], dtype=torch.long),
            "span_classes": torch.tensor([subj_class, obj_class], dtype=torch.long),
            "candidate_relations": torch.tensor([[0, 1]], dtype=torch.long),
            "relation_labels": torch.tensor([relation_idx], dtype=torch.long)
        }

    def _char_to_token_span(self, sentence, entity_text, entity_start):
        """Convert character span to token span"""
        text_before = sentence[:entity_start]
        tokens_before = self.tokenizer.tokenize(text_before)
        entity_tokens = self.tokenizer.tokenize(entity_text)

        token_start = len(tokens_before) + 1  # +1 for [CLS]
        token_end = token_start + len(entity_tokens) - 1

        return (token_start, token_end) if token_end < self.max_length else (None, None)

    def _create_dummy_sample(self, tokenized):
        """Create dummy sample for invalid cases"""
        return {
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "token_type_ids": tokenized["token_type_ids"][0],
            "spans": torch.zeros((2, 2), dtype=torch.long),
            "span_classes": torch.zeros(2, dtype=torch.long),
            "candidate_relations": torch.zeros((1, 2), dtype=torch.long),
            "relation_labels": torch.full((1,), -100, dtype=torch.long)
        }

class RelationClassifier(nn.Module):
    """Enhanced relation classification model with improved context extraction"""
    def __init__(self, model_name, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(model_name)

        bert_size = self.bert.config.hidden_size
        self.width_embedding = nn.Embedding(config.max_span_length + 1, config.span_hidden_size)
        self.entity_class_embedding = nn.Embedding(config.num_entity_classes, config.entity_class_embedding_size)

        # Classifier input size calculation
        span_rep_size = bert_size * 2 + config.span_hidden_size + config.entity_class_embedding_size
        classifier_input_size = span_rep_size * 2 + bert_size  # subj + obj + context

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, config.span_hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.span_hidden_size * 2, config.span_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.span_hidden_size, len(config.relation_types) + 1)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, spans, span_classes, candidate_relations, relation_labels=None):
        # Get BERT representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state

        # Extract span representations
        span_reps = self._extract_span_representations(sequence_output, spans, span_classes)

        # Classify relations with improved context
        relation_logits = self._classify_relations(sequence_output, span_reps, spans, candidate_relations)

        # Calculate loss if training
        loss = None
        if relation_labels is not None:
            # Create class weights tensor
            class_weights = torch.tensor(self.config.class_weights, device=relation_logits.device, dtype=torch.float)

            loss = F.cross_entropy(
                relation_logits.view(-1, len(self.config.relation_types) + 1),
                relation_labels.view(-1),
                weight=class_weights,
                ignore_index=-100
            )

        return {"loss": loss, "relation_logits": relation_logits}

    def _extract_span_representations(self, sequence_output, spans, span_classes):
        """Extract span representations with entity class info"""
        batch_size, num_spans = spans.shape[:2]

        # Get span boundaries
        span_starts, span_ends = spans[:, :, 0], spans[:, :, 1]

        # Extract start/end embeddings
        start_embs = self._batched_gather(sequence_output, span_starts)
        end_embs = self._batched_gather(sequence_output, span_ends)

        # Width and class embeddings
        widths = torch.clamp(span_ends - span_starts + 1, max=self.config.max_span_length)
        width_embs = self.width_embedding(widths)
        class_embs = self.entity_class_embedding(span_classes)

        return torch.cat([start_embs, end_embs, width_embs, class_embs], dim=-1)

    def _classify_relations(self, sequence_output, span_reps, spans, candidate_relations):
        """Enhanced relation classification with improved context extraction"""
        batch_size, num_candidates = candidate_relations.shape[:2]

        # Get subject/object representations
        subj_indices = candidate_relations[:, :, 0]
        obj_indices = candidate_relations[:, :, 1]
        subj_reps = self._batched_index_select(span_reps, subj_indices)
        obj_reps = self._batched_index_select(span_reps, obj_indices)

        # Extract context using Method 1: Between-token context
#         context_reps = self._extract_context_representations(sequence_output, spans, candidate_relations)

        # Method 2: Attention-weighted context (comment out Method 1 if using this)
        context_reps = self._extract_attention_weighted_context(sequence_output, spans, candidate_relations)

        # Concatenate all features
        relation_reps = torch.cat([subj_reps, context_reps, obj_reps], dim=-1)

        return self.classifier(relation_reps)

    def _extract_context_representations(self, sequence_output, spans, candidate_relations):
        """
        Method 1: Extract context between entity pairs
        """
        batch_size, num_candidates = candidate_relations.shape[:2]
        hidden_size = sequence_output.size(-1)

        # Get subject and object spans
        subj_indices = candidate_relations[:, :, 0]  # [batch, num_candidates]
        obj_indices = candidate_relations[:, :, 1]   # [batch, num_candidates]

        # Get span boundaries for subjects and objects
        subj_spans = self._batched_index_select(spans, subj_indices)  # [batch, num_candidates, 2]
        obj_spans = self._batched_index_select(spans, obj_indices)    # [batch, num_candidates, 2]

        context_reps = []

        for b in range(batch_size):
            batch_contexts = []
            for c in range(num_candidates):
                subj_start, subj_end = subj_spans[b, c, 0].item(), subj_spans[b, c, 1].item()
                obj_start, obj_end = obj_spans[b, c, 0].item(), obj_spans[b, c, 1].item()

                # Extract context between entities
                context_rep = self._extract_between_context(
                    sequence_output[b], subj_start, subj_end, obj_start, obj_end
                )

                batch_contexts.append(context_rep)

            context_reps.append(torch.stack(batch_contexts))

        return torch.stack(context_reps)

    def _extract_between_context(self, sequence_output, subj_start, subj_end, obj_start, obj_end):
        """
        Extract context representation between two entities
        """
        # Determine the order of entities in the sentence
        if subj_end < obj_start:
            # Subject comes before object: use tokens between them
            context_start = subj_end + 1
            context_end = obj_start - 1
        elif obj_end < subj_start:
            # Object comes before subject: use tokens between them
            context_start = obj_end + 1
            context_end = subj_start - 1
        else:
            # Overlapping entities: use surrounding context
            return self._extract_surrounding_context(sequence_output, subj_start, subj_end, obj_start, obj_end)

        # If there are tokens between entities
        if context_start <= context_end and context_end < sequence_output.size(0):
            between_tokens = sequence_output[context_start:context_end+1]
            # Average pooling of between tokens
            if between_tokens.size(0) > 0:
                return between_tokens.mean(dim=0)

        # Fallback: return zero vector if no between tokens
        return torch.zeros(sequence_output.size(-1), device=sequence_output.device)

    def _extract_surrounding_context(self, sequence_output, subj_start, subj_end, obj_start, obj_end):
        """
        Extract surrounding context when entities overlap or are adjacent
        """
        # Find the overall span covering both entities
        min_start = min(subj_start, obj_start)
        max_end = max(subj_end, obj_end)

        # Context window size
        context_window = 3

        # Left context
        left_start = max(1, min_start - context_window)  # Skip [CLS]
        left_end = min_start - 1
        left_context = sequence_output[left_start:left_end+1] if left_start <= left_end else torch.empty(0, sequence_output.size(-1), device=sequence_output.device)

        # Right context
        right_start = max_end + 1
        right_end = min(sequence_output.size(0) - 1, max_end + context_window)  # Avoid [SEP]
        right_context = sequence_output[right_start:right_end+1] if right_start <= right_end else torch.empty(0, sequence_output.size(-1), device=sequence_output.device)

        # Combine contexts
        if left_context.size(0) > 0 and right_context.size(0) > 0:
            combined_context = torch.cat([left_context, right_context], dim=0)
        elif left_context.size(0) > 0:
            combined_context = left_context
        elif right_context.size(0) > 0:
            combined_context = right_context
        else:
            return torch.zeros(sequence_output.size(-1), device=sequence_output.device)

        return combined_context.mean(dim=0)

    def _extract_attention_weighted_context(self, sequence_output, spans, candidate_relations):
        """
        Method 2: Use attention mechanism to weight context tokens
        Fixed version that properly handles batch dimensions
        """
        batch_size, num_candidates = candidate_relations.shape[:2]
        seq_len = sequence_output.size(1)
        hidden_size = sequence_output.size(-1)

        # Initialize attention layer if not exists
        if not hasattr(self, 'context_attention'):
            self.context_attention = nn.Linear(hidden_size, 1).to(sequence_output.device)

        context_reps = []

        for b in range(batch_size):
            batch_contexts = []
            for c in range(num_candidates):
                subj_idx = candidate_relations[b, c, 0].item()
                obj_idx = candidate_relations[b, c, 1].item()

                subj_start, subj_end = spans[b, subj_idx, 0].item(), spans[b, subj_idx, 1].item()
                obj_start, obj_end = spans[b, obj_idx, 0].item(), spans[b, obj_idx, 1].item()

                # Create mask for context tokens (exclude entity tokens)
                context_mask = torch.ones(seq_len, dtype=torch.bool, device=sequence_output.device)

                # Mask out entity tokens
                if subj_start < seq_len and subj_end < seq_len:
                    context_mask[subj_start:subj_end+1] = False
                if obj_start < seq_len and obj_end < seq_len:
                    context_mask[obj_start:obj_end+1] = False
                context_mask[0] = False  # [CLS]
                if seq_len > 1:
                    context_mask[-1] = False  # [SEP]

                # Get context tokens for this specific batch item
                context_tokens = sequence_output[b][context_mask]  # [num_context_tokens, hidden_size]

                if context_tokens.size(0) > 0:
                    # Apply attention weighting
                    attention_scores = self.context_attention(context_tokens).squeeze(-1)  # [num_context_tokens]
                    attention_weights = F.softmax(attention_scores, dim=0)  # [num_context_tokens]
                    context_rep = (context_tokens * attention_weights.unsqueeze(-1)).sum(dim=0)  # [hidden_size]
                else:
                    context_rep = torch.zeros(hidden_size, device=sequence_output.device)

                batch_contexts.append(context_rep)

            context_reps.append(torch.stack(batch_contexts))

        return torch.stack(context_reps)

    def _batched_gather(self, source, indices):
        """Gather operation with batch support"""
        batch_size, num_indices = indices.size()
        batch_indices = torch.arange(batch_size, device=source.device).unsqueeze(-1).expand(-1, num_indices)
        return source[batch_indices, indices]

    def _batched_index_select(self, source, indices):
        """Index select with batch support"""
        return self._batched_gather(source, indices)

def collate_fn(batch):
    """Custom collate function"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    collated = {key: torch.stack([item[key] for item in batch]) for key in batch[0].keys()}
    return collated

def load_config(entity_config_path=None):
    """Load entity configuration and handle both numeric IDs and text class names"""
    if not entity_config_path or not os.path.exists(entity_config_path):
        return None

    try:
        with open(entity_config_path, 'r') as f:
            entity_config = json.load(f)

        # Get basic info
        num_classes = entity_config.get('num_entity_classes', 5)
        class_map = entity_config.get('class_map', {})

        # Convert class_map to consistent format: {int_id: text_name}
        normalized_class_map = {}
        text_to_id_map = {}  # For reverse lookup

        for key, value in class_map.items():
            class_id = int(key)
            class_name = str(value)
            normalized_class_map[class_id] = class_name
            text_to_id_map[class_name] = class_id

        # Add class 0 if not present (typically for non-entity)
        if 0 not in normalized_class_map:
            normalized_class_map[0] = "O"
            text_to_id_map["O"] = 0

        logger.info(f"Loaded entity config with {num_classes} classes:")
        for class_id, class_name in normalized_class_map.items():
            logger.info(f"  {class_id}: {class_name}")

        return {
            'num_entity_classes': max(num_classes, len(normalized_class_map)),
            'entity_class_map': normalized_class_map,
            'text_to_id_map': text_to_id_map
        }
    except Exception as e:
        logger.warning(f"Failed to load entity config: {e}")
        return None

def create_balanced_batches(data, batch_size):
    """Create balanced batches across relation types"""
    # Group by relation type
    relation_groups = {}
    for idx, record in enumerate(data):
        relation = record.get("relation", "no_relation")
        relation_groups.setdefault(relation, []).append(idx)

    # Shuffle within groups
    for indices in relation_groups.values():
        random.shuffle(indices)

    # Create balanced batches
    relations = list(relation_groups.keys())
    if not relations:
        return []

    batches = []
    pointers = {rel: 0 for rel in relations}
    examples_per_relation = max(1, batch_size // len(relations))

    while any(pointers[rel] < len(relation_groups[rel]) for rel in relations):
        batch = []
        for rel in relations:
            if pointers[rel] < len(relation_groups[rel]):
                start_idx = pointers[rel]
                end_idx = min(start_idx + examples_per_relation, len(relation_groups[rel]))
                batch.extend(relation_groups[rel][start_idx:end_idx])
                pointers[rel] = end_idx

        if batch:
            random.shuffle(batch)
            batches.append(batch)

        # Break if all relations are exhausted
        if all(pointers[rel] >= len(relation_groups[rel]) for rel in relations):
            break

    return batches

def evaluate_model(model, dataloader, config, device):
    """Evaluate model and return metrics"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch is None:
                continue

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            preds = torch.argmax(outputs["relation_logits"], dim=-1)
            valid_mask = batch["relation_labels"] != -100

            all_preds.extend(preds[valid_mask].cpu().numpy())
            all_labels.extend(batch["relation_labels"][valid_mask].cpu().numpy())

    if not all_preds:
        return 0.0, "No valid predictions"

    # Define all possible classes and their names
    all_possible_classes = list(range(len(config.relation_types) + 1))
    all_target_names = config.relation_types + ["no_relation"]

    # Calculate F1 score
    f1 = f1_score(all_labels, all_preds, average='macro', labels=all_possible_classes, zero_division=0)

    # Generate classification report
    report = classification_report(
        all_labels, all_preds,
        labels=all_possible_classes,
        target_names=all_target_names,
        zero_division=0
    )

    return f1, report

def train_relation_model(train_data, val_data, relation_types, model_name='dmis-lab/biobert-base-cased-v1.1',
                        num_epochs=10, batch_size=8, learning_rate=3e-5, model_dir="models",
                        entity_config_path=None, class_weights=None):
    """Main training function"""
    # Setup
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = RelationConfig(relation_types, class_weights)

    # Load entity config if provided
    entity_info = load_config(entity_config_path)
    if entity_info:
        config.update_entity_config(entity_info)

    # Create datasets and dataloaders
    train_dataset = RelationDataset(train_data, tokenizer, config)
    val_dataset = RelationDataset(val_data, tokenizer, config)

    train_batches = create_balanced_batches(train_data, batch_size)
    val_batches = create_balanced_batches(val_data, batch_size)

    train_loader = DataLoader(train_dataset, batch_sampler=train_batches, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_sampler=val_batches, collate_fn=collate_fn)

    # Model setup
    model = RelationClassifier(model_name, config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    logger.info(f"Model loaded on device: {device}")
    logger.info(f"Improved context extraction (Method 1: Between-token context) enabled")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    # Create model directory and save config
    os.makedirs(model_dir, exist_ok=True)
    config_dict = {
        "relation_types": config.relation_types,
        "num_entity_classes": config.num_entity_classes,
        "entity_class_map": {str(k): v for k, v in config.entity_class_map.items()},
#         "context_method": "between_token_context"  # Document which method is used
        "context_method": "attention_weighted_context"
    }

    with open(os.path.join(model_dir, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Training loop
    best_f1 = 0
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            if batch is None:
                continue

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Training Loss: {avg_loss:.4f}")

        # Validation
        f1, report = evaluate_model(model, val_loader, config, device)
        logger.info(f"Epoch {epoch+1} - Validation F1: {f1:.4f}")
        logger.info(f"Classification Report:\n{report}")

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
            logger.info(f"Saved new best model with F1: {f1:.4f}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(model_dir, "final_model.pt"))
    logger.info("Training completed!")
    logger.info(f"Best validation F1: {best_f1:.4f}")

    return model, config

def analyze_dataset(data):
    """Analyze dataset and extract relation types and entity classes"""

    print(f"Analyzing {len(data)} records")
    print("="*60)

    # Collect statistics
    relations = []
    subject_classes = []
    object_classes = []

    for record in data:
        relations.append(record.get('relation', 'UNKNOWN'))
        subject_classes.append(record.get('subject_class', 0))
        object_classes.append(record.get('object_class', 0))

    # Count occurrences
    relation_counts = Counter(relations)
    subject_class_counts = Counter(subject_classes)
    object_class_counts = Counter(object_classes)

    # Print results
    print("RELATION TYPES:")
    for relation, count in relation_counts.most_common():
        print(f"  {relation}: {count} examples")

    print(f"\nENTITY CLASSES:")
    all_classes = sorted(set(subject_classes + object_classes))
    print(f"  Classes found: {all_classes}")
    print(f"  Subject classes: {dict(subject_class_counts)}")
    print(f"  Object classes: {dict(object_class_counts)}")

    # Generate command line arguments
    relation_types = sorted(relation_counts.keys())
    unique_classes = sorted(set(subject_classes + object_classes))

    print(f"\nFOR YOUR TRAINING COMMAND:")
    print("-"*60)
    print("Relation types argument:")
    print(f"{' '.join(relation_types)}")

    entity_config = {
        "num_entity_classes": len(unique_classes),
        "class_map": {str(cls): f"CLASS_{cls}" for cls in unique_classes}
    }

    return relation_types, unique_classes, entity_config

def main():
    parser = argparse.ArgumentParser(description="Train improved relation classification model with context extraction")
    parser.add_argument("--train", required=True, help="Training data JSON file")
    parser.add_argument("--val", required=True, help="Validation data JSON file")
    parser.add_argument("--relation_types", required=True, nargs='+', help="Relation types to classify")
    parser.add_argument("--model_dir", default="models", help="Model save directory")
    parser.add_argument("--base_model", default="dmis-lab/biobert-base-cased-v1.1", help="Base model")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--entity_config", default=None, help="Entity model config path")
    parser.add_argument("--class_weights", nargs='+', type=float, default=None,
                       help="Class weights for handling imbalanced data")

    args = parser.parse_args()

    # Validate and load data
    if not args.relation_types:
        raise ValueError("Must provide at least one relation type")

    # Filter out 'no_relation' from command line args (it's handled automatically)
    filtered_relation_types = [rel for rel in args.relation_types if rel.lower() != 'no_relation']

    # Validate class weights if provided
    if args.class_weights:
        expected_num_classes = len(filtered_relation_types) + 1  # +1 for no_relation
        if len(args.class_weights) != expected_num_classes:
            raise ValueError(f"Number of class weights ({len(args.class_weights)}) must match number of classes ({expected_num_classes})")
        logger.info(f"Using class weights: {args.class_weights}")

    with open(args.train, 'r') as f:
        train_data = json.load(f)
    with open(args.val, 'r') as f:
        val_data = json.load(f)

    logger.info(f"Loaded {len(train_data)} train, {len(val_data)} val examples")
    logger.info(f"Training for relations: {filtered_relation_types}")
    logger.info("Using improved context extraction: Method 1 (Between-token context)")

    # Train model
    train_relation_model(
        train_data, val_data, filtered_relation_types,
        model_name=args.base_model, num_epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        model_dir=args.model_dir, entity_config_path=args.entity_config,
        class_weights=args.class_weights
    )

if __name__ == "__main__":
    main()

# Chemotherapy TLINK Detection

A deep learning approach for temporal relation classification in chemotherapy clinical notes using fine-tuned BioBERT. This project classifies temporal relationships between medical events and time expressions in doctor's notes.

## Overview

This project implements a relation classification model that:
- Takes pairs of entities (events and time expressions) with their positions in clinical text
- Predicts temporal relationships (BEGINS-ON, CONTAINS, ENDS-ON, or no_relation) between entity pairs
- Uses BioBERT as the base model with custom span representation and context extraction

## Dataset

The model is designed for chemotherapy temporal annotation datasets containing:
- Clinical notes (doctor's notes)
- Entity annotations (EVENTs and TIMEX3 time expressions)
- TLINK temporal relations between entities
- Gold standard annotations in XML format

**Note**: Dataset is not included in this repository.

## Repository Structure

```
chemotherapy-TLINK-detection/
├── src/
│   ├── chemo_processor_no_relation.py    # Data preprocessing
│   ├── relation_model.py                 # Model training
│   └── test_relation_model.py           # Model evaluation
├── models/
│   └── config.json                      # Model configuration
└── README.md                           # This file
```

## Usage

### 1. Data Preprocessing

Process your chemotherapy dataset to extract relations and entities:

```bash
python src/chemo_processor_no_relation.py
```

**Configuration options:**
- Modify `base_directory` path in the script to point to your dataset
- Adjust `no_relation_ratio` (default: 0.3) to control negative sampling
- Set `no_relation_ratio=0.0` to disable negative examples

**Input format:**
- Directory structure with cancer types (breast, melanoma, ovarian)
- Each patient folder contains text files and corresponding XML annotations
- XML files contain entity spans and TLINK relations

**Output:**
- `chemo_train_relations.jsonl` - Training relations
- `chemo_dev_relations.jsonl` - Development relations  
- `train_entities.jsonl` - Training entities for NER
- `dev_entities.jsonl` - Development entities for NER

### 2. Model Training

Train the relation classification model:

```bash
python src/relation_model.py \
    --train data/processed/chemo_train_relations.jsonl \
    --val data/processed/chemo_dev_relations.jsonl \
    --relation_types BEGINS-ON CONTAINS ENDS-ON \
    --model_dir models/ \
    --epochs 20 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --entity_config models/entity_config.json
```

**Key parameters:**
- `--relation_types`: Space-separated list of relation types (excluding no_relation)
- `--class_weights`: Optional weights for handling class imbalance
- `--base_model`: BioBERT model (default: dmis-lab/biobert-base-cased-v1.1)
- `--entity_config`: Path to entity configuration file

**Advanced options:**
```bash
# With class weighting for imbalanced data
python src/relation_model.py \
    --train data/processed/chemo_train_relations.jsonl \
    --val data/processed/chemo_dev_relations.jsonl \
    --relation_types BEGINS-ON CONTAINS ENDS-ON \
    --class_weights 2.0 1.0 2.0 0.5 \
    --model_dir models/ \
    --epochs 15 \
    --entity_config models/entity_config.json
```

### 3. Model Evaluation

Test the trained model:

```bash
python src/test_relation_model.py \
    --test_data data/processed/test_relations.jsonl \
    --model_dir models/ \
    --output_dir results/
```

**Evaluation features:**
- Comprehensive metrics (accuracy, F1-macro/micro/weighted)
- Per-class precision, recall, and F1 scores
- Confusion matrix visualization
- Special analysis for no_relation class performance

## Model Architecture

### Core Components

1. **BioBERT Encoder**: Pre-trained biomedical BERT for contextual representations
2. **Span Representation**: Combines start/end tokens, width embeddings, and entity class embeddings
3. **Context Extraction**: Two methods available:
   - Between-token context: Uses tokens between entity pairs
   - Attention-weighted context: Learns to attend to relevant context tokens
4. **Classification Head**: Multi-layer feedforward network for relation prediction

### Key Features

- **Entity Class Embeddings**: Incorporates entity type information (EVENT vs TIMEX3)
- **Span Width Encoding**: Captures entity span length information
- **Balanced Training**: Custom batch sampling and class weighting
- **Context-Aware**: Sophisticated context extraction beyond simple concatenation

## Configuration

### Data Processing Configuration

In `chemo_processor_no_relation.py`:
```python
# Adjust negative sampling ratio
processor = ChemotherapyDataProcessor(no_relation_ratio=0.9)

# Entity distance filtering
MAX_DISTANCE = 100  # characters between entities
```

## Results and Metrics

The model provides comprehensive evaluation metrics:

- **Overall Performance**: Accuracy, F1-macro/micro/weighted scores
- **Positive Relations**: F1 scores excluding no_relation class
- **Per-Class Analysis**: Precision, recall, F1 for each relation type
- **Confusion Matrix**: Detailed breakdown of classification errors
- **No-Relation Analysis**: Special focus on negative class performance

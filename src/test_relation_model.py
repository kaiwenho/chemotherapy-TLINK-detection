import torch
import json
import logging
import os
import argparse
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Import from the main relation model file
from improved_relation_model import (
    RelationConfig, RelationDataset, RelationClassifier,
    collate_fn, load_config
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RelationModelTester:
    """Class to handle testing of trained relation models"""
    
    def __init__(self, model_dir: str, model_name: str = 'dmis-lab/biobert-base-cased-v1.1'):
        """
        Initialize the tester

        Args:
            model_dir: Directory containing the trained model and config
            model_name: Base model name used for training
        """
        self.model_dir = model_dir
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load configuration
        self.config = self._load_config()

        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # Load model
        self.model = self._load_model()

        logger.info(f"Loaded model for testing with {len(self.config.relation_types)} relation types")
        logger.info(f"Relation types: {self.config.relation_types}")
        logger.info(f"'no_relation' class mapped to index: {self.config.no_relation_idx}")

    def _load_config(self) -> RelationConfig:
        """Load model configuration"""
        config_path = os.path.join(self.model_dir, "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # Create config object
        relation_types = config_dict.get("relation_types", [])
        config = RelationConfig(relation_types=relation_types)

        # Store context method information
        self.context_method = config_dict.get("context_method", "between_token_context")
        logger.info(f"Model was trained with context method: {self.context_method}")

        # Update entity configuration
        if "entity_class_map" in config_dict:
            entity_info = {
                'num_entity_classes': config_dict.get("num_entity_classes", 5),
                'entity_class_map': {int(k): v for k, v in config_dict["entity_class_map"].items()},
                'text_to_id_map': {v: int(k) for k, v in config_dict["entity_class_map"].items()}
            }
            config.update_entity_config(entity_info)

        return config

    def _load_model(self) -> RelationClassifier:
        """Load the trained model with proper architecture matching"""
        # Try to load the best model first, then fall back to final model
        model_paths = [
            os.path.join(self.model_dir, "best_model.pt"),
            os.path.join(self.model_dir, "final_model.pt")
        ]

        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            raise FileNotFoundError(f"No model found in {self.model_dir}")

        # Load the state dict first to check what's in it
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)

        # Check if the model has context_attention layers
        has_context_attention = any(key.startswith('context_attention') for key in state_dict.keys())

        if has_context_attention:
            logger.info("Model was trained with attention-weighted context method")
            self.actual_context_method = "attention_weighted_context"
        else:
            logger.info("Model was trained with between-token context method")
            self.actual_context_method = "between_token_context"

        # Create model with the correct architecture
        model = RelationClassifier(self.model_name, self.config)

        # If the model needs context_attention but doesn't have it, create it
        if has_context_attention and not hasattr(model, 'context_attention'):
            # Add the context_attention layer to match the saved model
            bert_size = model.bert.config.hidden_size
            model.context_attention = torch.nn.Linear(bert_size, 1).to(self.device)

        # Load the state dict
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            logger.error(f"Error loading model: {e}")
            logger.error("This suggests a mismatch between training and testing architectures")
            raise

        model.to(self.device)
        model.eval()

        logger.info(f"Loaded model from: {model_path}")
        return model
    def analyze_test_data_distribution(self, test_data: List[Dict]) -> Dict:
        """Analyze the distribution of relation types in test data"""
        relation_counts = Counter()

        for example in test_data:
            relation = example.get("relation", "no_relation")
            # Normalize relation name
            if relation.lower() == 'no_relation':
                relation = 'no_relation'
            relation_counts[relation] += 1

        total = len(test_data)
        distribution = {}

        logger.info(f"Test data distribution ({total} examples):")
        for relation in sorted(relation_counts.keys()):
            count = relation_counts[relation]
            percentage = (count / total) * 100
            distribution[relation] = {"count": count, "percentage": percentage}
            logger.info(f"  {relation}: {count} ({percentage:.1f}%)")

        return distribution

    def test_model(self, test_data: List[Dict], batch_size: int = 16, detailed_output: bool = True) -> Dict:
        """
        Test the model on test data

        Args:
            test_data: List of test examples
            batch_size: Batch size for testing
            detailed_output: Whether to include detailed per-example predictions

        Returns:
            Dictionary containing test results
        """
        logger.info(f"Testing model on {len(test_data)} examples")

        # Analyze test data distribution
        data_distribution = self.analyze_test_data_distribution(test_data)

        # Create dataset and dataloader
        test_dataset = RelationDataset(test_data, self.tokenizer, self.config)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        # Collect predictions and labels
        all_preds = []
        all_labels = []
        all_probs = []
        detailed_predictions = []

        self.model.eval()
        with torch.no_grad():
            example_idx = 0  # Track the actual example index

            for batch in tqdm(test_loader, desc="Testing"):
                if batch is None:
                    continue

                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                relation_logits = outputs["relation_logits"]

                # Get predictions and probabilities
                relation_probs = F.softmax(relation_logits, dim=-1)
                relation_preds = torch.argmax(relation_logits, dim=-1)

                # Collect valid predictions (skip dummy samples with label -100)
                valid_mask = batch["relation_labels"] != -100

                batch_preds = relation_preds[valid_mask].cpu().numpy()
                batch_labels = batch["relation_labels"][valid_mask].cpu().numpy()
                batch_probs = relation_probs[valid_mask].cpu().numpy()

                all_preds.extend(batch_preds)
                all_labels.extend(batch_labels)
                all_probs.extend(batch_probs)

                # Store detailed predictions if requested
                if detailed_output:
                    for i, (pred, label, probs) in enumerate(zip(batch_preds, batch_labels, batch_probs)):
                        if example_idx < len(test_data):
                            example = test_data[example_idx]
                            detailed_predictions.append({
                                "example_idx": example_idx,
                                "subject_text": example["subject_text"],
                                "object_text": example["object_text"],
                                "source_sentence": example.get("source_sentence", ""),
                                "true_relation": self._get_relation_name(label),
                                "predicted_relation": self._get_relation_name(pred),
                                "confidence": float(probs[pred]),
                                "all_probabilities": {
                                    self._get_relation_name(j): float(prob)
                                    for j, prob in enumerate(probs)
                                },
                                "correct": pred == label
                            })
                        example_idx += 1

        # Log prediction distribution
        self._log_prediction_distribution(all_preds, all_labels)

        # Calculate metrics
        results = self._calculate_metrics(all_labels, all_preds, all_probs)

        # Add data distribution to results
        results["test_data_distribution"] = data_distribution

        if detailed_output:
            results["detailed_predictions"] = detailed_predictions

        return results

    def _log_prediction_distribution(self, predictions: List[int], labels: List[int]):
        """Log the distribution of predictions vs true labels"""
        class_names = self.config.relation_types + ["no_relation"]

        pred_counts = Counter(predictions)
        label_counts = Counter(labels)

        logger.info("Prediction distribution:")
        for i, class_name in enumerate(class_names):
            pred_count = pred_counts.get(i, 0)
            label_count = label_counts.get(i, 0)
            pred_pct = (pred_count / len(predictions)) * 100 if predictions else 0
            label_pct = (label_count / len(labels)) * 100 if labels else 0
            logger.info(f"  {class_name}: Predicted {pred_count} ({pred_pct:.1f}%), True {label_count} ({label_pct:.1f}%)")

    def _get_relation_name(self, relation_idx: int) -> str:
        """Convert relation index to name"""
        if relation_idx < len(self.config.relation_types):
            return self.config.relation_types[relation_idx]
        elif relation_idx == self.config.no_relation_idx:
            return "no_relation"
        else:
            logger.warning(f"Unknown relation index: {relation_idx}")
            return f"unknown_{relation_idx}"

    def _calculate_metrics(self, true_labels: List[int], pred_labels: List[int], pred_probs: List[List[float]]) -> Dict:
        """Calculate comprehensive metrics"""
        # Define all possible classes
        all_classes = list(range(len(self.config.relation_types) + 1))
        class_names = self.config.relation_types + ["no_relation"]

        # Basic metrics
        accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))

        # F1 scores
        f1_macro = f1_score(true_labels, pred_labels, labels=all_classes, average='macro', zero_division=0)
        f1_micro = f1_score(true_labels, pred_labels, labels=all_classes, average='micro', zero_division=0)
        f1_weighted = f1_score(true_labels, pred_labels, labels=all_classes, average='weighted', zero_division=0)

        # Per-class metrics
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            true_labels, pred_labels, labels=all_classes, zero_division=0
        )

        # Classification report
        class_report = classification_report(
            true_labels, pred_labels,
            labels=all_classes,
            target_names=class_names,
            zero_division=0,
            output_dict=True
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(true_labels, pred_labels, labels=all_classes)

        # Calculate metrics excluding no_relation for comparison
        positive_classes = list(range(len(self.config.relation_types)))  # Exclude no_relation

        # Filter data for positive relations only
        positive_indices = [i for i, label in enumerate(true_labels) if label in positive_classes]

        if positive_indices:
            positive_true = [true_labels[i] for i in positive_indices]
            positive_pred = [pred_labels[i] for i in positive_indices]

            f1_macro_positive = f1_score(positive_true, positive_pred, labels=positive_classes, average='macro', zero_division=0)
            f1_micro_positive = f1_score(positive_true, positive_pred, labels=positive_classes, average='micro', zero_division=0)
        else:
            f1_macro_positive = 0.0
            f1_micro_positive = 0.0

        results = {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_micro": float(f1_micro),
            "f1_weighted": float(f1_weighted),
            "f1_macro_positive_only": float(f1_macro_positive),
            "f1_micro_positive_only": float(f1_micro_positive),
            "per_class_metrics": {
                class_names[i]: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1_score": float(f1_per_class[i]),
                    "support": int(support[i])
                }
                for i in range(len(class_names))
            },
            "classification_report": class_report,
            "confusion_matrix": conf_matrix.tolist(),
            "class_names": class_names,
            "total_examples": len(true_labels)
        }

        return results

    def save_results(self, results: Dict, output_path: str):
        """Save test results to file"""
        # Create a copy and remove detailed predictions for main results file
        results_copy = results.copy()
        if "detailed_predictions" in results_copy:
            del results_copy["detailed_predictions"]

        # Save main results
        with open(output_path, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)

        logger.info(f"Test results saved to: {output_path}")

    def plot_confusion_matrix(self, results: Dict, output_path: str = None, figsize: Tuple[int, int] = (10, 8)):
        """Plot and save confusion matrix"""
        conf_matrix = np.array(results["confusion_matrix"])
        class_names = results["class_names"]

        plt.figure(figsize=figsize)
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to: {output_path}")
        else:
            plt.show()

    def analyze_no_relation_performance(self, results: Dict):
        """Analyze performance specifically for no_relation class"""
        if "no_relation" not in results["per_class_metrics"]:
            print("No 'no_relation' class found in results")
            return

        no_relation_metrics = results["per_class_metrics"]["no_relation"]

        print("\n" + "="*50)
        print("NO_RELATION CLASS ANALYSIS")
        print("="*50)
        print(f"Precision: {no_relation_metrics['precision']:.4f}")
        print(f"Recall: {no_relation_metrics['recall']:.4f}")
        print(f"F1-Score: {no_relation_metrics['f1_score']:.4f}")
        print(f"Support: {no_relation_metrics['support']}")

        # Analyze confusion matrix for no_relation
        conf_matrix = np.array(results["confusion_matrix"])
        class_names = results["class_names"]

        try:
            no_relation_idx = class_names.index("no_relation")
        except ValueError:
            print("'no_relation' class not found in class names")
            return

        print(f"\nConfusion Matrix for 'no_relation' (index {no_relation_idx}):")
        print(f"True Positives: {conf_matrix[no_relation_idx, no_relation_idx]}")
        print(f"False Negatives: {sum(conf_matrix[no_relation_idx, :]) - conf_matrix[no_relation_idx, no_relation_idx]}")
        print(f"False Positives: {sum(conf_matrix[:, no_relation_idx]) - conf_matrix[no_relation_idx, no_relation_idx]}")

        # Show what no_relation is being confused with
        print("\nMost common confusions with 'no_relation':")
        for i, class_name in enumerate(class_names):
            if i != no_relation_idx:
                false_pos = conf_matrix[i, no_relation_idx]  # True class i, predicted no_relation
                false_neg = conf_matrix[no_relation_idx, i]  # True no_relation, predicted class i
                if false_pos > 0 or false_neg > 0:
                    print(f"  {class_name}: {false_pos} predicted as no_relation, {false_neg} no_relation predicted as {class_name}")

    def print_summary(self, results: Dict):
        """Print a summary of test results"""
        print("\n" + "="*70)
        print("RELATION CLASSIFICATION TEST RESULTS")
        print("="*70)

        print(f"Total test examples: {results['total_examples']}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score (Macro): {results['f1_macro']:.4f}")
        print(f"F1 Score (Micro): {results['f1_micro']:.4f}")
        print(f"F1 Score (Weighted): {results['f1_weighted']:.4f}")
        print(f"F1 Score (Positive Relations Only): {results['f1_macro_positive_only']:.4f}")

        print("\nPer-Class Results:")
        print("-" * 70)
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 70)

        for class_name, metrics in results["per_class_metrics"].items():
            print(f"{class_name:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                  f"{metrics['f1_score']:<10.3f} {metrics['support']:<10}")

        # Show confusion matrix
        conf_matrix = np.array(results["confusion_matrix"])
        class_names = results["class_names"]

        # Calculate column width based on longest class name
        max_name_len = max(len(name) for name in class_names)
        col_width = max(max_name_len + 2, 8)  # At least 8 chars, more if needed

        print("\nConfusion Matrix:")
        print("-" * (15 + col_width * len(class_names)))
        # Print header
        header_label = "Actual \\ Pred"
        print(f"{header_label:<15}", end="")
        for name in class_names:
            print(f"{name:<{col_width}}", end="")
        print()

        # Print separator line
        print("-" * (15 + col_width * len(class_names)))

        # Print matrix
        for i, name in enumerate(class_names):
            print(f"{name:<15}", end="")
            for j in range(len(class_names)):
                print(f"{conf_matrix[i][j]:<{col_width}}", end="")
            print()

        # Add no_relation specific analysis
        self.analyze_no_relation_performance(results)

def main():
    parser = argparse.ArgumentParser(description="Test trained relation classification model")
    parser.add_argument("--test_data", required=True, help="Path to test data JSON file")
    parser.add_argument("--model_dir", required=True, help="Directory containing trained model")
    parser.add_argument("--output_dir", default="test_results", help="Directory to save test results")
    parser.add_argument("--base_model", default="dmis-lab/biobert-base-cased-v1.1", help="Base model name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing")
    parser.add_argument("--detailed", action="store_true", help="Include detailed per-example predictions")
    parser.add_argument("--plot_cm", action="store_true", help="Generate confusion matrix plot")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    logger.info(f"Loading test data from: {args.test_data}")
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)

    logger.info(f"Loaded {len(test_data)} test examples")

    # Initialize tester
    tester = RelationModelTester(args.model_dir, args.base_model)

    # Run testing
    results = tester.test_model(
        test_data,
        batch_size=args.batch_size,
        detailed_output=args.detailed
    )

    # Print summary
    tester.print_summary(results)

    # Save results
    results_path = os.path.join(args.output_dir, "test_results.json")
    tester.save_results(results, results_path)

    # Plot confusion matrix if requested
    if args.plot_cm:
        cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
        tester.plot_confusion_matrix(results, cm_path)

    # Save detailed predictions if included
    if args.detailed and "detailed_predictions" in results:
        detailed_path = os.path.join(args.output_dir, "detailed_predictions.json")
        with open(detailed_path, 'w') as f:
            json.dump(results["detailed_predictions"], f, indent=2)
        logger.info(f"Detailed predictions saved to: {detailed_path}")

    logger.info("Testing completed successfully!")

if __name__ == "__main__":
    main()

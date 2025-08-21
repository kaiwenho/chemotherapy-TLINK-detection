import json
import argparse
import logging
from typing import List, Dict
from collections import Counter
import numpy as np
import os

# Import your existing relation model testing components
from test_relation_model import RelationModelTester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPredictor:
    """Generate predictions for dataset without ground truth labels"""
    
    def __init__(self, model_dir: str):
        """
        Initialize the predictor
        
        Args:
            model_dir: Directory containing the trained relation model
        """
        self.model_dir = model_dir
        
        # Initialize the relation model tester
        logger.info(f"Loading relation model from: {model_dir}")
        self.relation_tester = RelationModelTester(model_dir)
        
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load the dataset for prediction"""
        logger.info(f"Loading dataset from: {dataset_path}")
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            logger.info(f"Loaded {len(dataset)} examples for prediction")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return []
    
    def analyze_dataset_info(self, dataset: List[Dict]):
        """Analyze basic dataset information without ground truth"""
        direction_counts = Counter()
        subject_class_counts = Counter()
        object_class_counts = Counter()
        
        for example in dataset:
            direction = example.get('pair_direction', 'unknown')
            subject_class = example.get('subject_class', 'unknown')
            object_class = example.get('object_class', 'unknown')
            
            direction_counts[direction] += 1
            subject_class_counts[subject_class] += 1
            object_class_counts[object_class] += 1
        
        return {
            'direction_distribution': dict(direction_counts),
            'subject_class_distribution': dict(subject_class_counts),
            'object_class_distribution': dict(object_class_counts),
            'total_examples': len(dataset)
        }
    
    def generate_predictions(self, dataset: List[Dict], batch_size: int = 16) -> Dict:
        """
        Generate predictions for the dataset
        
        Args:
            dataset: Dataset for prediction (without ground truth)
            batch_size: Batch size for model inference
            
        Returns:
            Prediction results
        """
        logger.info(f"Generating predictions for {len(dataset)} examples...")
        
        # The model expects a 'relation' field, but since we don't have ground truth,
        # we'll use the existing "unknown" values in your dataset
        # No need to modify the dataset since it already has 'relation': 'unknown'
        
        # Run prediction using the relation model tester
        # We only care about the predictions, not the evaluation metrics
        results = self.relation_tester.test_model(
            dataset, 
            batch_size=batch_size, 
            detailed_output=True
        )
        
        # Extract just the predictions
        predictions = results.get("detailed_predictions", [])
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        return {
            'predictions': predictions,
            'total_examples': len(dataset),
            'raw_results': results  # Keep the full results for reference
        }
    
    def analyze_predictions(self, predictions: List[Dict]) -> Dict:
        """Analyze the distribution of predictions"""
        relation_counts = Counter()
        confidence_scores = []
        
        for pred in predictions:
            predicted_relation = pred.get('predicted_relation', 'no_relation')
            confidence = pred.get('confidence', 0.0)
            
            relation_counts[predicted_relation] += 1
            confidence_scores.append(confidence)
        
        return {
            'relation_distribution': dict(relation_counts),
            'confidence_stats': {
                'count': len(confidence_scores),
                'mean': float(np.mean(confidence_scores)) if confidence_scores else 0.0,
                'min': float(np.min(confidence_scores)) if confidence_scores else 0.0,
                'max': float(np.max(confidence_scores)) if confidence_scores else 0.0,
                'median': float(np.median(confidence_scores)) if confidence_scores else 0.0
            }
        }
    
    def create_annotated_dataset(self, original_dataset: List[Dict], predictions: List[Dict]) -> List[Dict]:
        """Create dataset with predictions added"""
        if len(original_dataset) != len(predictions):
            logger.warning(f"Mismatch: {len(original_dataset)} examples, {len(predictions)} predictions")
            return []
        
        annotated_dataset = []
        for example, prediction in zip(original_dataset, predictions):
            annotated_example = example.copy()
            
            # Remove the long note for cleaner output
            if 'note' in annotated_example:
                del annotated_example['note']
            
            # Replace the "unknown" relation with the predicted relation
            annotated_example['relation'] = prediction.get('predicted_relation', 'no_relation')
            annotated_example['prediction_confidence'] = prediction.get('confidence', 0.0)
            
            # Add all prediction probabilities if available
            if 'all_probabilities' in prediction:
                annotated_example['prediction_probabilities'] = prediction['all_probabilities']
            
            annotated_dataset.append(annotated_example)
        
        return annotated_dataset
    
    def apply_pattern_filter(self, predictions: List[Dict], annotated_dataset: List[Dict]) -> List[Dict]:
        """Apply pattern-based filtering to predictions"""
        if len(predictions) != len(annotated_dataset):
            logger.warning(f"Mismatch in lengths: {len(predictions)} predictions, {len(annotated_dataset)} examples")
            return annotated_dataset
        
        filtered_dataset = []
        
        for pred, example in zip(predictions, annotated_dataset):
            filtered_example = example.copy()
            
            original_prediction = pred.get('predicted_relation', 'no_relation')
            subject_class = example.get('subject_class', -1)
            object_class = example.get('object_class', -1)
            
            # Validate pattern
            validated_prediction = self._validate_pattern(original_prediction, subject_class, object_class)
            filtered_example['relation'] = validated_prediction
            
            filtered_dataset.append(filtered_example)
        
        return filtered_dataset
    
    def _validate_pattern(self, predicted_relation: str, subject_class: int, object_class: int) -> str:
        """
        Validate if a prediction follows the correct biological pattern
        
        Args:
            predicted_relation: Original predicted relation
            subject_class: Class of subject entity (1=EVENT, 2=TIMEX)
            object_class: Class of object entity (1=EVENT, 2=TIMEX)
            
        Returns:
            Original prediction if valid, 'no_relation' if invalid
        """
        # Define valid patterns
        VALID_PATTERNS = {
            'BEGINS-ON': (1, 2),  # EVENT -> TIMEX
            'ENDS-ON': (1, 2),    # EVENT -> TIMEX
            'CONTAINS': (2, 1)    # TIMEX -> EVENT
        }
        
        # no_relation is always valid
        if predicted_relation == 'no_relation':
            return predicted_relation
        
        # Check if the predicted relation follows the correct pattern
        if predicted_relation in VALID_PATTERNS:
            expected_pattern = VALID_PATTERNS[predicted_relation]
            current_pattern = (subject_class, object_class)
            
            if current_pattern == expected_pattern:
                return predicted_relation
            else:
                # Pattern doesn't match - invalidate
                return 'no_relation'
        else:
            # Unknown relation - invalidate
            return 'no_relation'
    
    def analyze_pattern_filtering(self, original_predictions: List[Dict], 
                                filtered_dataset: List[Dict]) -> Dict:
        """Analyze the impact of pattern filtering"""
        logger.info("Analyzing pattern filtering impact...")
        
        changes = {'total_changed': 0, 'changes_by_type': {}, 'pattern_compliance': {}}
        relation_changes = Counter()
        
        for pred, filtered in zip(original_predictions, filtered_dataset):
            original = pred.get('predicted_relation', 'no_relation')
            new = filtered.get('relation', 'no_relation')
            
            if original != new:
                changes['total_changed'] += 1
                change_key = f"{original} -> {new}"
                relation_changes[change_key] += 1
        
        changes['changes_by_type'] = dict(relation_changes)
        
        # Analyze pattern compliance
        pattern_compliance = {'compliant': 0, 'non_compliant': 0}
        for example in filtered_dataset:
            subject_class = example.get('subject_class', -1)
            object_class = example.get('object_class', -1)
            relation = example.get('relation', 'no_relation')
            
            is_compliant = self._check_pattern_compliance(relation, subject_class, object_class)
            if is_compliant:
                pattern_compliance['compliant'] += 1
            else:
                pattern_compliance['non_compliant'] += 1
        
        changes['pattern_compliance'] = pattern_compliance
        
        total = len(filtered_dataset)
        logger.info(f"Pattern filtering results:")
        logger.info(f"  Total predictions changed: {changes['total_changed']}")
        logger.info(f"  Pattern compliant: {pattern_compliance['compliant']} ({pattern_compliance['compliant']/total*100:.1f}%)")
        logger.info(f"  Pattern non-compliant: {pattern_compliance['non_compliant']} ({pattern_compliance['non_compliant']/total*100:.1f}%)")
        
        if relation_changes:
            logger.info(f"  Most common changes:")
            for change, count in relation_changes.most_common(5):
                logger.info(f"    {change}: {count}")
        
        return changes
    
    def _check_pattern_compliance(self, relation: str, subject_class: int, object_class: int) -> bool:
        """Check if a relation follows the expected pattern"""
        VALID_PATTERNS = {
            'BEGINS-ON': (1, 2),  # EVENT -> TIMEX
            'ENDS-ON': (1, 2),    # EVENT -> TIMEX
            'CONTAINS': (2, 1)    # TIMEX -> EVENT
        }
        
        if relation == 'no_relation':
            return True
            
        if relation in VALID_PATTERNS:
            expected_pattern = VALID_PATTERNS[relation]
            return (subject_class, object_class) == expected_pattern
        
        return False
    
    def apply_confidence_filter(self, dataset: List[Dict], confidence_threshold: float = 0.5) -> List[Dict]:
        """Remove predictions below confidence threshold"""
        high_confidence_dataset = []
        for example in dataset:
            confidence = example.get('prediction_confidence', 0.0)
            relation = example.get('relation', 'no_relation')
            
            # Keep if confidence is above threshold OR if it's no_relation
            if confidence >= confidence_threshold or relation == 'no_relation':
                high_confidence_dataset.append(example)
        
        return high_confidence_dataset
    
    def remove_no_relation_predictions(self, dataset: List[Dict]) -> List[Dict]:
        """Remove all examples with 'no_relation' predictions"""
        filtered_dataset = [
            example for example in dataset 
            if example.get('relation', 'no_relation') != 'no_relation'
        ]
        
        return filtered_dataset
    
    def print_final_results(self, original_count: int, final_dataset: List[Dict]):
        """Print summary of final high-quality results"""
        final_count = len(final_dataset)
        
        logger.info("\n" + "="*60)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"Original examples: {original_count}")
        logger.info(f"Final high-quality predictions: {final_count}")
        logger.info(f"Kept: {final_count/original_count*100:.1f}% of original data")
        
        if final_count > 0:
            # Analyze final relation distribution
            relation_counts = Counter()
            confidences = []
            
            for example in final_dataset:
                relation = example.get('relation', 'unknown')
                confidence = example.get('prediction_confidence', 0.0)
                relation_counts[relation] += 1
                confidences.append(confidence)
            
            logger.info(f"\nFinal Relation Distribution:")
            for relation, count in sorted(relation_counts.items()):
                percentage = (count / final_count) * 100
                logger.info(f"  {relation}: {count} ({percentage:.1f}%)")
            
            logger.info(f"\nFinal Confidence Statistics:")
            logger.info(f"  Average: {np.mean(confidences):.3f}")
            logger.info(f"  Min: {np.min(confidences):.3f}")
            logger.info(f"  Max: {np.max(confidences):.3f}")
        
        logger.info("="*60)
    
    def save_predictions(self, original_dataset: List[Dict], predictions: List[Dict], 
                        dataset_stats: Dict, prediction_stats: Dict, output_dir: str,
                        filtered_dataset: List[Dict] = None, filtering_analysis: Dict = None,
                        high_confidence_dataset: List[Dict] = None, meaningful_dataset: List[Dict] = None) -> List[Dict]:
        """Save prediction results including filtered and meaningful versions"""
        logger.info(f"Saving prediction results to: {output_dir}")
        
        # Create regular annotated dataset (without filtering)
        annotated_dataset = self.create_annotated_dataset(original_dataset, predictions)
        
        # Save regular annotated dataset
        annotated_path = os.path.join(output_dir, "dataset_with_predictions.json")
        with open(annotated_path, 'w', encoding='utf-8') as f:
            json.dump(annotated_dataset, f, indent=2, ensure_ascii=False)
        logger.info(f"Raw predictions saved to: {annotated_path}")
        
        # Save filtered dataset (pattern-corrected)
        if filtered_dataset:
            filtered_path = os.path.join(output_dir, "dataset_with_filtered_predictions.json")
            with open(filtered_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_dataset, f, indent=2, ensure_ascii=False)
            logger.info(f"Pattern-filtered dataset saved to: {filtered_path}")
        
        # Save high-confidence dataset (pattern-corrected + confidence filtered)
        if high_confidence_dataset:
            high_conf_path = os.path.join(output_dir, "high_confidence_predictions.json")
            with open(high_conf_path, 'w', encoding='utf-8') as f:
                json.dump(high_confidence_dataset, f, indent=2, ensure_ascii=False)
            logger.info(f"High-confidence dataset saved to: {high_conf_path}")
        
        # Save meaningful dataset (no_relation removed) - BEST OUTPUT
        if meaningful_dataset:
            meaningful_path = os.path.join(output_dir, "meaningful_relations_only.json")
            with open(meaningful_path, 'w', encoding='utf-8') as f:
                json.dump(meaningful_dataset, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ MEANINGFUL relations saved to: {meaningful_path} (BEST OUTPUT)")
            logger.info(f"   → High-confidence (≥0.5) + Pattern-valid + No no_relation")
        
        # Save comparison data
        clean_predictions = []
        for i, (example, pred) in enumerate(zip(original_dataset, predictions)):
            clean_pred = {
                'example_index': i,
                'subject_text': example['subject_text'],
                'object_text': example['object_text'],
                'subject_class': example.get('subject_class', -1),
                'object_class': example.get('object_class', -1),
                'raw_prediction': pred.get('predicted_relation', 'no_relation'),
                'confidence': pred.get('confidence', 0.0),
                'pair_direction': example.get('pair_direction', 'unknown'),
                'source_sentence': example.get('source_sentence', ''),
                'note_path': example.get('note_path', ''),
                'all_probabilities': pred.get('all_probabilities', {})
            }
            
            # Add filtering information if available
            if filtered_dataset and i < len(filtered_dataset):
                filtered_example = filtered_dataset[i]
                clean_pred['filtered_prediction'] = filtered_example.get('relation', 'no_relation')
                clean_pred['pattern_filtered'] = filtered_example.get('pattern_filtered', False)
                clean_pred['meets_confidence_threshold'] = clean_pred['confidence'] >= 0.5
                clean_pred['included_in_meaningful'] = (
                    filtered_example.get('relation', 'no_relation') != 'no_relation' and 
                    clean_pred['confidence'] >= 0.5
                )
            
            clean_predictions.append(clean_pred)
        
        comparison_path = os.path.join(output_dir, "predictions_comparison.json")
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(clean_predictions, f, indent=2, ensure_ascii=False)
        logger.info(f"Prediction comparison saved to: {comparison_path}")
        
        # Calculate meaningful relation statistics
        meaningful_stats = {}
        confidence_stats = {}
        if meaningful_dataset:
            meaningful_relations = Counter()
            confidences = []
            for example in meaningful_dataset:
                relation = example.get('relation', 'unknown')
                confidence = example.get('prediction_confidence', 0.0)
                meaningful_relations[relation] += 1
                if confidence > 0.0:  # Exclude no_relation confidences
                    confidences.append(confidence)
            
            meaningful_stats = dict(meaningful_relations)
            if confidences:
                confidence_stats = {
                    'mean': float(np.mean(confidences)),
                    'min': float(np.min(confidences)),
                    'max': float(np.max(confidences)),
                    'count': len(confidences)
                }
        
        # Save summary statistics
        summary = {
            'dataset_statistics': dataset_stats,
            'prediction_statistics': prediction_stats,
            'filtering_analysis': filtering_analysis or {},
            'meaningful_relations_stats': meaningful_stats,
            'meaningful_confidence_stats': confidence_stats,
            'summary': {
                'total_examples': len(original_dataset),
                'total_predictions': len(predictions),
                'pattern_filtered_count': len(filtered_dataset) if filtered_dataset else 0,
                'high_confidence_count': len(high_confidence_dataset) if high_confidence_dataset else 0,
                'meaningful_relations_count': len(meaningful_dataset) if meaningful_dataset else 0,
                'confidence_threshold': 0.5,
                'no_relation_removed': len(high_confidence_dataset) - len(meaningful_dataset) if high_confidence_dataset and meaningful_dataset else 0,
                'predicted_relations': prediction_stats['relation_distribution'],
                'meaningful_relations_only': meaningful_stats,
                'meaningful_confidence_stats': confidence_stats,
                'average_confidence': prediction_stats['confidence_stats']['mean'],
                'filtering_changes': filtering_analysis.get('total_changed', 0) if filtering_analysis else 0
            }
        }
        
        summary_path = os.path.join(output_dir, "prediction_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Prediction summary saved to: {summary_path}")
        
        # Recommend the best file to use
        logger.info(f"\n{'='*70}")
        logger.info(f"📁 OUTPUT FILES SUMMARY:")
        logger.info(f"{'='*70}")
        if meaningful_dataset:
            logger.info(f"1. meaningful_relations_only.json - 🎯 BEST OUTPUT")
            logger.info(f"   → High-confidence (≥0.5) + Pattern-valid + Meaningful relations only")
            logger.info(f"   → {len(meaningful_dataset)} examples")
        if high_confidence_dataset:
            logger.info(f"2. high_confidence_predictions.json - High confidence (≥0.5) ({len(high_confidence_dataset)} examples)")
        if filtered_dataset:
            logger.info(f"3. dataset_with_filtered_predictions.json - Pattern-corrected ({len(filtered_dataset)} examples)")
        logger.info(f"4. dataset_with_predictions.json - Raw model output ({len(annotated_dataset)} examples)")
        logger.info(f"5. predictions_comparison.json - Side-by-side comparison")
        logger.info(f"6. prediction_summary.json - Detailed statistics")
        logger.info(f"{'='*70}")
        
        return meaningful_dataset if meaningful_dataset else (high_confidence_dataset if high_confidence_dataset else (filtered_dataset if filtered_dataset else annotated_dataset))
    
    def run_prediction_pipeline(self, dataset_path: str, output_dir: str = "prediction_results", 
                              batch_size: int = 16) -> Dict:
        """Run complete prediction pipeline"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Load dataset
        dataset = self.load_dataset(dataset_path)
        if not dataset:
            logger.error("Failed to load dataset")
            return {}
        
        original_count = len(dataset)
        logger.info(f"Processing {original_count} examples...")
        
        # Step 2: Generate predictions
        prediction_results = self.generate_predictions(dataset, batch_size)
        predictions = prediction_results['predictions']
        
        # Step 3: Create annotated dataset
        annotated_dataset = self.create_annotated_dataset(dataset, predictions)
        
        # Step 4: Apply pattern filtering
        filtered_dataset = self.apply_pattern_filter(predictions, annotated_dataset)
        
        # Step 5: Apply confidence threshold filter
        high_confidence_dataset = self.apply_confidence_filter(filtered_dataset, confidence_threshold=0.5)
        
        # Step 6: Remove no_relation predictions
        final_dataset = self.remove_no_relation_predictions(high_confidence_dataset)
        
        # Step 7: Print final results summary
        self.print_final_results(original_count, final_dataset)
        
        # Step 8: Save final results
        output_path = os.path.join(output_dir, "high_quality_predictions.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✅ High-quality predictions saved to: {output_path}")
        logger.info(f"   → {len(final_dataset)} predictions with confidence ≥0.5, pattern-valid, meaningful relations only")
        
        return {
            'final_dataset': final_dataset,
            'total_examples': original_count,
            'final_count': len(final_dataset)
        }

def main():
    parser = argparse.ArgumentParser(description="Generate predictions for dataset using relation model")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset JSON file")
    parser.add_argument("--model_dir", required=True, help="Directory containing trained relation model")
    parser.add_argument("--output_dir", default="prediction_results", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for prediction")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = DatasetPredictor(args.model_dir)
    
    # Run prediction pipeline
    results = predictor.run_prediction_pipeline(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    logger.info("Prediction completed successfully!")

if __name__ == "__main__":
    main()
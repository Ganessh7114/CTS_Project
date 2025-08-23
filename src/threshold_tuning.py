"""
Threshold Tuning for Predictive Maintenance
Optimizes decision thresholds for safety-critical classification while maintaining accuracy targets.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve, roc_curve
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

class ThresholdTuner:
    """
    Optimizes classification thresholds for safety-critical applications.
    Prioritizes minimizing false negatives for critical classes.
    """
    
    def __init__(self, safety_classes: List[str] = ['I', 'II'], 
                 accuracy_target: float = 0.90,
                 max_false_negative_rate: float = 0.05):
        """
        Initialize threshold tuner.
        
        Args:
            safety_classes: Classes that are safety-critical (should minimize false negatives)
            accuracy_target: Minimum overall accuracy to maintain
            max_false_negative_rate: Maximum allowed false negative rate for safety classes
        """
        self.safety_classes = safety_classes
        self.accuracy_target = accuracy_target
        self.max_false_negative_rate = max_false_negative_rate
        self.optimal_thresholds = {}
        self.threshold_results = {}
        
    def tune_thresholds_grid_search(self, y_true: pd.Series, y_prob: np.ndarray,
                                  class_names: List[str],
                                  threshold_grid: List[float] = None) -> Dict[str, Any]:
        """
        Grid search for optimal thresholds.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities (n_samples, n_classes)
            class_names: Names of classes
            threshold_grid: Grid of thresholds to search
            
        Returns:
            Dictionary with optimal thresholds and results
        """
        if threshold_grid is None:
            threshold_grid = np.arange(0.1, 0.9, 0.05)
        
        print(f"Grid searching {len(threshold_grid)} thresholds...")
        
        best_score = -np.inf
        best_thresholds = None
        best_predictions = None
        all_results = []
        
        # Grid search over threshold combinations
        for thresholds in product(threshold_grid, repeat=len(class_names)):
            # Apply thresholds
            y_pred = self._apply_thresholds(y_prob, thresholds, class_names)
            
            # Evaluate
            results = self._evaluate_thresholds(y_true, y_pred, y_prob, class_names)
            
            # Check constraints
            if self._check_constraints(results):
                score = self._calculate_score(results)
                all_results.append({
                    'thresholds': thresholds,
                    'score': score,
                    'results': results,
                    'predictions': y_pred
                })
                
                if score > best_score:
                    best_score = score
                    best_thresholds = thresholds
                    best_predictions = y_pred
        
        if best_thresholds is None:
            print("⚠️  No thresholds found meeting constraints. Relaxing constraints...")
            return self._relaxed_search(y_true, y_prob, class_names, threshold_grid)
        
        # Store results
        self.optimal_thresholds = dict(zip(class_names, best_thresholds))
        self.threshold_results = {
            'optimal_thresholds': self.optimal_thresholds,
            'best_score': best_score,
            'best_predictions': best_predictions,
            'all_results': all_results
        }
        
        print(f"✅ Optimal thresholds found:")
        for class_name, threshold in self.optimal_thresholds.items():
            print(f"  {class_name}: {threshold:.3f}")
        
        return self.threshold_results
    
    def _apply_thresholds(self, y_prob: np.ndarray, thresholds: Tuple[float, ...], 
                         class_names: List[str]) -> np.ndarray:
        """
        Apply thresholds to probability predictions.
        
        Args:
            y_prob: Predicted probabilities
            thresholds: Thresholds for each class
            class_names: Class names
            
        Returns:
            Predicted classes
        """
        y_pred = np.zeros(len(y_prob), dtype=int)
        
        for i, (class_name, threshold) in enumerate(zip(class_names, thresholds)):
            # Predict class i if probability >= threshold
            y_pred[y_prob[:, i] >= threshold] = i
        
        # If no class meets threshold, predict highest probability
        no_prediction = y_pred == 0
        if np.any(no_prediction):
            y_pred[no_prediction] = np.argmax(y_prob[no_prediction], axis=1)
        
        return y_pred
    
    def _evaluate_thresholds(self, y_true: pd.Series, y_pred: np.ndarray, 
                           y_prob: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
        """
        Evaluate threshold performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            class_names: Class names
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # False negative rates for safety classes
        fn_rates = {}
        for i, class_name in enumerate(class_names):
            if class_name in self.safety_classes:
                # Calculate false negative rate for this class
                true_positives = cm[i, i] if i < cm.shape[0] and i < cm.shape[1] else 0
                false_negatives = np.sum(cm[i, :]) - true_positives if i < cm.shape[0] else 0
                total_actual = np.sum(cm[i, :]) if i < cm.shape[0] else 0
                
                fn_rate = false_negatives / total_actual if total_actual > 0 else 0
                fn_rates[class_name] = fn_rate
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'false_negative_rates': fn_rates
        }
    
    def _check_constraints(self, results: Dict[str, Any]) -> bool:
        """
        Check if results meet constraints.
        
        Args:
            results: Evaluation results
            
        Returns:
            True if constraints are met
        """
        # Check accuracy target
        if results['accuracy'] < self.accuracy_target:
            return False
        
        # Check false negative rates for safety classes
        for class_name, fn_rate in results['false_negative_rates'].items():
            if fn_rate > self.max_false_negative_rate:
                return False
        
        return True
    
    def _calculate_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate score for threshold optimization.
        Higher score = better performance.
        
        Args:
            results: Evaluation results
            
        Returns:
            Optimization score
        """
        # Base score from accuracy
        score = results['accuracy']
        
        # Bonus for low false negative rates on safety classes
        for class_name, fn_rate in results['false_negative_rates'].items():
            # Higher bonus for lower false negative rates
            safety_bonus = (1 - fn_rate) * 0.2  # 20% bonus for perfect safety
            score += safety_bonus
        
        return score
    
    def _relaxed_search(self, y_true: pd.Series, y_prob: np.ndarray, 
                       class_names: List[str], threshold_grid: List[float]) -> Dict[str, Any]:
        """
        Relaxed search when constraints cannot be met.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            class_names: Class names
            threshold_grid: Threshold grid
            
        Returns:
            Best available results
        """
        print("Performing relaxed search...")
        
        best_score = -np.inf
        best_thresholds = None
        best_predictions = None
        all_results = []
        
        for thresholds in product(threshold_grid, repeat=len(class_names)):
            y_pred = self._apply_thresholds(y_prob, thresholds, class_names)
            results = self._evaluate_thresholds(y_true, y_pred, y_prob, class_names)
            
            # Calculate relaxed score (no constraints)
            score = self._calculate_score(results)
            all_results.append({
                'thresholds': thresholds,
                'score': score,
                'results': results,
                'predictions': y_pred
            })
            
            if score > best_score:
                best_score = score
                best_thresholds = thresholds
                best_predictions = y_pred
        
        self.optimal_thresholds = dict(zip(class_names, best_thresholds))
        self.threshold_results = {
            'optimal_thresholds': self.optimal_thresholds,
            'best_score': best_score,
            'best_predictions': best_predictions,
            'all_results': all_results,
            'constraints_met': False
        }
        
        return self.threshold_results
    
    def tune_thresholds_roc_optimization(self, y_true: pd.Series, y_prob: np.ndarray,
                                       class_names: List[str]) -> Dict[str, Any]:
        """
        Optimize thresholds using ROC curve analysis.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            class_names: Class names
            
        Returns:
            Dictionary with optimal thresholds
        """
        print("Optimizing thresholds using ROC analysis...")
        
        optimal_thresholds = {}
        
        for i, class_name in enumerate(class_names):
            if class_name in self.safety_classes:
                # For safety classes, optimize for high recall (low false negative rate)
                y_binary = (y_true == class_name).astype(int)
                
                # Calculate precision-recall curve
                precision, recall, thresholds = precision_recall_curve(y_binary, y_prob[:, i])
                
                # Find threshold that gives high recall while maintaining reasonable precision
                target_recall = 0.95  # 95% recall for safety classes
                recall_idx = np.argmax(recall >= target_recall)
                
                if recall_idx < len(thresholds):
                    optimal_threshold = thresholds[recall_idx]
                else:
                    # Fallback to Youden's J statistic
                    fpr, tpr, thresholds = roc_curve(y_binary, y_prob[:, i])
                    j_scores = tpr - fpr
                    optimal_threshold = thresholds[np.argmax(j_scores)]
                
                optimal_thresholds[class_name] = optimal_threshold
            else:
                # For non-safety classes, use default threshold
                optimal_thresholds[class_name] = 0.5
        
        self.optimal_thresholds = optimal_thresholds
        
        # Apply thresholds and evaluate
        y_pred = self._apply_thresholds(y_prob, 
                                      [optimal_thresholds[cn] for cn in class_names], 
                                      class_names)
        results = self._evaluate_thresholds(y_true, y_pred, y_prob, class_names)
        
        self.threshold_results = {
            'optimal_thresholds': optimal_thresholds,
            'results': results,
            'predictions': y_pred
        }
        
        return self.threshold_results
    
    def plot_threshold_analysis(self, y_true: pd.Series, y_prob: np.ndarray,
                              class_names: List[str], save_path: str = None):
        """
        Plot threshold analysis.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            class_names: Class names
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: ROC curves
        for i, class_name in enumerate(class_names):
            y_binary = (y_true == class_name).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
            axes[0, 0].plot(fpr, tpr, label=f'{class_name} (AUC = {np.trapz(tpr, fpr):.3f})')
        
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Precision-Recall curves
        for i, class_name in enumerate(class_names):
            y_binary = (y_true == class_name).astype(int)
            precision, recall, _ = precision_recall_curve(y_binary, y_prob[:, i])
            axes[0, 1].plot(recall, precision, label=f'{class_name}')
        
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Threshold vs metrics
        if hasattr(self, 'threshold_results') and 'all_results' in self.threshold_results:
            thresholds = [r['thresholds'] for r in self.threshold_results['all_results']]
            accuracies = [r['results']['accuracy'] for r in self.threshold_results['all_results']]
            scores = [r['score'] for r in self.threshold_results['all_results']]
            
            axes[1, 0].scatter(accuracies, scores, alpha=0.6)
            axes[1, 0].set_xlabel('Accuracy')
            axes[1, 0].set_ylabel('Optimization Score')
            axes[1, 0].set_title('Accuracy vs Optimization Score')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Confusion matrix with optimal thresholds
        if hasattr(self, 'threshold_results') and 'results' in self.threshold_results:
            cm = self.threshold_results['results']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names, ax=axes[1, 1])
            axes[1, 1].set_title('Confusion Matrix (Optimal Thresholds)')
            axes[1, 1].set_xlabel('Predicted')
            axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Threshold analysis plot saved to {save_path}")
        
        plt.show()
    
    def get_optimal_thresholds(self) -> Dict[str, float]:
        """Get optimal thresholds."""
        return self.optimal_thresholds
    
    def apply_optimal_thresholds(self, y_prob: np.ndarray, class_names: List[str]) -> np.ndarray:
        """
        Apply optimal thresholds to new predictions.
        
        Args:
            y_prob: Predicted probabilities
            class_names: Class names
            
        Returns:
            Predicted classes
        """
        if not self.optimal_thresholds:
            raise ValueError("No optimal thresholds found. Run tuning first.")
        
        thresholds = [self.optimal_thresholds[cn] for cn in class_names]
        return self._apply_thresholds(y_prob, thresholds, class_names)

def tune_classification_thresholds(y_true: pd.Series, y_prob: np.ndarray,
                                 class_names: List[str],
                                 method: str = 'grid_search') -> Dict[str, Any]:
    """
    Tune classification thresholds for safety-critical applications.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        class_names: Class names
        method: Tuning method ('grid_search' or 'roc_optimization')
        
    Returns:
        Dictionary with optimal thresholds and results
    """
    tuner = ThresholdTuner()
    
    if method == 'grid_search':
        return tuner.tune_thresholds_grid_search(y_true, y_prob, class_names)
    elif method == 'roc_optimization':
        return tuner.tune_thresholds_roc_optimization(y_true, y_prob, class_names)
    else:
        raise ValueError(f"Unknown method: {method}")

if __name__ == "__main__":
    # Example usage
    from data_simulation import create_synthetic_dataset
    from data_splitting import create_temporal_splits
    from feature_engineering import create_feature_pipeline
    from models import train_complete_model_suite
    
    # Generate and prepare data
    df = create_synthetic_dataset()
    splits = create_temporal_splits(df)
    
    # Create features
    pipeline = create_feature_pipeline()
    pipeline.fit(splits['train_df'])
    
    X_train = pipeline.transform(splits['train_df'])
    X_test = pipeline.transform(splits['test_df'])
    
    # Prepare targets
    y_train_class = splits['train_df']['stateClass']
    y_train_rul = splits['train_df']['lifeRemaining']
    y_test_class = splits['test_df']['stateClass']
    
    # Train models
    model_suite = train_complete_model_suite(X_train, y_train_class, y_train_rul)
    
    # Get predictions
    y_pred, y_prob = model_suite.predict_state(X_test, 'xgb')
    
    # Tune thresholds
    class_names = ['I', 'II', 'III']
    threshold_results = tune_classification_thresholds(
        y_test_class, y_prob, class_names, method='grid_search'
    )
    
    print("Threshold Tuning Results:")
    print(f"Optimal thresholds: {threshold_results['optimal_thresholds']}")
    print(f"Best score: {threshold_results['best_score']:.3f}")
    print(f"Accuracy: {threshold_results['results']['accuracy']:.3f}")
    
    # Plot analysis
    tuner = ThresholdTuner()
    tuner.plot_threshold_analysis(y_test_class, y_prob, class_names)
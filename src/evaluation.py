"""
Evaluation Module for Predictive Maintenance
Generates comprehensive metrics, plots, and reports for model evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve
from sklearn.calibration import calibration_curve
from typing import Dict, Any, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

class PredictiveMaintenanceEvaluator:
    """
    Comprehensive evaluator for predictive maintenance models.
    Generates metrics, plots, and reports for classification and regression tasks.
    """
    
    def __init__(self, class_names: List[str] = ['I', 'II', 'III'],
                 bucket_names: List[str] = ['Urgent', 'Warning', 'Safe']):
        """
        Initialize evaluator.
        
        Args:
            class_names: Names of state classes
            bucket_names: Names of RUL bucket classes
        """
        self.class_names = class_names
        self.bucket_names = bucket_names
        self.evaluation_results = {}
        
    def evaluate_classification(self, y_true: pd.Series, y_pred: np.ndarray, 
                              y_prob: np.ndarray, task_name: str = 'state_classification') -> Dict[str, Any]:
        """
        Evaluate classification model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            task_name: Name of the classification task
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating {task_name}...")
        
        # Ensure both y_true and y_pred are strings to avoid type mixing
        y_true_str = y_true.astype(str)
        y_pred_str = pd.Series(y_pred).astype(str)
        
        # Basic classification metrics
        report = classification_report(
            y_true_str, y_pred_str, 
            target_names=self.class_names,
            labels=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true_str, y_pred_str, labels=self.class_names)
        
        # ROC AUC for each class
        roc_auc = {}
        for i, class_name in enumerate(self.class_names):
            if len(np.unique(y_true)) > 1:
                try:
                    roc_auc[class_name] = roc_auc_score(
                        (y_true == class_name).astype(int), 
                        y_prob[:, i]
                    )
                except ValueError:
                    roc_auc[class_name] = np.nan
        
        # Overall accuracy
        accuracy = report.get('accuracy', 0.0)
        if accuracy == 0.0:
            # Calculate accuracy manually if not in report
            accuracy = (y_true_str.values == y_pred_str.values).mean()
        
        # Per-class metrics
        class_metrics = {}
        for class_name in self.class_names:
            if class_name in report:
                class_metrics[class_name] = {
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1_score': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                }
        
        # Safety-specific metrics
        safety_metrics = self._calculate_safety_metrics(y_true, y_pred, y_prob)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'class_metrics': class_metrics,
            'safety_metrics': safety_metrics,
            'predictions': y_pred,
            'probabilities': y_prob
        }
        
        self.evaluation_results[task_name] = results
        return results
    
    def evaluate_regression(self, y_true: pd.Series, y_pred: np.ndarray,
                          task_name: str = 'rul_regression') -> Dict[str, Any]:
        """
        Evaluate regression model performance.
        
        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            task_name: Name of the regression task
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating {task_name}...")
        
        # Basic regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Error distribution
        errors = y_true - y_pred
        error_stats = {
            'mean': np.mean(errors),
            'std': np.std(errors),
            'min': np.min(errors),
            'max': np.max(errors),
            'q25': np.percentile(errors, 25),
            'q75': np.percentile(errors, 75)
        }
        
        results = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'error_stats': error_stats,
            'predictions': y_pred,
            'errors': errors
        }
        
        self.evaluation_results[task_name] = results
        return results
    
    def _calculate_safety_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                y_prob: np.ndarray) -> Dict[str, Any]:
        """
        Calculate safety-specific metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary with safety metrics
        """
        safety_metrics = {}
        
        # False negative rates for safety-critical classes
        for i, class_name in enumerate(self.class_names):
            if class_name in ['I', 'II']:  # Safety-critical classes
                true_positives = np.sum((y_true == class_name) & (y_pred == class_name))
                false_negatives = np.sum((y_true == class_name) & (y_pred != class_name))
                total_actual = np.sum(y_true == class_name)
                
                if total_actual > 0:
                    fn_rate = false_negatives / total_actual
                    recall = true_positives / total_actual
                else:
                    fn_rate = 0.0
                    recall = 1.0
                
                safety_metrics[f'{class_name}_false_negative_rate'] = fn_rate
                safety_metrics[f'{class_name}_recall'] = recall
        
        # Overall safety score (weighted by class importance)
        safety_score = 0.0
        total_weight = 0.0
        
        for class_name in ['I', 'II']:
            if f'{class_name}_recall' in safety_metrics:
                weight = 2.0 if class_name == 'I' else 1.0  # Class I is more critical
                safety_score += safety_metrics[f'{class_name}_recall'] * weight
                total_weight += weight
        
        if total_weight > 0:
            safety_metrics['overall_safety_score'] = safety_score / total_weight
        else:
            safety_metrics['overall_safety_score'] = 0.0
        
        return safety_metrics
    
    def generate_classification_plots(self, task_name: str = 'state_classification',
                                    save_path: str = None) -> None:
        """
        Generate comprehensive classification plots.
        
        Args:
            task_name: Name of the classification task
            save_path: Path to save plots
        """
        if task_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for {task_name}")
        
        results = self.evaluation_results[task_name]
        y_true = results.get('y_true', None)
        y_pred = results['predictions']
        y_prob = results['probabilities']
        
        if y_true is None:
            print("Warning: No true labels provided for plotting")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Confusion Matrix
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # Plot 2: ROC Curves
        for i, class_name in enumerate(self.class_names):
            if class_name in results['roc_auc'] and not np.isnan(results['roc_auc'][class_name]):
                y_binary = (y_true == class_name).astype(int)
                fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
                auc = results['roc_auc'][class_name]
                axes[0, 1].plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Precision-Recall Curves
        for i, class_name in enumerate(self.class_names):
            y_binary = (y_true == class_name).astype(int)
            precision, recall, _ = precision_recall_curve(y_binary, y_prob[:, i])
            axes[0, 2].plot(recall, precision, label=f'{class_name}')
        
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curves')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Class Distribution
        class_counts = pd.Series(y_true).value_counts()
        axes[1, 0].bar(class_counts.index, class_counts.values)
        axes[1, 0].set_title('Class Distribution')
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Count')
        
        # Plot 5: Per-Class Metrics
        metrics_df = pd.DataFrame(results['class_metrics']).T
        metrics_df[['precision', 'recall', 'f1_score']].plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Per-Class Metrics')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Plot 6: Probability Distribution
        for i, class_name in enumerate(self.class_names):
            class_probs = y_prob[y_true == class_name, i]
            axes[1, 2].hist(class_probs, alpha=0.7, label=f'True {class_name}', bins=20)
        
        axes[1, 2].set_title('Probability Distribution by True Class')
        axes[1, 2].set_xlabel('Predicted Probability')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Classification plots saved to {save_path}")
        
        plt.show()
    
    def generate_regression_plots(self, y_true: pd.Series, task_name: str = 'rul_regression',
                                save_path: str = None) -> None:
        """
        Generate comprehensive regression plots.
        
        Args:
            y_true: True RUL values
            task_name: Name of the regression task
            save_path: Path to save plots
        """
        if task_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for {task_name}")
        
        results = self.evaluation_results[task_name]
        y_pred = results['predictions']
        errors = results['errors']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: True vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True RUL')
        axes[0, 0].set_ylabel('Predicted RUL')
        axes[0, 0].set_title(f'True vs Predicted RUL\nR² = {results["r2"]:.3f}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        axes[0, 1].scatter(y_pred, errors, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted RUL')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Error Distribution
        axes[0, 2].hist(errors, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('Prediction Error')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title(f'Error Distribution\nMAE = {results["mae"]:.2f}, RMSE = {results["rmse"]:.2f}')
        
        # Plot 4: Error vs True RUL
        axes[1, 0].scatter(y_true, errors, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('True RUL')
        axes[1, 0].set_ylabel('Prediction Error')
        axes[1, 0].set_title('Error vs True RUL')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Error Box Plot
        axes[1, 1].boxplot(errors)
        axes[1, 1].set_ylabel('Prediction Error')
        axes[1, 1].set_title('Error Distribution Box Plot')
        
        # Plot 6: Cumulative Error Distribution
        sorted_errors = np.sort(np.abs(errors))
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1, 2].plot(sorted_errors, cumulative)
        axes[1, 2].set_xlabel('Absolute Error')
        axes[1, 2].set_ylabel('Cumulative Probability')
        axes[1, 2].set_title('Cumulative Error Distribution')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Regression plots saved to {save_path}")
        
        plt.show()
    
    def generate_calibration_plots(self, y_true: pd.Series, y_prob: np.ndarray,
                                 task_name: str = 'state_classification',
                                 save_path: str = None) -> None:
        """
        Generate calibration plots for classification probabilities.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            task_name: Name of the classification task
            save_path: Path to save plots
        """
        fig, axes = plt.subplots(1, len(self.class_names), figsize=(5*len(self.class_names), 4))
        
        if len(self.class_names) == 1:
            axes = [axes]
        
        for i, class_name in enumerate(self.class_names):
            y_binary = (y_true == class_name).astype(int)
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_binary, y_prob[:, i], n_bins=10
            )
            
            # Plot calibration curve
            axes[i].plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
            axes[i].plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
            axes[i].set_xlabel('Mean Predicted Probability')
            axes[i].set_ylabel('Fraction of Positives')
            axes[i].set_title(f'Calibration Plot - Class {class_name}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Calibration plots saved to {save_path}")
        
        plt.show()
    
    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report.
        
        Returns:
            Formatted summary report string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("PREDICTIVE MAINTENANCE MODEL EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        for task_name, results in self.evaluation_results.items():
            report_lines.append(f"TASK: {task_name.upper()}")
            report_lines.append("-" * 40)
            
            if 'accuracy' in results:  # Classification task
                report_lines.append(f"Overall Accuracy: {results['accuracy']:.3f}")
                report_lines.append("")
                
                # Per-class metrics
                report_lines.append("Per-Class Metrics:")
                for class_name, metrics in results['class_metrics'].items():
                    report_lines.append(f"  {class_name}:")
                    report_lines.append(f"    Precision: {metrics['precision']:.3f}")
                    report_lines.append(f"    Recall: {metrics['recall']:.3f}")
                    report_lines.append(f"    F1-Score: {metrics['f1_score']:.3f}")
                    report_lines.append(f"    Support: {metrics['support']}")
                
                # Safety metrics
                if 'safety_metrics' in results:
                    report_lines.append("")
                    report_lines.append("Safety Metrics:")
                    for metric_name, value in results['safety_metrics'].items():
                        report_lines.append(f"  {metric_name}: {value:.3f}")
                
                # ROC AUC
                if 'roc_auc' in results:
                    report_lines.append("")
                    report_lines.append("ROC AUC Scores:")
                    for class_name, auc in results['roc_auc'].items():
                        if not np.isnan(auc):
                            report_lines.append(f"  {class_name}: {auc:.3f}")
            
            elif 'mae' in results:  # Regression task
                report_lines.append(f"MAE: {results['mae']:.2f}")
                report_lines.append(f"RMSE: {results['rmse']:.2f}")
                report_lines.append(f"R²: {results['r2']:.3f}")
                report_lines.append(f"MAPE: {results['mape']:.2f}%")
                
                # Error statistics
                report_lines.append("")
                report_lines.append("Error Statistics:")
                for stat_name, value in results['error_stats'].items():
                    report_lines.append(f"  {stat_name}: {value:.2f}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_evaluation_results(self, filepath: str):
        """Save evaluation results to file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for task_name, results in self.evaluation_results.items():
            serializable_results[task_name] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[task_name][key] = value.tolist()
                elif isinstance(value, np.integer):
                    serializable_results[task_name][key] = int(value)
                elif isinstance(value, np.floating):
                    serializable_results[task_name][key] = float(value)
                elif isinstance(value, pd.Series):
                    serializable_results[task_name][key] = value.tolist()
                else:
                    serializable_results[task_name][key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Evaluation results saved to {filepath}")

def create_evaluation_report(y_true_class: pd.Series, y_pred_class: np.ndarray, 
                           y_prob_class: np.ndarray,
                           y_true_rul: pd.Series, y_pred_rul: np.ndarray,
                           save_plots: bool = True) -> PredictiveMaintenanceEvaluator:
    """
    Create comprehensive evaluation report for predictive maintenance models.
    
    Args:
        y_true_class: True state labels
        y_pred_class: Predicted state labels
        y_prob_class: Predicted state probabilities
        y_true_rul: True RUL values
        y_pred_rul: Predicted RUL values
        save_plots: Whether to save plots
        
    Returns:
        Evaluator with all results
    """
    evaluator = PredictiveMaintenanceEvaluator()
    
    # Evaluate classification
    class_results = evaluator.evaluate_classification(
        y_true_class, y_pred_class, y_prob_class, 'state_classification'
    )
    
    # Evaluate regression
    reg_results = evaluator.evaluate_regression(
        y_true_rul, y_pred_rul, 'rul_regression'
    )
    
    # Generate plots
    if save_plots:
        evaluator.generate_classification_plots('state_classification', 'plots/classification_evaluation.png')
        evaluator.generate_regression_plots(y_true_rul, 'rul_regression', 'plots/regression_evaluation.png')
        evaluator.generate_calibration_plots(y_true_class, y_prob_class, 'state_classification', 'plots/calibration_evaluation.png')
    
    # Print summary
    print(evaluator.generate_summary_report())
    
    return evaluator

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
    y_test_rul = splits['test_df']['lifeRemaining']
    
    # Train models
    model_suite = train_complete_model_suite(X_train, y_train_class, y_train_rul)
    
    # Get predictions
    y_pred_class, y_prob_class = model_suite.predict_state(X_test, 'xgb')
    y_pred_rul = model_suite.predict_rul(X_test, 'xgb_reg')
    
    # Create evaluation report
    evaluator = create_evaluation_report(
        y_test_class, y_pred_class, y_prob_class,
        y_test_rul, y_pred_rul,
        save_plots=True
    )
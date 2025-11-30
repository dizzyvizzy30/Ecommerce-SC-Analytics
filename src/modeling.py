"""
Machine Learning Modeling Module
Trains and evaluates ML models for delivery delay prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')


class DeliveryDelayPredictor:
    """
    Machine learning model to predict delivery delays.
    """

    def __init__(self, random_state=42):
        """
        Initialize the predictor.

        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None
        self.label_encoders = {}
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}

    def prepare_data(self, df, target_column='is_delayed'):
        """
        Prepare data for modeling.

        Parameters:
        -----------
        df : pd.DataFrame
            Master dataset with features
        target_column : str
            Target variable column name

        Returns:
        --------
        tuple : X_train, X_test, y_train, y_test
        """
        print("Preparing data for modeling...")

        # Remove rows with missing target
        df_clean = df[df[target_column].notna()].copy()
        print(f"  Dataset size: {df_clean.shape}")

        # Select features
        exclude_cols = [
            'order_id', 'customer_id', 'order_purchase_timestamp',
            'order_approved_at', 'order_delivered_timestamp',
            'order_estimated_delivery_date', target_column,
            'delivery_delay_days', 'is_early', 'is_on_time',  # Related to target
            'order_status'  # May not exist in test set
        ]

        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

        X = df_clean[feature_cols].copy()
        y = df_clean[target_column].copy()

        print(f"  Features: {len(feature_cols)}")
        print(f"  Target distribution: {y.value_counts().to_dict()}")

        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        print(f"  Encoding {len(categorical_cols)} categorical features...")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = X[col].astype(str)
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le

        # Handle missing values
        X.fillna(X.median(), inplace=True)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        print(f"  Train set: {X_train.shape}")
        print(f"  Test set: {X_test.shape}")

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        self.X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        self.y_train = y_train.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        self.feature_columns = X_train.columns.tolist()

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_models(self):
        """
        Train multiple classification models.
        """
        print("\nTraining models...")

        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state
            )
        }

        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\n  Training {name}...")
            model.fit(self.X_train, self.y_train)

            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='binary')
            recall = recall_score(self.y_test, y_pred, average='binary')
            f1 = f1_score(self.y_test, y_pred, average='binary')

            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

            if y_pred_proba is not None:
                auc = roc_auc_score(self.y_test, y_pred_proba)
                self.results[name]['auc'] = auc
                print(f"    Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
            else:
                print(f"    Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        # Select best model based on F1 score
        best_model_name = max(self.results, key=lambda x: self.results[x]['f1'])
        self.best_model_name = best_model_name
        self.best_model = self.results[best_model_name]['model']

        print(f"\nBest model: {best_model_name} (F1: {self.results[best_model_name]['f1']:.4f})")

    def plot_confusion_matrix(self, model_name=None, save_path=None):
        """
        Plot confusion matrix for a model.

        Parameters:
        -----------
        model_name : str
            Model name (if None, uses best model)
        save_path : str
            Path to save figure
        """
        if model_name is None:
            model_name = self.best_model_name

        y_pred = self.results[model_name]['y_pred']
        cm = confusion_matrix(self.y_test, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {model_name}')

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved: {save_path}")

        return fig

    def plot_feature_importance(self, model_name=None, top_n=20, save_path=None):
        """
        Plot feature importance for tree-based models.

        Parameters:
        -----------
        model_name : str
            Model name (if None, uses best model)
        top_n : int
            Number of top features to display
        save_path : str
            Path to save figure
        """
        if model_name is None:
            model_name = self.best_model_name

        model = self.results[model_name]['model']

        if not hasattr(model, 'feature_importances_'):
            print(f"{model_name} does not have feature importances")
            return None

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(top_n), importances[indices], align='center')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([self.feature_columns[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importances - {model_name}')
        ax.invert_yaxis()

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved: {save_path}")

        return fig

    def plot_roc_curve(self, save_path=None):
        """
        Plot ROC curves for all models.

        Parameters:
        -----------
        save_path : str
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        for name, result in self.results.items():
            if result['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
                auc = result['auc']
                ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(alpha=0.3)

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved: {save_path}")

        return fig

    def get_model_comparison(self):
        """
        Get comparison dataframe of all models.

        Returns:
        --------
        pd.DataFrame : Model comparison dataframe
        """
        comparison_data = []

        for name, result in self.results.items():
            row = {
                'Model': name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1 Score': result['f1']
            }
            if 'auc' in result:
                row['AUC'] = result['auc']

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1 Score', ascending=False)

        return comparison_df

    def save_results(self, output_dir='results'):
        """
        Save model results and visualizations.

        Parameters:
        -----------
        output_dir : str
            Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)

        print("\nSaving results...")

        # Save model comparison
        comparison = self.get_model_comparison()
        comparison.to_csv(f"{output_dir}/model_comparison.csv", index=False)
        print(f"  Model comparison saved: {output_dir}/model_comparison.csv")

        # Save plots
        self.plot_confusion_matrix(save_path=f"{output_dir}/figures/confusion_matrix.png")
        self.plot_roc_curve(save_path=f"{output_dir}/figures/roc_curve.png")

        if hasattr(self.best_model, 'feature_importances_'):
            self.plot_feature_importance(save_path=f"{output_dir}/figures/feature_importance.png")

        # Save classification report
        y_pred = self.results[self.best_model_name]['y_pred']
        report = classification_report(self.y_test, y_pred)

        with open(f"{output_dir}/classification_report.txt", 'w') as f:
            f.write(f"Best Model: {self.best_model_name}\n")
            f.write("="*60 + "\n\n")
            f.write(report)

        print(f"  Classification report saved: {output_dir}/classification_report.txt")

        print("\nAll results saved successfully!")


def train_and_evaluate(df, target_column='is_delayed', save_results=True, output_dir='../results'):
    """
    Convenience function to train and evaluate models.

    Parameters:
    -----------
    df : pd.DataFrame
        Master dataset with features
    target_column : str
        Target variable column name
    save_results : bool
        Whether to save results
    output_dir : str
        Directory to save results (default: '../results' for notebooks)

    Returns:
    --------
    DeliveryDelayPredictor : Trained predictor object
    """
    predictor = DeliveryDelayPredictor()

    # Prepare data
    predictor.prepare_data(df, target_column)

    # Train models
    predictor.train_models()

    # Display results
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(predictor.get_model_comparison().to_string(index=False))

    # Save results
    if save_results:
        predictor.save_results(output_dir)

    return predictor

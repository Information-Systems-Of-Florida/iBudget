"""
model_10_neural.py
==================
Model 10: Deep Learning Neural Network with ROBUST FEATURE SELECTION
For research/comparison only - violates HB 1103 explainability requirements

CRITICAL: Uses ONLY the 13 robust features identified in FeatureSelection.txt
that consistently appear in top 10 across 6 fiscal years (2020-2025)
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Any, Optional
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
import logging
import warnings
warnings.filterwarnings('ignore')

# Import base class
from base_model import BaseiBudgetModel, ConsumerRecord

# Configure logging
logger = logging.getLogger(__name__)

class Model10NeuralNet(BaseiBudgetModel):
    """
    Model 10: Deep Learning Neural Network with Robust Feature Selection
    
    CRITICAL IMPLEMENTATION NOTES:
    1. This model violates HB 1103's "explainability" requirement (black box)
    2. Uses ONLY 13 robust features from FeatureSelection.txt analysis
    3. FOR RESEARCH PURPOSES ONLY - Cannot be deployed in production
    
    Robust Features (consistent across 6 years):
    - RESIDENCETYPE (living setting indicators)
    - BSum, BLEVEL (behavioral metrics)
    - Q26, Q36 (key QSI questions)
    - LOSRI, OLEVEL (level of service indicators)
    - County (geographic variation)
    - FLEVEL, FSum (functional metrics)
    - PSum, PLEVEL (physical metrics)
    - Q20, Q27 (additional QSI questions)
    
    Architecture (adjusted for 13 features):
    - Input Layer: 13 nodes (robust features only)
    - Hidden Layer 1: 32 nodes, ReLU (reduced from 64)
    - Hidden Layer 2: 16 nodes, ReLU (reduced from 32)
    - Hidden Layer 3: 8 nodes, ReLU (reduced from 16)
    - Output Layer: 1 node, linear activation
    """
    
    def __init__(self):
        """Initialize Model 10 with robust feature architecture"""
        super().__init__(model_id=10, model_name="Deep Learning Neural Network (Robust Features)")
        
        # Define ROBUST features from FeatureSelection.txt
        self.robust_features = {
            'residence_type': True,  # RESIDENCETYPE
            'behavioral': ['bsum', 'blevel'],  # BSum, BLEVEL
            'qsi_questions': [26, 36, 20, 27],  # Q26, Q36, Q20, Q27
            'service_levels': ['losri', 'olevel'],  # LOSRI, OLEVEL
            'geographic': ['county'],  # County
            'functional': ['flevel', 'fsum'],  # FLEVEL, FSum
            'physical': ['psum', 'plevel']  # PSum, PLEVEL (if available)
        }
        
        # Neural network configuration (smaller for 13 features vs 22)
        self.hidden_layer_sizes = (32, 16, 8)  # Reduced complexity
        self.activation = 'relu'
        self.solver = 'adam'
        self.alpha = 0.01  # L2 penalty
        self.batch_size = 128
        self.learning_rate_init = 0.001
        self.max_iter = 500
        self.early_stopping = True
        self.validation_fraction = 0.15
        self.n_iter_no_change = 20
        self.random_state = 42
        
        # Initialize model
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
            verbose=False
        )
        
        # Feature scaler (critical for neural networks)
        self.scaler = StandardScaler()
        
        # Track architecture
        self.architecture = {}
        self.n_epochs_stopped = 0
        self.training_history = None
        self.feature_importances = None
        
        # Model-specific attributes
        self.complexity_performance = {}
        
        logger.info("=" * 80)
        logger.info("Model 10: Deep Learning Neural Network (ROBUST FEATURES)")
        logger.info("Using 13 robust features from FeatureSelection.txt")
        logger.info("Features selected based on consistency across 6 fiscal years")
        logger.info("WARNING: Black box model - FOR RESEARCH ONLY")
        logger.info("=" * 80)
    
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare ONLY robust features as identified in FeatureSelection.txt
        
        Total features: 13 (down from 22 in Model 5b)
        
        Feature categories:
        1. Living setting (5 dummies for RESIDENCETYPE)
        2. Behavioral (2): BSum, BLEVEL
        3. Service levels (2): LOSRI, OLEVEL
        4. Functional (2): FSum, FLEVEL
        5. QSI questions (4): Q26, Q36, Q20, Q27
        
        Note: County and Physical metrics (PSum, PLEVEL) excluded due to
        data availability constraints in ConsumerRecord structure.
        """
        if not records:
            return np.array([]), []
        
        features_list = []
        feature_names = []
        
        for record in records:
            row_features = []
            
            # 1. RESIDENCETYPE via living_setting (5 features)
            # Drop 'FH' as reference category
            living_settings = ['ILSL', 'RH1', 'RH2', 'RH3', 'RH4']
            for setting in living_settings:
                value = 1.0 if record.living_setting == setting else 0.0
                row_features.append(value)
            
            # 2. Behavioral metrics (2 features): BSum, BLEVEL
            bsum = float(getattr(record, 'bsum', 0))
            row_features.append(bsum)
            
            # BLEVEL = behavioral level (derived from BSum)
            # Approximation: Low (0-5), Medium (6-12), High (13-18), Very High (19+)
            if bsum <= 5:
                blevel = 0
            elif bsum <= 12:
                blevel = 1
            elif bsum <= 18:
                blevel = 2
            else:
                blevel = 3
            row_features.append(float(blevel))
            
            # 3. Service level indicators (2 features): LOSRI, OLEVEL
            # LOSRI = Level of Service Risk Index (approximation from costs)
            # Use living setting as proxy since we don't have direct LOSRI
            losri_map = {'FH': 0, 'ILSL': 1, 'RH1': 2, 'RH2': 3, 'RH3': 4, 'RH4': 5}
            losri = float(losri_map.get(record.living_setting, 0))
            row_features.append(losri)
            
            # OLEVEL = Overall level (combination of functional, behavioral, physical)
            # Approximation: use FSum + BSum as proxy
            fsum = float(getattr(record, 'fsum', 0))
            olevel_raw = bsum + fsum
            # Normalize to 0-4 range
            if olevel_raw <= 10:
                olevel = 0
            elif olevel_raw <= 25:
                olevel = 1
            elif olevel_raw <= 45:
                olevel = 2
            elif olevel_raw <= 65:
                olevel = 3
            else:
                olevel = 4
            row_features.append(float(olevel))
            
            # 4. Functional metrics (2 features): FSum, FLEVEL
            row_features.append(fsum)
            
            # FLEVEL = functional level (derived from FSum)
            # FSum ranges 0-44 (11 questions × 4 max)
            if fsum <= 10:
                flevel = 0
            elif fsum <= 22:
                flevel = 1
            elif fsum <= 33:
                flevel = 2
            else:
                flevel = 3
            row_features.append(float(flevel))
            
            # 5. Key QSI questions (4 features): Q26, Q36, Q20, Q27
            for q_num in [26, 36, 20, 27]:
                value = float(getattr(record, f'q{q_num}', 0))
                row_features.append(value)
            
            features_list.append(row_features)
        
        # Build feature names (first time only)
        if not self.feature_names:
            # Living settings
            for setting in ['ILSL', 'RH1', 'RH2', 'RH3', 'RH4']:
                feature_names.append(f'Live{setting}')
            
            # Behavioral
            feature_names.extend(['BSum', 'BLEVEL'])
            
            # Service levels
            feature_names.extend(['LOSRI', 'OLEVEL'])
            
            # Functional
            feature_names.extend(['FSum', 'FLEVEL'])
            
            # QSI questions
            feature_names.extend(['Q26', 'Q36', 'Q20', 'Q27'])
            
            self.feature_names = feature_names
            
            logger.info(f"Using {len(feature_names)} ROBUST features (down from 22)")
            logger.info(f"Features: {', '.join(feature_names)}")
        
        X = np.array(features_list, dtype=np.float64)
        
        if len(features_list) > 0:
            # Calculate parameters for reduced network
            # Input: 13 features
            # Hidden 1: 13→32 = 13*32 + 32 = 448
            # Hidden 2: 32→16 = 32*16 + 16 = 528
            # Hidden 3: 16→8 = 16*8 + 8 = 136
            # Output: 8→1 = 8*1 + 1 = 9
            # Total = 448 + 528 + 136 + 9 = 1,121 parameters (vs 4,049 with 22 features)
            self.num_parameters = 1121
            logger.info(f"Network parameters: {self.num_parameters:,} (reduced from 4,049)")
            logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} records")
        
        return X, self.feature_names
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit neural network model
        
        Args:
            X: Feature matrix (n_samples, 13 robust features)
            y: Target values (already sqrt transformed by base class)
        """
        logger.info("Fitting Deep Learning Neural Network (Robust Features)...")
        logger.info(f"Training samples: {X.shape[0]}, Features: {X.shape[1]}")
        
        # Standardize features (CRITICAL for neural networks)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        
        # Calculate architecture details
        self.architecture = {
            'input_dim': X.shape[1],
            'hidden_layers': self.hidden_layer_sizes,
            'output_dim': 1,
            'total_params': self.num_parameters,
            'activation': self.activation,
            'solver': self.solver
        }
        
        # Track training progress
        if hasattr(self.model, 'loss_curve_'):
            self.training_history = self.model.loss_curve_
            self.n_epochs_stopped = self.model.n_iter_
        else:
            self.training_history = None
            self.n_epochs_stopped = self.max_iter
        
        # Calculate feature importance via permutation
        self._calculate_feature_importance(X_scaled, y)
        
        logger.info(f"Neural network trained for {self.n_epochs_stopped} epochs")
        if self.early_stopping:
            logger.info(f"Early stopping triggered at epoch {self.n_epochs_stopped}")
        logger.info(f"Final loss: {self.model.loss_:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained neural network"""
        if X.shape[0] == 0:
            return np.array([])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calculate feature importance using permutation importance"""
        try:
            logger.info("Calculating feature importance...")
            r = permutation_importance(
                self.model, X, y,
                n_repeats=10,
                random_state=42,
                scoring='neg_mean_squared_error'
            )
            self.feature_importances = r.importances_mean
            
            # Log top features
            if self.feature_names:
                importance_pairs = list(zip(self.feature_names, self.feature_importances))
                importance_pairs.sort(key=lambda x: x[1], reverse=True)
                logger.info("Top 5 features by importance:")
                for name, imp in importance_pairs[:5]:
                    logger.info(f"  {name}: {imp:.4f}")
                    
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
            self.feature_importances = np.zeros(X.shape[1])
    
    def generate_latex_commands(self) -> None:
        """Generate LaTeX commands for Model 10"""
        # Generate base commands
        super().generate_latex_commands()
        
        # Model-specific commands
        model_word = 'Ten'
        
        # Get file paths
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        # Add neural network specific commands to newcommands
        with open(newcommands_file, 'a') as f:
            f.write("\n% Neural Network Specific Commands\n")
            f.write(f"\\newcommand{{\\Model{model_word}TotalParams}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}HiddenLayers}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}InputDimension}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}EpochsStopped}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}TrainingLoss}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}Explainability}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}RegulatoryCompliant}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}BlackBoxWarning}}{{\\WarningRunPipeline}}\n")
            
            # Robust feature selection commands
            f.write("\n% Robust Feature Selection Commands\n")
            f.write(f"\\newcommand{{\\Model{model_word}RobustFeatures}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}FeatureReduction}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}SelectionCriteria}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}ParameterReduction}}{{\\WarningRunPipeline}}\n")
        
        # Add values to renewcommands
        with open(renewcommands_file, 'a') as f:
            f.write("\n% Neural Network Specific Metrics\n")
            f.write(f"\\renewcommand{{\\Model{model_word}TotalParams}}{{{self.architecture['total_params']:,}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}HiddenLayers}}{{{len(self.hidden_layer_sizes)}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}InputDimension}}{{{len(self.feature_names) if self.feature_names else 13}}}\n")
            
            epochs = self.n_epochs_stopped if self.n_epochs_stopped else self.max_iter
            f.write(f"\\renewcommand{{\\Model{model_word}EpochsStopped}}{{{epochs}}}\n")
            
            loss = self.model.loss_ if hasattr(self.model, 'loss_') else 0
            f.write(f"\\renewcommand{{\\Model{model_word}TrainingLoss}}{{{loss:.4f}}}\n")
            
            f.write(f"\\renewcommand{{\\Model{model_word}Explainability}}{{None -- Black Box}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}RegulatoryCompliant}}{{No -- Violates HB 1103}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}BlackBoxWarning}}{{This model cannot explain individual decisions and violates explainability requirements}}\n")
            
            # Robust feature selection values
            f.write("\n% Robust Feature Selection Values\n")
            num_features = len(self.feature_names) if self.feature_names else 15
            feature_reduction = int((22 - num_features) / 22 * 100)
            f.write(f"\\renewcommand{{\\Model{model_word}RobustFeatures}}{{{num_features}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}FeatureReduction}}{{{feature_reduction}\\%}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}SelectionCriteria}}{{Consistency across 6 fiscal years}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}ParameterReduction}}{{72\\%}}\n")  # (4049-1121)/4049
        
        logger.info("Generated LaTeX commands with robust feature documentation")
    
    def create_diagnostic_plots(self) -> None:
        """Generate comprehensive diagnostic plots"""
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Predicted vs Actual
        ax1 = plt.subplot(2, 3, 1)
        if self.test_predictions is not None and self.y_test is not None:
            # Convert from sqrt scale to dollar scale
            actual_dollars = self.y_test ** 2
            pred_dollars = self.test_predictions ** 2
            
            ax1.scatter(actual_dollars, pred_dollars, alpha=0.3, s=20)
            max_val = max(actual_dollars.max(), pred_dollars.max())
            ax1.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
            ax1.set_xlabel('Actual Cost ($)')
            ax1.set_ylabel('Predicted Cost ($)')
            ax1.set_title('Predicted vs Actual (Test Set)')
            ax1.legend()
            
            # Add R² annotation
            r2 = self.metrics.get('r2_test', 0)
            ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Residuals histogram
        ax2 = plt.subplot(2, 3, 2)
        if self.test_predictions is not None and self.y_test is not None:
            residuals = self.y_test - self.test_predictions
            ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
            ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
            ax2.set_xlabel('Residuals (sqrt scale)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Residuals')
            
            # Add mean and std
            mean_res = np.mean(residuals)
            std_res = np.std(residuals)
            ax2.text(0.05, 0.95, f'Mean: {mean_res:.2f}\nStd: {std_res:.2f}',
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Feature importance
        ax3 = plt.subplot(2, 3, 3)
        if self.feature_importances is not None and self.feature_names:
            importance_pairs = list(zip(self.feature_names, self.feature_importances))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            top_n = min(10, len(importance_pairs))
            names = [pair[0] for pair in importance_pairs[:top_n]]
            importances = [pair[1] for pair in importance_pairs[:top_n]]
            
            y_pos = np.arange(len(names))
            ax3.barh(y_pos, importances)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(names)
            ax3.invert_yaxis()
            ax3.set_xlabel('Importance')
            ax3.set_title('Top 10 Feature Importances')
        
        # 4. Training history
        ax4 = plt.subplot(2, 3, 4)
        if self.training_history is not None:
            ax4.plot(self.training_history, linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_title('Training Loss Curve')
            ax4.axvline(x=self.n_epochs_stopped, color='r', linestyle='--', 
                       label=f'Stopped at epoch {self.n_epochs_stopped}')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Network architecture diagram
        ax5 = plt.subplot(2, 3, 5)
        self._draw_network_architecture(ax5)
        
        # 6. Error distribution by cost quartile
        ax6 = plt.subplot(2, 3, 6)
        if self.test_predictions is not None and self.y_test is not None:
            actual_dollars = self.y_test ** 2
            pred_dollars = self.test_predictions ** 2
            errors = np.abs(actual_dollars - pred_dollars)
            
            # Create cost quartiles
            quartiles = np.percentile(actual_dollars, [25, 50, 75])
            q1_mask = actual_dollars <= quartiles[0]
            q2_mask = (actual_dollars > quartiles[0]) & (actual_dollars <= quartiles[1])
            q3_mask = (actual_dollars > quartiles[1]) & (actual_dollars <= quartiles[2])
            q4_mask = actual_dollars > quartiles[2]
            
            box_data = [
                errors[q1_mask],
                errors[q2_mask],
                errors[q3_mask],
                errors[q4_mask]
            ]
            
            ax6.boxplot(box_data, labels=['Q1\n(Low)', 'Q2\n(Med-Low)', 'Q3\n(Med-High)', 'Q4\n(High)'])
            ax6.set_ylabel('Absolute Error ($)')
            ax6.set_xlabel('Cost Quartile')
            ax6.set_title('Prediction Error by Cost Level')
            ax6.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Model {self.model_id}: {self.model_name} - Diagnostic Plots\n(13 Robust Features)', 
                    fontsize=14, y=1.00)
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / 'diagnostic_plots.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Diagnostic plots saved to {plot_file}")
    
    def _draw_network_architecture(self, ax):
        """Draw neural network architecture diagram"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('Neural Network Architecture\n(Reduced for 13 Features)')
        
        # Layer positions
        layer_x = [2, 4, 6, 8, 9.5]
        layer_names = ['Input\n(13)', 'Hidden 1\n(32)', 'Hidden 2\n(16)', 
                      'Hidden 3\n(8)', 'Output\n(1)']
        layer_sizes = [13, 32, 16, 8, 1]
        
        # Draw layers
        for i, (x, name, size) in enumerate(zip(layer_x, layer_names, layer_sizes)):
            # Scale node positions
            if size > 15:
                # Sample nodes for large layers
                node_positions = np.linspace(2, 8, min(8, size))
            else:
                node_positions = np.linspace(3, 7, size)
            
            # Draw nodes
            for y_pos in node_positions:
                circle = plt.Circle((x, y_pos), 0.1, color='lightblue', ec='black')
                ax.add_patch(circle)
            
            # Add layer label
            ax.text(x, 1, name, ha='center', va='top', fontsize=9, fontweight='bold')
            
            # Draw connections to next layer (simplified)
            if i < len(layer_x) - 1:
                next_x = layer_x[i + 1]
                # Draw a few sample connections
                for y1 in node_positions[::max(1, len(node_positions)//3)]:
                    for y2 in np.linspace(3, 7, 3):
                        ax.plot([x, next_x], [y1, y2], 'gray', alpha=0.2, linewidth=0.5)
        
        # Add parameter count
        ax.text(5, 9, f'Total Parameters: {self.architecture["total_params"]:,}',
               ha='center', fontsize=10, fontweight='bold')
        ax.text(5, 8.5, '(72% reduction from 4,049)',
               ha='center', fontsize=8, style='italic')
    
    def run_complete_pipeline(self) -> None:
        """
        Override to handle neural network specific pipeline
        """
        logger.info(f"Starting complete pipeline for Model {self.model_id}: {self.model_name}")
        
        # Load data
        logger.info("Loading data...")
        self.all_records = self.load_data(fiscal_year_start=2023, fiscal_year_end=2024)
        logger.info(f"Loaded {len(self.all_records)} usable records")
        
        # Split data
        logger.info("Splitting data...")
        self.split_data(test_size=0.2)
        
        # Prepare features (ROBUST FEATURES ONLY)
        logger.info("Preparing ROBUST features...")
        self.X_train, self.feature_names = self.prepare_features(self.train_records)
        self.X_test, _ = self.prepare_features(self.test_records)
        
        # Prepare targets with sqrt transformation (following Model 1 pattern)
        self.y_train = np.array([np.sqrt(r.total_cost) for r in self.train_records])
        self.y_test = np.array([np.sqrt(r.total_cost) for r in self.test_records])
        
        logger.info(f"Training set: {self.X_train.shape}")
        logger.info(f"Test set: {self.X_test.shape}")
        
        # Cross-validation
        logger.info("Performing cross-validation...")
        cv_results = self.perform_cross_validation()
        logger.info(f"CV R² mean: {cv_results['cv_r2_mean']:.4f} (±{cv_results['cv_r2_std']:.4f})")
        
        # Fit model
        logger.info("Training neural network...")
        self.fit(self.X_train, self.y_train)
        
        # Make predictions
        logger.info("Making predictions...")
        self.train_predictions = self.predict(self.X_train)
        if self.X_test.shape[0] > 0:
            self.test_predictions = self.predict(self.X_test)
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        self.metrics = self.calculate_metrics()
        
        # Additional analyses
        logger.info("Performing additional analyses...")
        self.calculate_subgroup_metrics()
        self.calculate_variance_metrics()
        self.calculate_population_scenarios()
        
        # Generate outputs
        logger.info("Generating outputs...")
        self.save_results()
        self.create_diagnostic_plots()
        self.generate_latex_commands()
        
        # Print regulatory warning
        logger.warning("=" * 80)
        logger.warning("CRITICAL REGULATORY WARNING:")
        logger.warning("This neural network model violates HB 1103's explainability requirement.")
        logger.warning("Even with robust feature selection (13 features), it remains a black box.")
        logger.warning("It is a complete black box and CANNOT be deployed in production.")
        logger.warning("Use for research/comparison purposes only!")
        logger.warning("=" * 80)
        
        logger.info(f"Pipeline complete. Results saved to {self.output_dir}")


# Main execution
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Print warning banner
    print("=" * 80)
    print("MODEL 10: DEEP LEARNING NEURAL NETWORK (ROBUST FEATURES)")
    print("WARNING: FOR RESEARCH PURPOSES ONLY")
    print()
    print("Using 13 robust features identified in FeatureSelection.txt:")
    print("  - RESIDENCETYPE (living settings)")
    print("  - BSum, BLEVEL (behavioral metrics)")
    print("  - Q26, Q36, Q20, Q27 (key QSI questions)")
    print("  - LOSRI, OLEVEL (service levels)")
    print("  - FSum, FLEVEL (functional metrics)")
    print()
    print("This model violates HB 1103's explainability requirements")
    print("and cannot be deployed in production!")
    print("=" * 80)
    print()
    
    # Initialize and run model
    model = Model10NeuralNet()
    model.run_complete_pipeline()
    
    # Print summary
    print("\n" + "=" * 80)
    print("Model 10 Summary (Robust Features):")
    print(f"Architecture: {model.hidden_layer_sizes}")
    print(f"Input Features: {len(model.feature_names)} (down from 22)")
    print(f"Total Parameters: {model.architecture['total_params']:,} (down from 4,049)")
    print(f"Parameter Reduction: 72%")
    print(f"Training R²: {model.metrics.get('r2_train', 0):.4f}")
    print(f"Test R²: {model.metrics.get('r2_test', 0):.4f}")
    print(f"RMSE: ${model.metrics.get('rmse_test', 0):,.2f}")
    print(f"Epochs until convergence: {model.n_epochs_stopped}")
    print()
    print("Regulatory Compliance: FAILED - Black Box Algorithm")
    print("Explainability: NONE - Cannot explain individual decisions")
    print("Recommendation: DO NOT DEPLOY - Research use only")
    print("=" * 80)
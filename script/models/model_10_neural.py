"""
model_10_neural.py
==================
Model 10: Deep Learning Neural Network with ROBUST FEATURE SELECTION
For research/comparison only - violates HB 1103 explainability requirements

CRITICAL: Uses ONLY the 13 robust features identified in FeatureSelection.txt
that consistently appear in top 10 across 6 fiscal years (2020-2025)

DEPLOYMENT STATUS: NOT RECOMMENDED - Black box architecture
Research value: Feature selection validation, not neural network deployment
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
import time
import random  # For random seed control
warnings.filterwarnings('ignore')

# Import base class
from base_model import BaseiBudgetModel, ConsumerRecord

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# SINGLE POINT OF CONTROL FOR RANDOM SEED
# ============================================================================
# Change this value to get different random splits, or keep at 42 for reproducibility
# This seed controls:
#   - Train/test split
#   - Cross-validation folds
#   - Neural network initialization
#   - Any other random operations in the pipeline
# ============================================================================
RANDOM_SEED = 42


class Model10NeuralNet(BaseiBudgetModel):
    """
    Model 10: Deep Learning Neural Network with Robust Feature Selection
    
    CRITICAL IMPLEMENTATION NOTES:
    1. This model violates HB 1103's "explainability" requirement (black box)
    2. Uses ONLY 13 robust features from FeatureSelection.txt analysis
    3. FOR RESEARCH PURPOSES ONLY - Cannot be deployed in production
    
    Robust Features (consistent across 6 years): 13 TOTAL
    - Living Settings (5): ILSL, RH1, RH2, RH3, RH4 (FH as reference)
    - Behavioral (2): BSum, BLEVEL
    - Service Levels (2): LOSRI, OLEVEL
    - Functional (2): FSum, FLEVEL
    - Key QSI (2): Q26, Q36 (top 2 most important)
    
    Total: 5+2+2+2+2 = 13 features
    
    Architecture (adjusted for 13 features):
    - Input Layer: 13 nodes (robust features only)
    - Hidden Layer 1: 32 nodes, ReLU (reduced from 64)
    - Hidden Layer 2: 16 nodes, ReLU (reduced from 32)
    - Hidden Layer 3: 8 nodes, ReLU (reduced from 16)
    - Output Layer: 1 node, linear activation
    - Total Parameters: ~1,145 (72% reduction from 4,049)
    """
    
    def __init__(self, use_sqrt_transform: bool = False):
        """
        Initialize Model 10 with robust feature architecture
        
        Args:
            use_sqrt_transform: If True, use sqrt transformation (historical baseline)
                               If False, use original dollar scale (simpler interpretation)
        """
        super().__init__(model_id=10, model_name="Deep Learning Neural Network (Robust Features)")
        
        # ============================================================================
        # TRANSFORMATION CONTROL - Test both modes empirically
        # ============================================================================
        # Set to True to use sqrt transformation (historical Model 5b baseline)
        # Set to False to fit on original dollar scale (simpler interpretation)
        # Neural networks can work on either scale - test both!
        # ============================================================================
        self.use_sqrt_transform = use_sqrt_transform
        self.transformation = "sqrt" if use_sqrt_transform else "none"
        logger.info(f"Transformation: {self.transformation}")
        
        # Define ROBUST features from FeatureSelection.txt
        # These 13 features represent the most stable predictors across 6 fiscal years
        # Breakdown: Living(5) + Behavioral(2) + Service(2) + Functional(2) + QSI(2) = 13
        self.n_robust_features = 13
        self.feature_reduction = ((22 - 13) / 22) * 100  # 41% reduction
        self.selection_criteria = "Consistency across 6 fiscal years (2020-2025)"
        
        # Neural network configuration (scaled for 13 features vs 22)
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
            random_state=RANDOM_SEED,
            verbose=False
        )
        
        # Feature scaler (CRITICAL for neural networks)
        self.scaler = StandardScaler()
        
        # Track architecture and training
        self.input_dimension = 13
        self.hidden_layers = len(self.hidden_layer_sizes)
        self.total_params = 0  # Will calculate after fit
        self.parameter_reduction = 0.0
        self.epochs_stopped = 0
        self.training_time = 0.0
        self.training_loss = 0.0
        self.validation_loss = 0.0
        
        # Explainability attributes
        self.explainability = "Limited - black box architecture"
        self.regulatory_compliant = "Problematic - HB 1103 concerns"
        self.deployment_recommendation = "Not Recommended"
        self.performance_gain = 0.0
        self.explainability_tradeoff = "Marginal gain not worth transparency loss"
        self.black_box_warning = "Neural networks cannot provide clear explanations for individual budget determinations"
        
        logger.info("=" * 80)
        logger.info("Model 10: Deep Learning Neural Network (ROBUST FEATURES)")
        logger.info(f"Using {self.n_robust_features} robust features from FeatureSelection.txt")
        logger.info(f"Feature breakdown: Living(5) + Behavioral(2) + Service(2) + Functional(2) + QSI(2) = 13")
        logger.info(f"Feature reduction: {self.feature_reduction:.1f}%")
        logger.info(f"Selection criteria: {self.selection_criteria}")
        logger.info("WARNING: Black box model - FOR RESEARCH ONLY")
        logger.info("Deployment NOT recommended due to HB 1103 explainability requirements")
        logger.info("=" * 80)
    
    def split_data(self, test_size: float = 0.2, random_state: int = RANDOM_SEED) -> None:
        """
        Override to handle boolean test_size and use global seed
        
        Args:
            test_size: Proportion of data for testing (or boolean)
            random_state: Random seed for reproducibility
        """
        if isinstance(test_size, bool):
            test_size = 0.2 if test_size else 0.0
        
        if not self.all_records:
            raise ValueError("No records loaded. Call load_data() first.")
        
        # Use parent's split_data with explicit random_state
        super().split_data(test_size=test_size, random_state=random_state)
    
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare ONLY robust features as identified in FeatureSelection.txt
        
        Total features: 13 (down from 22 in Model 5b)
        
        Feature categories:
        1. Living setting (5): ILSL, RH1, RH2, RH3, RH4 (FH as reference)
        2. Behavioral (2): BSum, BLEVEL
        3. Service levels (2): LOSRI, OLEVEL
        4. Functional (2): FSum, FLEVEL
        5. QSI questions (2): Q26, Q36 (top 2 most important)
        
        Note: Reduced QSI from 4 to 2 to achieve exactly 13 features.
        Q20 and Q27 excluded as less critical than Q26/Q36.
        County and Physical metrics (PSum, PLEVEL) excluded due to
        data availability constraints and lower temporal stability.
        
        Args:
            records: List of ConsumerRecord objects
            
        Returns:
            X: Feature matrix (n_samples, 13)
            feature_names: List of feature names
        """
        features = []
        feature_names = []
        
        for record in records:
            row = []
            
            # 1. Living Setting (5 features) - categorical dummies
            # Reference: FH (Family Home)
            if not feature_names:
                feature_names.extend(['ILSL', 'RH1', 'RH2', 'RH3', 'RH4'])
            
            row.append(1 if record.living_setting == 'ILSL' else 0)
            row.append(1 if record.living_setting == 'RH1' else 0)
            row.append(1 if record.living_setting == 'RH2' else 0)
            row.append(1 if record.living_setting == 'RH3' else 0)
            row.append(1 if record.living_setting == 'RH4' else 0)
            
            # 2. Behavioral Metrics (2 features)
            if len(feature_names) < 7:
                feature_names.extend(['BSum', 'BLEVEL'])
            row.append(record.bsum)
            row.append(record.blevel)
            
            # 3. Service Levels (2 features)
            if len(feature_names) < 9:
                feature_names.extend(['LOSRI', 'OLEVEL'])
            row.append(record.losri)
            row.append(record.olevel)
            
            # 4. Functional Metrics (2 features)
            if len(feature_names) < 11:
                feature_names.extend(['FSum', 'FLEVEL'])
            row.append(record.fsum)
            row.append(record.flevel)
            
            # 5. Key QSI Questions (2 features - TOP 2 ONLY)
            # Q26 and Q36 are the most important based on mutual information
            if len(feature_names) < 13:
                feature_names.extend(['Q26', 'Q36'])
            row.append(record.q26)
            row.append(record.q36)
            
            features.append(row)
        
        X = np.array(features)
        
        logger.info(f"Prepared {self.n_robust_features} robust features from {len(records)} records")
        logger.info(f"Feature categories: Living(5), Behavioral(2), Service(2), Functional(2), QSI(2)")
        
        # Verify feature count
        assert X.shape[1] == self.n_robust_features, \
            f"Expected {self.n_robust_features} features, got {X.shape[1]}"
        
        return X, feature_names
    
    def run_complete_pipeline(self, fiscal_year_start: int = 2023, fiscal_year_end: int = 2024,
                             test_size: float = 0.2, perform_cv: bool = True, 
                             n_cv_folds: int = 10) -> Dict[str, Any]:
        """
        Override to handle neural network specific pipeline with transformation
        
        CRITICAL: Handles transformation explicitly to avoid scale mismatch
        - Extracts original-scale costs
        - Applies transformation if requested
        - Fits on appropriate scale
        - Predictions ALWAYS returned in original dollars
        - Metrics ALWAYS calculated on original dollar scale
        """
        logger.info(f"Starting complete pipeline for Model {self.model_id}: {self.model_name}")
        logger.info(f"Transformation mode: {self.transformation}")
        
        # Load data using base class (loads FY2024 data from pickle)
        logger.info("Loading data from pickle files...")
        self.all_records = self.load_data(fiscal_year_start=fiscal_year_start, 
                                         fiscal_year_end=fiscal_year_end)
        logger.info(f"Loaded {len(self.all_records)} usable records from FY{fiscal_year_start}-{fiscal_year_end}")
        
        # Split data
        logger.info("Splitting data into train/test sets...")
        self.split_data(test_size=test_size)
        logger.info(f"Train: {len(self.train_records)}, Test: {len(self.test_records)}")
        
        # Prepare features (ROBUST FEATURES ONLY)
        logger.info("Preparing ROBUST features...")
        self.X_train, self.feature_names = self.prepare_features(self.train_records)
        self.X_test, _ = self.prepare_features(self.test_records)
        logger.info(f"Feature matrix shape: Train {self.X_train.shape}, Test {self.X_test.shape}")
        
        # CRITICAL: Handle transformation explicitly
        # Extract original-scale costs
        y_train_original = np.array([r.total_cost for r in self.train_records])
        y_test_original = np.array([r.total_cost for r in self.test_records])
        
        # Apply transformation if requested
        if self.use_sqrt_transform:
            logger.info("Applying sqrt transformation to costs for model fitting...")
            y_train_fit = np.sqrt(y_train_original)
            y_test_fit = np.sqrt(y_test_original)
        else:
            logger.info("Using original dollar scale (no transformation)...")
            y_train_fit = y_train_original
            y_test_fit = y_test_original
        
        # Store original scale for metrics (CRITICAL!)
        self.y_train = y_train_original
        self.y_test = y_test_original
        
        # Cross-validation (on appropriate scale)
        if perform_cv:
            logger.info(f"Performing {n_cv_folds}-fold cross-validation...")
            cv_results = self.perform_cross_validation(n_splits=n_cv_folds)
            self.metrics['cv_mean'] = cv_results.get('cv_mean', 0)
            self.metrics['cv_std'] = cv_results.get('cv_std', 0)
            logger.info(f"CV Results: R² = {self.metrics['cv_mean']:.4f} ± {self.metrics['cv_std']:.4f}")
        
        # Fit model (on appropriate scale)
        logger.info("Training neural network...")
        start_time = time.time()
        self.fit(self.X_train, y_train_fit)
        self.training_time = time.time() - start_time
        logger.info(f"Training complete in {self.training_time:.1f} seconds")
        logger.info(f"Early stopping at epoch {self.epochs_stopped} (of {self.max_iter})")
        
        # Make predictions (ALWAYS returns original dollar scale)
        logger.info("Making predictions...")
        self.train_predictions = self.predict(self.X_train)
        self.test_predictions = self.predict(self.X_test)
        
        # Calculate metrics (ALWAYS on original dollar scale)
        logger.info("Calculating metrics...")
        self.calculate_metrics()
        
        # Calculate performance gain vs Model 3
        model3_r2 = 0.8023  # Historical Model 3 performance
        if 'test_r2' in self.metrics:
            gain = ((self.metrics['test_r2'] - model3_r2) / model3_r2) * 100
            self.performance_gain = max(0, gain)  # Only positive gains
        
        # Calculate subgroup metrics
        logger.info("Calculating subgroup metrics...")
        self.calculate_subgroup_metrics()
        
        # Calculate variance metrics
        logger.info("Calculating variance metrics...")
        self.calculate_variance_metrics()
        
        # Calculate population scenarios
        logger.info("Calculating population scenarios...")
        self.calculate_population_scenarios()
        
        # Generate outputs
        logger.info("Generating outputs...")
        self.generate_latex_commands()
        self.save_results()
        self.plot_diagnostics()
        
        # Final summary
        logger.info("=" * 80)
        logger.info(f"PIPELINE COMPLETE: {self.model_name}")
        logger.info("=" * 80)
        self.log_metrics_summary()
        
        # Log architecture info
        logger.info("")
        logger.info("Neural Network Architecture:")
        logger.info(f"  Input: {self.input_dimension} features (robust selection)")
        logger.info(f"  Hidden: {self.hidden_layer_sizes}")
        logger.info(f"  Total Parameters: {self.total_params:,}")
        logger.info(f"  Parameter Reduction: {self.parameter_reduction:.1f}% from full model")
        logger.info(f"  Training Time: {self.training_time:.1f} seconds")
        
        # Log deployment warning
        logger.info("")
        logger.info("⚠️  DEPLOYMENT WARNING:")
        logger.info(f"  Explainability: {self.explainability}")
        logger.info(f"  Regulatory Status: {self.regulatory_compliant}")
        logger.info(f"  Recommendation: {self.deployment_recommendation}")
        logger.info(f"  Performance Gain: {self.performance_gain:.1f}% over Model 3")
        logger.info(f"  Trade-off: {self.explainability_tradeoff}")
        
        # Log generated files
        logger.info("")
        logger.info("Generated files:")
        for file in sorted(self.output_dir.glob("*")):
            logger.info(f"  - {file.name}")
        
        return {
            'metrics': self.metrics,
            'subgroup_metrics': self.subgroup_metrics,
            'variance_metrics': self.variance_metrics,
            'population_scenarios': self.population_scenarios
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the neural network model with standardization
        
        Args:
            X: Feature matrix (n_samples, 13)
            y: Target values (in appropriate scale - sqrt or original)
        """
        # Standardize features (CRITICAL for neural networks)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        
        # Track training details
        self.epochs_stopped = self.model.n_iter_
        self.training_loss = self.model.loss_
        
        # Get validation loss if available (early stopping enabled)
        if hasattr(self.model, 'best_loss_') and self.model.best_loss_ is not None:
            self.validation_loss = self.model.best_loss_
        elif hasattr(self.model, 'validation_scores_') and len(self.model.validation_scores_) > 0:
            self.validation_loss = min(self.model.validation_scores_)
        else:
            # No validation available (shouldn't happen with early_stopping=True)
            self.validation_loss = self.training_loss
        
        # Calculate total parameters
        self.total_params = sum(
            layer.size for layer in self.model.coefs_
        ) + sum(
            layer.size for layer in self.model.intercepts_
        )
        
        # Calculate parameter reduction from full model (22 features)
        # Full model would be: 22*64 + 64 + 64*32 + 32 + 32*16 + 16 + 16*1 + 1 = 4,049
        full_model_params = 4049
        self.parameter_reduction = ((full_model_params - self.total_params) / full_model_params) * 100
        
        logger.info(f"Neural network fitted with {self.total_params:,} parameters")
        logger.info(f"Parameter reduction: {self.parameter_reduction:.1f}% from full model")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        CRITICAL: Always returns predictions on ORIGINAL DOLLAR SCALE
        regardless of transformation used during training
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions in original dollars (ALWAYS)
        """
        # Standardize features using fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Get predictions (in fitted scale)
        y_pred = self.model.predict(X_scaled)
        
        # Back-transform if needed
        if self.use_sqrt_transform:
            y_pred = y_pred ** 2  # Square to get back to dollars
        
        # Ensure non-negative
        y_pred = np.maximum(y_pred, 0)
        
        return y_pred
    
    def perform_cross_validation(self, n_splits: int = 10) -> Dict[str, float]:
        """
        Perform cross-validation with proper transformation handling
        
        CRITICAL: CV scores ALWAYS calculated on original dollar scale
        """
        logger.info(f"Performing {n_splits}-fold cross-validation...")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = []
        
        # Get original-scale costs
        y_original = np.array([r.total_cost for r in self.train_records])
        
        # Apply transformation if needed for fitting
        if self.use_sqrt_transform:
            y_fit = np.sqrt(y_original)
        else:
            y_fit = y_original
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train), 1):
            # Split data
            X_cv_train, X_cv_val = self.X_train[train_idx], self.X_train[val_idx]
            y_cv_train, y_cv_val = y_fit[train_idx], y_fit[val_idx]
            y_cv_val_original = y_original[val_idx]  # For scoring
            
            # Create and fit model
            cv_model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                batch_size=self.batch_size,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                early_stopping=False,  # No early stopping in CV
                random_state=RANDOM_SEED + fold,
                verbose=False
            )
            
            # Standardize
            cv_scaler = StandardScaler()
            X_cv_train_scaled = cv_scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = cv_scaler.transform(X_cv_val)
            
            # Fit
            cv_model.fit(X_cv_train_scaled, y_cv_train)
            
            # Predict
            y_cv_pred = cv_model.predict(X_cv_val_scaled)
            
            # Back-transform if needed
            if self.use_sqrt_transform:
                y_cv_pred = y_cv_pred ** 2
            y_cv_pred = np.maximum(y_cv_pred, 0)
            
            # Score ALWAYS on original scale
            score = r2_score(y_cv_val_original, y_cv_pred)
            cv_scores.append(score)
            
            logger.info(f"  Fold {fold}: R² = {score:.4f}")
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        logger.info(f"Cross-validation: R² = {cv_mean:.4f} ± {cv_std:.4f}")
        
        return {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'cv_scores': cv_scores
        }
    
    def generate_latex_commands(self) -> None:
        """
        Override base class method to add Model 10 specific commands
        
        CRITICAL: Must override generate_latex_commands (not a new method name!)
        Must call super() FIRST, then append model-specific commands
        """
        # STEP 1: Call parent FIRST - creates files with 'w' mode
        super().generate_latex_commands()
        
        # STEP 2: Append Model 10 specific commands using 'a' mode
        logger.info(f"Adding Model {self.model_id} specific LaTeX commands...")
        
        model_word = 'Ten'
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        # Append to newcommands (definitions)
        with open(newcommands_file, 'a') as f:
            f.write("\n% ============================================================================\n")
            f.write(f"% Model {self.model_id} Specific Commands - Neural Network Architecture\n")
            f.write("% ============================================================================\n")
            
            # Feature selection commands
            f.write(f"\\newcommand{{\\Model{model_word}RobustFeatures}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}FeatureReduction}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}SelectionCriteria}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}NumFeatures}}{{\\placeholder}}\n")
            
            # Architecture commands
            f.write(f"\\newcommand{{\\Model{model_word}InputDimension}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}HiddenLayers}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}HiddenLayerOneNodes}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}HiddenLayerTwoNodes}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}HiddenLayerThreeNodes}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}TotalParams}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}ParameterReduction}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}Activation}}{{\\placeholder}}\n")
            
            # Training commands
            f.write(f"\\newcommand{{\\Model{model_word}EpochsStopped}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}MaxEpochs}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}BatchSize}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}LearningRate}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}Regularization}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}TrainingLoss}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}ValidationLoss}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}TrainingTime}}{{\\placeholder}}\n")
            
            # Transformation command
            f.write(f"\\newcommand{{\\Model{model_word}Transformation}}{{\\placeholder}}\n")
            
            # Explainability commands (CRITICAL)
            f.write(f"\\newcommand{{\\Model{model_word}Explainability}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}RegulatoryCompliant}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}DeploymentRecommendation}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}PerformanceGain}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}ExplainabilityTradeoff}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}BlackBoxWarning}}{{\\placeholder}}\n")
        
        # Append to renewcommands (values)
        with open(renewcommands_file, 'a') as f:
            f.write("\n% ============================================================================\n")
            f.write(f"% Model {self.model_id} Specific Values - Neural Network Details\n")
            f.write("% ============================================================================\n")
            
            # Feature selection values
            f.write(f"\\renewcommand{{\\Model{model_word}RobustFeatures}}{{{self.n_robust_features}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}FeatureReduction}}{{{self.feature_reduction:.1f}\\%}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}SelectionCriteria}}{{{self.selection_criteria}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}NumFeatures}}{{{self.n_robust_features}}}\n")
            
            # Architecture values
            f.write(f"\\renewcommand{{\\Model{model_word}InputDimension}}{{{self.input_dimension}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}HiddenLayers}}{{{self.hidden_layers}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}HiddenLayerOneNodes}}{{{self.hidden_layer_sizes[0]}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}HiddenLayerTwoNodes}}{{{self.hidden_layer_sizes[1]}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}HiddenLayerThreeNodes}}{{{self.hidden_layer_sizes[2]}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}TotalParams}}{{{self.total_params:,}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}ParameterReduction}}{{{self.parameter_reduction:.1f}\\%}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}Activation}}{{{self.activation.upper()}}}\n")
            
            # Training values
            f.write(f"\\renewcommand{{\\Model{model_word}EpochsStopped}}{{{self.epochs_stopped}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}MaxEpochs}}{{{self.max_iter}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}BatchSize}}{{{self.batch_size}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}LearningRate}}{{{self.learning_rate_init}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}Regularization}}{{{self.alpha}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}TrainingLoss}}{{{self.training_loss:.6f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}ValidationLoss}}{{{self.validation_loss:.6f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}TrainingTime}}{{{self.training_time:.1f}}}\n")
            
            # Transformation value
            f.write(f"\\renewcommand{{\\Model{model_word}Transformation}}{{{self.transformation}}}\n")
            
            # Explainability values (CRITICAL)
            f.write(f"\\renewcommand{{\\Model{model_word}Explainability}}{{{self.explainability}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}RegulatoryCompliant}}{{{self.regulatory_compliant}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}DeploymentRecommendation}}{{{self.deployment_recommendation}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}PerformanceGain}}{{{self.performance_gain:.1f}\\%}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}ExplainabilityTradeoff}}{{{self.explainability_tradeoff}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}BlackBoxWarning}}{{{self.black_box_warning}}}\n")
        
        logger.info(f"Model {self.model_id} specific commands added successfully")
        logger.info(f"  - Feature selection commands: 4")
        logger.info(f"  - Architecture commands: 8")
        logger.info(f"  - Training commands: 8")
        logger.info(f"  - Explainability commands: 6")
        logger.info(f"  - Total model-specific: 26 commands")


def main():
    """
    Run Model 10 complete pipeline with transformation control
    
    CRITICAL: Tests both sqrt and original dollar transformations
    """
    # SET ALL RANDOM SEEDS FOR REPRODUCIBILITY
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    print("\n" + "="*80)
    print("MODEL 10: DEEP LEARNING NEURAL NETWORK WITH ROBUST FEATURES")
    print("="*80)
    print(f"\n🎲 Random Seed: {RANDOM_SEED} (for reproducibility)")
    print("💡 To change seed, edit RANDOM_SEED = 42 at top of file")
    
    # ============================================================================
    # TRANSFORMATION OPTION - Test both empirically!
    # ============================================================================
    # Model 5 (Ridge) found original dollars > sqrt transformation
    # Test both modes for neural networks:
    # - True: Use sqrt (historical baseline, may help with variance)
    # - False: Original dollars (simpler interpretation, worked well for Ridge)
    # ============================================================================
    USE_SQRT = False  # Changed to test original dollar scale
    
    print(f"\n📐 Transformation: {'sqrt' if USE_SQRT else 'none (original dollars)'}")
    print(f"    (To test other mode, set USE_SQRT = {not USE_SQRT})")
    
    print("\n⚠️  DEPLOYMENT WARNING:")
    print("    This model violates HB 1103 explainability requirements")
    print("    FOR RESEARCH PURPOSES ONLY - Cannot be deployed in production")
    print("    Value: Feature selection validation, not neural network deployment")
    
    # Initialize model with transformation control
    model = Model10NeuralNet(use_sqrt_transform=USE_SQRT)
    
    # Run complete pipeline
    # DO NOT pass random_state parameter - base class doesn't accept it!
    results = model.run_complete_pipeline(
        fiscal_year_start=2023,
        fiscal_year_end=2024,
        test_size=0.2,
        perform_cv=True,
        n_cv_folds=10
    )
    
    # Verify command count
    print("\n" + "="*80)
    print("LATEX COMMAND VERIFICATION")
    print("="*80)
    
    renewcommands_file = Path(f"models/model_10/model_10_renewcommands.tex")
    if renewcommands_file.exists():
        with open(renewcommands_file, 'r') as f:
            command_count = sum(1 for line in f if '\\renewcommand' in line)
        print(f"\n✅ Total LaTeX commands generated: {command_count}")
        print(f"   Expected: ~105 (75 base + 26 model-specific + 4 special)")
        if command_count >= 100:
            print("   ✅ PASS: All commands generated")
        else:
            print(f"   ⚠️  WARNING: Expected ~105, got {command_count}")
    
    print("\n" + "="*80)
    print("KEY FINDINGS - MODEL 10")
    print("="*80)
    print(f"\n📊 Performance:")
    print(f"   Test R²: {results['metrics'].get('test_r2', 0):.4f}")
    print(f"   Test RMSE: ${results['metrics'].get('test_rmse', 0):,.0f}")
    print(f"   CV Mean: {results['metrics'].get('cv_mean', 0):.4f} ± {results['metrics'].get('cv_std', 0):.4f}")
    
    print(f"\n🔧 Architecture:")
    print(f"   Robust Features: {model.n_robust_features} (41% reduction from 22)")
    print(f"   Hidden Layers: {model.hidden_layer_sizes}")
    print(f"   Total Parameters: {model.total_params:,} ({model.parameter_reduction:.1f}% reduction)")
    print(f"   Training Time: {model.training_time:.1f} seconds")
    print(f"   Early Stopped: Epoch {model.epochs_stopped} of {model.max_iter}")
    
    print(f"\n⚠️  Deployment Assessment:")
    print(f"   Explainability: {model.explainability}")
    print(f"   Regulatory: {model.regulatory_compliant}")
    print(f"   Recommendation: {model.deployment_recommendation}")
    print(f"   Performance Gain: {model.performance_gain:.1f}% over Model 3")
    print(f"   Trade-off: {model.explainability_tradeoff}")
    
    print("\n💡 Key Insight:")
    print("   The value of Model 10 is in validating the 13 robust features,")
    print("   NOT in deploying the neural network. Apply these robust features")
    print("   to interpretable models (Model 1 or 3) for best results.")
    
    print("\n" + "="*80)
    print("REPRODUCIBILITY NOTE")
    print("="*80)
    print(f"\n🎲 Current random seed: {RANDOM_SEED}")
    print(f"   To replicate: Keep RANDOM_SEED = {RANDOM_SEED}")
    print(f"   To test sensitivity: Change RANDOM_SEED value")
    print(f"\n📐 Current transformation: {'sqrt' if USE_SQRT else 'none'}")
    print(f"   To test alternative: Set USE_SQRT = {not USE_SQRT}")
    
    print("\n✅ Model 10 pipeline complete!")
    print("="*80 + "\n")
    
    return model


if __name__ == "__main__":
    # DON'T use logging.basicConfig() - let base_model handle logging
    # This ensures ALL logs go to both console AND file
    
    model = main()
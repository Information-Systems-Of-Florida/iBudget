"""
Main Pipeline for ISF/iBudget Model Calibration
This script orchestrates the calibration of all 10 alternative models
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import base classes (these would be in separate module files)
# from base_model_framework import BaseISFModel, ModelComparison

class ModelPipeline:
    """Main pipeline for model calibration and comparison"""
    
    def __init__(self, config_path: str = "configs/master_config.json"):
        """
        Initialize pipeline
        
        Parameters:
        -----------
        config_path : str
            Path to master configuration file
        """
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.models = []
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def setup_logging(self):
        """Setup logging for pipeline"""
        log_dir = Path(self.config['master_config']['output_settings']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                ),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("ModelPipeline")
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data"""
        self.logger.info("Loading data...")
        
        data_path = self.config['master_config']['data_settings']['data_path']
        self.data = pd.read_csv(data_path)
        
        self.logger.info(f"Data loaded: {len(self.data)} rows, {len(self.data.columns)} columns")
        
        # Basic data validation
        self.validate_data()
        
        return self.data
    
    def validate_data(self):
        """Validate data quality and requirements"""
        self.logger.info("Validating data...")
        
        # Check for required columns
        dependent_var = self.config['master_config']['data_settings']['dependent_variable']
        if dependent_var not in self.data.columns:
            raise ValueError(f"Dependent variable {dependent_var} not found in data")
        
        # Check for missing values
        missing_rates = self.data.isnull().mean()
        high_missing = missing_rates[missing_rates > 
                                     self.config['master_config']['validation_thresholds']['acceptable_missing_rate']]
        
        if len(high_missing) > 0:
            self.logger.warning(f"High missing rates detected in: {high_missing.index.tolist()}")
        
        # Check for outliers in dependent variable
        dependent_data = self.data[dependent_var]
        Q1 = dependent_data.quantile(0.25)
        Q3 = dependent_data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((dependent_data < Q1 - 3 * IQR) | (dependent_data > Q3 + 3 * IQR)).sum()
        
        self.logger.info(f"Outliers detected in dependent variable: {outliers} ({100*outliers/len(self.data):.2f}%)")
        
    def prepare_common_features(self) -> pd.DataFrame:
        """Prepare common features for all models"""
        self.logger.info("Preparing common features...")
        
        features = pd.DataFrame()
        
        # Get common features from config
        common_features = self.config['master_config']['common_features']
        
        # Add demographic features
        for feature in common_features['demographic']:
            if feature in self.data.columns:
                # One-hot encode categorical variables
                if self.data[feature].dtype == 'object':
                    encoded = pd.get_dummies(self.data[feature], prefix=feature)
                    features = pd.concat([features, encoded], axis=1)
                else:
                    features[feature] = self.data[feature]
        
        # Add QSI variables
        for var in common_features['qsi_variables']:
            if var in self.data.columns:
                features[var] = self.data[var]
        
        # Add sum scores
        for score in common_features['sum_scores']:
            if score in self.data.columns:
                features[score] = self.data[score]
        
        self.logger.info(f"Common features prepared: {features.shape}")
        
        return features
    
    def split_data(self, features: pd.DataFrame, target: pd.Series):
        """Split data into train and test sets"""
        self.logger.info("Splitting data into train/test sets...")
        
        data_settings = self.config['master_config']['data_settings']
        
        # Stratified split
        stratify_var = data_settings.get('stratification_variable')
        if stratify_var and stratify_var in self.data.columns:
            stratify = self.data[stratify_var]
        else:
            stratify = None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, target,
            test_size=data_settings['test_size'],
            random_state=data_settings['random_seed'],
            stratify=stratify
        )
        
        self.logger.info(f"Train set: {len(self.X_train)} samples")
        self.logger.info(f"Test set: {len(self.X_test)} samples")
        
    def load_model_class(self, model_config: Dict[str, Any]):
        """
        Dynamically load model class based on configuration
        
        Parameters:
        -----------
        model_config : dict
            Model configuration
            
        Returns:
        --------
        Model class instance
        """
        model_type = model_config['model_type']
        model_id = model_config['model_id']
        model_name = model_config['model_name']
        
        # Import the appropriate model class
        # In practice, these would be in separate files
        if model_type == 'linear_regression':
            from models.alternative_1 import Alternative1Model
            return Alternative1Model(model_id, model_name, model_config)
        elif model_type == 'glm_gamma':
            from models.alternative_2 import Alternative2Model
            return Alternative2Model(model_id, model_name, model_config)
        # ... continue for all 10 models
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def calibrate_single_model(self, model_config: Dict[str, Any]):
        """
        Calibrate a single model
        
        Parameters:
        -----------
        model_config : dict
            Model configuration
            
        Returns:
        --------
        Calibrated model instance
        """
        model_id = model_config['model_id']
        self.logger.info(f"="*50)
        self.logger.info(f"Calibrating Model {model_id}: {model_config['model_name']}")
        self.logger.info(f"="*50)
        
        try:
            # Load model class
            model = self.load_model_class(model_config)
            
            # Prepare features (may be model-specific)
            X_train = model.prepare_features(self.X_train.copy())
            X_test = model.prepare_features(self.X_test.copy())
            
            # Fit model
            model.fit(X_train, self.y_train)
            
            # Evaluate on test set
            model.evaluate(X_test, self.y_test)
            
            # Cross-validation
            X_full = pd.concat([X_train, X_test])
            y_full = pd.concat([self.y_train, self.y_test])
            cv_folds = self.config['master_config']['data_settings']['cv_folds']
            model.cross_validate(X_full, y_full, cv=cv_folds)
            
            # Generate plots
            model.generate_diagnostic_plots(X_test, self.y_test)
            
            # Save outputs
            model.save_metrics()
            model.save_predictions(X_test, self.y_test)
            model.to_tex()
            
            self.logger.info(f"Model {model_id} calibration completed successfully")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error calibrating model {model_id}: {str(e)}")
            raise
    
    def calibrate_all_models(self):
        """Calibrate all models"""
        self.logger.info("Starting calibration of all models...")
        
        model_configs = self.config['model_configs']
        
        for model_key in sorted(model_configs.keys()):
            model_config = model_configs[model_key]
            
            try:
                model = self.calibrate_single_model(model_config)
                self.models.append(model)
            except Exception as e:
                self.logger.error(f"Failed to calibrate {model_key}: {str(e)}")
                # Continue with other models
                continue
        
        self.logger.info(f"Successfully calibrated {len(self.models)} models")
    
    def generate_comparisons(self):
        """Generate model comparisons"""
        if len(self.models) == 0:
            self.logger.warning("No models to compare")
            return
        
        self.logger.info("Generating model comparisons...")
        
        # Create comparison instance
        comparison = ModelComparison(self.models)
        
        # Generate comparison outputs
        comparison.generate_comparison_matrix()
        comparison.generate_performance_summary()
        comparison.generate_master_tex()
        
        self.logger.info("Comparison generation completed")
    
    def generate_common_tex_files(self):
        """Generate common TeX parameter files"""
        self.logger.info("Generating common TeX files...")
        
        common_dir = Path(self.config['master_config']['output_settings']['common_dir'])
        common_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate common demographic parameters
        demographic_tex = """% Common Demographic Parameters
\\newcommand{\\CommonAgeGroups}{0-20, 21-30, 31+}
\\newcommand{\\CommonLivingSettings}{FM, iLSL, RH1, RH2, RH3, RH4}
\\newcommand{\\CommonDiagnoses}{Intellectual Disability, Autism Spectrum, Cerebral Palsy, Other}
"""
        
        with open(common_dir / 'common_demographic_params.tex', 'w') as f:
            f.write(demographic_tex)
        
        # Generate common QSI parameters
        qsi_tex = """% Common QSI Parameters
\\newcommand{\\CommonQSIVariables}{Q16, Q18, Q20, Q21, Q23, Q28, Q33, Q34, Q36, Q43}
\\newcommand{\\CommonSumScores}{BSum, FHFSum, SLFSum, SLBSum}
"""
        
        with open(common_dir / 'common_qsi_params.tex', 'w') as f:
            f.write(qsi_tex)
        
        self.logger.info("Common TeX files generated")
    
    def run(self):
        """Execute the complete pipeline"""
        self.logger.info("="*60)
        self.logger.info("Starting ISF/iBudget Model Calibration Pipeline")
        self.logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("="*60)
        
        try:
            # Load and prepare data
            self.load_data()
            
            # Prepare common features
            features = self.prepare_common_features()
            
            # Get target variable
            target_var = self.config['master_config']['data_settings']['dependent_variable']
            target = self.data[target_var]
            
            # Split data
            self.split_data(features, target)
            
            # Generate common TeX files
            self.generate_common_tex_files()
            
            # Calibrate all models
            self.calibrate_all_models()
            
            # Generate comparisons
            self.generate_comparisons()
            
            self.logger.info("="*60)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info("="*60)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def generate_summary_report(self):
        """Generate executive summary report"""
        self.logger.info("Generating summary report...")
        
        if len(self.models) == 0:
            self.logger.warning("No models available for summary")
            return
        
        # Collect best model based on test R^2
        best_model = max(self.models, 
                        key=lambda m: m.metrics.get('test', {}).get('r2', 0))
        
        summary = f"""
# ISF/iBudget Model Calibration Summary
## Date: {datetime.now().strftime('%Y-%m-%d')}

## Models Calibrated: {len(self.models)}

## Best Performing Model:
- **Model ID**: {best_model.model_id}
- **Name**: {best_model.model_name}
- **Test R^2**: {best_model.metrics['test']['r2']:.4f}
- **Test RMSE**: ${best_model.metrics['test']['rmse']:.2f}

## All Models Performance:
"""
        
        for model in sorted(self.models, key=lambda m: m.model_id):
            if 'test' in model.metrics:
                summary += f"""
### Alternative {model.model_id}: {model.model_name}
- Test R^2: {model.metrics['test']['r2']:.4f}
- Test RMSE: ${model.metrics['test']['rmse']:.2f}
- CV R^2 (mean +- std): {model.metrics.get('cv', {}).get('r2_mean', 'N/A'):.4f} +- {model.metrics.get('cv', {}).get('r2_std', 'N/A'):.4f}
"""
        
        # Save summary
        summary_path = Path(self.config['master_config']['output_settings']['report_dir']) / 'calibration_summary.md'
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        self.logger.info(f"Summary report saved to {summary_path}")


def main():
    """Main entry point"""
    
    # Parse command line arguments if needed
    import argparse
    parser = argparse.ArgumentParser(description='ISF/iBudget Model Calibration Pipeline')
    parser.add_argument('--config', type=str, default='configs/master_config.json',
                       help='Path to master configuration file')
    parser.add_argument('--models', type=str, nargs='+', 
                       help='Specific models to calibrate (e.g., 1 2 3)')
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = ModelPipeline(config_path=args.config)
    
    # Optionally filter models
    if args.models:
        model_ids = [int(m) for m in args.models]
        # Filter config to only include specified models
        all_configs = pipeline.config['model_configs']
        filtered_configs = {k: v for k, v in all_configs.items() 
                           if v['model_id'] in model_ids}
        pipeline.config['model_configs'] = filtered_configs
    
    # Run pipeline
    pipeline.run()
    
    # Generate summary
    pipeline.generate_summary_report()


if __name__ == "__main__":
    main()

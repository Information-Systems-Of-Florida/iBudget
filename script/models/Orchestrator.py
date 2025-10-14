"""
orchestrator.py
===============
Master orchestrator for running multiple iBudget models with optimal feature configuration.

Features:
- JSON-based configuration system
- Feature set from Feature Selection analysis
- Runs models 1, 2, 3, 4, 5, 6, 9
- Stops immediately on any error
- Generates combined predictions CSV
- Creates comparison reports
- Calls generate_comparison_plots.py

Usage:
    python orchestrator.py                              # Uses ConsensusConfiguration.json
    python orchestrator.py --config MyConfig.json       # Uses custom config
"""

import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import logging

# Import model classes
from model_1_reeval import Model1Linear
from model_2_gamma import Model2GLMGamma
from model_3_robust import Model3Robust
from model_4_wls import Model4WLS
from model_5_ridge import Model5Ridge
from model_6_lognormal import Model6LogNormal
from model_9_random_forest import Model9RandomForest

# ============================================================================
# FEATURE CONFIGURATION (Optimal from Feature Selection)
# ============================================================================
# Interaction terms use lambda functions and cannot be in JSON

FEATURE_CONFIG_OPTIMAL = {
    'categorical': {
        'living_setting': {
            'categories': ['ILSL', 'RH1', 'RH2', 'RH3', 'RH4'],
            'reference': 'FH'
        },
        'age_group': {
            'categories': ['Age21_30', 'Age31Plus'],
            'reference': 'Age3_20'
        },
        'county': {
            'use_all': True,  # Special flag for base class to include all counties
            'reference': None  # Will auto-select most common
        }
    },
    'numeric': [
        'age',      # Continuous age
        'bsum',     # Behavioral sum score
        'fsum',     # Functional sum score
        'psum',     # Physical sum score
        'losri',    # Level of Support Rating Index
        'olevel',   # Overall support level
        'blevel',   # Behavioral support level
        'flevel',   # Functional support level
        'plevel'    # Physical support level
    ],
    'qsi': [19, 21, 26, 27, 30, 36, 44],  # 7 items from Feature Selection
    'interactions': [
        # Family Home × Functional Sum
        ('FH_x_FSum', lambda r: (1 if r.living_setting == 'FH' else 0) * float(r.fsum)),
        # Supported Living × Functional Sum
        ('SL_x_FSum', lambda r: (1 if r.living_setting in ['RH1','RH2','RH3','RH4'] else 0) * float(r.fsum)),
        # Supported Living × Behavioral Sum
        ('SL_x_BSum', lambda r: (1 if r.living_setting in ['RH1','RH2','RH3','RH4'] else 0) * float(r.bsum))
    ]
}

# Model class registry
MODEL_CLASSES = {
    1: Model1Linear,
    2: Model2GLMGamma,
    3: Model3Robust,
    4: Model4WLS,
    5: Model5Ridge,
    6: Model6LogNormal,
    9: Model9RandomForest
}

# ============================================================================
# ORCHESTRATOR CLASS
# ============================================================================

class ModelOrchestrator:
    """
    Master orchestrator for running multiple iBudget models
    """
    
    def __init__(self, config_path: str = "ConsensusConfiguration.json"):
        """
        Initialize orchestrator
        
        Args:
            config_path: Path to JSON configuration file
        """
        self.config_path = Path(config_path)
        self.config = self.load_configuration()
        self.results = {}
        self.predictions = {}
        
        # Setup logging
        self.setup_logging()
        
        # Create output directories
        self.setup_output_dirs()
        
    def load_configuration(self) -> Dict:
        """Load and validate JSON configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please create ConsensusConfiguration.json or specify --config"
            )
        
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required = ['scenario_name', 'data_settings', 'models_to_run', 'model_settings']
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")
        
        return config
    
    def setup_logging(self):
        """Setup logging system"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create logs directory
        log_dir = Path('../../report/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f'orchestrator_{timestamp}.log'
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('Orchestrator')
        
        self.logger.info("=" * 80)
        self.logger.info("IBUDGET MODEL ORCHESTRATOR")
        self.logger.info("=" * 80)
        self.logger.info(f"Configuration: {self.config_path}")
        self.logger.info(f"Scenario: {self.config['scenario_name']}")
        self.logger.info(f"Log file: {log_file}")
        
    def setup_output_dirs(self):
        """Create output directory structure"""
        output_settings = self.config.get('output_settings', {})
        base_dir = Path(output_settings.get('output_base_dir', '../../report/models'))
        model_id = output_settings.get('orchestration_model_id', 70)
        
        self.output_dir = base_dir / f'model_{model_id}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def run_single_model(self, model_id: int) -> Dict[str, Any]:
        """
        Run a single model
        
        Args:
            model_id: Model number (1-9)
            
        Returns:
            Dictionary with model results
            
        Raises:
            Exception: If model fails (stops orchestrator)
        """
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"RUNNING MODEL {model_id}")
        self.logger.info("=" * 80)
        
        # Get model configuration
        model_config = self.config['model_settings'][str(model_id)]
        self.logger.info(f"Model: {model_config.get('name', f'Model {model_id}')}")
        self.logger.info(f"Transformation: {model_config.get('transformation', 'none')}")
        self.logger.info(f"Outlier removal: {model_config.get('use_outlier_removal', False)}")
        
        try:
            # Get model class
            ModelClass = MODEL_CLASSES[model_id]
            
            # Prepare model parameters
            model_params = {
                'random_seed': self.config['data_settings']['random_seed'],
                'log_suffix': f"model_{model_id}_orchestrated"
            }
            
            # Add model-specific settings
            if 'transformation' in model_config:
                trans = model_config['transformation']
                if trans == 'sqrt':
                    model_params['use_sqrt_transform'] = True
                # log and none handled by base class
            
            if model_config.get('use_outlier_removal', False):
                model_params['use_outlier_removal'] = True
                if 'outlier_threshold' in model_config:
                    model_params['outlier_threshold'] = model_config['outlier_threshold']
            
            # Initialize model
            self.logger.info(f"Initializing {ModelClass.__name__}...")
            model = ModelClass(**model_params)
            
            # Set feature configuration
            model.feature_config = FEATURE_CONFIG_OPTIMAL
            self.logger.info("Applied optimal feature configuration from Feature Selection")
            
            # Run pipeline
            data_settings = self.config['data_settings']
            pipeline_settings = self.config.get('pipeline_settings', {})
            
            self.logger.info("Running complete pipeline...")
            results = model.run_complete_pipeline(
                fiscal_year_start=data_settings['fiscal_year_start'],
                fiscal_year_end=data_settings['fiscal_year_end'],
                test_size=data_settings['test_size'],
                perform_cv=pipeline_settings.get('perform_cv', True),
                n_cv_folds=pipeline_settings.get('cv_folds', 10)
            )
            
            # Store predictions
            if hasattr(model, 'y_train_pred') and hasattr(model, 'y_test_pred'):
                self.predictions[model_id] = {
                    'train_records': model.train_records,
                    'test_records': model.test_records,
                    'train_actual': model.y_train,
                    'train_pred': model.y_train_pred,
                    'test_actual': model.y_test,
                    'test_pred': model.y_test_pred
                }
            
            # Extract key metrics
            model_results = {
                'model_id': model_id,
                'model_name': model_config.get('name', f'Model {model_id}'),
                'r2_train': model.metrics.get('r2_train'),
                'r2_test': model.metrics.get('r2_test'),
                'rmse_train': model.metrics.get('rmse_train'),
                'rmse_test': model.metrics.get('rmse_test'),
                'mae_train': model.metrics.get('mae_train'),
                'mae_test': model.metrics.get('mae_test'),
                'mape_train': model.metrics.get('mape_train'),
                'mape_test': model.metrics.get('mape_test'),
                'cv_mean': model.metrics.get('cv_mean'),
                'cv_std': model.metrics.get('cv_std'),
                'n_features': len(model.feature_names) if hasattr(model, 'feature_names') else None,
                'n_train': len(model.train_records) if hasattr(model, 'train_records') else None,
                'n_test': len(model.test_records) if hasattr(model, 'test_records') else None,
                'success': True,
                'error': None
            }
            
            self.logger.info(f"✓ Model {model_id} completed successfully")
            self.logger.info(f"  R² Test: {model_results['r2_test']:.4f}")
            self.logger.info(f"  RMSE Test: ${model_results['rmse_test']:,.0f}")
            self.logger.info(f"  Features: {model_results['n_features']}")
            
            return model_results
            
        except Exception as e:
            # Log error and re-raise to stop orchestrator
            error_msg = f"FATAL ERROR in Model {model_id}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            # Re-raise exception to stop orchestrator
            raise RuntimeError(
                f"Model {model_id} failed. Stopping orchestrator.\n"
                f"Error: {str(e)}\n"
                f"See log file for details."
            ) from e
    
    def run_all_models(self):
        """Run all configured models"""
        models_to_run = self.config['models_to_run']
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"RUNNING {len(models_to_run)} MODELS")
        self.logger.info("=" * 80)
        self.logger.info(f"Models: {models_to_run}")
        
        # Run each model sequentially
        for i, model_id in enumerate(models_to_run, 1):
            self.logger.info(f"\n[{i}/{len(models_to_run)}] Starting Model {model_id}...")
            
            # Run model (will raise exception if fails)
            result = self.run_single_model(model_id)
            self.results[model_id] = result
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("ALL MODELS COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 80)
    
    def generate_combined_predictions(self):
        """Generate combined predictions CSV with all models"""
        self.logger.info("")
        self.logger.info("Generating combined predictions CSV...")
        
        if not self.predictions:
            self.logger.warning("No predictions available")
            return
        
        # Combine train and test predictions
        all_records = []
        
        for model_id in sorted(self.predictions.keys()):
            pred_data = self.predictions[model_id]
            
            # Process training set
            for i, record in enumerate(pred_data['train_records']):
                row = {
                    'CaseNo': record.consumer_id,
                    'Dataset': 'Train',
                    'Actual_Cost': pred_data['train_actual'][i],
                    f'Model_{model_id}': pred_data['train_pred'][i]
                }
                all_records.append(row)
            
            # Process test set
            for i, record in enumerate(pred_data['test_records']):
                row = {
                    'CaseNo': record.consumer_id,
                    'Dataset': 'Test',
                    'Actual_Cost': pred_data['test_actual'][i],
                    f'Model_{model_id}': pred_data['test_pred'][i]
                }
                all_records.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_records)
        
        # Pivot to get one row per CaseNo
        model_cols = [c for c in df.columns if c.startswith('Model_')]
        
        df_pivot = df.groupby('CaseNo').agg({
            'Dataset': 'first',
            'Actual_Cost': 'first',
            **{col: 'first' for col in model_cols}
        }).reset_index()
        
        # Reorder columns
        cols = ['CaseNo', 'Dataset', 'Actual_Cost'] + sorted(model_cols)
        df_pivot = df_pivot[cols]
        
        # Save
        output_file = self.output_dir / 'predictions.csv'
        df_pivot.to_csv(output_file, index=False)
        
        self.logger.info(f"✓ Saved combined predictions: {output_file}")
        self.logger.info(f"  Total records: {len(df_pivot)}")
        self.logger.info(f"  Train records: {(df_pivot['Dataset'] == 'Train').sum()}")
        self.logger.info(f"  Test records: {(df_pivot['Dataset'] == 'Test').sum()}")
    
    def generate_comparison_report(self):
        """Generate comparison CSV and summary JSON"""
        self.logger.info("")
        self.logger.info("Generating comparison report...")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_id in sorted(self.results.keys()):
            result = self.results[model_id]
            comparison_data.append({
                'Model_ID': result['model_id'],
                'Model_Name': result['model_name'],
                'R2_Train': result['r2_train'],
                'R2_Test': result['r2_test'],
                'RMSE_Train': result['rmse_train'],
                'RMSE_Test': result['rmse_test'],
                'MAE_Train': result['mae_train'],
                'MAE_Test': result['mae_test'],
                'MAPE_Train': result['mape_train'],
                'MAPE_Test': result['mape_test'],
                'CV_Mean': result['cv_mean'],
                'CV_Std': result['cv_std'],
                'N_Features': result['n_features'],
                'N_Train': result['n_train'],
                'N_Test': result['n_test']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by R² Test (descending)
        df = df.sort_values('R2_Test', ascending=False)
        
        # Save comparison CSV
        comparison_file = self.output_dir / 'orchestration_comparison.csv'
        df.to_csv(comparison_file, index=False)
        
        self.logger.info(f"✓ Saved comparison table: {comparison_file}")
        
        # Create summary JSON
        summary = {
            'timestamp': datetime.now().isoformat(),
            'scenario': self.config['scenario_name'],
            'description': self.config.get('description', ''),
            'configuration': self.config_path.name,
            'total_models': len(self.results),
            'successful_models': len(self.results),
            'failed_models': 0,
            'best_model': {
                'id': int(df.iloc[0]['Model_ID']),
                'name': df.iloc[0]['Model_Name'],
                'r2_test': float(df.iloc[0]['R2_Test']),
                'rmse_test': float(df.iloc[0]['RMSE_Test'])
            },
            'all_results': {int(k): v for k, v in self.results.items()}
        }
        
        # Save summary JSON
        summary_file = self.output_dir / 'orchestration_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"✓ Saved summary: {summary_file}")
        
        # Print comparison table
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("MODEL PERFORMANCE COMPARISON")
        self.logger.info("=" * 80)
        self.logger.info("")
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.4f}'.format)
        
        self.logger.info(df.to_string(index=False))
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("BEST MODEL")
        self.logger.info("=" * 80)
        best = summary['best_model']
        self.logger.info(f"Model {best['id']}: {best['name']}")
        self.logger.info(f"R² Test: {best['r2_test']:.4f}")
        self.logger.info(f"RMSE Test: ${best['rmse_test']:,.0f}")
    
    def call_comparison_plots(self):
        """Call generate_comparison_plots.py to create visualizations"""
        self.logger.info("")
        self.logger.info("Generating comparison plots...")
        
        try:
            # Import and run the plotting script
            import generate_comparison_plots as gcp
            
            # Run the main function
            gcp.main()
            
            self.logger.info("✓ Comparison plots generated successfully")
            
        except Exception as e:
            self.logger.warning(f"Could not generate comparison plots: {e}")
            self.logger.warning("You can run generate_comparison_plots.py manually")
    
    def run(self):
        """Run complete orchestration pipeline"""
        try:
            start_time = datetime.now()
            
            # Run all models
            self.run_all_models()
            
            # Generate outputs
            self.generate_combined_predictions()
            self.generate_comparison_report()
            
            # Generate plots
            if self.config.get('pipeline_settings', {}).get('generate_plots', True):
                self.call_comparison_plots()
            
            # Final summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("ORCHESTRATION COMPLETE")
            self.logger.info("=" * 80)
            self.logger.info(f"Total time: {duration}")
            self.logger.info(f"Models run: {len(self.results)}")
            self.logger.info(f"Output directory: {self.output_dir}")
            self.logger.info("=" * 80)
            
            return 0  # Success
            
        except Exception as e:
            self.logger.error("")
            self.logger.error("=" * 80)
            self.logger.error("ORCHESTRATION FAILED")
            self.logger.error("=" * 80)
            self.logger.error(str(e))
            self.logger.error("=" * 80)
            
            return 1  # Failure

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='iBudget Model Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orchestrator.py                              # Use ConsensusConfiguration.json
  python orchestrator.py --config MyConfig.json       # Use custom configuration
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='ConsensusConfiguration.json',
        help='Path to JSON configuration file (default: ConsensusConfiguration.json)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("IBUDGET MODEL ORCHESTRATOR")
    print("=" * 80)
    print()
    
    try:
        # Initialize and run orchestrator
        orchestrator = ModelOrchestrator(config_path=args.config)
        exit_code = orchestrator.run()
        
        sys.exit(exit_code)
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print()
        print("Please create a configuration file or specify --config")
        sys.exit(1)
        
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        print()
        print("See log file for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
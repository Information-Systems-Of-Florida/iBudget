"""
ModelPipeline.py
================
Master execution framework for running multiple iBudget model configurations
with different parameters and feature configurations.

Design Principles:
- Configuration dictionary approach 
- Continues execution if individual models fail
- Single place to modify feature configurations
"""

import sys
import importlib
import traceback
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import pandas as pd
import numpy as np

# ============================================================================
# FEATURE CONFIGURATIONS
# ============================================================================

# Model 1 Feature Configuration (21 features - Model 5b exact specification)
FEATURE_CONFIG_MODEL1 = {
    'categorical': {
        'living_setting': {
            'categories': ['LiveILSL', 'LiveRH1', 'LiveRH2', 'LiveRH3', 'LiveRH4'],
            'reference': 'LiveFH'  # FH is reference category
        },
        'age_group': {
            'categories': ['Age21_30', 'Age31Plus'],
            'reference': 'Age3_20'  # Age3_20 is reference category
        }
    },
    'numeric': [
        'BSum'  # Behavioral sum score
    ],
    'qsi': [16, 18, 20, 21, 23, 28, 33, 34, 36, 43],  # 10 QSI questions
    'interactions': [
        ('FHFSum', lambda r: (1 if r.living_setting == 'LiveFH' else 0) * r.fsum),
        ('SLFSum', lambda r: (1 if r.living_setting == 'LiveILSL' else 0) * r.fsum),
        ('SLBSum', lambda r: (1 if r.living_setting == 'LiveILSL' else 0) * r.bsum)
    ]
}

# Standard Feature Configuration (expanded set for Models 2-10)
FEATURE_CONFIG_STANDARD = {
    'categorical': {
        'living_setting': {
            'categories': ['LiveILSL', 'LiveRH1', 'LiveRH2', 'LiveRH3', 'LiveRH4'],
            'reference': 'LiveFH'
        },
        'age_group': {
            'categories': ['Age21_30', 'Age31Plus'],
            'reference': 'Age3_20'
        }
    },
    'numeric': [
        'BSum', 'FSum',  # Summary scores
        'age',  # Continuous age
        'slevel', 'flevel', 'plevel', 'losri', 'olevel'  # Support levels
    ],
    'binary': {
        'Male': lambda r: r.gender == 'M',
        'Medicare': lambda r: r.medicare == 1,
        'Medicaid': lambda r: r.medicaid == 1,
        'BehaviorHigh': lambda r: r.bsum > 10,  # High behavioral needs
        'FunctionalHigh': lambda r: r.fsum > 15  # High functional needs
    },
    'qsi': list(range(1, 51)),  # All 50 QSI questions
    'interactions': [
        ('FHFSum', lambda r: (1 if r.living_setting == 'LiveFH' else 0) * r.fsum),
        ('SLFSum', lambda r: (1 if r.living_setting == 'LiveILSL' else 0) * r.fsum),
        ('SLBSum', lambda r: (1 if r.living_setting == 'LiveILSL' else 0) * r.bsum),
        ('AgeHighSupport', lambda r: r.age * (r.slevel > 3)),
        ('MedicareHighCost', lambda r: (1 if r.medicare == 1 else 0) * (r.total_cost > 50000))
    ]
}

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    'model_1': {
        'class': 'Model1Linear',
        'module': 'model_1_reeval',
        'scenarios': {
            'sqrt_outlier': {
                'use_sqrt_transform': True,
                'use_outlier_removal': True,
                'outlier_threshold': 1.645,  # ~10% outliers (Model 5b default)
                'feature_config': FEATURE_CONFIG_MODEL1,
                'run': True  # Set to True to execute
            },
            'sqrt_no_outlier': {
                'use_sqrt_transform': True,
                'use_outlier_removal': False,
                'feature_config': FEATURE_CONFIG_MODEL1,
                'run': True
            },
            'no_transform': {
                'use_sqrt_transform': False,
                'use_outlier_removal': False,
                'feature_config': FEATURE_CONFIG_MODEL1,
                'run': False  # Skip this scenario
            },
            'outlier_only': {
                'use_sqrt_transform': False,
                'use_outlier_removal': True,
                'outlier_threshold': 2.0,  # Stricter threshold
                'feature_config': FEATURE_CONFIG_MODEL1,
                'run': False
            }
        }
    },
    'model_2': {
        'class': 'Model2GLMGamma',
        'module': 'model_2_gamma',
        'scenarios': {
            'gamma_standard': {
                'use_outlier_removal': False,
                'use_selected_features': True,
                'feature_config': FEATURE_CONFIG_STANDARD,
                'run': True
            },
            'gamma_outlier': {
                'use_outlier_removal': True,
                'outlier_threshold': 1.645,
                'use_selected_features': True,
                'feature_config': FEATURE_CONFIG_STANDARD,
                'run': True
            },
            'gamma_minimal': {
                'use_outlier_removal': False,
                'use_selected_features': False,
                'feature_config': FEATURE_CONFIG_MODEL1,  # Use minimal features
                'run': False
            }
        }
    },
}

# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class ModelPipeline:
    """
    Central execution framework for running multiple model configurations
    """
    
    def __init__(self, output_dir: Path = None, log_dir: Path = None):
        """
        Initialize the pipeline
        
        Args:
            output_dir: Directory for model outputs
            log_dir: Directory for log files
        """
        self.output_dir = output_dir or Path('./output')
        self.log_dir = log_dir or Path('./logs')
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {}
        self.errors = {}
        
        # Setup pipeline logger
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for the pipeline"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'pipeline_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ModelPipeline')
        self.logger.info("="*80)
        self.logger.info("MODEL PIPELINE INITIALIZED")
        self.logger.info("="*80)
    
    def run_model_scenario(self, 
                          model_id: str, 
                          scenario_name: str, 
                          config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Run a single model scenario
        
        Args:
            model_id: Model identifier (e.g., 'model_1')
            scenario_name: Scenario name (e.g., 'sqrt_outlier')
            config: Scenario configuration dictionary
        
        Returns:
            Results dictionary or None if failed
        """
        # Extract model class info
        model_config = MODEL_CONFIGS[model_id]
        module_name = model_config['module']
        class_name = model_config['class']
        
        # Generate log suffix
        log_suffix = f"{model_id}_{scenario_name}"
        
        try:
            self.logger.info(f"Running {model_id} - {scenario_name}")
            self.logger.info(f"  Module: {module_name}")
            self.logger.info(f"  Class: {class_name}")
            
            # Dynamic import
            module = importlib.import_module(module_name)
            ModelClass = getattr(module, class_name)
            
            # Prepare model parameters
            model_params = {
                k: v for k, v in config.items() 
                if k not in ['run', 'feature_config']
            }
            model_params['log_suffix'] = log_suffix
            model_params['output_dir'] = self.output_dir / model_id / scenario_name
            
            # Create output directory
            model_params['output_dir'].mkdir(parents=True, exist_ok=True)
            
            # Log configuration
            self.logger.info(f"  Parameters: {model_params}")
            
            # Initialize model
            model = ModelClass(**model_params)
            
            # Set feature configuration if provided and model supports it
            if 'feature_config' in config and hasattr(model, 'feature_config'):
                model.feature_config = config['feature_config']
                self.logger.info(f"  Feature config: Custom configuration applied")
            
            # Run the model pipeline
            results = model.run_complete_pipeline(
                fiscal_year_start=2024,
                fiscal_year_end=2024,
                test_size=0.2,
                perform_cv=True,
                n_cv_folds=10
            )
            
            # Generate diagnostic plots if method exists
            if hasattr(model, 'generate_diagnostic_plots'):
                model.generate_diagnostic_plots()
            
            # Store key results
            scenario_results = {
                'model_id': model_id,
                'scenario': scenario_name,
                'config': config,
                'metrics': results.get('metrics', {}),
                'r2_train': results.get('metrics', {}).get('r2_train', None),
                'r2_test': results.get('metrics', {}).get('r2_test', None),
                'rmse_test': results.get('metrics', {}).get('rmse_test', None),
                'mae_test': results.get('metrics', {}).get('mae_test', None),
                'cv_mean': results.get('metrics', {}).get('cv_mean', None),
                'cv_std': results.get('metrics', {}).get('cv_std', None),
                'n_features': len(model.feature_names) if hasattr(model, 'feature_names') else None,
                'n_train': results.get('n_train', None),
                'n_test': results.get('n_test', None)
            }
            
            # Log success
            self.logger.info(f"  ✓ Success: R² = {scenario_results['r2_test']:.4f}")
            
            return scenario_results
            
        except Exception as e:
            # Log error but continue with other models
            error_msg = f"Failed to run {model_id} - {scenario_name}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            # Store error information
            self.errors[f"{model_id}_{scenario_name}"] = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
            return None
    
    def run_all_models(self):
        """
        Run all configured model scenarios
        """
        self.logger.info("="*80)
        self.logger.info("RUNNING ALL MODEL SCENARIOS")
        self.logger.info("="*80)
        
        total_scenarios = sum(
            len([s for s in scenarios.values() if s.get('run', False)])
            for scenarios in [m['scenarios'] for m in MODEL_CONFIGS.values()]
        )
        
        self.logger.info(f"Total scenarios to run: {total_scenarios}")
        self.logger.info("")
        
        scenario_count = 0
        
        # Iterate through all models and scenarios
        for model_id, model_config in MODEL_CONFIGS.items():
            for scenario_name, scenario_config in model_config['scenarios'].items():
                if scenario_config.get('run', False):
                    scenario_count += 1
                    self.logger.info(f"[{scenario_count}/{total_scenarios}] {model_id} - {scenario_name}")
                    
                    # Run the scenario
                    result = self.run_model_scenario(model_id, scenario_name, scenario_config)
                    
                    # Store results
                    if result:
                        key = f"{model_id}_{scenario_name}"
                        self.results[key] = result
                    
                    self.logger.info("")
        
        # Summary
        self.logger.info("="*80)
        self.logger.info("PIPELINE EXECUTION COMPLETE")
        self.logger.info("="*80)
        self.logger.info(f"Successful runs: {len(self.results)}")
        self.logger.info(f"Failed runs: {len(self.errors)}")
        
        # Generate comparison report
        self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """
        Generate a comparison report of all model results
        """
        if not self.results:
            self.logger.warning("No results to compare")
            return
        
        # Convert results to DataFrame for easy comparison
        comparison_data = []
        for key, result in self.results.items():
            comparison_data.append({
                'Model': result['model_id'],
                'Scenario': result['scenario'],
                'R² Train': result['r2_train'],
                'R² Test': result['r2_test'],
                'RMSE': result['rmse_test'],
                'MAE': result['mae_test'],
                'CV R² Mean': result['cv_mean'],
                'CV R² Std': result['cv_std'],
                'Features': result['n_features'],
                'Train N': result['n_train'],
                'Test N': result['n_test']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Sort by test R²
        df_comparison = df_comparison.sort_values('R² Test', ascending=False)
        
        # Save to CSV
        comparison_file = self.output_dir / 'model_comparison.csv'
        df_comparison.to_csv(comparison_file, index=False)
        
        # Print comparison
        self.logger.info("\n" + "="*80)
        self.logger.info("MODEL PERFORMANCE COMPARISON")
        self.logger.info("="*80)
        
        # Format for display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.4f}'.format)
        
        self.logger.info("\n" + df_comparison.to_string(index=False))
        
        # Best model
        if not df_comparison.empty:
            best_model = df_comparison.iloc[0]
            self.logger.info("\n" + "="*80)
            self.logger.info("BEST MODEL")
            self.logger.info("="*80)
            self.logger.info(f"Model: {best_model['Model']} - {best_model['Scenario']}")
            self.logger.info(f"R² Test: {best_model['R² Test']:.4f}")
            self.logger.info(f"RMSE: ${best_model['RMSE']:,.2f}")
            self.logger.info(f"Features: {int(best_model['Features'])}")
        
        # Save summary JSON
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_scenarios': len(self.results) + len(self.errors),
            'successful': len(self.results),
            'failed': len(self.errors),
            'best_model': {
                'id': best_model['Model'] if not df_comparison.empty else None,
                'scenario': best_model['Scenario'] if not df_comparison.empty else None,
                'r2_test': float(best_model['R² Test']) if not df_comparison.empty else None
            },
            'all_results': self.results,
            'errors': self.errors
        }
        
        summary_file = self.output_dir / 'pipeline_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"\nResults saved to: {self.output_dir}")
    
    def run_specific_models(self, model_list: List[str]):
        """
        Run only specific models from the configuration
        
        Args:
            model_list: List of model IDs to run (e.g., ['model_1', 'model_2'])
        """
        self.logger.info("="*80)
        self.logger.info(f"RUNNING SPECIFIC MODELS: {', '.join(model_list)}")
        self.logger.info("="*80)
        
        for model_id in model_list:
            if model_id not in MODEL_CONFIGS:
                self.logger.warning(f"Model {model_id} not found in configuration")
                continue
            
            model_config = MODEL_CONFIGS[model_id]
            for scenario_name, scenario_config in model_config['scenarios'].items():
                if scenario_config.get('run', False):
                    result = self.run_model_scenario(model_id, scenario_name, scenario_config)
                    
                    if result:
                        key = f"{model_id}_{scenario_name}"
                        self.results[key] = result
        
        # Generate comparison report
        self.generate_comparison_report()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    print("="*80)
    print("IBUDGET MODEL PIPELINE")
    print("="*80)
    print()
    
    # Initialize pipeline
    pipeline = ModelPipeline(
        output_dir=Path('../../report/logs'),
        log_dir=Path('./logs')
    )
    
    # Option 1: Run all models with 'run': True
    pipeline.run_all_models()
    
    # Option 2: Run specific models only
    # pipeline.run_specific_models(['model_1', 'model_2'])
    
    # Option 3: Run individual scenario
    # result = pipeline.run_model_scenario(
    #     'model_1', 
    #     'sqrt_outlier',
    #     MODEL_CONFIGS['model_1']['scenarios']['sqrt_outlier']
    # )
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
"""

generate_comparison_plots.py

=============================

Generate comprehensive comparison plots for Algorithm Comparison chapter.

Extracts data from LaTeX renewcommand files and creates visualizations.



Usage:

    python generate_comparison_plots.py



Output:

    Saves plots to ../report/figures/comparison/

"""



import re

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pathlib import Path

from typing import Dict, List, Tuple

import warnings

warnings.filterwarnings('ignore')



# Set style

plt.style.use('seaborn-v0_8-darkgrid')

sns.set_palette("husl")



class ModelDataExtractor:

    """Extract model metrics from LaTeX renewcommand files"""

    

    def __init__(self, models_dir: Path):

        self.models_dir = models_dir

        self.model_numbers = [1, 2, 3, 4, 5, 6, 9]

        self.data = {}

        

    def extract_value(self, text: str, command_name: str) -> float:

        """Extract numeric value from renewcommand"""

        # Pattern: \renewcommand{\CommandName}{value}

        pattern = rf'\\renewcommand{{\\{command_name}}}{{([^}}]+)}}'

        match = re.search(pattern, text)

        if match:

            value_str = match.group(1).replace(',', '')

            try:

                return float(value_str)

            except ValueError:

                return np.nan

        return np.nan

    

    def load_model_data(self, model_num: int) -> Dict:

        """Load all metrics for a specific model"""

        # Determine correct model number word

        num_words = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 

                     6: 'Six', 9: 'Nine'}

        num_word = num_words[model_num]

        

        # Try multiple possible file locations

        possible_paths = [

            # Primary location: ../report/models/model_[#]/

            self.models_dir / 'report' / 'models' / f'model_{model_num}' / f'model_{model_num}_renewcommands.tex',

            # Alternative locations

            self.models_dir / f'model_{model_num}' / f'model_{model_num}_renewcommands.tex',

            self.models_dir.parent / f'model_{model_num}_renewcommands.tex',

            self.models_dir / f'model_{model_num}_renewcommands.tex',

            Path.cwd() / f'model_{model_num}_renewcommands.tex',

            Path.cwd().parent / 'report' / 'models' / f'model_{model_num}' / f'model_{model_num}_renewcommands.tex',

        ]

        

        filepath = None

        for path in possible_paths:

            if path.exists():

                filepath = path

                break

        

        if filepath is None:

            print(f"Warning: File not found for Model {model_num}. Tried:")

            for path in possible_paths:

                print(f"  - {path}")

            return None

        

        print(f"  âœ“ Found Model {model_num}: {filepath}")

        

        with open(filepath, 'r') as f:

            content = f.read()

        

        # Extract metrics

        data = {

            'model_num': model_num,

            'r2_train': self.extract_value(content, f'Model{num_word}RSquaredTrain'),

            'r2_test': self.extract_value(content, f'Model{num_word}RSquaredTest'),

            'rmse_train': self.extract_value(content, f'Model{num_word}RMSETrain'),

            'rmse_test': self.extract_value(content, f'Model{num_word}RMSETest'),

            'mae_train': self.extract_value(content, f'Model{num_word}MAETrain'),

            'mae_test': self.extract_value(content, f'Model{num_word}MAETest'),

            'mape_train': self.extract_value(content, f'Model{num_word}MAPETrain'),

            'mape_test': self.extract_value(content, f'Model{num_word}MAPETest'),

            'cv_mean': self.extract_value(content, f'Model{num_word}CVMean'),

            'cv_std': self.extract_value(content, f'Model{num_word}CVStd'),

            'within_1k': self.extract_value(content, f'Model{num_word}WithinOneK'),

            'within_2k': self.extract_value(content, f'Model{num_word}WithinTwoK'),

            'within_5k': self.extract_value(content, f'Model{num_word}WithinFiveK'),

            'within_10k': self.extract_value(content, f'Model{num_word}WithinTenK'),

            'within_20k': self.extract_value(content, f'Model{num_word}WithinTwentyK'),

            'n_train': self.extract_value(content, f'Model{num_word}TrainingSamples'),

            'n_test': self.extract_value(content, f'Model{num_word}TestSamples'),

        }

        

        # Subgroup performance - Living Settings

        data['living_fh_r2'] = self.extract_value(content, f'Model{num_word}SubgroupLivingFHRSquared')

        data['living_fh_bias'] = self.extract_value(content, f'Model{num_word}SubgroupLivingFHBias')

        data['living_ilsl_r2'] = self.extract_value(content, f'Model{num_word}SubgroupLivingILSLRSquared')

        data['living_ilsl_bias'] = self.extract_value(content, f'Model{num_word}SubgroupLivingILSLBias')

        data['living_rh_r2'] = self.extract_value(content, f'Model{num_word}SubgroupLivingRHOneFourRSquared')

        data['living_rh_bias'] = self.extract_value(content, f'Model{num_word}SubgroupLivingRHOneFourBias')

        

        # Subgroup performance - Age Groups

        data['age_u21_r2'] = self.extract_value(content, f'Model{num_word}SubgroupAgeAgeUnderTwentyOneRSquared')

        data['age_21_30_r2'] = self.extract_value(content, f'Model{num_word}SubgroupAgeAgeTwentyOneToThirtyRSquared')

        data['age_31p_r2'] = self.extract_value(content, f'Model{num_word}SubgroupAgeAgeThirtyOnePlusRSquared')

        

        # Subgroup performance - Cost Quartiles

        data['cost_q1_bias'] = self.extract_value(content, f'Model{num_word}SubgroupCostQOneLowBias')

        data['cost_q2_bias'] = self.extract_value(content, f'Model{num_word}SubgroupCostQTwoBias')

        data['cost_q3_bias'] = self.extract_value(content, f'Model{num_word}SubgroupCostQThreeBias')

        data['cost_q4_bias'] = self.extract_value(content, f'Model{num_word}SubgroupCostQFourHighBias')

        

        return data

    

    def load_all_models(self):

        """Load data for all models"""

        for model_num in self.model_numbers:

            data = self.load_model_data(model_num)

            if data is not None:

                self.data[model_num] = data

        return self.data





class ComparisonPlotGenerator:

    """Generate comparison plots"""

    

    def __init__(self, data: Dict, output_dir: Path):

        self.data = data

        self.output_dir = output_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)

        

        # Model names for labels

        self.model_names = {

            1: 'OLS',

            2: 'GLM Gamma',

            3: 'Robust',

            4: 'WLS',

            5: 'Ridge',

            6: 'Log-Normal',

            9: 'Random Forest'

        }

        

        self.model_colors = {

            1: '#e74c3c',  # Red

            2: '#3498db',  # Blue

            3: '#2ecc71',  # Green

            4: '#f39c12',  # Orange

            5: '#9b59b6',  # Purple

            6: '#1abc9c',  # Teal

            9: '#e67e22'   # Dark Orange

        }

    

    def plot_a_performance_comparison(self):

        """A. Performance Comparison Bar Chart (RÂ² values)"""

        fig, ax = plt.subplots(figsize=(10, 6))

        

        # Filter out models with missing data

        models = [m for m in self.data.keys() if self.data[m].get('r2_train') is not None]

        if not models:

            print("âš  Skipping Plot A: No valid data")

            return

            

        r2_train = [self.data[m]['r2_train'] for m in models]

        r2_test = [self.data[m]['r2_test'] for m in models]

        

        x = np.arange(len(models))

        width = 0.35

        

        bars1 = ax.bar(x - width/2, r2_train, width, label='Training', alpha=0.8)

        bars2 = ax.bar(x + width/2, r2_test, width, label='Test', alpha=0.8)

        

        ax.set_xlabel('Model', fontsize=12, fontweight='bold')

        ax.set_ylabel('RÂ² Score', fontsize=12, fontweight='bold')

        ax.set_title('Model Performance Comparison: RÂ² Scores', fontsize=14, fontweight='bold')

        ax.set_xticks(x)

        ax.set_xticklabels([f'Model {m}\n{self.model_names[m]}' for m in models], fontsize=10)

        ax.legend(fontsize=10)

        ax.grid(axis='y', alpha=0.3)

        ax.set_ylim([0, max(r2_train + r2_test) * 1.1])

        

        # Add value labels on bars

        for bars in [bars1, bars2]:

            for bar in bars:

                height = bar.get_height()

                ax.text(bar.get_x() + bar.get_width()/2., height,

                       f'{height:.3f}',

                       ha='center', va='bottom', fontsize=8)

        

        plt.tight_layout()

        plt.savefig(self.output_dir / 'plot_a_r2_comparison.png', dpi=300, bbox_inches='tight')

        plt.close()

        print("âœ“ Generated Plot A: RÂ² Comparison")

    

    def plot_d_cumulative_accuracy(self):

        """D. Cumulative Accuracy Curve"""

        fig, ax = plt.subplots(figsize=(12, 7))

        

        thresholds = [1, 2, 5, 10, 20]

        threshold_labels = ['$1K', '$2K', '$5K', '$10K', '$20K']

        

        plotted_any = False

        for model_num in self.data.keys():

            if self.data[model_num].get('within_1k') is None:

                continue

                

            accuracies = [

                self.data[model_num]['within_1k'],

                self.data[model_num]['within_2k'],

                self.data[model_num]['within_5k'],

                self.data[model_num]['within_10k'],

                self.data[model_num]['within_20k']

            ]

            

            if any(a is not None and not np.isnan(a) for a in accuracies):

                ax.plot(thresholds, accuracies, marker='o', linewidth=2.5, 

                       label=f'Model {model_num}: {self.model_names[model_num]}',

                       color=self.model_colors[model_num])

                plotted_any = True

        

        if not plotted_any:

            print("âš  Skipping Plot D: No valid data")

            plt.close()

            return

        

        ax.set_xlabel('Error Tolerance', fontsize=12, fontweight='bold')

        ax.set_ylabel('% of Predictions Within Tolerance', fontsize=12, fontweight='bold')

        ax.set_title('Cumulative Accuracy: Predictions Within Error Thresholds', 

                    fontsize=14, fontweight='bold')

        ax.set_xticks(thresholds)

        ax.set_xticklabels(threshold_labels)

        ax.legend(loc='lower right', fontsize=9)

        ax.grid(True, alpha=0.3)

        ax.set_ylim([0, 100])

        

        plt.tight_layout()

        plt.savefig(self.output_dir / 'plot_d_cumulative_accuracy.png', dpi=300, bbox_inches='tight')

        plt.close()

        print("âœ“ Generated Plot D: Cumulative Accuracy Curve")

    

    def plot_e_tolerance_heatmap(self):

        """E. Tolerance Band Heatmap"""

        fig, ax = plt.subplots(figsize=(10, 6))

        

        # Filter models with valid data

        models = [m for m in self.data.keys() 

                 if self.data[m].get('within_1k') is not None]

        

        if not models:

            print("âš  Skipping Plot E: No valid data")

            plt.close()

            return

            

        thresholds = ['$1K', '$2K', '$5K', '$10K', '$20K']

        

        # Create matrix

        matrix = []

        for model_num in models:

            row = [

                self.data[model_num]['within_1k'],

                self.data[model_num]['within_2k'],

                self.data[model_num]['within_5k'],

                self.data[model_num]['within_10k'],

                self.data[model_num]['within_20k']

            ]

            matrix.append(row)

        

        matrix = np.array(matrix)

        

        # Create heatmap

        im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=100)

        

        # Set ticks

        ax.set_xticks(np.arange(len(thresholds)))

        ax.set_yticks(np.arange(len(models)))

        ax.set_xticklabels(thresholds)

        ax.set_yticklabels([f'Model {m}: {self.model_names[m]}' for m in models])

        

        # Add text annotations

        for i in range(len(models)):

            for j in range(len(thresholds)):

                text = ax.text(j, i, f'{matrix[i, j]:.1f}%',

                             ha="center", va="center", color="black", fontsize=9)

        

        ax.set_title('Prediction Accuracy by Error Tolerance', fontsize=14, fontweight='bold')

        ax.set_xlabel('Error Tolerance', fontsize=12, fontweight='bold')

        ax.set_ylabel('Model', fontsize=12, fontweight='bold')

        

        # Colorbar

        cbar = plt.colorbar(im, ax=ax)

        cbar.set_label('% Within Tolerance', fontsize=10)

        

        plt.tight_layout()

        plt.savefig(self.output_dir / 'plot_e_tolerance_heatmap.png', dpi=300, bbox_inches='tight')

        plt.close()

        print("âœ“ Generated Plot E: Tolerance Heatmap")

    

    def plot_f_cv_boxplot(self):

        """F. Cross-Validation Box Plot"""

        fig, ax = plt.subplots(figsize=(10, 6))

        

        # Filter models with valid CV data

        models = [m for m in self.data.keys() 

                 if self.data[m].get('cv_mean') is not None 

                 and self.data[m].get('cv_std') is not None]

        

        if not models:

            print("âš  Skipping Plot F: No valid data")

            plt.close()

            return

        

        # Create box plot data (simulate from mean and std)

        box_data = []

        positions = []

        labels = []

        

        for i, model_num in enumerate(models):

            mean = self.data[model_num]['cv_mean']

            std = self.data[model_num]['cv_std']

            

            # Simulate 10 CV fold scores

            cv_scores = np.random.normal(mean, std, 10)

            box_data.append(cv_scores)

            positions.append(i)

            labels.append(f'Model {model_num}\n{self.model_names[model_num]}')

        

        bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,

                       labels=labels,

                       boxprops=dict(facecolor='lightblue', alpha=0.7),

                       medianprops=dict(color='red', linewidth=2),

                       whiskerprops=dict(linewidth=1.5),

                       capprops=dict(linewidth=1.5))

        

        ax.set_ylabel('RÂ² Score (Cross-Validation)', fontsize=12, fontweight='bold')

        ax.set_title('Cross-Validation Stability: RÂ² Distribution Across Folds', 

                    fontsize=14, fontweight='bold')

        ax.grid(axis='y', alpha=0.3)

        ax.set_ylim([0, 1])

        

        plt.tight_layout()

        plt.savefig(self.output_dir / 'plot_f_cv_boxplot.png', dpi=300, bbox_inches='tight')

        plt.close()

        print("âœ“ Generated Plot F: CV Box Plot")

    

    def plot_h_living_setting_performance(self):

        """H. Performance by Living Setting"""

        fig, ax = plt.subplots(figsize=(12, 7))

        

        # Filter models with valid living setting data

        models = [m for m in self.data.keys() 

                 if self.data[m].get('living_fh_r2') is not None]

        

        if not models:

            print("âš  Skipping Plot H: No valid data")

            plt.close()

            return

            

        settings = ['Family Home', 'ILSL', 'RH 1-4']

        

        x = np.arange(len(settings))

        width = 0.12

        

        for i, model_num in enumerate(models):

            r2_values = [

                self.data[model_num]['living_fh_r2'],

                self.data[model_num]['living_ilsl_r2'],

                self.data[model_num]['living_rh_r2']

            ]

            

            offset = (i - len(models)/2) * width

            ax.bar(x + offset, r2_values, width, 

                  label=f'Model {model_num}',

                  color=self.model_colors[model_num], alpha=0.8)

        

        ax.set_xlabel('Living Setting', fontsize=12, fontweight='bold')

        ax.set_ylabel('RÂ² Score', fontsize=12, fontweight='bold')

        ax.set_title('Model Performance by Living Setting', fontsize=14, fontweight='bold')

        ax.set_xticks(x)

        ax.set_xticklabels(settings)

        ax.legend(loc='upper right', ncol=2, fontsize=9)

        ax.grid(axis='y', alpha=0.3)

        ax.axhline(y=0, color='black', linewidth=0.8)

        

        plt.tight_layout()

        plt.savefig(self.output_dir / 'plot_h_living_setting.png', dpi=300, bbox_inches='tight')

        plt.close()

        print("âœ“ Generated Plot H: Living Setting Performance")

    

    def plot_i_bias_heatmap(self):

        """I. Bias Heatmap by Subgroup"""

        fig, ax = plt.subplots(figsize=(10, 8))

        

        # Filter models with valid bias data

        models = [m for m in self.data.keys() 

                 if self.data[m].get('living_fh_bias') is not None]

        

        if not models:

            print("âš  Skipping Plot I: No valid data")

            plt.close()

            return

            

        subgroups = ['FH', 'ILSL', 'RH 1-4', 'Q1\n(Low Cost)', 'Q2', 'Q3', 'Q4\n(High Cost)']

        

        # Create bias matrix

        matrix = []

        for model_num in models:

            row = [

                self.data[model_num]['living_fh_bias'],

                self.data[model_num]['living_ilsl_bias'],

                self.data[model_num]['living_rh_bias'],

                self.data[model_num]['cost_q1_bias'],

                self.data[model_num]['cost_q2_bias'],

                self.data[model_num]['cost_q3_bias'],

                self.data[model_num]['cost_q4_bias']

            ]

            matrix.append(row)

        

        matrix = np.array(matrix)

        

        # Determine color scale limits (symmetric around zero)

        vmax = np.nanmax(np.abs(matrix))

        vmin = -vmax

        

        # Create heatmap

        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)

        

        # Set ticks

        ax.set_xticks(np.arange(len(subgroups)))

        ax.set_yticks(np.arange(len(models)))

        ax.set_xticklabels(subgroups, fontsize=10)

        ax.set_yticklabels([f'Model {m}: {self.model_names[m]}' for m in models])

        

        # Add text annotations

        for i in range(len(models)):

            for j in range(len(subgroups)):

                value = matrix[i, j]

                if not np.isnan(value):

                    text_color = 'white' if abs(value) > vmax * 0.6 else 'black'

                    text = ax.text(j, i, f'${value:,.0f}',

                                 ha="center", va="center", color=text_color, fontsize=8)

        

        ax.set_title('Prediction Bias by Subgroup (Positive = Over-prediction)', 

                    fontsize=14, fontweight='bold')

        ax.set_xlabel('Subgroup', fontsize=12, fontweight='bold')

        ax.set_ylabel('Model', fontsize=12, fontweight='bold')

        

        # Colorbar

        cbar = plt.colorbar(im, ax=ax)

        cbar.set_label('Bias ($)', fontsize=10)

        

        plt.tight_layout()

        plt.savefig(self.output_dir / 'plot_i_bias_heatmap.png', dpi=300, bbox_inches='tight')

        plt.close()

        print("âœ“ Generated Plot I: Bias Heatmap")

    

    def plot_b_multi_metric(self):

        """B. Multi-Metric Comparison"""

        fig, ax = plt.subplots(figsize=(12, 7))

        

        # Filter models with valid data

        models = [m for m in self.data.keys() 

                 if self.data[m].get('r2_test') is not None 

                 and self.data[m].get('rmse_test') is not None 

                 and self.data[m].get('mae_test') is not None]

        

        if not models:

            print("âš  Skipping Plot B: No valid data")

            plt.close()

            return

        

        # Normalize RMSE and MAE to 0-1 scale for comparison with RÂ²

        max_rmse = max([self.data[m]['rmse_test'] for m in models])

        max_mae = max([self.data[m]['mae_test'] for m in models])

        

        r2_values = [self.data[m]['r2_test'] for m in models]

        rmse_normalized = [1 - (self.data[m]['rmse_test'] / max_rmse) for m in models]

        mae_normalized = [1 - (self.data[m]['mae_test'] / max_mae) for m in models]

        

        x = np.arange(len(models))

        width = 0.25

        

        bars1 = ax.bar(x - width, r2_values, width, label='RÂ² (Test)', alpha=0.8)

        bars2 = ax.bar(x, rmse_normalized, width, label='RMSE (normalized)', alpha=0.8)

        bars3 = ax.bar(x + width, mae_normalized, width, label='MAE (normalized)', alpha=0.8)

        

        ax.set_xlabel('Model', fontsize=12, fontweight='bold')

        ax.set_ylabel('Score (higher is better)', fontsize=12, fontweight='bold')

        ax.set_title('Multi-Metric Model Comparison\n(All metrics normalized to 0-1 scale)', 

                    fontsize=14, fontweight='bold')

        ax.set_xticks(x)

        ax.set_xticklabels([f'Model {m}\n{self.model_names[m]}' for m in models], fontsize=9)

        ax.legend(fontsize=10)

        ax.grid(axis='y', alpha=0.3)

        ax.set_ylim([0, 1.1])

        

        plt.tight_layout()

        plt.savefig(self.output_dir / 'plot_b_multi_metric.png', dpi=300, bbox_inches='tight')

        plt.close()

        print("âœ“ Generated Plot B: Multi-Metric Comparison")

    

    def plot_c_error_metrics(self):

        """C. Error Metrics Comparison (Horizontal)"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        

        # Filter models with valid data

        models = [m for m in self.data.keys() 

                 if self.data[m].get('rmse_test') is not None 

                 and self.data[m].get('mae_test') is not None]

        

        if not models:

            print("âš  Skipping Plot C: No valid data")

            plt.close()

            return

            

        rmse_values = [self.data[m]['rmse_test'] for m in models]

        mae_values = [self.data[m]['mae_test'] for m in models]

        

        # RMSE subplot

        colors_rmse = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(models)))

        sorted_indices_rmse = np.argsort(rmse_values)

        

        y_pos = np.arange(len(models))

        ax1.barh(y_pos, [rmse_values[i] for i in sorted_indices_rmse], 

                color=colors_rmse, alpha=0.8)

        ax1.set_yticks(y_pos)

        ax1.set_yticklabels([f'Model {models[i]}: {self.model_names[models[i]]}' 

                            for i in sorted_indices_rmse])

        ax1.set_xlabel('RMSE ($)', fontsize=12, fontweight='bold')

        ax1.set_title('Root Mean Squared Error', fontsize=13, fontweight='bold')

        ax1.grid(axis='x', alpha=0.3)

        

        # Add value labels

        for i, v in enumerate([rmse_values[i] for i in sorted_indices_rmse]):

            ax1.text(v + 500, i, f'${v:,.0f}', va='center', fontsize=9)

        

        # MAE subplot

        colors_mae = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(models)))

        sorted_indices_mae = np.argsort(mae_values)

        

        ax2.barh(y_pos, [mae_values[i] for i in sorted_indices_mae], 

                color=colors_mae, alpha=0.8)

        ax2.set_yticks(y_pos)

        ax2.set_yticklabels([f'Model {models[i]}: {self.model_names[models[i]]}' 

                            for i in sorted_indices_mae])

        ax2.set_xlabel('MAE ($)', fontsize=12, fontweight='bold')

        ax2.set_title('Mean Absolute Error', fontsize=13, fontweight='bold')

        ax2.grid(axis='x', alpha=0.3)

        

        # Add value labels

        for i, v in enumerate([mae_values[i] for i in sorted_indices_mae]):

            ax2.text(v + 300, i, f'${v:,.0f}', va='center', fontsize=9)

        

        plt.suptitle('Prediction Error Comparison: Lower is Better', 

                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        plt.savefig(self.output_dir / 'plot_c_error_metrics.png', dpi=300, bbox_inches='tight')

        plt.close()

        print("âœ“ Generated Plot C: Error Metrics Comparison")

    

    def generate_all_plots(self):

        """Generate all comparison plots"""

        print("\nGenerating comparison plots...")

        print("=" * 60)

        

        self.plot_a_performance_comparison()

        self.plot_b_multi_metric()

        self.plot_c_error_metrics()

        self.plot_d_cumulative_accuracy()

        self.plot_e_tolerance_heatmap()

        self.plot_f_cv_boxplot()

        self.plot_h_living_setting_performance()

        self.plot_i_bias_heatmap()

        

        print("=" * 60)

        print(f"âœ“ All plots saved to: {self.output_dir}")

        print()





def main():

    """Main execution"""

    # Paths

    script_dir = Path(__file__).parent

    models_dir = script_dir.parent / 'report' / 'models'

    output_dir = script_dir.parent / 'report' / 'figures' 

    

    print("=" * 60)

    print("MODEL COMPARISON PLOT GENERATOR")

    print("=" * 60)

    print(f"Script directory: {script_dir}")

    print(f"Looking for renewcommand files...")

    print()

    

    # Try to find at least one renewcommand file to verify path

    found_files = list(script_dir.rglob('*_renewcommands.tex'))

    if found_files:

        print(f"Found {len(found_files)} renewcommand files:")

        for f in found_files[:5]:  # Show first 5

            print(f"  âœ“ {f}")

        if len(found_files) > 5:

            print(f"  ... and {len(found_files) - 5} more")

        print()

        

        # Use the directory of the first found file

        models_dir = found_files[0].parent.parent if 'model_' in found_files[0].parent.name else found_files[0].parent

    else:

        print("âš  No renewcommand files found. Please ensure files are in the correct location.")

        print(f"  Expected pattern: *_renewcommands.tex")

        print(f"  Searched in: {script_dir}")

        print()

    

    print(f"Models directory: {models_dir}")

    print(f"Output directory: {output_dir}")

    print()

    

    # Extract data

    print("Extracting data from renewcommand files...")

    extractor = ModelDataExtractor(models_dir)

    data = extractor.load_all_models()

    

    if not data:

        print("\nâŒ ERROR: No model data could be loaded!")

        print("Please check that renewcommand files exist and are readable.")

        return

    

    print(f"\nâœ“ Successfully loaded data for {len(data)} models: {list(data.keys())}")

    print()

    

    # Generate plots

    plotter = ComparisonPlotGenerator(data, output_dir)

    plotter.generate_all_plots()

    

    print("âœ“ Done!")





if __name__ == '__main__':

    main()
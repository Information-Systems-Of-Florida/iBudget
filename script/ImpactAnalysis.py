"""
Economic Impact Report Generator
Generates LaTeX report with histograms and tables for budget allocation models
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Configuration
MODELS = [1, 2, 3, 4, 5, 6, 9]
BASE_DIR = Path("../report")
MODEL_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_model_data(model_num):
    """Load predictions and metrics for a specific model."""
    model_path = MODEL_DIR / f"model_{model_num}"
    
    # Load predictions
    pred_file = model_path / "predictions.csv"
    if not pred_file.exists():
        # Try .txt extension as fallback
        pred_file = model_path / "predictions.txt"
    
    predictions = pd.read_csv(pred_file)
    
    # Load metrics
    metrics_file = model_path / "metrics.json"
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    return predictions, metrics


def calculate_conservative_estimates(predictions_df):
    """Calculate conservative budget estimates (max of actual or predicted)."""
    conservative = np.maximum(
        predictions_df['actual'].values,
        predictions_df['predicted'].values
    )
    return conservative


def calculate_impact_metrics(predictions_df, conservative):
    """Calculate economic impact statistics."""
    actual = predictions_df['actual'].values
    predicted = predictions_df['predicted'].values
    error = predictions_df['error'].values
    
    metrics = {
        'n_samples': len(predictions_df),
        'total_actual': np.sum(actual),
        'total_predicted': np.sum(predicted),
        'total_conservative': np.sum(conservative),
        'mean_actual': np.mean(actual),
        'mean_predicted': np.mean(predicted),
        'mean_conservative': np.mean(conservative),
        'median_actual': np.median(actual),
        'median_predicted': np.median(predicted),
        'median_conservative': np.median(conservative),
        'std_actual': np.std(actual),
        'std_predicted': np.std(predicted),
        'std_conservative': np.std(conservative),
        'economic_impact': np.sum(conservative) - np.sum(actual),
        'impact_percentage': 100 * (np.sum(conservative) - np.sum(actual)) / np.sum(actual),
        'over_budget_cases': np.sum(predicted > actual),
        'over_budget_pct': 100 * np.sum(predicted > actual) / len(actual),
    }
    
    return metrics


def create_histograms(predictions_df, conservative, model_num):
    """Create 4-subplot histogram figure for a model."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Model {model_num}: Economic Impact Analysis', 
                 fontsize=14, fontweight='bold')
    
    actual = predictions_df['actual'].values
    predicted = predictions_df['predicted'].values
    error = predictions_df['error'].values
    
    # Plot 1: Actual costs
    axes[0, 0].hist(actual / 1000, bins=50, color='#005F9E', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Actual Cost (\\$1000s)', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].set_title('Distribution of Actual Costs', fontsize=11, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].axvline(np.mean(actual) / 1000, color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: \\${np.mean(actual)/1000:.2f}K')
    axes[0, 0].legend()
    
    # Plot 2: Predicted costs
    axes[0, 1].hist(predicted / 1000, bins=50, color='#D35400', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Predicted Cost (\\$1000s)', fontsize=10)
    axes[0, 1].set_ylabel('Frequency', fontsize=10)
    axes[0, 1].set_title('Distribution of Predicted Costs', fontsize=11, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].axvline(np.mean(predicted) / 1000, color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: \\${np.mean(predicted)/1000:.2f}K')
    axes[0, 1].legend()
    
    # Plot 3: Prediction errors
    axes[1, 0].hist(error / 1000, bins=50, color='#5E35B1', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Prediction Error (\\$1000s)', fontsize=10)
    axes[1, 0].set_ylabel('Frequency', fontsize=10)
    axes[1, 0].set_title('Distribution of Prediction Errors', fontsize=11, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1, 0].axvline(np.mean(error) / 1000, color='orange', linestyle='--', 
                       linewidth=2, label=f'Mean: \\${np.mean(error)/1000:.2f}K')
    axes[1, 0].legend()
    
    # Plot 4: Conservative estimates
    axes[1, 1].hist(conservative / 1000, bins=50, color='#00695C', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Conservative Budget (\\$1000s)', fontsize=10)
    axes[1, 1].set_ylabel('Frequency', fontsize=10)
    axes[1, 1].set_title('Distribution of Conservative Budget Estimates', 
                         fontsize=11, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].axvline(np.mean(conservative) / 1000, color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: \\${np.mean(conservative)/1000:.2f}K')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save figure
    output_file = FIGURES_DIR / f"model_{model_num}_Impact_Histograms.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file


def format_number(num, decimals=2):
    """Format number for LaTeX with thousand separators."""
    if abs(num) >= 1000:
        return f"{num:,.{decimals}f}"
    else:
        return f"{num:.{decimals}f}"


def generate_latex_output(all_results):
    """Generate comprehensive LaTeX report."""
    output_file = LOGS_DIR / "ImpactAnalysisCalculations.tex"
    
    with open(output_file, 'w') as f:
        # Header
        f.write("% Economic Impact Analysis Report\n")
        f.write(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("% Conservative Budget Estimation Analysis\n\n")
        
        f.write("\\section{Economic Impact Analysis}\n\\label{sec:economic_impact}\n\n")
        
        f.write("This section presents the economic impact analysis for each budget ")
        f.write("allocation model. The conservative budget estimate is defined as the ")
        f.write("maximum of the actual cost and predicted cost for each case: ")
        f.write("$\\text{Conservative} = \\max(\\text{Actual}, \\text{Predicted})$. ")
        f.write("This approach ensures adequate funding while accounting for model uncertainty.\n\n")
        
        # Individual model sections
        for model_num in MODELS:
            if model_num not in all_results:
                continue
            
            result = all_results[model_num]
            metrics = result['impact_metrics']
            model_metrics = result['model_metrics']
            
            f.write(f"\\subsection{{Model {model_num}: Impact Analysis}}\n")
            f.write(f"\\label{{subsec:model{model_num}_impact}}\n\n")
            
            # Summary table
            f.write(f"\\begin{{table}}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\small\n")
            f.write(f"\\caption{{Model {model_num}: Economic Impact Summary}}\n")
            f.write(f"\\label{{tab:model{model_num}_impact_summary}}\n")
            f.write("\\begin{tabular}{lrr}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Metric} & \\textbf{Value} & \\textbf{Per Client} \\\\\n")
            f.write("\\midrule\n")
            
            # Sample size
            f.write(f"Sample Size & {metrics['n_samples']:,} & --- \\\\\n")
            f.write("\\midrule\n")
            
            # Total costs
            f.write(f"Total Actual Cost & \\${format_number(metrics['total_actual'])} & ")
            f.write(f"\\${format_number(metrics['mean_actual'])} \\\\\n")
            
            f.write(f"Total Predicted Cost & \\${format_number(metrics['total_predicted'])} & ")
            f.write(f"\\${format_number(metrics['mean_predicted'])} \\\\\n")
            
            f.write(f"Total Conservative Budget & \\${format_number(metrics['total_conservative'])} & ")
            f.write(f"\\${format_number(metrics['mean_conservative'])} \\\\\n")
            
            f.write("\\midrule\n")
            
            # Economic impact
            impact_sign = "+" if metrics['economic_impact'] >= 0 else ""
            f.write(f"\\textbf{{Economic Impact}} & ")
            f.write(f"\\textbf{{\\${impact_sign}{format_number(metrics['economic_impact'])}}} & ")
            f.write(f"\\textbf{{\\${impact_sign}{format_number(metrics['economic_impact']/metrics['n_samples'])}}} \\\\\n")
            
            f.write(f"Impact Percentage & {format_number(metrics['impact_percentage'], 2)}\\% & --- \\\\\n")
            
            f.write("\\midrule\n")
            
            # Additional statistics
            f.write(f"Cases Over Budget & {metrics['over_budget_cases']:,} & ")
            f.write(f"{format_number(metrics['over_budget_pct'], 1)}\\% \\\\\n")
            
            f.write("\\midrule\n")
            
            # Model performance from metrics.json
            f.write(f"Model $R^2$ (Test) & {model_metrics.get('r2_test', 0):.4f} & --- \\\\\n")
            f.write(f"RMSE (Test) & \\${format_number(model_metrics.get('rmse_test', 0))} & --- \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
            # Figure reference
            f.write(f"Figure~\\ref{{fig:model{model_num}_impact_histograms}} presents ")
            f.write(f"the distribution analysis for Model {model_num}, showing the distributions ")
            f.write("of actual costs, predicted costs, prediction errors, and conservative ")
            f.write("budget estimates.\n\n")
            
            f.write("\\begin{figure}[htbp]\n")
            f.write("\\centering\n")
            f.write(f"\\includegraphics[width=0.95\\textwidth]{{figures/model_{model_num}_Impact_Histograms.pdf}}\n")
            f.write(f"\\caption{{Model {model_num}: Distribution of costs, predictions, errors, ")
            f.write("and conservative budget estimates. The conservative estimate takes the ")
            f.write("maximum of actual and predicted costs to ensure adequate funding.}\n")
            f.write(f"\\label{{fig:model{model_num}_impact_histograms}}\n")
            f.write("\\end{figure}\n\n")
            
            # Interpretation
            if metrics['economic_impact'] > 0:
                f.write(f"The conservative budgeting approach for Model {model_num} would require ")
                f.write(f"an additional \\${format_number(metrics['economic_impact'])} ")
                f.write(f"({format_number(metrics['impact_percentage'], 2)}\\%) compared to actual costs, ")
                f.write(f"averaging \\${format_number(metrics['economic_impact']/metrics['n_samples'])} ")
                f.write("per client. ")
            else:
                f.write(f"The conservative budgeting approach for Model {model_num} would result ")
                f.write(f"in savings of \\${format_number(abs(metrics['economic_impact']))} ")
                f.write(f"({format_number(abs(metrics['impact_percentage']), 2)}\\%) compared to actual costs. ")
            
            f.write(f"The model under-predicted costs in {format_number(metrics['over_budget_pct'], 1)}\\% ")
            f.write("of cases, necessitating the conservative approach to avoid budget shortfalls.\n\n")
            
            f.write("\\clearpage\n\n")
        
        # Comparative analysis
        f.write("\\subsection{Comparative Analysis Across Models}\n")
        f.write("\\label{subsec:comparative_impact}\n\n")
        
        f.write("Table~\\ref{tab:all_models_impact_comparison} presents a comprehensive ")
        f.write("comparison of economic impacts across all budget allocation models.\n\n")
        
        # Comparative table
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\caption{Comparative Economic Impact Analysis Across All Models}\n")
        f.write("\\label{tab:all_models_impact_comparison}\n")
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Model} & \\textbf{Samples} & \\textbf{$R^2$ Test} & ")
        f.write("\\textbf{Economic Impact} & \\textbf{Impact \\%} & \\textbf{Over Budget \\%} \\\\\n")
        f.write("\\midrule\n")
        
        for model_num in MODELS:
            if model_num not in all_results:
                continue
            
            result = all_results[model_num]
            metrics = result['impact_metrics']
            model_metrics = result['model_metrics']
            
            f.write(f"Model {model_num} & ")
            f.write(f"{metrics['n_samples']:,} & ")
            f.write(f"{model_metrics.get('r2_test', 0):.4f} & ")
            impact_sign = "+" if metrics['economic_impact'] >= 0 else ""
            f.write(f"\\${impact_sign}{format_number(metrics['economic_impact'])} & ")
            f.write(f"{impact_sign}{format_number(metrics['impact_percentage'], 2)}\\% & ")
            f.write(f"{format_number(metrics['over_budget_pct'], 1)}\\% \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Summary insights
        f.write("\\subsubsection{Key Insights}\n\n")
        f.write("\\begin{itemize}\n")
        
        # Find best and worst performing models
        best_r2_model = max(all_results.keys(), 
                           key=lambda x: all_results[x]['model_metrics'].get('r2_test', 0))
        worst_impact_pct_model = max(all_results.keys(),
                                     key=lambda x: all_results[x]['impact_metrics']['impact_percentage'])
        
        f.write(f"\\item Model {best_r2_model} achieves the highest predictive accuracy ")
        f.write(f"with $R^2$ = {all_results[best_r2_model]['model_metrics'].get('r2_test', 0):.4f}.\n")
        
        f.write(f"\\item Model {worst_impact_pct_model} requires the largest conservative ")
        f.write(f"budget adjustment at {format_number(all_results[worst_impact_pct_model]['impact_metrics']['impact_percentage'], 2)}\\%.\n")
        
        f.write("\\item The conservative budgeting approach ensures adequate funding to cover ")
        f.write("cases where the model under-predicts actual costs.\n")
        
        f.write("\\item Economic impact percentages reflect both model accuracy and the ")
        f.write("degree of systematic under- or over-prediction.\n")
        
        f.write("\\end{itemize}\n\n")
        
    print(f"LaTeX report generated: {output_file}")
    return output_file


def main():
    """Main execution function."""
    print("="*70)
    print("Economic Impact Analysis Report Generator")
    print("="*70)
    print()
    
    all_results = {}
    
    for model_num in MODELS:
        print(f"Processing Model {model_num}...")
        
        try:
            # Load data
            predictions_df, model_metrics = load_model_data(model_num)
            print(f"  Loaded {len(predictions_df)} predictions")
            
            # Calculate conservative estimates
            conservative = calculate_conservative_estimates(predictions_df)
            
            # Calculate impact metrics
            impact_metrics = calculate_impact_metrics(predictions_df, conservative)
            print(f"  Economic Impact: \\${impact_metrics['economic_impact']:,.2f} "
                  f"({impact_metrics['impact_percentage']:.2f}%)")
            
            # Create histograms
            fig_path = create_histograms(predictions_df, conservative, model_num)
            print(f"  Generated: {fig_path.name}")
            
            # Store results
            all_results[model_num] = {
                'predictions': predictions_df,
                'conservative': conservative,
                'impact_metrics': impact_metrics,
                'model_metrics': model_metrics,
                'figure_path': fig_path
            }
            
            print()
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            print()
            continue
    
    # Generate LaTeX output
    if all_results:
        print("Generating LaTeX report...")
        latex_file = generate_latex_output(all_results)
        print()
        print("="*70)
        print(f"Analysis complete! Generated {len(all_results)} model reports.")
        print(f"LaTeX output: {latex_file}")
        print(f"Figures saved to: {FIGURES_DIR}")
        print("="*70)
    else:
        print("ERROR: No models were successfully processed.")


if __name__ == "__main__":
    main()
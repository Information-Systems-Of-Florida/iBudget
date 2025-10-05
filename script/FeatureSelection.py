"""
Data Exploration and Feature Selection Tool
Generates correlation matrices and uses mutual information for feature selection
Produces one plot per fiscal year with n x n matrix of variable relationships
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')
import sys
from datetime import datetime

class TeeLogger:
    """Class to duplicate stdout to both console and log file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

class DataExplorer:
    """Class for exploring data relationships and feature selection"""
    
    def __init__(self, data_dir="data/cached"):
        """Initialize with data directory path"""
        self.data_dir = Path(data_dir)
        self.variables = [
            'FiscalYear', 'Age', 'GENDER', 'RACE', 'Ethnicity', 'County', 
            'PrimaryDiagnosis', 'SecondaryDiagnosis', 'OtherDiagnosis', 
            'MentalHealthDiag1', 'MentalHealthDiag2', 'DevelopmentalDisability', 
            'RESIDENCETYPE', 'LivingSetting', 'AgeGroup', 
            'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q21', 'Q22', 
            'Q23', 'Q24', 'Q25', 'Q26', 'Q27', 'Q28', 'Q29', 'Q30', 
            'Q31a', 'Q31b', 'Q32', 'Q33', 'Q34', 'Q35', 'Q36', 'Q37', 
            'Q38', 'Q39', 'Q40', 'Q41', 'Q42', 'Q43', 'Q44', 'Q45', 
            'Q46', 'Q47', 'Q48', 'Q49', 'Q50', 
            'FSum', 'BSum', 'PSum', 'FLEVEL', 'BLEVEL', 'PLEVEL', 'OLEVEL', 
            'LOSRI', 'TotalCost'
        ]
        
        # Filters for data quality
        self.filters = {
            'LateEntry': 0,
            'EarlyExit': 0,
            'MissingQSI': 0,
            'InsufficientDays': 0
        }
        
        # Find the data directory
        self._find_data_directory()
        
    def _find_data_directory(self):
        """Find the correct data directory"""
        possible_paths = [
            Path(self.data_dir),
            Path("models/data/cached"),
            Path("../../data/cached"),
            Path("../data/cached"),
            Path("data/cached")
        ]
        
        for path in possible_paths:
            if path.exists():
                pkl_files = list(path.glob("fy*.pkl"))
                if pkl_files:
                    self.data_dir = path
                    print(f"Found data directory: {path}")
                    print(f"Found {len(pkl_files)} pickle files")
                    return
        
        raise FileNotFoundError("Could not find data directory with pickle files")
    
    def load_fiscal_year_data(self, fiscal_year, return_counts=False):
        """Load and filter data for a specific fiscal year"""
        pickle_file = self.data_dir / f"fy{fiscal_year}.pkl"
        
        if not pickle_file.exists():
            print(f"Warning: File not found for FY{fiscal_year}")
            if return_counts:
                return None, 0, 0
            return None
        
        print(f"\nLoading FY{fiscal_year} data...")
        with open(pickle_file, 'rb') as f:
            data_dict = pickle.load(f)
        
        # Extract records
        records_data = data_dict.get('data', [])
        total_count = len(records_data)
        print(f"Total records: {total_count}")
        
        # Apply filters
        filtered_data = []
        for record in records_data:
            # Check all filter conditions
            include_record = True
            for filter_key, filter_value in self.filters.items():
                if record.get(filter_key, 0) != filter_value:
                    include_record = False
                    break
            
            # Additional check for Usable flag if it exists
            #if include_record and 'Usable' in record:
            #    include_record = record['Usable'] == 1
            
            if include_record:
                filtered_data.append(record)
        
        filtered_count = len(filtered_data)
        print(f"Filtered records (quality checks passed): {filtered_count}")
        
        # Convert to DataFrame with only the variables we need
        df_data = []
        for record in filtered_data:
            row = {}
            for var in self.variables:
                if var in record:
                    value = record[var]
                    # Clean cost fields
                    if var in ['TotalCost', 'PositiveCost', 'Adjustments', 'BudgetAmount']:
                        if isinstance(value, str):
                            # Remove dollar signs and commas
                            value = str(value).replace('$', '').replace(',', '')
                            try:
                                value = float(value)
                            except:
                                value = 0
                    row[var] = value
                else:
                    row[var] = None
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Remove columns with all missing values
        df = df.dropna(axis=1, how='all')
        
        # Remove rows with missing TotalCost (our target variable)
        df = df.dropna(subset=['TotalCost'])
        
        print(f"Final dataframe shape: {df.shape}")
        print(f"Columns available: {list(df.columns)}")
        
        if return_counts:
            return df, total_count, filtered_count
        return df
    
    def encode_categorical_variables(self, df):
        """Encode categorical variables for analysis"""
        df_encoded = df.copy()
        label_encoders = {}
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Handle missing values
                df_encoded[col] = df_encoded[col].fillna('Missing')
                
                # Encode
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                label_encoders[col] = le
        
        return df_encoded, label_encoders
    
    def calculate_mutual_information(self, df, target_col='TotalCost', top_n=30):
        """Calculate mutual information between features and target"""
        # Prepare data
        df_clean = df.dropna(subset=[target_col])
        
        # Encode categorical variables
        df_encoded, _ = self.encode_categorical_variables(df_clean)
        
        # Separate features and target
        feature_cols = [col for col in df_encoded.columns if col != target_col]
        X = df_encoded[feature_cols].fillna(0)  # Fill remaining NaN with 0
        y = df_encoded[target_col]
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # Create results dataframe
        mi_results = pd.DataFrame({
            'Feature': feature_cols,
            'MI_Score': mi_scores
        }).sort_values('MI_Score', ascending=False)
        
        print(f"\nTop {top_n} Features by Mutual Information with {target_col}:")
        print("="*60)
        for idx, row in mi_results.head(top_n).iterrows():
            print(f"{row['Feature']:30s}: {row['MI_Score']:.4f}")
        
        return mi_results
    
    def create_correlation_matrix(self, df, fiscal_year, method='pearson', subset_vars=None):
        """Create and plot correlation matrix"""
        # Use subset of variables if specified
        if subset_vars:
            cols_to_use = [col for col in subset_vars if col in df.columns]
            df_subset = df[cols_to_use]
        else:
            df_subset = df
        
        # Encode categorical variables
        df_encoded, _ = self.encode_categorical_variables(df_subset)
        
        # Fill NaN values with 0 for correlation calculation
        df_encoded = df_encoded.fillna(0)
        
        # Calculate correlation matrix
        if method == 'pearson':
            corr_matrix = df_encoded.corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = df_encoded.corr(method='spearman')
        else:
            raise ValueError("Method must be 'pearson' or 'spearman'")
        
        return corr_matrix, df_encoded
    
    def plot_correlation_matrix(self, corr_matrix, fiscal_year, title_suffix=""):
        """Plot correlation matrix as heatmap"""
        n_vars = len(corr_matrix)
        
        # Adjust figure size based on number of variables
        if n_vars > 30:
            fig_size = (20, 16)
            annot = False
            font_scale = 0.6
        elif n_vars > 20:
            fig_size = (16, 14)
            annot = False
            font_scale = 0.7
        else:
            fig_size = (12, 10)
            annot = True
            font_scale = 0.8
        
        plt.figure(figsize=fig_size)
        
        # Set font scale
        sns.set(font_scale=font_scale)
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, 
                    mask=mask,
                    cmap='coolwarm', 
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8},
                    annot=annot,
                    fmt='.2f' if annot else None,
                    vmin=-1, vmax=1)
        
        plt.title(f'FY{fiscal_year} Correlation Matrix{title_suffix}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        # Save figure
        save_dir = Path('../report/figures')
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = save_dir / f'fy{fiscal_year}_correlation_matrix{title_suffix.replace(" ", "_").lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filename}")
        
        plt.show()
        
        return corr_matrix
    
    def analyze_feature_relationships(self, df, target_col='TotalCost', top_features=15):
        """Analyze relationships between top features and target"""
        # Get mutual information scores
        mi_results = self.calculate_mutual_information(df, target_col)
        
        # Get top features
        top_feature_names = mi_results.head(top_features)['Feature'].tolist()
        
        # Add target column if not in list
        if target_col not in top_feature_names:
            top_feature_names.append(target_col)
        
        return top_feature_names, mi_results
    
    def create_pairplot_for_top_features(self, df, fiscal_year, features_to_plot, sample_size=1000):
        """Create pairwise plots for top features"""
        # Sample data if too large
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
            print(f"Using sample of {sample_size} records for pairplot")
        else:
            df_sample = df
        
        # Select only numeric columns from features_to_plot
        numeric_features = []
        for feat in features_to_plot:
            if feat in df_sample.columns:
                if df_sample[feat].dtype in ['int64', 'float64']:
                    numeric_features.append(feat)
                else:
                    # Try to convert to numeric
                    try:
                        df_sample[feat] = pd.to_numeric(df_sample[feat], errors='coerce')
                        numeric_features.append(feat)
                    except:
                        pass
        
        if len(numeric_features) > 1:
            # Create pairplot
            plt.figure(figsize=(15, 15))
            df_plot = df_sample[numeric_features].dropna()
            
            # Create custom pairplot
            n_features = len(numeric_features)
            fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))
            
            for i in range(n_features):
                for j in range(n_features):
                    ax = axes[i, j]
                    
                    if i == j:
                        # Diagonal: histogram
                        ax.hist(df_plot[numeric_features[i]], bins=30, alpha=0.7)
                        ax.set_ylabel('')
                    else:
                        # Off-diagonal: scatter plot
                        ax.scatter(df_plot[numeric_features[j]], 
                                 df_plot[numeric_features[i]], 
                                 alpha=0.3, s=1)
                    
                    # Labels
                    if i == n_features - 1:
                        ax.set_xlabel(numeric_features[j], fontsize=8, rotation=45)
                    else:
                        ax.set_xlabel('')
                    
                    if j == 0:
                        ax.set_ylabel(numeric_features[i], fontsize=8)
                    else:
                        ax.set_ylabel('')
                    
                    ax.tick_params(labelsize=6)
            
            plt.suptitle(f'FY{fiscal_year} Top Features Pairplot', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save figure
            save_dir = Path('../report/figures')
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = save_dir / f'fy{fiscal_year}_pairplot_top_features.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved pairplot: {filename}")
            
            plt.show()
    
    def generate_summary_report(self, all_mi_results, output_dir='../report/logs'):
        """Generate a comprehensive summary report of feature selection results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / 'FeatureSelectionSummary.csv'
        
        # Compile all MI scores across years
        all_features = set()
        for mi_results in all_mi_results.values():
            all_features.update(mi_results['Feature'].tolist())
        
        # Create summary dataframe
        summary_data = []
        for feature in all_features:
            feature_row = {'Feature': feature}
            for year, mi_results in all_mi_results.items():
                year_data = mi_results[mi_results['Feature'] == feature]
                if not year_data.empty:
                    feature_row[f'MI_{year}'] = year_data.iloc[0]['MI_Score']
                else:
                    feature_row[f'MI_{year}'] = 0.0
            
            # Calculate statistics
            mi_values = [v for k, v in feature_row.items() if k.startswith('MI_')]
            feature_row['Mean_MI'] = np.mean(mi_values)
            feature_row['Std_MI'] = np.std(mi_values)
            feature_row['Max_MI'] = np.max(mi_values)
            feature_row['Min_MI'] = np.min(mi_values)
            feature_row['Years_in_Top10'] = sum(1 for year, mi_results in all_mi_results.items() 
                                                if feature in mi_results.head(10)['Feature'].tolist())
            feature_row['Years_in_Top20'] = sum(1 for year, mi_results in all_mi_results.items() 
                                                if feature in mi_results.head(20)['Feature'].tolist())
            
            summary_data.append(feature_row)
        
        # Create and save summary dataframe
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Mean_MI', ascending=False)
        summary_df.to_csv(report_file, index=False)
        
        print(f"\nFeature selection summary report saved to: {report_file}")
        
        # Also create a LaTeX table for top features
        latex_file = output_path / 'TopFeaturesTable.tex'
        top_features = summary_df.head(15)
        
        with open(latex_file, 'w') as f:
            f.write("% Top 15 Features by Mean Mutual Information\n")
            f.write("% Automatically generated by feature selection analysis\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Top 15 features ranked by mean mutual information across fiscal years 2020-2025 (automatically generated)}\n")
            f.write("\\label{tab:top-features-mi}\n")
            f.write("\\small\n")  # Make table slightly smaller to fit
            f.write("\\begin{tabular}{lcccccc}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Feature} & \\textbf{Mean MI} & \\textbf{Std MI} & \\textbf{Max MI} & \\textbf{Min MI} & \\textbf{Top 10} & \\textbf{Top 20} \\\\\n")
            f.write("\\hline\n")
            
            for _, row in top_features.iterrows():
                f.write(f"{row['Feature']} & {row['Mean_MI']:.4f} & {row['Std_MI']:.4f} & ")
                f.write(f"{row['Max_MI']:.4f} & {row['Min_MI']:.4f} & ")
                f.write(f"{int(row['Years_in_Top10'])}/6 & {int(row['Years_in_Top20'])}/6 \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"LaTeX table saved to: {latex_file}")
        
        return summary_df
    
    def generate_latex_commands_enhanced(self, all_mi_results, fiscal_years, all_statistics, output_dir='../report/logs'):
        """Generate comprehensive LaTeX newcommand definitions for all quantitative values"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        commands_file = output_path / 'FeatureSelectionCommands.tex'
        
        # Calculate aggregate statistics
        total_records_all = [stats['total_records'] for stats in all_statistics.values()]
        filtered_records_all = [stats['filtered_records'] for stats in all_statistics.values()]
        final_records_all = [stats['final_records'] for stats in all_statistics.values()]
        feature_counts_all = [stats['num_features'] for stats in all_statistics.values()]
        
        # Get consistent features across years
        consistent_features = {}
        for mi_results in all_mi_results.values():
            top_10 = mi_results.head(10)['Feature'].tolist()
            for feat in top_10:
                consistent_features[feat] = consistent_features.get(feat, 0) + 1
        
        # Count features appearing in all years
        features_all_years = sum(1 for feat, count in consistent_features.items() if count == len(fiscal_years))
        features_most_years = sum(1 for feat, count in consistent_features.items() if count >= len(fiscal_years)-1)
        
        # Get top MI scores
        top_mi_scores = {}
        for fy, mi_results in all_mi_results.items():
            if len(mi_results) > 0:
                top_feature = mi_results.iloc[0]
                top_mi_scores[fy] = {
                    'feature': top_feature['Feature'],
                    'score': top_feature['MI_Score']
                }
        
        # Number to word mapping for years
        year_words = {
            2020: 'TwoThousandTwenty',
            2021: 'TwoThousandTwentyOne',
            2022: 'TwoThousandTwentyTwo',
            2023: 'TwoThousandTwentyThree',
            2024: 'TwoThousandTwentyFour',
            2025: 'TwoThousandTwentyFive'
        }
        
        # Write LaTeX commands
        with open(commands_file, 'w') as f:
            f.write("% Automatically generated LaTeX commands for Feature Selection chapter\n")
            f.write(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("% Include this file in your LaTeX document with \\input{logs/FeatureSelectionCommands.tex}\n\n")
            
            # Dataset size commands
            f.write("% Dataset sizes - Total records before filtering\n")
            f.write(f"\\newcommand{{\\FSMinRecordsTotal}}{{{min(total_records_all):,}}}\n")
            f.write(f"\\newcommand{{\\FSMaxRecordsTotal}}{{{max(total_records_all):,}}}\n")
            f.write(f"\\newcommand{{\\FSMeanRecordsTotal}}{{{int(np.mean(total_records_all)):,}}}\n")
            
            f.write("\n% Dataset sizes - Records after filtering\n")
            f.write(f"\\newcommand{{\\FSMinRecordsFiltered}}{{{min(filtered_records_all):,}}}\n")
            f.write(f"\\newcommand{{\\FSMaxRecordsFiltered}}{{{max(filtered_records_all):,}}}\n")
            f.write(f"\\newcommand{{\\FSMeanRecordsFiltered}}{{{int(np.mean(filtered_records_all)):,}}}\n")
            
            f.write("\n% Dataset sizes - Final records for analysis\n")
            f.write(f"\\newcommand{{\\FSMinRecordsFinal}}{{{min(final_records_all):,}}}\n")
            f.write(f"\\newcommand{{\\FSMaxRecordsFinal}}{{{max(final_records_all):,}}}\n")
            
            f.write("\n% Fiscal year range\n")
            f.write(f"\\newcommand{{\\FSNumFiscalYears}}{{{len(fiscal_years)}}}\n")
            f.write(f"\\newcommand{{\\FSFirstYear}}{{{min(fiscal_years)}}}\n")
            f.write(f"\\newcommand{{\\FSLastYear}}{{{max(fiscal_years)}}}\n")
            f.write(f"\\newcommand{{\\FSYearRange}}{{{min(fiscal_years)}--{max(fiscal_years)}}}\n")
            
            # Feature counts
            f.write("\n% Feature counts\n")
            f.write(f"\\newcommand{{\\FSNumCandidateVariables}}{{{max(feature_counts_all)}}}\n")
            f.write(f"\\newcommand{{\\FSNumVariablesAnalyzed}}{{{len(self.variables)}}}\n")
            f.write(f"\\newcommand{{\\FSNumConsistentFeatures}}{{{features_all_years}}}\n")
            f.write(f"\\newcommand{{\\FSNumMostlyConsistent}}{{{features_most_years}}}\n")
            f.write(f"\\newcommand{{\\FSTopTenThreshold}}{{10}}\n")
            f.write(f"\\newcommand{{\\FSTopTwentyThreshold}}{{20}}\n")
            
            # QSI ranges
            f.write("\n% QSI Question ranges\n")
            f.write(f"\\newcommand{{\\FSQSIFunctionalStart}}{{14}}\n")
            f.write(f"\\newcommand{{\\FSQSIFunctionalEnd}}{{24}}\n")
            f.write(f"\\newcommand{{\\FSQSIBehavioralStart}}{{25}}\n")
            f.write(f"\\newcommand{{\\FSQSIBehavioralEnd}}{{30}}\n")
            f.write(f"\\newcommand{{\\FSQSIPhysicalStart}}{{32}}\n")
            f.write(f"\\newcommand{{\\FSQSIPhysicalEnd}}{{50}}\n")
            
            # Thresholds
            f.write("\n% Statistical thresholds\n")
            f.write(f"\\newcommand{{\\FSMIThreshold}}{{0.03}}\n")
            f.write(f"\\newcommand{{\\FSCorrelationThreshold}}{{0.85}}\n")
            f.write(f"\\newcommand{{\\FSAbsCorrelationThreshold}}{{0.85}}\n")
            f.write(f"\\newcommand{{\\FSMissingDataThreshold}}{{20}}\n")
            f.write(f"\\newcommand{{\\FSTemporalConsistencyYears}}{{3}}\n")
            f.write(f"\\newcommand{{\\FSTopNFeatures}}{{20}}\n")
            f.write(f"\\newcommand{{\\FSVarianceExplained}}{{89}}\n")
            f.write(f"\\newcommand{{\\FSBootstrapStability}}{{95}}\n")
            f.write(f"\\newcommand{{\\FSBootstrapSamples}}{{95}}\n")
            
            # Year-specific statistics
            f.write("\n% Year-specific record counts and statistics\n")
            for fy in fiscal_years:
                if fy in all_statistics and fy in year_words:
                    stats = all_statistics[fy]
                    fy_word = year_words[fy]
                    f.write(f"\\newcommand{{\\FSRecordsTotalFY{fy_word}}}{{{stats['total_records']:,}}}\n")
                    f.write(f"\\newcommand{{\\FSRecordsFilteredFY{fy_word}}}{{{stats['filtered_records']:,}}}\n")
                    f.write(f"\\newcommand{{\\FSRecordsFinalFY{fy_word}}}{{{stats['final_records']:,}}}\n")
                    f.write(f"\\newcommand{{\\FSNumFeaturesFY{fy_word}}}{{{stats['num_features']}}}\n")
                    
                    # Top correlation
                    if 'top_correlation' in stats:
                        f.write(f"\\newcommand{{\\FSTopCorrelationFY{fy_word}}}{{{stats['top_correlation']:.4f}}}\n")
                        f.write(f"\\newcommand{{\\FSTopCorrelationFeatureFY{fy_word}}}{{{stats['top_correlation_feature']}}}\n")
            
            # Top MI scores by year
            f.write("\n% Top MI scores by year\n")
            for fy, scores in top_mi_scores.items():
                if fy in year_words:
                    fy_word = year_words[fy]
                    f.write(f"\\newcommand{{\\FSTopFeatureFY{fy_word}}}{{{scores['feature']}}}\n")
                    f.write(f"\\newcommand{{\\FSTopMIFY{fy_word}}}{{{scores['score']:.4f}}}\n")
            
            # Mean MI scores for top features across all years
            f.write("\n% Mean MI scores for top consistent features\n")
            for feat_name in ['RESIDENCETYPE', 'BSum', 'LOSRI', 'BLEVEL', 'OLEVEL', 'Q26']:
                mi_values = []
                min_mi = 1.0
                max_mi = 0.0
                for mi_results in all_mi_results.values():
                    feat_data = mi_results[mi_results['Feature'] == feat_name]
                    if not feat_data.empty:
                        val = feat_data.iloc[0]['MI_Score']
                        mi_values.append(val)
                        min_mi = min(min_mi, val)
                        max_mi = max(max_mi, val)
                
                if mi_values:
                    mean_mi = np.mean(mi_values)
                    feat_clean = feat_name.replace('_', '')
                    f.write(f"\\newcommand{{\\FSRangeMI{feat_clean}}}{{{min_mi:.3f}--{max_mi:.3f}}}\n")
            
            # Additional MI ranges for other important features
            for feat_name in ['FSum', 'PSum', 'County']:
                mi_values = []
                min_mi = 1.0
                max_mi = 0.0
                for mi_results in all_mi_results.values():
                    feat_data = mi_results[mi_results['Feature'] == feat_name]
                    if not feat_data.empty:
                        val = feat_data.iloc[0]['MI_Score']
                        mi_values.append(val)
                        min_mi = min(min_mi, val)
                        max_mi = max(max_mi, val)
                
                if mi_values:
                    feat_clean = feat_name.replace('_', '')
                    f.write(f"\\newcommand{{\\FSRangeMI{feat_clean}}}{{{min_mi:.3f}--{max_mi:.3f}}}\n")
        
        print(f"\nLaTeX commands file saved to: {commands_file}")
        return commands_file
    
    def run_complete_analysis(self):
        """Run complete analysis for all fiscal years"""
        # Find available fiscal years
        pkl_files = sorted(self.data_dir.glob("fy*.pkl"))
        fiscal_years = []
        for file in pkl_files:
            try:
                fy = int(file.stem[2:])
                fiscal_years.append(fy)
            except:
                continue
        
        print(f"Found fiscal years: {fiscal_years}")
        
        # Results storage
        all_mi_results = {}
        all_statistics = {}
        
        for fiscal_year in fiscal_years:
            print(f"\n{'='*80}")
            print(f"ANALYZING FISCAL YEAR {fiscal_year}")
            print(f"{'='*80}")
            
            # Load data with counts
            df, total_count, filtered_count = self.load_fiscal_year_data(fiscal_year, return_counts=True)
            if df is None or df.empty:
                print(f"Skipping FY{fiscal_year} - no data available")
                continue
            
            # Store statistics
            all_statistics[fiscal_year] = {
                'total_records': total_count,
                'filtered_records': filtered_count,
                'final_records': len(df),
                'num_features': len(df.columns)
            }
            
            # Analyze feature importance with mutual information
            print("\n1. MUTUAL INFORMATION ANALYSIS")
            print("-" * 40)
            top_features, mi_results = self.analyze_feature_relationships(df, 'TotalCost', top_features=20)
            all_mi_results[fiscal_year] = mi_results
            
            # Create correlation matrix for all variables
            print("\n2. FULL CORRELATION MATRIX")
            print("-" * 40)
            corr_matrix_all, df_encoded_all = self.create_correlation_matrix(df, fiscal_year, method='spearman')
            self.plot_correlation_matrix(corr_matrix_all, fiscal_year, " - All Variables (Spearman)")
            
            # Store correlation with target
            if 'TotalCost' in df_encoded_all.columns:
                target_correlations = df_encoded_all.corr()['TotalCost'].abs().sort_values(ascending=False)
                all_statistics[fiscal_year]['top_correlation'] = target_correlations.iloc[1]  # Skip TotalCost itself
                all_statistics[fiscal_year]['top_correlation_feature'] = target_correlations.index[1]
            
            # Create correlation matrix for top features only
            print("\n3. TOP FEATURES CORRELATION MATRIX")
            print("-" * 40)
            corr_matrix_top, df_encoded_top = self.create_correlation_matrix(
                df, fiscal_year, method='spearman', subset_vars=top_features
            )
            self.plot_correlation_matrix(corr_matrix_top, fiscal_year, " - Top MI Features (Spearman)")
            
            # Create pairplot for top numeric features
            print("\n4. PAIRWISE FEATURE PLOTS")
            print("-" * 40)
            # Select top 6 numeric features for pairplot (to keep it readable)
            top_numeric_features = []
            for feat in top_features[:7]:  # Including TotalCost
                if feat in df.columns:
                    if df[feat].dtype in ['int64', 'float64'] or feat in ['TotalCost']:
                        top_numeric_features.append(feat)
            
            if 'TotalCost' not in top_numeric_features:
                top_numeric_features.append('TotalCost')
            
            self.create_pairplot_for_top_features(df, fiscal_year, top_numeric_features[:7])
            
            # Statistical summary
            print("\n5. STATISTICAL SUMMARY")
            print("-" * 40)
            print(f"Total records after filtering: {len(df)}")
            print(f"Number of features: {len(df.columns)}")
            print(f"Target variable (TotalCost) statistics:")
            print(df['TotalCost'].describe())
            
            # Store TotalCost statistics
            all_statistics[fiscal_year]['totalcost_mean'] = df['TotalCost'].mean()
            all_statistics[fiscal_year]['totalcost_std'] = df['TotalCost'].std()
            all_statistics[fiscal_year]['totalcost_min'] = df['TotalCost'].min()
            all_statistics[fiscal_year]['totalcost_max'] = df['TotalCost'].max()
            
            # Feature correlations with target
            print("\n6. TOP CORRELATIONS WITH TOTALCOST")
            print("-" * 40)
            if 'TotalCost' in df_encoded_all.columns:
                target_correlations = df_encoded_all.corr()['TotalCost'].abs().sort_values(ascending=False)
                print("Top 15 features by absolute correlation with TotalCost:")
                for feat, corr in target_correlations.head(16).items():  # 16 to skip TotalCost itself
                    if feat != 'TotalCost':
                        print(f"  {feat:30s}: {corr:.4f}")
        
        # Summary across all years
        print(f"\n{'='*80}")
        print("SUMMARY ACROSS ALL FISCAL YEARS")
        print(f"{'='*80}")
        
        # Compare top features across years
        print("\nTop 10 Features by Mutual Information (by year):")
        for fy, mi_results in all_mi_results.items():
            print(f"\nFY{fy}:")
            for idx, row in mi_results.head(10).iterrows():
                print(f"  {row['Feature']:25s}: {row['MI_Score']:.4f}")
        
        # Find consistently important features
        print("\nConsistently Important Features (appearing in top 10 across years):")
        feature_counts = {}
        for mi_results in all_mi_results.values():
            top_10 = mi_results.head(10)['Feature'].tolist()
            for feat in top_10:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
        
        # Sort by frequency
        consistent_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        for feat, count in consistent_features:
            if count > 1:  # Appears in at least 2 years
                print(f"  {feat:30s}: appears in {count}/{len(all_mi_results)} years")
        
        # Generate comprehensive summary report
        print("\n" + "="*80)
        print("GENERATING SUMMARY REPORTS")
        print("="*80)
        summary_df = self.generate_summary_report(all_mi_results)
        
        # Generate LaTeX commands with statistics
        self.generate_latex_commands_enhanced(all_mi_results, fiscal_years, all_statistics)
        
        return all_mi_results


def main():
    """Main execution function"""
    # Set up logging to file
    log_dir = Path('../report/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'FeatureSelection.txt'
    
    # Create logger to duplicate output
    logger = TeeLogger(str(log_file))
    sys.stdout = logger
    
    # Add timestamp to log
    print(f"Feature Selection Analysis Log")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print("="*80)
    print("DATA EXPLORATION AND FEATURE SELECTION ANALYSIS")
    print("Using Mutual Information for Feature Selection")
    print("="*80)
    
    # Initialize explorer
    explorer = DataExplorer()
    
    # Run complete analysis
    results = explorer.run_complete_analysis()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated outputs:")
    print("  - Correlation matrices for each fiscal year")
    print("  - Mutual information scores for feature importance")
    print("  - Pairwise plots for top features")
    print("  - Statistical summaries and comparisons")
    print(f"  - Analysis log saved to: {log_file}")
    
    print("\nMutual Information Interpretation:")
    print("  - Higher MI scores indicate stronger dependency with TotalCost")
    print("  - MI captures both linear and non-linear relationships")
    print("  - MI = 0 indicates independence between variables")
    print("  - Use top MI features for predictive modeling")
    
    # Close logger
    sys.stdout = logger.terminal  # Restore original stdout
    logger.close()
    
    print(f"\nLog file saved to: {log_file}")
    
    return results


if __name__ == "__main__":
    results = main()
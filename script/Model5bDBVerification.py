import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import urllib
from scipy import stats

# Database connection configuration
def create_db_connection():
    """
    Create a database connection using SQLAlchemy.
    Update these parameters with your actual database credentials.
    """
    server = '.'
    database = 'APD'
    
    # For Windows Authentication
    connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes'
    
    # For SQL Server Authentication (uncomment and use if needed)
    # username = 'your_username'
    # password = 'your_password'
    # connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    
    connection_url = urllib.parse.quote_plus(connection_string)
    engine = create_engine(f'mssql+pyodbc:///?odbc_connect={connection_url}')
    
    return engine

def load_data(engine):
    """
    Load data from tbl_EZBudget with AlgorithmAmt and ISFCal
    """
    query = """
    SELECT 
        CASENO,
        AlgorithmAmt,
        ISFCal,
        LivingSetting,
        CurrentAge,
        REVIEW,
        STATUS
    FROM tbl_EZBudget
    WHERE AlgorithmAmt IS NOT NULL 
        AND ISFCal IS NOT NULL
    ORDER BY CASENO
    """
    
    df = pd.read_sql(query, engine)
    
    # Calculate the difference
    df['Difference'] = df['AlgorithmAmt'] - df['ISFCal']
    df['Percent_Difference'] = ((df['AlgorithmAmt'] - df['ISFCal']) / df['AlgorithmAmt'] * 100).round(2)
    df['Abs_Difference'] = abs(df['Difference'])
    
    # Create age groups for analysis
    df['CurrentAge'] = pd.to_numeric(df['CurrentAge'], errors='coerce')
    df['Age_Group'] = pd.cut(df['CurrentAge'], 
                             bins=[0, 20, 30, 40, 50, 60, 100],
                             labels=['<21', '21-30', '31-40', '41-50', '51-60', '60+'],
                             include_lowest=True)
    
    return df

def create_plots(df):
    """
    Create comprehensive visualization of AlgorithmAmt vs ISFCal discrepancies
    """
    # Set the style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. CASENO vs Difference (as requested)
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(df['CASENO'], df['Difference'], alpha=0.5, s=10)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Zero Difference')
    ax1.set_xlabel('CASENO')
    ax1.set_ylabel('AlgorithmAmt - ISFCal ($)')
    ax1.set_title('Difference between AlgorithmAmt and ISFCal by Case Number')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter plot: AlgorithmAmt vs ISFCal with perfect agreement line
    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(df['AlgorithmAmt'], df['ISFCal'], alpha=0.5, s=10)
    
    # Add perfect agreement line
    min_val = min(df['AlgorithmAmt'].min(), df['ISFCal'].min())
    max_val = max(df['AlgorithmAmt'].max(), df['ISFCal'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Agreement')
    
    # Add regression line
    z = np.polyfit(df['AlgorithmAmt'], df['ISFCal'], 1)
    p = np.poly1d(z)
    ax2.plot(df['AlgorithmAmt'].sort_values(), p(df['AlgorithmAmt'].sort_values()), 
             'g-', alpha=0.8, label=f'Regression (slope={z[0]:.3f})')
    
    ax2.set_xlabel('AlgorithmAmt ($)')
    ax2.set_ylabel('ISFCal ($)')
    ax2.set_title('AlgorithmAmt vs ISFCal Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Bland-Altman Plot (recommended for method comparison)
    ax3 = plt.subplot(3, 3, 3)
    mean_values = (df['AlgorithmAmt'] + df['ISFCal']) / 2
    diff_values = df['AlgorithmAmt'] - df['ISFCal']
    
    ax3.scatter(mean_values, diff_values, alpha=0.5, s=10)
    
    # Calculate limits of agreement
    mean_diff = diff_values.mean()
    std_diff = diff_values.std()
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff
    
    ax3.axhline(y=mean_diff, color='red', linestyle='-', label=f'Mean Diff: ${mean_diff:,.0f}')
    ax3.axhline(y=upper_limit, color='red', linestyle='--', label=f'Upper LoA: ${upper_limit:,.0f}')
    ax3.axhline(y=lower_limit, color='red', linestyle='--', label=f'Lower LoA: ${lower_limit:,.0f}')
    ax3.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    
    ax3.set_xlabel('Average of AlgorithmAmt and ISFCal ($)')
    ax3.set_ylabel('AlgorithmAmt - ISFCal ($)')
    ax3.set_title('Bland-Altman Plot\n(Agreement between Methods)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribution of differences
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(df['Difference'], bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax4.axvline(x=df['Difference'].median(), color='green', linestyle='--', 
                linewidth=2, label=f'Median: ${df["Difference"].median():,.0f}')
    ax4.set_xlabel('Difference ($)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Differences\n(AlgorithmAmt - ISFCal)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Percentage difference distribution
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(df['Percent_Difference'], bins=50, edgecolor='black', alpha=0.7)
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax5.axvline(x=df['Percent_Difference'].median(), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {df["Percent_Difference"].median():.1f}%')
    ax5.set_xlabel('Percentage Difference (%)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Distribution of Percentage Differences')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Box plot by Living Setting
    ax6 = plt.subplot(3, 3, 6)
    living_settings = df.groupby('LivingSetting')['Difference'].apply(list)
    ax6.boxplot(living_settings.values, labels=[ls[:20] + '...' if len(ls) > 20 else ls 
                                                  for ls in living_settings.index])
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
    ax6.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax6.set_ylabel('Difference ($)')
    ax6.set_title('Difference by Living Setting')
    ax6.grid(True, alpha=0.3)
    
    # 7. Difference vs AlgorithmAmt (to see if error is proportional)
    ax7 = plt.subplot(3, 3, 7)
    ax7.scatter(df['AlgorithmAmt'], df['Abs_Difference'], alpha=0.5, s=10)
    
    # Add trend line
    z = np.polyfit(df['AlgorithmAmt'], df['Abs_Difference'], 1)
    p = np.poly1d(z)
    ax7.plot(df['AlgorithmAmt'].sort_values(), p(df['AlgorithmAmt'].sort_values()), 
             'r-', alpha=0.8, label=f'Trend (slope={z[0]:.5f})')
    
    ax7.set_xlabel('AlgorithmAmt ($)')
    ax7.set_ylabel('Absolute Difference ($)')
    ax7.set_title('Absolute Difference vs AlgorithmAmt\n(Error Proportionality Check)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Q-Q Plot to check normality of differences
    ax8 = plt.subplot(3, 3, 8)
    stats.probplot(df['Difference'], dist="norm", plot=ax8)
    ax8.set_title('Q-Q Plot of Differences\n(Normality Check)')
    ax8.grid(True, alpha=0.3)
    
    # 9. Cumulative distribution of absolute differences
    ax9 = plt.subplot(3, 3, 9)
    sorted_abs_diff = np.sort(df['Abs_Difference'])
    cumulative = np.arange(1, len(sorted_abs_diff) + 1) / len(sorted_abs_diff) * 100
    
    ax9.plot(sorted_abs_diff, cumulative, linewidth=2)
    ax9.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50th percentile')
    ax9.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90th percentile')
    ax9.axhline(y=95, color='darkred', linestyle='--', alpha=0.5, label='95th percentile')
    
    # Add vertical lines for specific thresholds
    for threshold in [1000, 5000, 10000, 20000]:
        pct = (df['Abs_Difference'] <= threshold).mean() * 100
        ax9.axvline(x=threshold, color='gray', linestyle=':', alpha=0.5)
        ax9.text(threshold, 5, f'${threshold/1000:.0f}k\n({pct:.1f}%)', 
                ha='center', fontsize=8)
    
    ax9.set_xlabel('Absolute Difference ($)')
    ax9.set_ylabel('Cumulative Percentage (%)')
    ax9.set_title('Cumulative Distribution of Absolute Differences')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.set_xlim(0, df['Abs_Difference'].quantile(0.99))
    
    plt.suptitle('ISFCal vs AlgorithmAmt Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig

def print_summary_statistics(df):
    """
    Print summary statistics of the comparison
    """
    print("=" * 80)
    print("SUMMARY STATISTICS: AlgorithmAmt vs ISFCal Comparison")
    print("=" * 80)
    
    print(f"\nNumber of cases analyzed: {len(df):,}")
    
    print("\n--- AlgorithmAmt Statistics ---")
    print(f"Mean: ${df['AlgorithmAmt'].mean():,.2f}")
    print(f"Median: ${df['AlgorithmAmt'].median():,.2f}")
    print(f"Std Dev: ${df['AlgorithmAmt'].std():,.2f}")
    print(f"Min: ${df['AlgorithmAmt'].min():,.2f}")
    print(f"Max: ${df['AlgorithmAmt'].max():,.2f}")
    
    print("\n--- ISFCal Statistics ---")
    print(f"Mean: ${df['ISFCal'].mean():,.2f}")
    print(f"Median: ${df['ISFCal'].median():,.2f}")
    print(f"Std Dev: ${df['ISFCal'].std():,.2f}")
    print(f"Min: ${df['ISFCal'].min():,.2f}")
    print(f"Max: ${df['ISFCal'].max():,.2f}")
    
    print("\n--- Difference Statistics (AlgorithmAmt - ISFCal) ---")
    print(f"Mean Difference: ${df['Difference'].mean():,.2f}")
    print(f"Median Difference: ${df['Difference'].median():,.2f}")
    print(f"Std Dev of Difference: ${df['Difference'].std():,.2f}")
    print(f"Min Difference: ${df['Difference'].min():,.2f}")
    print(f"Max Difference: ${df['Difference'].max():,.2f}")
    
    print("\n--- Absolute Difference Statistics ---")
    print(f"Mean Absolute Difference: ${df['Abs_Difference'].mean():,.2f}")
    print(f"Median Absolute Difference: ${df['Abs_Difference'].median():,.2f}")
    
    print("\n--- Percentage Difference Statistics ---")
    print(f"Mean % Difference: {df['Percent_Difference'].mean():.2f}%")
    print(f"Median % Difference: {df['Percent_Difference'].median():.2f}%")
    
    print("\n--- Agreement Metrics ---")
    within_5_pct = (df['Abs_Difference'] <= abs(df['AlgorithmAmt'] * 0.05)).mean() * 100
    within_10_pct = (df['Abs_Difference'] <= abs(df['AlgorithmAmt'] * 0.10)).mean() * 100
    within_1000 = (df['Abs_Difference'] <= 1000).mean() * 100
    within_5000 = (df['Abs_Difference'] <= 5000).mean() * 100
    within_10000 = (df['Abs_Difference'] <= 10000).mean() * 100
    
    print(f"Cases within 5% of AlgorithmAmt: {within_5_pct:.1f}%")
    print(f"Cases within 10% of AlgorithmAmt: {within_10_pct:.1f}%")
    print(f"Cases within $1,000: {within_1000:.1f}%")
    print(f"Cases within $5,000: {within_5000:.1f}%")
    print(f"Cases within $10,000: {within_10000:.1f}%")
    
    # Correlation
    correlation = df['AlgorithmAmt'].corr(df['ISFCal'])
    print(f"\nCorrelation coefficient: {correlation:.4f}")
    
    # Cases where ISFCal is higher/lower
    higher = (df['ISFCal'] > df['AlgorithmAmt']).mean() * 100
    lower = (df['ISFCal'] < df['AlgorithmAmt']).mean() * 100
    equal = (df['ISFCal'] == df['AlgorithmAmt']).mean() * 100
    
    print(f"\nISFCal > AlgorithmAmt: {higher:.1f}% of cases")
    print(f"ISFCal < AlgorithmAmt: {lower:.1f}% of cases")
    print(f"ISFCal = AlgorithmAmt: {equal:.1f}% of cases")
    
    print("\n--- Top 10 Largest Positive Differences (AlgorithmAmt > ISFCal) ---")
    top_positive = df.nlargest(10, 'Difference')[['CASENO', 'AlgorithmAmt', 'ISFCal', 'Difference', 'LivingSetting']]
    print(top_positive.to_string(index=False))
    
    print("\n--- Top 10 Largest Negative Differences (ISFCal > AlgorithmAmt) ---")
    top_negative = df.nsmallest(10, 'Difference')[['CASENO', 'AlgorithmAmt', 'ISFCal', 'Difference', 'LivingSetting']]
    print(top_negative.to_string(index=False))
    
    print("=" * 80)

def main():
    """
    Main function to run the analysis
    """
    try:
        # Create database connection
        print("Connecting to database...")
        engine = create_db_connection()
        
        # Load data
        print("Loading data...")
        df = load_data(engine)
        
        if df.empty:
            print("No data found with both AlgorithmAmt and ISFCal values.")
            return
        
        print(f"Loaded {len(df):,} records for analysis.")
        
        # Print summary statistics
        print_summary_statistics(df)
        
        # Create visualizations
        print("\nGenerating visualizations...")
        fig = create_plots(df)
        
        # Optional: Save the figure
        save_option = input("\nDo you want to save the plots? (y/n): ")
        if save_option.lower() == 'y':
            filename = input("Enter filename (without extension): ")
            fig.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
            print(f"Plots saved as {filename}.png")
        
        # Optional: Export data to CSV
        export_option = input("\nDo you want to export the data to CSV? (y/n): ")
        if export_option.lower() == 'y':
            filename = input("Enter filename (without extension): ")
            df.to_csv(f'{filename}.csv', index=False)
            print(f"Data exported to {filename}.csv")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nPlease check your database connection settings and ensure:")
        print("1. SQL Server is running")
        print("2. The server name and database name are correct")
        print("3. You have the necessary permissions")
        print("4. The ISFCal calculations have been completed")

if __name__ == "__main__":
    main()
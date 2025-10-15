"""
SHAP Values Implementation for Random Forest iBudget Model
===========================================================
This module demonstrates how to calculate and visualize SHAP values 
for the Random Forest model (Model 9) in the Florida APD iBudget system.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ============================================
# 1. BASIC SHAP CALCULATION
# ============================================

def calculate_shap_values(model, X_train, X_test):
    """
    Calculate SHAP values for a Random Forest model.
    
    Parameters:
    -----------
    model : RandomForestRegressor
        Trained Random Forest model
    X_train : pd.DataFrame
        Training features (needed for background sampling)
    X_test : pd.DataFrame
        Test features to explain
    
    Returns:
    --------
    shap_values : np.array
        SHAP values for each feature and instance
    explainer : shap.Explainer
        SHAP explainer object for further analysis
    """
    
    # Method 1: TreeExplainer (Exact for tree-based models)
    # This is fast and exact for Random Forests
    explainer = shap.TreeExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)
    
    return shap_values, explainer


def calculate_shap_values_sampling(model, X_train, X_test, n_samples=100):
    """
    Alternative: Calculate SHAP using sampling (works for any model).
    Slower but more general - useful if switching to neural networks later.
    """
    
    # Create explainer with sampled background
    background = shap.sample(X_train, n_samples)
    explainer = shap.KernelExplainer(model.predict, background)
    
    # Calculate SHAP values (slower than TreeExplainer)
    shap_values = explainer.shap_values(X_test)
    
    return shap_values, explainer


# ============================================
# 2. PRACTICAL IMPLEMENTATION FOR iBUDGET
# ============================================

class iBudgetSHAPExplainer:
    """
    SHAP explainer specifically designed for the iBudget Random Forest model.
    """
    
    def __init__(self, model, feature_names, training_data):
        """
        Initialize the explainer with the Random Forest model.
        
        Parameters:
        -----------
        model : RandomForestRegressor
            Trained Model 9 Random Forest
        feature_names : list
            Names of features (QSI items, demographics, etc.)
        training_data : pd.DataFrame
            Training data used to fit the model
        """
        self.model = model
        self.feature_names = feature_names
        self.training_data = training_data
        self.explainer = shap.TreeExplainer(model, training_data)
        
        # Calculate expected value (baseline)
        self.expected_value = self.explainer.expected_value
        
    def explain_individual(self, consumer_data):
        """
        Generate SHAP explanation for a single consumer.
        
        Parameters:
        -----------
        consumer_data : pd.Series or dict
            Single consumer's features
        
        Returns:
        --------
        explanation : dict
            Detailed explanation with feature contributions
        """
        
        # Convert to DataFrame if needed
        if isinstance(consumer_data, dict):
            consumer_df = pd.DataFrame([consumer_data])
        else:
            consumer_df = consumer_data.to_frame().T
            
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(consumer_df)[0]
        
        # Get prediction
        prediction = self.model.predict(consumer_df)[0]
        
        # Create explanation dictionary
        explanation = {
            'consumer_id': consumer_df.index[0] if hasattr(consumer_df.index, '__getitem__') else 'Unknown',
            'predicted_budget': prediction,
            'baseline_budget': self.expected_value,
            'difference_from_baseline': prediction - self.expected_value,
            'feature_contributions': {}
        }
        
        # Add individual feature contributions
        for i, feature in enumerate(self.feature_names):
            contribution = shap_values[i]
            explanation['feature_contributions'][feature] = {
                'value': consumer_df.iloc[0, i],
                'contribution': contribution,
                'contribution_pct': (contribution / prediction) * 100 if prediction != 0 else 0
            }
            
        # Sort features by absolute contribution
        explanation['ranked_features'] = sorted(
            explanation['feature_contributions'].items(),
            key=lambda x: abs(x[1]['contribution']),
            reverse=True
        )
        
        return explanation
    
    def explain_subgroup(self, consumers_data, group_name="Subgroup"):
        """
        Generate aggregated SHAP explanations for a subgroup of consumers.
        
        Parameters:
        -----------
        consumers_data : pd.DataFrame
            Multiple consumers' features
        group_name : str
            Name of the subgroup for reporting
        
        Returns:
        --------
        group_explanation : dict
            Aggregated explanation for the subgroup
        """
        
        # Calculate SHAP values for all consumers
        shap_values = self.explainer.shap_values(consumers_data)
        
        # Calculate mean absolute SHAP values (feature importance)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Get predictions
        predictions = self.model.predict(consumers_data)
        
        group_explanation = {
            'group_name': group_name,
            'n_consumers': len(consumers_data),
            'mean_predicted_budget': predictions.mean(),
            'std_predicted_budget': predictions.std(),
            'baseline_budget': self.expected_value,
            'feature_importance': {}
        }
        
        # Calculate feature importance
        for i, feature in enumerate(self.feature_names):
            group_explanation['feature_importance'][feature] = {
                'mean_abs_contribution': mean_abs_shap[i],
                'mean_contribution': shap_values[:, i].mean(),
                'std_contribution': shap_values[:, i].std(),
                'importance_rank': 0  # Will be filled next
            }
        
        # Rank features by importance
        ranked = sorted(
            group_explanation['feature_importance'].items(),
            key=lambda x: x[1]['mean_abs_contribution'],
            reverse=True
        )
        
        for rank, (feature, _) in enumerate(ranked, 1):
            group_explanation['feature_importance'][feature]['importance_rank'] = rank
            
        return group_explanation
    
    def generate_report(self, consumer_data):
        """
        Generate a human-readable report for a consumer's budget allocation.
        
        Parameters:
        -----------
        consumer_data : pd.Series or dict
            Single consumer's features
        
        Returns:
        --------
        report : str
            Formatted explanation report
        """
        
        explanation = self.explain_individual(consumer_data)
        
        report = f"""
        iBUDGET ALLOCATION EXPLANATION REPORT
        =====================================
        
        Consumer ID: {explanation['consumer_id']}
        Predicted Budget: ${explanation['predicted_budget']:,.2f}
        Baseline Budget: ${explanation['baseline_budget']:,.2f}
        Difference from Baseline: ${explanation['difference_from_baseline']:+,.2f}
        
        TOP FACTORS INFLUENCING THIS ALLOCATION:
        -----------------------------------------
        """
        
        # Add top 10 features
        for i, (feature, details) in enumerate(explanation['ranked_features'][:10], 1):
            contrib = details['contribution']
            value = details['value']
            
            # Format based on contribution direction
            if contrib > 0:
                direction = "increases"
                symbol = "+"
            else:
                direction = "decreases"
                symbol = ""
            
            report += f"""
        {i}. {feature}: {value}
           → {direction} budget by ${symbol}{contrib:,.2f}
        """
        
        report += """
        
        INTERPRETATION:
        ---------------
        This allocation is personalized based on the individual's specific 
        combination of support needs. The factors above show how each 
        characteristic moves the budget from the population average 
        to this individual's specific allocation.
        
        Note: These contributions are calculated using SHAP (SHapley Additive 
        exPlanations), ensuring fair and consistent attribution of each factor's 
        impact on the final budget.
        """
        
        return report


# ============================================
# 3. VISUALIZATION FUNCTIONS
# ============================================

def plot_waterfall(shap_values, feature_names, feature_values, max_display=15):
    """
    Create a waterfall plot showing how each feature moves the prediction
    from the baseline to the final value.
    """
    
    # Create SHAP Explanation object
    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=shap_values.base_values if hasattr(shap_values, 'base_values') else 0,
        data=feature_values,
        feature_names=feature_names
    )
    
    # Create waterfall plot
    shap.waterfall_plot(shap_explanation, max_display=max_display)
    plt.title("iBudget Allocation Breakdown - Feature Contributions")
    plt.xlabel("Contribution to Budget ($)")
    

def plot_force(shap_values, feature_names, feature_values):
    """
    Create a force plot showing push/pull of features on the prediction.
    """
    
    shap.force_plot(
        base_value=shap_values.base_values if hasattr(shap_values, 'base_values') else 0,
        shap_values=shap_values,
        features=feature_values,
        feature_names=feature_names
    )


def plot_summary(shap_values, feature_names, feature_values):
    """
    Create a summary plot showing feature importance across all predictions.
    """
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, 
        feature_values,
        feature_names=feature_names,
        show=False
    )
    plt.title("Feature Impact on iBudget Allocations")
    plt.tight_layout()
    plt.show()


# ============================================
# 4. EXAMPLE USAGE WITH iBUDGET DATA
# ============================================

def example_ibudget_shap_analysis():
    """
    Complete example showing SHAP analysis for the iBudget Random Forest model.
    """
    
    # Simulate iBudget data (in practice, load actual data)
    np.random.seed(42)
    n_samples = 1000
    
    # Feature names matching the iBudget model
    feature_names = [
        'BSum', 'FSum', 'PSum',  # Summary scores
        'Living_RH1', 'Living_RH2', 'Living_RH3', 'Living_RH4', 'Living_ILSL',  # Living settings
        'Age21_30', 'Age31Plus',  # Age groups
        'Q19', 'Q21', 'Q26', 'Q27', 'Q30', 'Q36', 'Q44',  # Key QSI items
        'County_Urban', 'County_Rural'  # Simplified county
    ]
    
    # Generate synthetic data
    X = pd.DataFrame(
        np.random.randn(n_samples, len(feature_names)),
        columns=feature_names
    )
    
    # Make some features binary (living settings, age groups)
    for col in ['Living_RH1', 'Living_RH2', 'Living_RH3', 'Living_RH4', 
                'Living_ILSL', 'Age21_30', 'Age31Plus', 'County_Urban', 'County_Rural']:
        X[col] = (X[col] > 0).astype(int)
    
    # Make QSI items 0-3 scale
    for col in ['Q19', 'Q21', 'Q26', 'Q27', 'Q30', 'Q36', 'Q44']:
        X[col] = np.random.randint(0, 4, n_samples)
    
    # Make summary scores positive
    for col in ['BSum', 'FSum', 'PSum']:
        X[col] = np.abs(X[col] * 10)
    
    # Create target variable (simulated budget)
    y = (
        30000 +  # Base
        X['BSum'] * 1000 +
        X['FSum'] * 800 +
        X['Living_RH4'] * 25000 +
        X['Living_RH3'] * 20000 +
        X['Age31Plus'] * 5000 +
        np.random.normal(0, 5000, n_samples)  # Noise
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest (Model 9 configuration)
    model = RandomForestRegressor(
        n_estimators=500,  # ModelNineNTrees
        max_depth=20,      # ModelNineMaxDepth
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Initialize SHAP explainer
    explainer = iBudgetSHAPExplainer(model, feature_names, X_train)
    
    # Example 1: Explain a single consumer
    print("=" * 60)
    print("EXAMPLE 1: Individual Consumer Explanation")
    print("=" * 60)
    
    sample_consumer = X_test.iloc[0]
    report = explainer.generate_report(sample_consumer)
    print(report)
    
    # Example 2: Explain a subgroup
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Subgroup Analysis (High Support Needs)")
    print("=" * 60)
    
    high_support = X_test[X_test['BSum'] > X_test['BSum'].quantile(0.75)]
    group_explanation = explainer.explain_subgroup(high_support, "High Behavioral Support")
    
    print(f"\nSubgroup: {group_explanation['group_name']}")
    print(f"Number of Consumers: {group_explanation['n_consumers']}")
    print(f"Mean Budget: ${group_explanation['mean_predicted_budget']:,.2f}")
    print(f"\nTop 5 Important Features:")
    
    top_features = sorted(
        group_explanation['feature_importance'].items(),
        key=lambda x: x[1]['importance_rank']
    )[:5]
    
    for feature, details in top_features:
        print(f"  {details['importance_rank']}. {feature}: "
              f"Mean contribution = ${details['mean_contribution']:,.2f}")
    
    # Example 3: Visualizations
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Generating Visualizations")
    print("=" * 60)
    
    # Calculate SHAP values for test set
    shap_values, tree_explainer = calculate_shap_values(model, X_train, X_test)
    
    # Create summary plot
    print("\nGenerating summary plot...")
    plot_summary(shap_values, feature_names, X_test)
    
    return explainer, model, X_test


# ============================================
# 5. PRODUCTION INTEGRATION
# ============================================

class ProductionSHAPService:
    """
    Production-ready SHAP service for real-time iBudget explanations.
    """
    
    def __init__(self, model_path, training_data_path):
        """
        Initialize service with pre-trained model and training data.
        """
        import joblib
        
        # Load model and training data
        self.model = joblib.load(model_path)
        self.training_data = pd.read_csv(training_data_path)
        self.feature_names = self.training_data.columns.tolist()
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model, self.training_data)
        
        # Cache for performance
        self.explanation_cache = {}
        
    def explain_allocation(self, consumer_id, consumer_features, use_cache=True):
        """
        Generate real-time explanation for a consumer's budget allocation.
        
        Parameters:
        -----------
        consumer_id : str
            Unique identifier for the consumer
        consumer_features : dict
            Feature values for the consumer
        use_cache : bool
            Whether to use cached explanations
            
        Returns:
        --------
        explanation : dict
            Detailed explanation with API-friendly format
        """
        
        # Check cache
        if use_cache and consumer_id in self.explanation_cache:
            return self.explanation_cache[consumer_id]
        
        # Prepare features
        feature_df = pd.DataFrame([consumer_features])
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(feature_df)[0]
        prediction = self.model.predict(feature_df)[0]
        
        # Build API response
        explanation = {
            'consumer_id': consumer_id,
            'allocation': float(prediction),
            'baseline_allocation': float(self.explainer.expected_value),
            'timestamp': pd.Timestamp.now().isoformat(),
            'factors': []
        }
        
        # Add top contributing factors
        contributions = sorted(
            zip(self.feature_names, shap_values),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        for feature, contribution in contributions[:10]:
            explanation['factors'].append({
                'feature': feature,
                'value': consumer_features.get(feature, 0),
                'contribution': float(contribution),
                'percentage': float((contribution / prediction) * 100) if prediction != 0 else 0
            })
        
        # Cache if enabled
        if use_cache:
            self.explanation_cache[consumer_id] = explanation
            
        return explanation
    
    def batch_explain(self, consumer_batch):
        """
        Generate explanations for multiple consumers efficiently.
        """
        explanations = []
        
        for consumer_id, features in consumer_batch.items():
            explanation = self.explain_allocation(consumer_id, features)
            explanations.append(explanation)
            
        return explanations


# Run example if script is executed directly
if __name__ == "__main__":
    explainer, model, test_data = example_ibudget_shap_analysis()
    print("\nSHAP analysis complete!")

#!/usr/bin/env python3
"""
Model 5b Implementation for Florida APD iBudget Algorithm

This module implements the final Model 5b from the UpdateStatisticalModelsiBudget document.
The model uses square-root transformation and multiple linear regression to predict 
individual budget allocations based on QSI assessment data.

Model 5b uses the following coefficients (from Table 4):
- Intercept: 27.5720
- Living Settings: ILSL (35.8220), RH1 (90.6294), RH2 (131.7576), RH3 (209.4558), RH4 (267.0995)
- Age Groups: Age21-30 (47.8473), Age31+ (48.9634)
- Behavioral/Functional Sums: BSum (0.4954), FHFSum (0.6349), SLFSum (2.0529), SLBSum (1.4501)
- QSI Questions: Q16 (2.4984), Q18 (5.8537), Q20 (2.6772), Q21 (2.7878), Q23 (6.3555), 
                 Q28 (2.2803), Q33 (1.2233), Q34 (2.1764), Q36 (2.6734), Q43 (1.9304)

Reference levels (coefficients = 0):
- Living Setting: Family Home (FH)
- Age: Under 21
"""

import json
import math
import sys
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Model5bCoefficients:
    """Model 5b regression coefficients from the final algorithm."""
    
    # Intercept
    intercept: float = 27.5720
    
    # Living Setting coefficients (FH is reference level with 0)
    live_ilsl: float = 35.8220  # Independent Living & Supported Living
    live_rh1: float = 90.6294   # Residential Habilitation, Standard and Live In
    live_rh2: float = 131.7576  # Residential Habilitation, Behavior Focus
    live_rh3: float = 209.4558  # Residential Habilitation, Intensive Behavior
    live_rh4: float = 267.0995  # Residential Habilitation, CTEP and Special Medical Home Care
    
    # Age Group coefficients (Under 21 is reference level with 0)
    age_21_30: float = 47.8473  # Age 21-30
    age_31_plus: float = 48.9634  # Age 31+
    
    # Sum and interaction coefficients
    bsum: float = 0.4954        # Behavioral status sum score
    fhfsum: float = 0.6349      # Family Home by Functional status interaction
    slfsum: float = 2.0529      # ILSL by Functional status interaction
    slbsum: float = 1.4501      # ILSL by Behavioral status interaction
    
    # QSI Question coefficients
    q16: float = 2.4984         # Eating
    q18: float = 5.8537         # Transfers
    q20: float = 2.6772         # Hygiene
    q21: float = 2.7878         # Dressing
    q23: float = 6.3555         # Self-protection
    q28: float = 2.2803         # Inappropriate Sexual Behavior
    q33: float = 1.2233         # Injury to Person Caused by Aggression
    q34: float = 2.1764         # Use of Mechanical Restraints
    q36: float = 2.6734         # Use of Psychotropic Medications
    q43: float = 1.9304         # Treatment (Physician Prescribed)


class TeeOutput:
    """
    Helper class to write output to both console and file simultaneously.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


class Model5b:
    """
    Implementation of Model 5b for Florida APD iBudget Algorithm.
    
    This class implements the final regression model with square-root transformation
    that achieved R-squared = 0.7998 after removing 9.40% outliers.
    """
    
    def __init__(self):
        self.coefficients = Model5bCoefficients()
        self.model_info = {
            "name": "Model 5b",
            "r_squared": 0.7998,
            "outliers_removed": 0.094,
            "residual_standard_error": 30.82,
            "degrees_of_freedom": 23193,
            "f_statistic": 4412,
            "p_value": "< 2.2e-16"
        }
    
    def validate_input(self, qsi_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize QSI input data.
        
        Args:
            qsi_data: Dictionary containing QSI assessment data
            
        Returns:
            Validated and normalized data dictionary
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        required_fields = ['living_setting', 'age', 'bsum', 'fsum', 'psum']
        qsi_questions = ['Q16', 'Q18', 'Q20', 'Q21', 'Q23', 'Q28', 'Q33', 'Q34', 'Q36', 'Q43']
        
        # Check required fields
        for field in required_fields:
            if field not in qsi_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Check QSI questions
        for q in qsi_questions:
            if q not in qsi_data:
                raise ValueError(f"Missing required QSI question: {q}")
        
        # Validate living setting
        valid_living_settings = ['FH', 'ILSL', 'RH1', 'RH2', 'RH3', 'RH4']
        if qsi_data['living_setting'] not in valid_living_settings:
            raise ValueError(f"Invalid living_setting. Must be one of: {valid_living_settings}")
        
        # Validate age
        if not isinstance(qsi_data['age'], (int, float)) or qsi_data['age'] < 0:
            raise ValueError("Age must be a non-negative number")
        
        # Validate QSI scores (0-4 scale)
        for q in qsi_questions:
            score = qsi_data[q]
            if not isinstance(score, (int, float)) or score < 0 or score > 4:
                raise ValueError(f"{q} must be between 0 and 4, got: {score}")
        
        # Validate sum scores
        if not (0 <= qsi_data['bsum'] <= 24):  # 6 questions by 4 max score
            raise ValueError("BSum must be between 0 and 24")
        if not (0 <= qsi_data['fsum'] <= 44):  # 11 questions by 4 max score
            raise ValueError("FSum must be between 0 and 44")
        if not (0 <= qsi_data['psum'] <= 76):  # 19 questions by 4 max score
            raise ValueError("PSum must be between 0 and 76")
        
        return qsi_data
    
    def calculate_interaction_terms(self, qsi_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate interaction terms between living setting and sum scores.
        
        Args:
            qsi_data: Validated QSI data
            
        Returns:
            Dictionary containing interaction term values
        """
        living_setting = qsi_data['living_setting']
        fsum = qsi_data['fsum']
        bsum = qsi_data['bsum']
        
        interactions = {
            'fhfsum': 0,  # Family Home by Functional Sum
            'slfsum': 0,  # ILSL by Functional Sum  
            'slbsum': 0   # ILSL by Behavioral Sum
        }
        
        if living_setting == 'FH':
            interactions['fhfsum'] = fsum
        elif living_setting == 'ILSL':
            interactions['slfsum'] = fsum
            interactions['slbsum'] = bsum
        
        return interactions
    
    def predict_square_root_scale(self, qsi_data: Dict[str, Any]) -> float:
        """
        Calculate prediction in square-root scale using Model 5b coefficients.
        
        Args:
            qsi_data: Validated QSI assessment data
            
        Returns:
            Predicted value in square-root scale
        """
        # Start with intercept
        prediction = self.coefficients.intercept
        
        # Add living setting effects (FH is reference level)
        living_setting = qsi_data['living_setting']
        if living_setting == 'ILSL':
            prediction += self.coefficients.live_ilsl
        elif living_setting == 'RH1':
            prediction += self.coefficients.live_rh1
        elif living_setting == 'RH2':
            prediction += self.coefficients.live_rh2
        elif living_setting == 'RH3':
            prediction += self.coefficients.live_rh3
        elif living_setting == 'RH4':
            prediction += self.coefficients.live_rh4
        # FH has coefficient 0 (reference level)
        
        # Add age effects (Under 21 is reference level)
        age = qsi_data['age']
        if 21 <= age <= 30:
            prediction += self.coefficients.age_21_30
        elif age >= 31:
            prediction += self.coefficients.age_31_plus
        # Under 21 has coefficient 0 (reference level)
        
        # Add behavioral sum effect
        prediction += self.coefficients.bsum * qsi_data['bsum']
        
        # Add interaction terms
        interactions = self.calculate_interaction_terms(qsi_data)
        prediction += self.coefficients.fhfsum * interactions['fhfsum']
        prediction += self.coefficients.slfsum * interactions['slfsum']
        prediction += self.coefficients.slbsum * interactions['slbsum']
        
        # Add QSI question effects
        prediction += self.coefficients.q16 * qsi_data['Q16']
        prediction += self.coefficients.q18 * qsi_data['Q18']
        prediction += self.coefficients.q20 * qsi_data['Q20']
        prediction += self.coefficients.q21 * qsi_data['Q21']
        prediction += self.coefficients.q23 * qsi_data['Q23']
        prediction += self.coefficients.q28 * qsi_data['Q28']
        prediction += self.coefficients.q33 * qsi_data['Q33']
        prediction += self.coefficients.q34 * qsi_data['Q34']
        prediction += self.coefficients.q36 * qsi_data['Q36']
        prediction += self.coefficients.q43 * qsi_data['Q43']
        
        return prediction
    
    def predict_budget(self, qsi_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict individual budget allocation using Model 5b.
        
        Args:
            qsi_data: QSI assessment data
            
        Returns:
            Dictionary containing prediction results
        """
        # Validate input
        validated_data = self.validate_input(qsi_data)
        
        # Calculate prediction in square-root scale
        sqrt_prediction = self.predict_square_root_scale(validated_data)
        
        # Transform back to dollar scale by squaring
        budget_prediction = sqrt_prediction ** 2
        
        # Calculate interaction terms for transparency
        interactions = self.calculate_interaction_terms(validated_data)
        
        return {
            'predicted_budget': round(budget_prediction, 2),
            'sqrt_scale_prediction': round(sqrt_prediction, 4),
            'model_info': self.model_info,
            'input_data': validated_data,
            'interaction_terms': interactions,
            'coefficients_used': {
                'living_setting': validated_data['living_setting'],
                'age_group': self._get_age_group(validated_data['age']),
                'qsi_scores': {q: validated_data[q] for q in ['Q16', 'Q18', 'Q20', 'Q21', 'Q23', 'Q28', 'Q33', 'Q34', 'Q36', 'Q43']}
            }
        }
    
    def _get_age_group(self, age: float) -> str:
        """Helper function to determine age group."""
        if age < 21:
            return "Under 21 (reference)"
        elif 21 <= age <= 30:
            return "21-30"
        else:
            return "31+"
    
    def predict_batch(self, qsi_data_list: list) -> list:
        """
        Predict budgets for multiple individuals.
        
        Args:
            qsi_data_list: List of QSI assessment data dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for i, qsi_data in enumerate(qsi_data_list):
            try:
                result = self.predict_budget(qsi_data)
                result['record_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'record_index': i,
                    'error': str(e),
                    'input_data': qsi_data
                })
        return results


def main():
    """
    Main function to test Model 5b implementation using QSI-unit-test1.json
    Output is written to both console and model5b_output.txt
    """
    # Set up dual output to console and file
    output_filename = 'model5b_output.txt'
    tee = TeeOutput(output_filename)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        # Add timestamp to output
        print("Florida APD iBudget Algorithm - Model 5b Implementation")
        print("=" * 60)
        print(f"Execution Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output File: {output_filename}")
        print("=" * 60)
        
        # Initialize the model
        model = Model5b()
        
        try:
            # Load test data
            with open('QSI-unit-test1.json', 'r') as f:
                test_data = json.load(f)
            
            print(f"\nLoaded {len(test_data['test_cases'])} test cases from QSI-unit-test1.json")
            print(f"Test data description: {test_data['description']}")
            
            # Run predictions
            results = model.predict_batch(test_data['test_cases'])
            
            # Display results
            print(f"\nModel 5b Prediction Results:")
            print("-" * 40)
            
            for result in results:
                if 'error' in result:
                    print(f"Record {result['record_index']}: ERROR - {result['error']}")
                else:
                    data = result['input_data']
                    print(f"\nRecord {result['record_index']}:")
                    print(f"  Individual: {data.get('individual_id', 'N/A')}")
                    print(f"  Living Setting: {data['living_setting']}")
                    print(f"  Age: {data['age']} ({result['coefficients_used']['age_group']})")
                    print(f"  Predicted Budget: ${result['predicted_budget']:,.2f}")
                    print(f"  Square-root Scale: {result['sqrt_scale_prediction']}")
            
            # Summary statistics
            successful_predictions = [r for r in results if 'error' not in r]
            if successful_predictions:
                budgets = [r['predicted_budget'] for r in successful_predictions]
                print(f"\nSummary Statistics:")
                print(f"  Successful predictions: {len(successful_predictions)}")
                print(f"  Average predicted budget: ${sum(budgets)/len(budgets):,.2f}")
                print(f"  Minimum predicted budget: ${min(budgets):,.2f}")
                print(f"  Maximum predicted budget: ${max(budgets):,.2f}")
            
            print(f"\nModel Information:")
            print(f"  R-squared: {model.model_info['r_squared']}")
            print(f"  Outliers removed: {model.model_info['outliers_removed']*100:.1f}%")
            print(f"  Residual standard error: {model.model_info['residual_standard_error']}")
            
            print(f"\n" + "=" * 60)
            print(f"Execution completed successfully.")
            print(f"Results saved to: {output_filename}")
            
        except FileNotFoundError:
            print("\nError: QSI-unit-test1.json not found.")
            print("Please ensure the test data file is in the same directory.")
        except json.JSONDecodeError as e:
            print(f"\nError reading JSON file: {e}")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
    
    finally:
        # Restore original stdout and close file
        sys.stdout = original_stdout
        tee.close()
        print(f"\nOutput has been written to both console and {output_filename}")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from scipy.stats import shapiro
import scipy.stats as stats

def AB_Test(df, group, target):
    '''
    Perform an A/B test given the following:
    1. dataframe
    2. group: a column in the dataframe containing the A/B groups
    3. target: a column in the dataframe containing the value
    '''
    # Split A/B
    group_A = df[df[group] == "A"][target]
    group_B = df[df[group] == "B"][target]
    
    # Test assumption: Normality
    norm_test_A = shapiro(group_A)[1] < 0.05
    norm_test_B = shapiro(group_B)[1] < 0.05
    # H0: Distribution is Normal - False
    # H1: Distribution is not Normal - True
    
    if (norm_test_A == False) & (norm_test_B == False): # "H0: Normal Distribution"
        # Parametric Test
        test_type = "Parametric"
        # Test assumption: Homogeneity of variances
        leveneTest = stats.levene(group_A, group_B)[1] < 0.05
        # H0: Homogeneous: False
        # H1: Heterogeneous: True
        
        if leveneTest == False:
            # Homogeneous
            homogeneous = "Yes"
            ttest = stats.ttest_ind(group_A, group_B, equal_var=True)[1]
            # H0: M1 == M2 - False
            # H1: M1 != M2 - True
        else:
            # Heterogeneous
            homogeneous = "No"
            ttest = stats.ttest_ind(group_A, group_B, equal_var=False)[1]
            # H0: M1 == M2 - False
            # H1: M1 != M2 - True
    else:
        # Non-Parametric Test
        test_type = "Non-Parametric"
        ttest = stats.mannwhitneyu(group_A, group_B)[1]
        # H0: M1 == M2 - False
        # H1: M1 != M2 - True
    
    if ttest < 0.05:
        result = "Reject H0"
        comment = "A/B groups are not similar"
    else: 
        result = "Fail to Reject H0"
        comment = "A/B groups are similar"
        
    # Result
    temp = pd.DataFrame({
        "Test Type": [test_type],
        "AB Hypothesis":[result], 
        "p-value":[ttest],
        "Comment": [comment]
    })
    
    # Columns
    if (norm_test_A == False) & (norm_test_B == False):
        temp["Homogeneous"] = [homogeneous]
        temp = temp[["Test Type", "Homogeneous","AB Hypothesis", "p-value", "Comment"]]
    else:
        temp = temp[["Test Type", "AB Hypothesis", "p-value", "Comment"]]
    
    # Print Hypothesis
    print("A/B Test Hypothesis")
    print("H0: A == B")
    print("H1: A != B")
    
    return temp 
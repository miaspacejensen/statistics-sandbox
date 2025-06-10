import numpy as np
import pandas as pd
from scipy.stats import shapiro, stats, f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

def Multivariate_Test(df, group, target):
    '''
    Perform a statistical test comparing multiple groups given the following:
    1. dataframe
    2. group: a column in the dataframe containing the group identifiers
    3. target: a column in the dataframe containing the values
    '''
    # Extract unique groups
    unique_groups = df[group].unique()

    # Split into groups based on unique group identifiers and store in a list
    groups_data = [df[df[group] == grp][target] for grp in unique_groups]
    
    # Test assumption: Normality for each group
    norm_tests = [shapiro(data)[1] < 0.05 for data in groups_data]
    # H0: Distribution is Normal - False
    # H1: Distribution is not Normal - True
        
    if all(norm == False for norm in norm_tests): # "H0: Normal Distribution"
        # Parametric Test: ANOVA
        test_type = "Parametric"
        # Test assumption: Homogeneity of variances
        leveneTest = levene(*groups_data)[1] < 0.05
        # H0: Homogeneous: False
        # H1: Heterogeneous: True
        
        if leveneTest == False:
            # Homogeneous
            homogeneous = "Yes"
            f_stat, p_value = f_oneway(*groups_data) # ANOVA Test
            # H0: Groups have equal means - False
            # H1: At least one group mean is different - True
        else:
            # Heterogeneous
            homogeneous = "No"
            # ANOVA is used despite heterogeneity; could consider Welch's ANOVA (not implemented here).
            f_stat, p_value = f_oneway(*groups_data) # ANOVA Test
        print(f"ANOVA Test Statistic: {f_stat}")

    else:
        # Non-Parametric Test: Kruskal-Wallis
        test_type = "Non-Parametric"
        h_stat, p_value = kruskal(*groups_data) # Kruskal-Wallis Test
        # H0: Groups have equal distributions - False
        # H1: At least one group distribution is different - True
        print(f"Kruskal-Wallis Test Statistic: {h_stat}")

    print(f"P-value: {p_value}\n")
    
    if p_value < 0.05:
        result = "Reject H0"
        comment = "At least one group differs"
        # Tukey's HSD test to determine which group/s are different
        tukey_result = pairwise_tukeyhsd(endog=df[target], groups=df[group], alpha=0.05)
        df_tukey = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])
    else: 
        result = "Fail to Reject H0"
        comment = "Groups are similar"
        df_tukey = pd.DataFrame()
        
    # Result
    df_summary = pd.DataFrame({
        "Test Type": [test_type],
        "Hypothesis":[result], 
        "p-value":[p_value],
        "Comment": [comment]
    })
    
    # Columns
    if test_type == "Parametric":
        df_summary["Homogeneous"] = [homogeneous]
        df_summary = df_summary[["Test Type", "Homogeneous","Hypothesis", "p-value", "Comment"]]
    else:
        df_summary = df_summary[["Test Type", "Hypothesis", "p-value", "Comment"]]
    
    # Print Hypothesis
    print("Multi-Group Test Hypothesis")
    print("H0: All groups are equal")
    print("H1: At least one group is different")

    return df_summary, df_tukey
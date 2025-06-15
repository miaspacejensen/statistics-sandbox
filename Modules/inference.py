import numpy as np
from scipy.stats import norm, t, chi2

def z_test_known_variance(sample_data, population_mean, known_variance):
    '''
    Inference on the Mean (µ) When the Variance (σ^2) is Known.

    Use a z-test since the variance is known.

    Example usage:
    sample_data = [980, 1005, 995, 1010, 1000]
    population_mean = 1000
    known_variance = 100
    z_score, p_value = z_test_known_variance(sample_data, population_mean, known_variance)
    
    Assumptions: 
    - Normality
    '''
    sample_mean = np.mean(sample_data)
    n = len(sample_data)
    standard_error = np.sqrt(known_variance / n)
    z_score = (sample_mean - population_mean) / standard_error
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    return z_score, p_value

def t_test_unknown_variance(sample_data, population_mean):
    '''
    Inference on the Mean (µ) When the Variance (σ^2) is Unknown.

    Use a t-test since the variance is unknown.

    Example usage:
    sample_data = [980, 1005, 995, 1010, 1000]
    population_mean = 1000
    t_score, p_value = t_test_unknown_variance(sample_data, population_mean)
    
    Assumptions: 
    - Normality
    - Independence
    '''
    sample_mean = np.mean(sample_data)
    sample_std = np.std(sample_data, ddof=1)
    n = len(sample_data)
    standard_error = sample_std / np.sqrt(n)
    t_score = (sample_mean - population_mean) / standard_error
    p_value = 2 * (1 - t.cdf(abs(t_score), df=n-1))
    return t_score, p_value[0]

def chi_square_variance_test(sample_data, hypothesized_variance):
    '''
    Inference on the Variance (σ^2) or Standard Deviation (σ)

    Use a chi-square test to infer about variance.

    Example usage:
    sample_data = [4.2, 4.0, 3.8, 4.4, 4.1]
    hypothesized_variance = 0.25
    chi_square_stat, p_value = chi_square_variance_test(sample_data, hypothesized_variance)
    
    Assumptions: 
    - Normality
    '''
    n = len(sample_data)
    sample_variance = np.var(sample_data, ddof=1)
    chi_square_stat = (n - 1) * sample_variance / hypothesized_variance
    p_value = 1 - chi2.cdf(chi_square_stat, df=n-1)
    return chi_square_stat, p_value

def pooled_t_test(sample1, sample2):
    '''
    Inference on the Means (µ1 - µ2) of Two Samples When Their 
    Standard Deviations (σ1, σ2) are Unknown.
    The Common Variance Case (σ1 = σ2 = σ).

    Use a pooled t-test.

    Example usage:
    sample1 = [20, 21, 22, 19, 18]
    sample2 = [17, 18, 16, 15, 19]
    t_score, p_value = pooled_t_test(sample1, sample2)
    
    Assumptions: 
    - Normality
    - Homogeneity of variances
    '''
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
    
    pooled_variance = (((n1 - 1) * std1**2) + ((n2 - 1) * std2**2)) / (n1 + n2 - 2)
    standard_error = np.sqrt(pooled_variance * (1/n1 + 1/n2))
    t_score = (mean1 - mean2) / standard_error
    p_value = 2 * (1 - t.cdf(abs(t_score), df=n1 + n2 - 2))
    return t_score, p_value

def paired_t_test(sample1, sample2):
    '''
    Inference on the Means (µ1 - µ2) of Two Samples When Their 
    Standard Deviations (σ1, σ2) are Unknown.
    The Paired Observations Case.

    Use a paired t-test.

    Example usage:
    before = [200, 215, 210, 195, 198]
    after = [190, 205, 202, 190, 185]
    t_score, p_value = paired_t_test(before, after)
    
    Assumptions: 
    - Normality
    - Independence of pairs
    '''
    differences = np.array(sample1) - np.array(sample2)
    mean_difference = np.mean(differences)
    std_difference = np.std(differences, ddof=1)
    n = len(differences)
    standard_error = std_difference / np.sqrt(n)
    t_score = mean_difference / standard_error
    p_value = 2 * (1 - t.cdf(abs(t_score), df=n-1))
    return t_score, p_value

def f_test_variance_ratio(sample1, sample2):
    '''
    Inference on the Variance ((σ1^2)/(σ2^2)) or ((σ2^2)/(σ1^2)) or 
    Standard Deviations ((σ1/σ2) or (σ2/σ1)) of Two Samples

    Use an F-test for variances.

    Example usage:
    sample1 = [1.5, 1.7, 1.6, 1.5, 1.8]
    sample2 = [2.2, 2.1, 2.3, 2.5, 2.4]
    f_statistic, p_value = f_test_variance_ratio(sample1, sample2)
    
    Assumptions: 
    - Normality
    - Independence
    '''
    variance1 = np.var(sample1, ddof=1)
    variance2 = np.var(sample2, ddof=1)
    f_statistic = variance1 / variance2 if variance1 > variance2 else variance2 / variance1
    df1, df2 = len(sample1) - 1, len(sample2) - 1
    p_value = 1 - f.cdf(f_statistic, df1=df1, df2=df2)
    return f_statistic, p_value
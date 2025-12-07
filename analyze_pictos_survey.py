#!/usr/bin/env python3
"""
Comprehensive Analysis of Pictos Survey Data

This script analyzes survey data on danger symbols/pictograms including:
- Color associations
- Symbol recognition and remembrance rates
- Demographic patterns
- Cross-comparisons with statistical tests
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (
    ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact,
    shapiro, levene, f_oneway, kruskal, normaltest
)
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.weightstats import ttest_ind as ttest_ind_sm
from statsmodels.stats.multitest import multipletests
try:
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    ORDINAL_AVAILABLE = True
except ImportError:
    ORDINAL_AVAILABLE = False
    warnings.warn("statsmodels OrderedModel not available. Ordinal logistic regression will use fallback methods.")
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Bayesian analysis imports
try:
    import pymc as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    warnings.warn("PyMC and/or ArviZ not available. Bayesian analysis will be skipped.")

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Create results directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
BAYESIAN_DIR = RESULTS_DIR / "bayesian"
BAYESIAN_DIR.mkdir(exist_ok=True)
BAYESIAN_FIGURES_DIR = FIGURES_DIR / "bayesian"
BAYESIAN_FIGURES_DIR.mkdir(exist_ok=True)


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_codebook(codebook_path):
    """Load and parse the codebook CSV file."""
    codebook = pd.read_csv(codebook_path, sep=';')
    return codebook


def load_data(data_path):
    """Load the survey results data."""
    df = pd.read_csv(data_path)
    return df


def preprocess_data(df, codebook):
    """Preprocess the data: handle missing values, create derived variables."""
    df_clean = df.copy()
    
    # Replace -9 (Not answered) with NaN for numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].replace(-9, np.nan)
    
    # Convert age to numeric
    if 'DD01_01' in df_clean.columns:
        df_clean['DD01_01'] = pd.to_numeric(df_clean['DD01_01'], errors='coerce')
    
    # Create Europe vs Rest of World variable
    if 'DD04' in df_clean.columns:
        df_clean['is_europe'] = (df_clean['DD04'] == 1).astype(int)
        df_clean['region_group'] = df_clean['DD04'].map({
            1: 'Europe',
            2: 'America',
            4: 'Asia',
            6: 'Africa',
            7: 'Oceania'
        }).fillna('Other')
        df_clean['region_binary'] = df_clean['region_group'].map({
            'Europe': 'Europe',
            'America': 'Rest of World',
            'Asia': 'Rest of World',
            'Africa': 'Rest of World',
            'Oceania': 'Rest of World',
            'Other': 'Rest of World'
        })
    
    # Create international student status binary
    if 'DD03' in df_clean.columns:
        df_clean['is_international_student'] = (df_clean['DD03'] == 1).astype(int)
        df_clean['international_student_status'] = df_clean['DD03'].map({
            1: 'International Student',
            2: 'Non-International Student'
        }).fillna('Unknown')
    
    # Create gender labels
    if 'DD02' in df_clean.columns:
        df_clean['gender_label'] = df_clean['DD02'].map({
            1: 'Female',
            2: 'Male',
            3: 'Diverse'
        }).fillna('Unknown')
    
    # Create age groups
    if 'DD01_01' in df_clean.columns:
        df_clean['age_group'] = pd.cut(
            df_clean['DD01_01'],
            bins=[0, 22, 27, 100],
            labels=['18-22', '23-27', '28+']
        )
    
    # Create symbol recognition binary (PK variables: 1-2 = not recognized, 3-5 = recognized)
    pk_cols = [col for col in df_clean.columns if col.startswith('PK')]
    for col in pk_cols:
        if col in df_clean.columns:
            # Explicitly define: 3-5 = recognized, 1-2 = not recognized
            df_clean[f'{col}_recognized'] = ((df_clean[col] >= 3) & (df_clean[col] <= 5)).astype(int)
            df_clean[f'{col}_not_recognized'] = ((df_clean[col] >= 1) & (df_clean[col] <= 2)).astype(int)
    
    # Create remembrance accuracy (M_RichtigFalsch: 1 = correct)
    m_cols = [col for col in df_clean.columns if col.startswith('M') and col.endswith('_RichtigFalsch')]
    for col in m_cols:
        if col in df_clean.columns:
            df_clean[f'{col}_correct'] = (df_clean[col] == 1).astype(int)
    
    return df_clean


# ============================================================================
# GENERAL STATISTICS
# ============================================================================

def calculate_recognition_rates(df):
    """Calculate symbol recognition rates from PK variables."""
    pk_cols = [col for col in df.columns if col.startswith('PK') and not col.endswith('_recognized')]
    
    recognition_data = []
    
    # Symbol mapping based on codebook
    symbol_map = {
        'PK01': 'Noteworthy 1',
        'PK02': 'Noteworthy 2',
        'PK03': 'Camera 1',
        'PK04': 'Camera 2',
        'PK05': 'Payment 1',
        'PK06': 'Payment 2',
        'PK07': 'Location 1',
        'PK08': 'Location 2',
        'PK09': 'Data Collection 1',
        'PK10': 'Data Collection 2',
        'PK11': 'Contacts 1',
        'PK12': 'Contacts 2',
        'PK13': 'Contacts 3',
        'PK14': 'Rights 1',
        'PK15': 'Rights 2',
        'PK16': 'Rights 3',
        'PK17': 'Data Sharing 1',
        'PK18': 'Data Sharing 2',
        'PK19': 'Data Sharing 3'
    }
    
    for col in pk_cols:
        if col in df.columns:
            # Recognition: 3-5 = recognized, 1-2 = not recognized
            recognized = ((df[col] >= 3) & (df[col] <= 5)).sum()
            not_recognized = ((df[col] >= 1) & (df[col] <= 2)).sum()
            total = df[col].notna().sum()
            
            if total > 0:
                rate = recognized / total * 100
                # Wilson score confidence interval
                ci_lower, ci_upper = proportion_confint(recognized, total, alpha=0.05, method='wilson')
                
                recognition_data.append({
                    'Symbol': symbol_map.get(col, col),
                    'Variable': col,
                    'Recognized': recognized,
                    'Not_Recognized': not_recognized,
                    'Total': total,
                    'Recognition_Rate': rate,
                    'CI_Lower': ci_lower * 100,
                    'CI_Upper': ci_upper * 100
                })
    
    return pd.DataFrame(recognition_data)


def calculate_remembrance_rates(df):
    """Calculate symbol remembrance accuracy from M_RichtigFalsch variables."""
    m_cols = [col for col in df.columns if col.startswith('M') and col.endswith('_RichtigFalsch')]
    
    remembrance_data = []
    
    # Symbol mapping
    symbol_map = {
        'M1_RichtigFalsch': 'Noteworthy 1',
        'M2_RichtigFalsch': 'Noteworthy 2',
        'M3_RichtigFalsch': 'Camera 1',
        'M4_RichtigFalsch': 'Camera 2',
        'M5_RichtigFalsch': 'Payment 1',
        'M6_RichtigFalsch': 'Payment 2',
        'M7_RichtigFalsch': 'Location 1',
        'M8_RichtigFalsch': 'Location 2',
        'M9_RichtigFalsch': 'Data Collection 1',
        'M10_RichtigFalsch': 'Data Collection 2',
        'M11_RichtigFalsch': 'Contacts 1',
        'M12_RichtigFalsch': 'Contacts 2',
        'M13_RichtigFalsch': 'Contacts 3',
        'M14_RichtigFalsch': 'Rights 1',
        'M15_RichtigFalsch': 'Rights 2',
        'M16_RichtigFalsch': 'Rights 3',
        'M17_RichtigFalsch': 'Data Sharing 1',
        'M18_RichtigFalsch': 'Data Sharing 2',
        'M19_RichtigFalsch': 'Data Sharing 3'
    }
    
    for col in m_cols:
        if col in df.columns:
            correct = (df[col] == 1).sum()
            total = df[col].notna().sum()
            
            if total > 0:
                rate = correct / total * 100
                ci_lower, ci_upper = proportion_confint(correct, total, alpha=0.05, method='wilson')
                
                remembrance_data.append({
                    'Symbol': symbol_map.get(col, col),
                    'Variable': col,
                    'Correct': correct,
                    'Total': total,
                    'Remembrance_Rate': rate,
                    'CI_Lower': ci_lower * 100,
                    'CI_Upper': ci_upper * 100
                })
    
    return pd.DataFrame(remembrance_data)


def calculate_color_associations(df):
    """Calculate color association statistics."""
    color_map = {
        'CA01': 'Red',
        'CA02': 'Blue',
        'CA03': 'Yellow',
        'CA04': 'Purple',
        'CA05': 'Orange',
        'CA06': 'Green'
    }
    
    dimension_map = {
        '01': 'Dangerous/Safe\n(lower=more dangerous)',
        '02': 'Positive/Negative\n(lower=more positive)',
        '03': 'Alarming/Reassuring\n(lower=more alarming)',
        '04': 'Important/Inconsequential\n(lower=more important)',
        '05': 'Intense/Neutral\n(lower=more intense)'
    }
    
    color_data = []
    
    for color_code, color_name in color_map.items():
        for dim_code, dim_name in dimension_map.items():
            col = f'{color_code}_{dim_code}'
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    median_val = values.median()
                    std_val = values.std()
                    
                    # 95% CI for mean
                    sem = stats.sem(values)
                    ci_lower = mean_val - 1.96 * sem
                    ci_upper = mean_val + 1.96 * sem
                    
                    color_data.append({
                        'Color': color_name,
                        'Color_Code': color_code,
                        'Dimension': dim_name,
                        'Dimension_Code': dim_code,
                        'Mean': mean_val,
                        'Median': median_val,
                        'Std': std_val,
                        'N': len(values),
                        'CI_Lower': ci_lower,
                        'CI_Upper': ci_upper
                    })
    
    return pd.DataFrame(color_data)


def calculate_demographics(df):
    """Calculate demographic statistics."""
    demo_data = {}
    
    # Gender distribution
    if 'gender_label' in df.columns:
        demo_data['gender'] = df['gender_label'].value_counts().to_dict()
        demo_data['gender_pct'] = (df['gender_label'].value_counts(normalize=True) * 100).to_dict()
    
    # International student status
    if 'international_student_status' in df.columns:
        demo_data['international_student'] = df['international_student_status'].value_counts().to_dict()
        demo_data['international_student_pct'] = (df['international_student_status'].value_counts(normalize=True) * 100).to_dict()
    
    # Region
    if 'region_group' in df.columns:
        demo_data['region'] = df['region_group'].value_counts().to_dict()
        demo_data['region_pct'] = (df['region_group'].value_counts(normalize=True) * 100).to_dict()
    
    # Age statistics
    if 'DD01_01' in df.columns:
        age_values = df['DD01_01'].dropna()
        if len(age_values) > 0:
            demo_data['age'] = {
                'mean': age_values.mean(),
                'median': age_values.median(),
                'std': age_values.std(),
                'min': age_values.min(),
                'max': age_values.max(),
                'n': len(age_values)
            }
    
    return demo_data


# ============================================================================
# STATISTICAL TESTS AND EFFECT SIZES
# ============================================================================

# ============================================================================
# BAYESIAN ANALYSIS FUNCTIONS
# ============================================================================

def bayesian_ttest(group1, group2, var_name, prior_sd=1.0, draws=2000, chains=4):
    """
    Bayesian t-test for comparing two groups on a continuous variable.
    
    Parameters:
    -----------
    group1 : array-like
        Data for group 1
    group2 : array-like
        Data for group 2
    var_name : str
        Name of the variable being compared
    prior_sd : float
        Standard deviation for prior on mean difference (default: 1.0)
    draws : int
        Number of MCMC draws (default: 2000)
    chains : int
        Number of MCMC chains (default: 2)
    
    Returns:
    --------
    dict : Dictionary containing Bayesian analysis results
    """
    if not BAYESIAN_AVAILABLE:
        return None
    
    g1_clean = np.array(group1.dropna() if hasattr(group1, 'dropna') else group1)
    g2_clean = np.array(group2.dropna() if hasattr(group2, 'dropna') else group2)
    
    if len(g1_clean) < 2 or len(g2_clean) < 2:
        return None
    
    try:
        with pm.Model() as model:
            # Priors
            mu1 = pm.Normal('mu1', mu=g1_clean.mean(), sigma=prior_sd * 3)
            mu2 = pm.Normal('mu2', mu=g2_clean.mean(), sigma=prior_sd * 3)
            sigma = pm.HalfNormal('sigma', sigma=prior_sd * 2)
            
            # Likelihood
            group1_obs = pm.Normal('group1_obs', mu=mu1, sigma=sigma, observed=g1_clean)
            group2_obs = pm.Normal('group2_obs', mu=mu2, sigma=sigma, observed=g2_clean)
            
            # Derived quantities
            mean_diff = pm.Deterministic('mean_diff', mu1 - mu2)
            
            # Effect size (Cohen's d)
            pooled_std = sigma
            cohens_d_bayes = pm.Deterministic('cohens_d', mean_diff / pooled_std)
            
            # Sample from posterior
            trace = pm.sample(draws=draws, chains=chains, return_inferencedata=True, 
                            progressbar=False, random_seed=42)
        
        # Extract posterior samples
        posterior = trace.posterior
        
        # Calculate summary statistics
        mean_diff_samples = posterior['mean_diff'].values.flatten()
        cohens_d_samples = posterior['cohens_d'].values.flatten()
        mu1_samples = posterior['mu1'].values.flatten()
        mu2_samples = posterior['mu2'].values.flatten()
        
        # Calculate 95% HDI
        mean_diff_hdi = az.hdi(trace, var_names=['mean_diff'], hdi_prob=0.95)['mean_diff'].values
        cohens_d_hdi = az.hdi(trace, var_names=['cohens_d'], hdi_prob=0.95)['cohens_d'].values
        
        # Calculate probabilities
        prob_positive = (mean_diff_samples > 0).mean()
        prob_large_effect = (np.abs(cohens_d_samples) > 0.5).mean()
        prob_medium_effect = (np.abs(cohens_d_samples) > 0.2).mean()
        
        # Convergence diagnostics
        rhat = az.rhat(trace)
        ess = az.ess(trace)
        
        return {
            'Variable': var_name,
            'Method': 'Bayesian t-test',
            'Mean_Difference_Posterior_Mean': mean_diff_samples.mean(),
            'Mean_Difference_HDI_Lower': mean_diff_hdi[0],
            'Mean_Difference_HDI_Upper': mean_diff_hdi[1],
            'Cohens_D_Posterior_Mean': cohens_d_samples.mean(),
            'Cohens_D_HDI_Lower': cohens_d_hdi[0],
            'Cohens_D_HDI_Upper': cohens_d_hdi[1],
            'Prob_Effect_Positive': prob_positive,
            'Prob_Large_Effect': prob_large_effect,
            'Prob_Medium_Effect': prob_medium_effect,
            'Group1_Mean_Posterior': mu1_samples.mean(),
            'Group2_Mean_Posterior': mu2_samples.mean(),
            'R_hat_mean_diff': float(rhat['mean_diff'].values),
            'ESS_mean_diff': float(ess['mean_diff'].values),
            'Trace': trace,
            'Posterior_Samples': {
                'mean_diff': mean_diff_samples,
                'cohens_d': cohens_d_samples,
                'mu1': mu1_samples,
                'mu2': mu2_samples
            }
        }
    except Exception as e:
        warnings.warn(f"Bayesian t-test failed for {var_name}: {str(e)}")
        return None


def bayesian_ordinal_model(group1, group2, var_name, n_categories=5, draws=2000, chains=4):
    """
    Bayesian ordinal model (ordered probit) for comparing two groups on Likert-scale variables.
    
    Parameters:
    -----------
    group1 : array-like
        Data for group 1 (ordinal values, typically 1-5)
    group2 : array-like
        Data for group 2 (ordinal values, typically 1-5)
    var_name : str
        Name of the variable being compared
    n_categories : int
        Number of ordinal categories (default: 5)
    draws : int
        Number of MCMC draws (default: 2000)
    chains : int
        Number of MCMC chains (default: 2)
    
    Returns:
    --------
    dict : Dictionary containing Bayesian ordinal analysis results
    """
    if not BAYESIAN_AVAILABLE:
        return None
    
    g1_clean = np.array(group1.dropna() if hasattr(group1, 'dropna') else group1)
    g2_clean = np.array(group2.dropna() if hasattr(group2, 'dropna') else group2)
    
    if len(g1_clean) < 2 or len(g2_clean) < 2:
        return None
    
    # Ensure values are in valid range
    g1_clean = g1_clean[(g1_clean >= 1) & (g1_clean <= n_categories)]
    g2_clean = g2_clean[(g2_clean >= 1) & (g2_clean <= n_categories)]
    
    if len(g1_clean) < 2 or len(g2_clean) < 2:
        return None
    
    try:
        with pm.Model() as model:
            # Priors for group means on latent scale
            mu1 = pm.Normal('mu1', mu=0, sigma=2)
            mu2 = pm.Normal('mu2', mu=0, sigma=2)
            
            # Cutpoints for ordered probit (n_categories - 1 cutpoints)
            # First cutpoint fixed at 0 for identifiability
            cutpoints = pm.Normal('cutpoints', mu=np.linspace(-1, 1, n_categories-2), 
                                sigma=1, shape=n_categories-2)
            cutpoints_ordered = pm.math.concatenate([[0], pm.math.sort(cutpoints)])
            
            # Likelihood for group 1
            group1_obs = pm.OrderedProbit('group1_obs', eta=mu1, cutpoints=cutpoints_ordered, 
                                         observed=g1_clean - 1)  # PyMC uses 0-indexed
            
            # Likelihood for group 2
            group2_obs = pm.OrderedProbit('group2_obs', eta=mu2, cutpoints=cutpoints_ordered, 
                                         observed=g2_clean - 1)
            
            # Derived quantities
            mean_diff = pm.Deterministic('mean_diff', mu1 - mu2)
            
            # Sample from posterior
            trace = pm.sample(draws=draws, chains=chains, return_inferencedata=True,
                            progressbar=False, random_seed=42, target_accept=0.9)
        
        # Extract posterior samples
        posterior = trace.posterior
        
        # Calculate summary statistics
        mu1_samples = posterior['mu1'].values.flatten()
        mu2_samples = posterior['mu2'].values.flatten()
        mean_diff_samples = posterior['mean_diff'].values.flatten()
        
        # Calculate 95% HDI
        mu1_hdi = az.hdi(trace, var_names=['mu1'], hdi_prob=0.95)['mu1'].values
        mu2_hdi = az.hdi(trace, var_names=['mu2'], hdi_prob=0.95)['mu2'].values
        mean_diff_hdi = az.hdi(trace, var_names=['mean_diff'], hdi_prob=0.95)['mean_diff'].values
        
        # Calculate probabilities
        prob_positive = (mean_diff_samples > 0).mean()
        prob_large_diff = (np.abs(mean_diff_samples) > 1.0).mean()
        prob_medium_diff = (np.abs(mean_diff_samples) > 0.5).mean()
        
        # Convergence diagnostics
        rhat = az.rhat(trace)
        ess = az.ess(trace)
        
        return {
            'Variable': var_name,
            'Method': 'Bayesian Ordinal (Ordered Probit)',
            'Mu1_Posterior_Mean': mu1_samples.mean(),
            'Mu1_HDI_Lower': mu1_hdi[0],
            'Mu1_HDI_Upper': mu1_hdi[1],
            'Mu2_Posterior_Mean': mu2_samples.mean(),
            'Mu2_HDI_Lower': mu2_hdi[0],
            'Mu2_HDI_Upper': mu2_hdi[1],
            'Mean_Diff_Posterior_Mean': mean_diff_samples.mean(),
            'Mean_Diff_HDI_Lower': mean_diff_hdi[0],
            'Mean_Diff_HDI_Upper': mean_diff_hdi[1],
            'Prob_Effect_Positive': prob_positive,
            'Prob_Large_Diff': prob_large_diff,
            'Prob_Medium_Diff': prob_medium_diff,
            'R_hat_mean_diff': float(rhat['mean_diff'].values),
            'ESS_mean_diff': float(ess['mean_diff'].values),
            'Trace': trace,
            'Posterior_Samples': {
                'mu1': mu1_samples,
                'mu2': mu2_samples,
                'mean_diff': mean_diff_samples
            }
        }
    except Exception as e:
        warnings.warn(f"Bayesian ordinal model failed for {var_name}: {str(e)}")
        return None


def bayesian_proportion_test(group1_success, group1_total, group2_success, group2_total, 
                             var_name, draws=2000, chains=4):
    """
    Bayesian proportion test for comparing two groups on a binary variable.
    
    Parameters:
    -----------
    group1_success : int
        Number of successes in group 1
    group1_total : int
        Total number of observations in group 1
    group2_success : int
        Number of successes in group 2
    group2_total : int
        Total number of observations in group 2
    var_name : str
        Name of the variable being compared
    draws : int
        Number of MCMC draws (default: 2000)
    chains : int
        Number of MCMC chains (default: 2)
    
    Returns:
    --------
    dict : Dictionary containing Bayesian analysis results
    """
    if not BAYESIAN_AVAILABLE:
        return None
    
    if group1_total < 1 or group2_total < 1:
        return None
    
    try:
        with pm.Model() as model:
            # Priors: Beta(1, 1) = Uniform(0, 1)
            p1 = pm.Beta('p1', alpha=1, beta=1)
            p2 = pm.Beta('p2', alpha=1, beta=1)
            
            # Likelihood: Binomial
            group1_obs = pm.Binomial('group1_obs', n=group1_total, p=p1, observed=group1_success)
            group2_obs = pm.Binomial('group2_obs', n=group2_total, p=p2, observed=group2_success)
            
            # Derived quantities
            prop_diff = pm.Deterministic('prop_diff', p1 - p2)
            odds_ratio = pm.Deterministic('odds_ratio', (p1 / (1 - p1)) / (p2 / (1 - p2)))
            log_odds_ratio = pm.Deterministic('log_odds_ratio', pm.math.log(odds_ratio))
            
            # Sample from posterior
            trace = pm.sample(draws=draws, chains=chains, return_inferencedata=True,
                            progressbar=False, random_seed=42)
        
        # Extract posterior samples
        posterior = trace.posterior
        
        # Calculate summary statistics
        p1_samples = posterior['p1'].values.flatten()
        p2_samples = posterior['p2'].values.flatten()
        prop_diff_samples = posterior['prop_diff'].values.flatten()
        odds_ratio_samples = posterior['odds_ratio'].values.flatten()
        
        # Calculate 95% HDI
        p1_hdi = az.hdi(trace, var_names=['p1'], hdi_prob=0.95)['p1'].values
        p2_hdi = az.hdi(trace, var_names=['p2'], hdi_prob=0.95)['p2'].values
        prop_diff_hdi = az.hdi(trace, var_names=['prop_diff'], hdi_prob=0.95)['prop_diff'].values
        odds_ratio_hdi = az.hdi(trace, var_names=['odds_ratio'], hdi_prob=0.95)['odds_ratio'].values
        
        # Calculate probabilities
        prob_p1_greater = (p1_samples > p2_samples).mean()
        prob_odds_ratio_gt_1 = (odds_ratio_samples > 1).mean()
        
        # Convergence diagnostics
        rhat = az.rhat(trace)
        ess = az.ess(trace)
        
        return {
            'Variable': var_name,
            'Method': 'Bayesian Proportion Test',
            'P1_Posterior_Mean': p1_samples.mean(),
            'P1_HDI_Lower': p1_hdi[0],
            'P1_HDI_Upper': p1_hdi[1],
            'P2_Posterior_Mean': p2_samples.mean(),
            'P2_HDI_Lower': p2_hdi[0],
            'P2_HDI_Upper': p2_hdi[1],
            'Proportion_Diff_Posterior_Mean': prop_diff_samples.mean(),
            'Proportion_Diff_HDI_Lower': prop_diff_hdi[0],
            'Proportion_Diff_HDI_Upper': prop_diff_hdi[1],
            'Odds_Ratio_Posterior_Mean': odds_ratio_samples.mean(),
            'Odds_Ratio_HDI_Lower': odds_ratio_hdi[0],
            'Odds_Ratio_HDI_Upper': odds_ratio_hdi[1],
            'Prob_P1_Greater': prob_p1_greater,
            'Prob_Odds_Ratio_GT_1': prob_odds_ratio_gt_1,
            'R_hat_p1': float(rhat['p1'].values),
            'ESS_p1': float(ess['p1'].values),
            'Trace': trace,
            'Posterior_Samples': {
                'p1': p1_samples,
                'p2': p2_samples,
                'prop_diff': prop_diff_samples,
                'odds_ratio': odds_ratio_samples
            }
        }
    except Exception as e:
        warnings.warn(f"Bayesian proportion test failed for {var_name}: {str(e)}")
        return None


def bayesian_anova(groups_dict, var_name, draws=2000, chains=4):
    """
    Bayesian ANOVA for comparing multiple groups.
    
    Parameters:
    -----------
    groups_dict : dict
        Dictionary with group names as keys and data arrays as values
    var_name : str
        Name of the variable being compared
    draws : int
        Number of MCMC draws (default: 2000)
    chains : int
        Number of MCMC chains (default: 2)
    
    Returns:
    --------
    dict : Dictionary containing Bayesian analysis results
    """
    if not BAYESIAN_AVAILABLE:
        return None
    
    if len(groups_dict) < 2:
        return None
    
    # Clean data
    groups_clean = {}
    for name, data in groups_dict.items():
        clean_data = np.array(data.dropna() if hasattr(data, 'dropna') else data)
        if len(clean_data) >= 2:
            groups_clean[name] = clean_data
    
    if len(groups_clean) < 2:
        return None
    
    try:
        group_names = list(groups_clean.keys())
        group_data = list(groups_clean.values())
        
        with pm.Model() as model:
            # Hierarchical priors for group means
            mu_overall = pm.Normal('mu_overall', mu=np.mean([d.mean() for d in group_data]), 
                                  sigma=2.0)
            sigma_mu = pm.HalfNormal('sigma_mu', sigma=1.0)
            
            # Group means
            mu_groups = pm.Normal('mu_groups', mu=mu_overall, sigma=sigma_mu, 
                                 shape=len(group_names))
            
            # Common variance
            sigma = pm.HalfNormal('sigma', sigma=2.0)
            
            # Likelihood for each group
            for i, (name, data) in enumerate(groups_clean.items()):
                pm.Normal(f'obs_{i}', mu=mu_groups[i], sigma=sigma, observed=data)
            
            # Pairwise differences
            pairwise_diffs = {}
            for i in range(len(group_names)):
                for j in range(i + 1, len(group_names)):
                    diff_name = f'diff_{group_names[i]}_vs_{group_names[j]}'
                    pairwise_diffs[diff_name] = pm.Deterministic(
                        diff_name, mu_groups[i] - mu_groups[j]
                    )
            
            # Sample from posterior
            trace = pm.sample(draws=draws, chains=chains, return_inferencedata=True,
                            progressbar=False, random_seed=42)
        
        # Extract results
        posterior = trace.posterior
        
        # Calculate group means and HDIs
        results = {
            'Variable': var_name,
            'Method': 'Bayesian ANOVA',
            'Group_Means': {},
            'Pairwise_Differences': {},
            'Trace': trace
        }
        
        for i, name in enumerate(group_names):
            mu_samples = posterior['mu_groups'].values[:, :, i].flatten()
            mu_hdi = az.hdi(trace, var_names=[f'mu_groups'], hdi_prob=0.95)
            results['Group_Means'][name] = {
                'Posterior_Mean': mu_samples.mean(),
                'HDI_Lower': float(mu_hdi['mu_groups'].values[i, 0]),
                'HDI_Upper': float(mu_hdi['mu_groups'].values[i, 1])
            }
        
        # Pairwise differences
        for diff_name in pairwise_diffs.keys():
            diff_samples = posterior[diff_name].values.flatten()
            diff_hdi = az.hdi(trace, var_names=[diff_name], hdi_prob=0.95)[diff_name].values
            results['Pairwise_Differences'][diff_name] = {
                'Posterior_Mean': diff_samples.mean(),
                'HDI_Lower': diff_hdi[0],
                'HDI_Upper': diff_hdi[1],
                'Prob_Diff_Positive': (diff_samples > 0).mean()
            }
        
        return results
    except Exception as e:
        warnings.warn(f"Bayesian ANOVA failed for {var_name}: {str(e)}")
        return None


def bayesian_regression_models(df, dependent_var, independent_vars, draws=2000, chains=4):
    """
    Bayesian linear regression models.
    
    Parameters:
    -----------
    df : DataFrame
        Data containing all variables
    dependent_var : str
        Name of dependent variable
    independent_vars : list
        List of independent variable names
    draws : int
        Number of MCMC draws (default: 2000)
    chains : int
        Number of MCMC chains (default: 2)
    
    Returns:
    --------
    dict : Dictionary containing Bayesian regression results
    """
    if not BAYESIAN_AVAILABLE:
        return None
    
    # Prepare data
    data_clean = df[[dependent_var] + independent_vars].dropna()
    
    if len(data_clean) < 3:
        return None
    
    y = data_clean[dependent_var].values
    X = data_clean[independent_vars].values
    
    try:
        with pm.Model() as model:
            # Priors
            alpha = pm.Normal('alpha', mu=0, sigma=1)
            beta = pm.Normal('beta', mu=0, sigma=1, shape=len(independent_vars))
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Linear model
            mu = alpha + pm.math.dot(X, beta)
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
            
            # Sample from posterior
            trace = pm.sample(draws=draws, chains=chains, return_inferencedata=True,
                            progressbar=False, random_seed=42)
        
        # Extract results
        posterior = trace.posterior
        
        results = {
            'Dependent_Variable': dependent_var,
            'Independent_Variables': independent_vars,
            'Method': 'Bayesian Linear Regression',
            'Coefficients': {},
            'Trace': trace
        }
        
        # Alpha (intercept)
        alpha_samples = posterior['alpha'].values.flatten()
        alpha_hdi = az.hdi(trace, var_names=['alpha'], hdi_prob=0.95)['alpha'].values
        results['Coefficients']['Intercept'] = {
            'Posterior_Mean': alpha_samples.mean(),
            'HDI_Lower': alpha_hdi[0],
            'HDI_Upper': alpha_hdi[1],
            'Prob_Positive': (alpha_samples > 0).mean()
        }
        
        # Beta coefficients
        for i, var_name in enumerate(independent_vars):
            beta_samples = posterior['beta'].values[:, :, i].flatten()
            beta_hdi = az.hdi(trace, var_names=['beta'], hdi_prob=0.95)
            results['Coefficients'][var_name] = {
                'Posterior_Mean': beta_samples.mean(),
                'HDI_Lower': float(beta_hdi['beta'].values[i, 0]),
                'HDI_Upper': float(beta_hdi['beta'].values[i, 1]),
                'Prob_Positive': (beta_samples > 0).mean()
            }
        
        return results
    except Exception as e:
        warnings.warn(f"Bayesian regression failed for {dependent_var}: {str(e)}")
        return None


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0
    
    d = (group1.mean() - group2.mean()) / pooled_std
    return d


def cramers_v(contingency_table):
    """Calculate Cramér's V effect size for chi-square tests."""
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    
    if min_dim == 0 or n == 0:
        return 0
    
    v = np.sqrt(chi2 / (n * min_dim))
    return v


def test_normality(data):
    """Test for normality using Shapiro-Wilk (for n<50) or D'Agostino's test."""
    data_clean = data.dropna()
    if len(data_clean) < 3:
        return None, None
    
    if len(data_clean) < 50:
        stat, p = shapiro(data_clean)
        test_name = 'Shapiro-Wilk'
    else:
        stat, p = normaltest(data_clean)
        test_name = "D'Agostino"
    
    return test_name, p


def compare_groups_ordinal(group1, group2, var_name):
    """
    Compare two groups on an ordinal variable using ordinal logistic regression.
    
    Parameters:
    -----------
    group1 : array-like
        Data for group 1 (ordinal values, typically 1-5)
    group2 : array-like
        Data for group 2 (ordinal values, typically 1-5)
    var_name : str
        Name of the variable being compared
    
    Returns:
    --------
    dict : Dictionary containing ordinal logistic regression results
    """
    g1_clean = np.array(group1.dropna() if hasattr(group1, 'dropna') else group1)
    g2_clean = np.array(group2.dropna() if hasattr(group2, 'dropna') else group2)
    
    if len(g1_clean) < 2 or len(g2_clean) < 2:
        return None
    
    # Ensure values are valid ordinal categories
    all_values = np.concatenate([g1_clean, g2_clean])
    unique_values = np.unique(all_values)
    n_categories = len(unique_values)
    
    if n_categories < 2:
        return None
    
    # Create combined dataset for ordinal logistic regression
    y = np.concatenate([g1_clean, g2_clean])
    group_indicator = np.concatenate([np.zeros(len(g1_clean)), np.ones(len(g2_clean))])
    
    # Prepare data for statsmodels
    data_df = pd.DataFrame({
        'y': y,
        'group': group_indicator
    })
    
    try:
        if ORDINAL_AVAILABLE:
            # Use ordinal logistic regression (proportional odds model)
            mod = OrderedModel(data_df['y'], data_df[['group']], distr='logit')
            result = mod.fit(method='bfgs', disp=0)
            
            # Extract coefficient for group difference
            coef = result.params['group']
            se = result.bse['group']
            p_value = result.pvalues['group']
            
            # Calculate odds ratio
            odds_ratio = np.exp(coef)
            or_ci_lower = np.exp(coef - 1.96 * se)
            or_ci_upper = np.exp(coef + 1.96 * se)
            
            # Effect size: convert odds ratio to approximate Cohen's d
            # Using log odds ratio as effect size metric
            log_or = coef
            # Approximate Cohen's d from log odds ratio (rough approximation)
            # For ordinal data, we can use the log odds ratio as an effect size
            approx_d = log_or / 1.81  # Rough conversion factor
            
            return {
                'Variable': var_name,
                'Test': 'Ordinal Logistic Regression',
                'Statistic': coef,
                'P_Value': p_value,
                'Effect_Size': approx_d,
                'Effect_Size_Name': "Approx. Cohen's d (from log OR)",
                'Odds_Ratio': odds_ratio,
                'OR_CI_Lower': or_ci_lower,
                'OR_CI_Upper': or_ci_upper,
                'Log_Odds_Ratio': log_or,
                'Group1_Median': np.median(g1_clean),
                'Group1_N': len(g1_clean),
                'Group2_Median': np.median(g2_clean),
                'Group2_N': len(g2_clean),
                'Group1_Mean': g1_clean.mean(),  # Keep for compatibility
                'Group2_Mean': g2_clean.mean(),  # Keep for compatibility
                'Mean_Difference': g1_clean.mean() - g2_clean.mean(),  # Keep for compatibility
                'CI_Lower': or_ci_lower - 1,  # Approximate CI for difference
                'CI_Upper': or_ci_upper - 1,
                'Significant': p_value < 0.05
            }
        else:
            # Fallback to Mann-Whitney U if ordinal regression not available
            return None
    except Exception as e:
        warnings.warn(f"Ordinal logistic regression failed for {var_name}: {str(e)}. Falling back to Mann-Whitney U.")
        return None


def compare_groups_continuous(group1, group2, var_name):
    """Compare two groups on a continuous variable with comprehensive statistics."""
    g1_clean = group1.dropna()
    g2_clean = group2.dropna()
    
    if len(g1_clean) < 2 or len(g2_clean) < 2:
        return None
    
    # Test normality
    g1_normal = test_normality(g1_clean)
    g2_normal = test_normality(g2_clean)
    
    # Test homogeneity of variance
    try:
        levene_stat, levene_p = levene(g1_clean, g2_clean)
        equal_var = levene_p > 0.05
    except:
        equal_var = True
    
    # Choose appropriate test
    use_parametric = (g1_normal[1] > 0.05 if g1_normal else False) and \
                    (g2_normal[1] > 0.05 if g2_normal else False)
    
    if use_parametric:
        # t-test
        stat, p_value = ttest_ind(g1_clean, g2_clean, equal_var=equal_var)
        test_name = 't-test'
    else:
        # Mann-Whitney U
        stat, p_value = mannwhitneyu(g1_clean, g2_clean, alternative='two-sided')
        test_name = 'Mann-Whitney U'
    
    # Effect size
    if use_parametric:
        effect_size = cohens_d(g1_clean, g2_clean)
        effect_name = "Cohen's d"
    else:
        # Rank-biserial correlation as effect size for Mann-Whitney
        u_stat = stat
        n1, n2 = len(g1_clean), len(g2_clean)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)
        effect_name = "Rank-biserial r"
    
    # Confidence interval for mean difference
    diff_mean = g1_clean.mean() - g2_clean.mean()
    if use_parametric:
        # Use t-test CI
        try:
            result = ttest_ind_sm(g1_clean, g2_clean, usevar='pooled')
            if hasattr(result, 'confint'):
                ci_lower = result.confint[0]
                ci_upper = result.confint[1]
            else:
                # Fallback: manual CI calculation
                n1, n2 = len(g1_clean), len(g2_clean)
                s1, s2 = g1_clean.std(ddof=1), g2_clean.std(ddof=1)
                pooled_se = np.sqrt((s1**2/n1) + (s2**2/n2))
                t_crit = stats.t.ppf(0.975, n1 + n2 - 2)
                ci_lower = diff_mean - t_crit * pooled_se
                ci_upper = diff_mean + t_crit * pooled_se
        except:
            # Fallback: simple CI
            n1, n2 = len(g1_clean), len(g2_clean)
            s1, s2 = g1_clean.std(ddof=1), g2_clean.std(ddof=1)
            pooled_se = np.sqrt((s1**2/n1) + (s2**2/n2))
            t_crit = stats.t.ppf(0.975, min(n1, n2) - 1) if min(n1, n2) > 1 else 1.96
            ci_lower = diff_mean - t_crit * pooled_se
            ci_upper = diff_mean + t_crit * pooled_se
    else:
        # Bootstrap CI for median difference (or use simple approximation for very small samples)
        if len(g1_clean) >= 3 and len(g2_clean) >= 3:
            n_iterations = 1000
            diffs = []
            for _ in range(n_iterations):
                g1_sample = np.random.choice(g1_clean, size=len(g1_clean), replace=True)
                g2_sample = np.random.choice(g2_clean, size=len(g2_clean), replace=True)
                diffs.append(np.median(g1_sample) - np.median(g2_sample))
            ci_lower = np.percentile(diffs, 2.5)
            ci_upper = np.percentile(diffs, 97.5)
        else:
            # For very small samples, use a simple approximation
            ci_lower = diff_mean - 1.96 * (g1_clean.std() + g2_clean.std())
            ci_upper = diff_mean + 1.96 * (g1_clean.std() + g2_clean.std())
    
    return {
        'Variable': var_name,
        'Test': test_name,
        'Statistic': stat,
        'P_Value': p_value,
        'Effect_Size': effect_size,
        'Effect_Size_Name': effect_name,
        'Group1_Mean': g1_clean.mean(),
        'Group1_N': len(g1_clean),
        'Group2_Mean': g2_clean.mean(),
        'Group2_N': len(g2_clean),
        'Mean_Difference': diff_mean,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper,
        'Significant': p_value < 0.05
    }


def compare_groups_categorical(group1, group2, var_name):
    """Compare two groups on a categorical variable."""
    g1_clean = group1.dropna()
    g2_clean = group2.dropna()
    
    if len(g1_clean) < 2 or len(g2_clean) < 2:
        return None
    
    # Create contingency table
    g1_counts = g1_clean.value_counts()
    g2_counts = g2_clean.value_counts()
    
    all_categories = set(g1_counts.index) | set(g2_counts.index)
    contingency = pd.DataFrame({
        'Group1': [g1_counts.get(cat, 0) for cat in all_categories],
        'Group2': [g2_counts.get(cat, 0) for cat in all_categories]
    }, index=list(all_categories))
    
    # Chi-square or Fisher's exact test
    if contingency.shape[0] == 2 and contingency.shape[1] == 2:
        # 2x2 table - use Fisher's exact
        stat, p_value = fisher_exact(contingency)
        test_name = "Fisher's Exact"
        effect_size = None  # Calculate odds ratio
        odds_ratio = (contingency.iloc[0, 0] * contingency.iloc[1, 1]) / \
                     (contingency.iloc[0, 1] * contingency.iloc[1, 0]) if \
                     (contingency.iloc[0, 1] * contingency.iloc[1, 0]) > 0 else np.nan
        effect_name = "Odds Ratio"
    else:
        # Larger table - use chi-square
        stat, p_value, dof, expected = chi2_contingency(contingency)
        test_name = "Chi-square"
        effect_size = cramers_v(contingency)
        effect_name = "Cramér's V"
        odds_ratio = None
    
    return {
        'Variable': var_name,
        'Test': test_name,
        'Statistic': stat,
        'P_Value': p_value,
        'Effect_Size': effect_size if effect_size is not None else odds_ratio,
        'Effect_Size_Name': effect_name,
        'Group1_N': len(g1_clean),
        'Group2_N': len(g2_clean),
        'Significant': p_value < 0.05
    }


# ============================================================================
# CROSS-COMPARISON ANALYSES
# ============================================================================

def compare_europe_vs_rest(df):
    """Compare Europe vs Rest of World on all variables."""
    if 'region_binary' not in df.columns:
        return None
    
    europe = df[df['region_binary'] == 'Europe']
    rest = df[df['region_binary'] == 'Rest of World']
    
    if len(europe) < 2 or len(rest) < 2:
        return None
    
    comparisons = []
    
    # Color associations (ordinal Likert scales)
    color_cols = [col for col in df.columns if col.startswith('CA') and '_' in col]
    for col in color_cols:
        # Try ordinal logistic regression first
        result = compare_groups_ordinal(europe[col], rest[col], col)
        # Fallback to continuous comparison if ordinal fails
        if result is None:
            result = compare_groups_continuous(europe[col], rest[col], col)
        if result:
            result['Comparison'] = 'Europe vs Rest of World'
            comparisons.append(result)
    
    # Symbol recognition (binary)
    pk_recog_cols = [col for col in df.columns if col.endswith('_recognized')]
    for col in pk_recog_cols:
        result = compare_groups_categorical(europe[col], rest[col], col)
        if result:
            result['Comparison'] = 'Europe vs Rest of World'
            comparisons.append(result)
    
    # Remembrance accuracy (binary)
    m_correct_cols = [col for col in df.columns if col.endswith('_correct')]
    for col in m_correct_cols:
        result = compare_groups_categorical(europe[col], rest[col], col)
        if result:
            result['Comparison'] = 'Europe vs Rest of World'
            comparisons.append(result)
    
    return pd.DataFrame(comparisons)


def compare_international_vs_noninternational_student(df):
    """Compare International Students vs Non-International Students on all variables."""
    if 'is_international_student' not in df.columns:
        return None
    
    international_students = df[df['is_international_student'] == 1]
    noninternational_students = df[df['is_international_student'] == 0]
    
    if len(international_students) < 2 or len(noninternational_students) < 2:
        return None
    
    comparisons = []
    
    # Color associations (ordinal Likert scales)
    color_cols = [col for col in df.columns if col.startswith('CA') and '_' in col]
    for col in color_cols:
        # Try ordinal logistic regression first
        result = compare_groups_ordinal(international_students[col], noninternational_students[col], col)
        # Fallback to continuous comparison if ordinal fails
        if result is None:
            result = compare_groups_continuous(international_students[col], noninternational_students[col], col)
        if result:
            result['Comparison'] = 'International Student vs Non-International Student'
            comparisons.append(result)
    
    # Symbol recognition
    pk_recog_cols = [col for col in df.columns if col.endswith('_recognized')]
    for col in pk_recog_cols:
        result = compare_groups_categorical(international_students[col], noninternational_students[col], col)
        if result:
            result['Comparison'] = 'International Student vs Non-International Student'
            comparisons.append(result)
    
    # Remembrance accuracy
    m_correct_cols = [col for col in df.columns if col.endswith('_correct')]
    for col in m_correct_cols:
        result = compare_groups_categorical(international_students[col], noninternational_students[col], col)
        if result:
            result['Comparison'] = 'International Student vs Non-International Student'
            comparisons.append(result)
    
    return pd.DataFrame(comparisons)


def compare_gender(df):
    """Compare by gender (focusing on Male vs Female, excluding Diverse if small sample)."""
    if 'gender_label' not in df.columns:
        return None
    
    # Focus on Male vs Female
    gender_df = df[df['gender_label'].isin(['Male', 'Female'])]
    male = gender_df[gender_df['gender_label'] == 'Male']
    female = gender_df[gender_df['gender_label'] == 'Female']
    
    if len(male) < 2 or len(female) < 2:
        return None
    
    comparisons = []
    
    # Color associations (ordinal Likert scales)
    color_cols = [col for col in df.columns if col.startswith('CA') and '_' in col]
    for col in color_cols:
        # Try ordinal logistic regression first
        result = compare_groups_ordinal(male[col], female[col], col)
        # Fallback to continuous comparison if ordinal fails
        if result is None:
            result = compare_groups_continuous(male[col], female[col], col)
        if result:
            result['Comparison'] = 'Male vs Female'
            comparisons.append(result)
    
    # Symbol recognition
    pk_recog_cols = [col for col in df.columns if col.endswith('_recognized')]
    for col in pk_recog_cols:
        result = compare_groups_categorical(male[col], female[col], col)
        if result:
            result['Comparison'] = 'Male vs Female'
            comparisons.append(result)
    
    # Remembrance accuracy
    m_correct_cols = [col for col in df.columns if col.endswith('_correct')]
    for col in m_correct_cols:
        result = compare_groups_categorical(male[col], female[col], col)
        if result:
            result['Comparison'] = 'Male vs Female'
            comparisons.append(result)
    
    return pd.DataFrame(comparisons)


def compare_europe_vs_africa(df):
    """Compare Europe vs Africa on all variables."""
    return compare_regions(df, region1_code=1, region2_code=6, comparison_name='Europe vs Africa')


def compare_europe_vs_asia(df):
    """Compare Europe vs Asia on all variables."""
    return compare_regions(df, region1_code=1, region2_code=4, comparison_name='Europe vs Asia')


def compare_regions(df, region1_code, region2_code, comparison_name):
    """General function to compare any two regions on all variables."""
    if 'DD04' not in df.columns:
        return None
    
    # Region name mapping
    region_names = {
        1: 'Europe',
        2: 'America',
        4: 'Asia',
        6: 'Africa',
        7: 'Oceania'
    }
    
    region1_name = region_names.get(region1_code, f'Region_{region1_code}')
    region2_name = region_names.get(region2_code, f'Region_{region2_code}')
    
    region1 = df[df['DD04'] == region1_code]
    region2 = df[df['DD04'] == region2_code]
    
    if len(region1) < 2 or len(region2) < 2:
        return None
    
    comparisons = []
    
    # Color associations (ordinal Likert scales)
    color_cols = [col for col in df.columns if col.startswith('CA') and '_' in col]
    for col in color_cols:
        # Try ordinal logistic regression first
        result = compare_groups_ordinal(region1[col], region2[col], col)
        # Fallback to continuous comparison if ordinal fails
        if result is None:
            result = compare_groups_continuous(region1[col], region2[col], col)
        if result:
            result['Comparison'] = comparison_name
            result['Group1_Region'] = region1_name
            result['Group2_Region'] = region2_name
            comparisons.append(result)
    
    # Symbol recognition (binary)
    pk_recog_cols = [col for col in df.columns if col.endswith('_recognized')]
    for col in pk_recog_cols:
        result = compare_groups_categorical(region1[col], region2[col], col)
        if result:
            result['Comparison'] = comparison_name
            result['Group1_Region'] = region1_name
            result['Group2_Region'] = region2_name
            comparisons.append(result)
    
    # Remembrance accuracy (binary)
    m_correct_cols = [col for col in df.columns if col.endswith('_correct')]
    for col in m_correct_cols:
        result = compare_groups_categorical(region1[col], region2[col], col)
        if result:
            result['Comparison'] = comparison_name
            result['Group1_Region'] = region1_name
            result['Group2_Region'] = region2_name
            comparisons.append(result)
    
    return pd.DataFrame(comparisons)


def apply_multiple_comparisons_correction(comparisons_df, method='fdr_bh'):
    """Apply multiple comparisons correction to p-values."""
    if comparisons_df is None or len(comparisons_df) == 0:
        return comparisons_df
    
    if 'P_Value' not in comparisons_df.columns:
        return comparisons_df
    
    p_values = comparisons_df['P_Value'].values
    rejected, p_adjusted, _, _ = multipletests(p_values, alpha=0.1, method=method)
    
    comparisons_df['P_Value_Adjusted'] = p_adjusted
    comparisons_df['Significant_Adjusted'] = rejected
    
    return comparisons_df


def run_bayesian_comparisons(df, comparisons_df, use_ordinal=True):
    """
    Run Bayesian analysis for all comparisons in comparisons_df.
    
    Parameters:
    -----------
    df : DataFrame
        Original cleaned data
    comparisons_df : DataFrame
        DataFrame with frequentist comparison results
    use_ordinal : bool
        Whether to use ordinal models for Likert-scale variables (default: True)
    
    Returns:
    --------
    DataFrame : Bayesian comparison results
    """
    if not BAYESIAN_AVAILABLE or comparisons_df is None or len(comparisons_df) == 0:
        return None
    
    bayesian_results = []
    
    for idx, row in comparisons_df.iterrows():
        var = row['Variable']
        comparison = row['Comparison']
        
        # Skip if not enough data
        if pd.isna(row.get('Group1_N')) or pd.isna(row.get('Group2_N')):
            continue
        
        # Determine groups based on comparison type
        if 'Europe vs Rest of World' in comparison:
            if 'region_binary' in df.columns:
                group1 = df[df['region_binary'] == 'Europe']
                group2 = df[df['region_binary'] == 'Rest of World']
            else:
                continue
        elif 'International Student vs Non-International Student' in comparison:
            if 'is_international_student' in df.columns:
                group1 = df[df['is_international_student'] == 1]
                group2 = df[df['is_international_student'] == 0]
            else:
                continue
        elif 'Male vs Female' in comparison:
            if 'gender_label' in df.columns:
                group1 = df[df['gender_label'] == 'Male']
                group2 = df[df['gender_label'] == 'Female']
            else:
                continue
        elif 'Europe vs Asia' in comparison:
            if 'DD04' in df.columns:
                group1 = df[df['DD04'] == 1]
                group2 = df[df['DD04'] == 4]
            else:
                continue
        elif 'Europe vs Africa' in comparison:
            if 'DD04' in df.columns:
                group1 = df[df['DD04'] == 1]
                group2 = df[df['DD04'] == 6]
            else:
                continue
        elif 'Europe vs America' in comparison:
            if 'DD04' in df.columns:
                group1 = df[df['DD04'] == 1]
                group2 = df[df['DD04'] == 2]
            else:
                continue
        elif 'Asia vs Africa' in comparison:
            if 'DD04' in df.columns:
                group1 = df[df['DD04'] == 4]
                group2 = df[df['DD04'] == 6]
            else:
                continue
        elif 'Asia vs America' in comparison:
            if 'DD04' in df.columns:
                group1 = df[df['DD04'] == 4]
                group2 = df[df['DD04'] == 2]
            else:
                continue
        elif 'Africa vs America' in comparison:
            if 'DD04' in df.columns:
                group1 = df[df['DD04'] == 6]
                group2 = df[df['DD04'] == 2]
            else:
                continue
        else:
            continue
        
        # Run appropriate Bayesian test
        if var.startswith('CA'):
            # Color association variables are Likert scales (1-5)
            if use_ordinal:
                # Use ordinal model for Likert-scale data
                bayes_result = bayesian_ordinal_model(group1[var], group2[var], var, n_categories=5)
            else:
                # Use continuous t-test (treating as continuous)
                bayes_result = bayesian_ttest(group1[var], group2[var], var)
            
            if bayes_result:
                bayes_result['Comparison'] = comparison
                # Convert ordinal results to similar format as continuous
                if use_ordinal and 'Mean_Diff_Posterior_Mean' in bayes_result:
                    # Map ordinal results to effect size format
                    bayes_result['Mean_Difference_Posterior_Mean'] = bayes_result.get('Mean_Diff_Posterior_Mean')
                    bayes_result['Mean_Difference_HDI_Lower'] = bayes_result.get('Mean_Diff_HDI_Lower')
                    bayes_result['Mean_Difference_HDI_Upper'] = bayes_result.get('Mean_Diff_HDI_Upper')
                    # For ordinal, we report mean difference on latent scale
                    # Approximate Cohen's d by dividing by typical scale (e.g., 1.0)
                    mean_diff_samples = bayes_result.get('Posterior_Samples', {}).get('mean_diff', [])
                    if len(mean_diff_samples) > 0:
                        # Approximate effect size
                        bayes_result['Cohens_D_Posterior_Mean'] = np.mean(mean_diff_samples) / 1.0
                        hdi = az.hdi(bayes_result.get('Trace'), var_names=['mean_diff'], hdi_prob=0.95)['mean_diff'].values
                        bayes_result['Cohens_D_HDI_Lower'] = hdi[0] / 1.0
                        bayes_result['Cohens_D_HDI_Upper'] = hdi[1] / 1.0
                bayesian_results.append(bayes_result)
        elif var.startswith('PK'):
            # PK variables are also Likert scales (1-5) for recognition
            if use_ordinal:
                bayes_result = bayesian_ordinal_model(group1[var], group2[var], var, n_categories=5)
                if bayes_result:
                    bayes_result['Comparison'] = comparison
                    # Map ordinal results
                    mean_diff_samples = bayes_result.get('Posterior_Samples', {}).get('mean_diff', [])
                    if len(mean_diff_samples) > 0:
                        bayes_result['Mean_Difference_Posterior_Mean'] = bayes_result.get('Mean_Diff_Posterior_Mean')
                        bayes_result['Mean_Difference_HDI_Lower'] = bayes_result.get('Mean_Diff_HDI_Lower')
                        bayes_result['Mean_Difference_HDI_Upper'] = bayes_result.get('Mean_Diff_HDI_Upper')
                        bayes_result['Cohens_D_Posterior_Mean'] = np.mean(mean_diff_samples) / 1.0
                        hdi = az.hdi(bayes_result.get('Trace'), var_names=['mean_diff'], hdi_prob=0.95)['mean_diff'].values
                        bayes_result['Cohens_D_HDI_Lower'] = hdi[0] / 1.0
                        bayes_result['Cohens_D_HDI_Upper'] = hdi[1] / 1.0
                    bayesian_results.append(bayes_result)
        elif var.endswith('_correct') or var.endswith('_recognized'):
            # Binary variable (remembrance or recognition)
            if var.endswith('_correct'):
                base_var = var.replace('_correct', '')
            else:
                base_var = var.replace('_recognized', '')
            
            if base_var in df.columns:
                g1_success = (group1[base_var] == 1).sum() if var.endswith('_correct') else group1[var].sum()
                g1_total = group1[base_var].notna().sum() if var.endswith('_correct') else group1[var].notna().sum()
                g2_success = (group2[base_var] == 1).sum() if var.endswith('_correct') else group2[var].sum()
                g2_total = group2[base_var].notna().sum() if var.endswith('_correct') else group2[var].notna().sum()
                
                bayes_result = bayesian_proportion_test(g1_success, g1_total, g2_success, g2_total, var)
                if bayes_result:
                    bayes_result['Comparison'] = comparison
                    bayesian_results.append(bayes_result)
    
    if len(bayesian_results) == 0:
        return None
    
    # Convert to DataFrame
    bayesian_df = pd.DataFrame(bayesian_results)
    return bayesian_df


def bayesian_analysis_fdr_significant(comparisons_df, df_clean):
    """
    Run comprehensive Bayesian analysis on FDR-significant results.
    
    Parameters:
    -----------
    comparisons_df : DataFrame
        DataFrame with frequentist comparison results
    df_clean : DataFrame
        Cleaned data
    
    Returns:
    --------
    DataFrame : Detailed Bayesian analysis of FDR-significant results
    """
    if not BAYESIAN_AVAILABLE:
        return None
    
    if 'Significant_Adjusted' not in comparisons_df.columns:
        return None
    
    fdr_sig = comparisons_df[comparisons_df['Significant_Adjusted'] == True].copy()
    
    if len(fdr_sig) == 0:
        return None
    
    detailed_results = []
    
    for idx, row in fdr_sig.iterrows():
        var = row['Variable']
        comparison = row['Comparison']
        
        # Get groups
        if 'Europe vs Asia' in comparison:
            group1 = df_clean[df_clean['DD04'] == 1]
            group2 = df_clean[df_clean['DD04'] == 4]
            g1_name, g2_name = 'Europe', 'Asia'
        elif 'Europe vs Rest of World' in comparison:
            group1 = df_clean[df_clean['DD04'] == 1]
            group2 = df_clean[df_clean['DD04'] != 1]
            g1_name, g2_name = 'Europe', 'Rest of World'
        elif 'Europe vs America' in comparison:
            group1 = df_clean[df_clean['DD04'] == 1]
            group2 = df_clean[df_clean['DD04'] == 2]
            g1_name, g2_name = 'Europe', 'America'
        elif 'Asia vs America' in comparison:
            group1 = df_clean[df_clean['DD04'] == 4]
            group2 = df_clean[df_clean['DD04'] == 2]
            g1_name, g2_name = 'Asia', 'America'
        elif 'Africa vs America' in comparison:
            group1 = df_clean[df_clean['DD04'] == 6]
            group2 = df_clean[df_clean['DD04'] == 2]
            g1_name, g2_name = 'Africa', 'America'
        else:
            continue
        
        result_dict = {
            'Variable': var,
            'Comparison': comparison,
            'Group1_Name': g1_name,
            'Group2_Name': g2_name
        }
        
        # Run Bayesian analysis
        if var.startswith('CA'):
            # Color associations are Likert scales - use ordinal model
            bayes_result = bayesian_ordinal_model(group1[var], group2[var], var, n_categories=5)
            if bayes_result:
                # Map ordinal results to effect size format
                mean_diff_samples = bayes_result.get('Posterior_Samples', {}).get('mean_diff', [])
                if len(mean_diff_samples) > 0:
                    # Approximate effect size for ordinal (latent scale difference)
                    approx_d = np.mean(mean_diff_samples) / 1.0
                    hdi = az.hdi(bayes_result.get('Trace'), var_names=['mean_diff'], hdi_prob=0.95)['mean_diff'].values
                    result_dict.update({
                        'Analysis_Type': 'Ordinal (Likert)',
                        'Effect_Size_Mean': approx_d,
                        'Effect_Size_HDI_Lower': hdi[0] / 1.0,
                        'Effect_Size_HDI_Upper': hdi[1] / 1.0,
                        'Mean_Diff_Mean': bayes_result.get('Mean_Diff_Posterior_Mean'),
                        'Mean_Diff_HDI_Lower': bayes_result.get('Mean_Diff_HDI_Lower'),
                        'Mean_Diff_HDI_Upper': bayes_result.get('Mean_Diff_HDI_Upper'),
                        'Prob_Effect_Positive': bayes_result.get('Prob_Effect_Positive'),
                        'Prob_Large_Effect': bayes_result.get('Prob_Large_Diff'),
                        'Trace': bayes_result.get('Trace')
                    })
        elif var.endswith('_correct'):
            # Binary (remembrance)
            base_var = var.replace('_correct', '')
            if base_var in df_clean.columns:
                g1_success = (group1[base_var] == 1).sum()
                g1_total = group1[base_var].notna().sum()
                g2_success = (group2[base_var] == 1).sum()
                g2_total = group2[base_var].notna().sum()
                
                bayes_result = bayesian_proportion_test(g1_success, g1_total, g2_success, g2_total, var)
                if bayes_result:
                    result_dict.update({
                        'Analysis_Type': 'Binary',
                        'P1_Mean': bayes_result.get('P1_Posterior_Mean'),
                        'P1_HDI_Lower': bayes_result.get('P1_HDI_Lower'),
                        'P1_HDI_Upper': bayes_result.get('P1_HDI_Upper'),
                        'P2_Mean': bayes_result.get('P2_Posterior_Mean'),
                        'P2_HDI_Lower': bayes_result.get('P2_HDI_Lower'),
                        'P2_HDI_Upper': bayes_result.get('P2_HDI_Upper'),
                        'Odds_Ratio_Mean': bayes_result.get('Odds_Ratio_Posterior_Mean'),
                        'Odds_Ratio_HDI_Lower': bayes_result.get('Odds_Ratio_HDI_Lower'),
                        'Odds_Ratio_HDI_Upper': bayes_result.get('Odds_Ratio_HDI_Upper'),
                        'Prob_P1_Greater': bayes_result.get('Prob_P1_Greater'),
                        'Trace': bayes_result.get('Trace')
                    })
        
        if len(result_dict) > 4:  # More than just basic info
            detailed_results.append(result_dict)
    
    if len(detailed_results) == 0:
        return None
    
    return pd.DataFrame(detailed_results)


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_recognition_rates(recognition_df, save_path):
    """Plot symbol recognition rates with confidence intervals."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    recognition_df_sorted = recognition_df.sort_values('Recognition_Rate', ascending=True)
    
    y_pos = np.arange(len(recognition_df_sorted))
    bars = ax.barh(y_pos, recognition_df_sorted['Recognition_Rate'], 
                   xerr=[recognition_df_sorted['Recognition_Rate'] - recognition_df_sorted['CI_Lower'],
                         recognition_df_sorted['CI_Upper'] - recognition_df_sorted['Recognition_Rate']],
                   capsize=5, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(recognition_df_sorted['Symbol'], fontsize=9)
    ax.set_xlabel('Recognition Rate (%)', fontsize=12)
    ax.set_title('Symbol Recognition Rates with 95% Confidence Intervals', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (rate, ci_l, ci_u) in enumerate(zip(recognition_df_sorted['Recognition_Rate'],
                                                 recognition_df_sorted['CI_Lower'],
                                                 recognition_df_sorted['CI_Upper'])):
        ax.text(rate + 2, i, f'{rate:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_remembrance_rates(remembrance_df, save_path):
    """Plot symbol remembrance rates with confidence intervals."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    remembrance_df_sorted = remembrance_df.sort_values('Remembrance_Rate', ascending=True)
    
    y_pos = np.arange(len(remembrance_df_sorted))
    bars = ax.barh(y_pos, remembrance_df_sorted['Remembrance_Rate'],
                   xerr=[remembrance_df_sorted['Remembrance_Rate'] - remembrance_df_sorted['CI_Lower'],
                         remembrance_df_sorted['CI_Upper'] - remembrance_df_sorted['Remembrance_Rate']],
                   capsize=5, alpha=0.7, color='orange')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(remembrance_df_sorted['Symbol'], fontsize=9)
    ax.set_xlabel('Remembrance Rate (%)', fontsize=12)
    ax.set_title('Symbol Remembrance Rates with 95% Confidence Intervals', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (rate, ci_l, ci_u) in enumerate(zip(remembrance_df_sorted['Remembrance_Rate'],
                                                 remembrance_df_sorted['CI_Lower'],
                                                 remembrance_df_sorted['CI_Upper'])):
        ax.text(rate + 2, i, f'{rate:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_color_associations_heatmap(color_df, save_path):
    """Plot color associations as a heatmap."""
    # Create a copy for plotting
    color_df_plot = color_df.copy()
    
    # Remove German text from color names (already in English format from calculate_color_associations)
    # But handle if there are any remaining German names
    color_df_plot['Color'] = color_df_plot['Color'].str.replace(r'\(.*?\)', '', regex=True).str.strip()
    color_df_plot['Color'] = color_df_plot['Color'].replace({
        'Rot (Red)': 'Red',
        'Blau (Blue)': 'Blue',
        'Gelb (Yellow)': 'Yellow',
        'Lila (Purple)': 'Purple',
        'Grün (Green)': 'Green'
    })
    
    # Pivot data for heatmap
    pivot_data = color_df_plot.pivot(index='Color', columns='Dimension', values='Mean')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn_r', center=3,
                cbar_kws={'label': 'Mean Rating (1-5 scale, lower=left pole, higher=right pole)'}, 
                ax=ax, vmin=1, vmax=5)
    ax.set_title('Color Associations Heatmap\n(Mean Ratings Across Dimensions)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_color_distributions(df, save_path):
    """Plot violin plots for color association distributions."""
    color_cols = [col for col in df.columns if col.startswith('CA') and '_' in col]
    
    # Create a long-format dataframe for plotting
    plot_data = []
    for col in color_cols:
        color_code = col.split('_')[0]
        dim_code = col.split('_')[1]
        
        color_map = {
            'CA01': 'Red', 'CA02': 'Blue', 'CA03': 'Yellow',
            'CA04': 'Purple', 'CA05': 'Orange', 'CA06': 'Green'
        }
        dim_map = {
            '01': 'Danger/Safe\n(lower=danger)', 
            '02': 'Pos/Neg\n(lower=positive)', 
            '03': 'Alarm/Reassure\n(lower=alarm)',
            '04': 'Important\n(lower=important)', 
            '05': 'Intense/Neutral\n(lower=intense)'
        }
        
        for val in df[col].dropna():
            plot_data.append({
                'Color': color_map.get(color_code, color_code),
                'Dimension': dim_map.get(dim_code, dim_code),
                'Value': val
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    if len(plot_df) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        colors = ['Red', 'Blue', 'Yellow', 'Purple', 'Orange', 'Green']
        for idx, color in enumerate(colors):
            if idx < len(axes):
                color_data = plot_df[plot_df['Color'] == color]
                if len(color_data) > 0:
                    sns.violinplot(data=color_data, x='Dimension', y='Value', ax=axes[idx])
                    axes[idx].set_title(f'{color}', fontweight='bold')
                    axes[idx].set_ylim(1, 5)
                    axes[idx].set_ylabel('Rating (1-5 scale, lower=left pole, higher=right pole)', fontsize=9)
                    axes[idx].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Color Association Distributions by Dimension', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_comparison_summary(comparisons_df, save_path):
    """Plot summary of significant comparisons."""
    if comparisons_df is None or len(comparisons_df) == 0:
        return
    
    # Filter significant results
    sig_results = comparisons_df[comparisons_df['Significant'] == True].copy()
    
    if len(sig_results) == 0:
        return
    
    # Group by comparison type
    fig, ax = plt.subplots(figsize=(12, 8))
    
    comparison_counts = sig_results['Comparison'].value_counts()
    bars = ax.barh(comparison_counts.index, comparison_counts.values, alpha=0.7)
    
    ax.set_xlabel('Number of Significant Differences', fontsize=12)
    ax.set_title('Summary of Significant Cross-Comparisons', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(comparison_counts.values):
        ax.text(v + 0.5, i, str(v), va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_fdr_significant_results(comparisons_df, df_clean, save_path):
    """Plot detailed visualization of results significant after FDR correction."""
    if comparisons_df is None or len(comparisons_df) == 0:
        return
    
    # Filter to FDR-corrected significant results
    if 'Significant_Adjusted' not in comparisons_df.columns:
        return
    
    sig_fdr = comparisons_df[comparisons_df['Significant_Adjusted'] == True].copy()
    
    if len(sig_fdr) == 0:
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    ax_main = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis('off')
    
    # Prepare data for plotting
    plot_data = []
    
    # Variable labels mapping with scale directionality
    var_labels = {
        'M6_RichtigFalsch_correct': 'Payment 2\nRemembrance',
        'M8_RichtigFalsch_correct': 'Location 2\nRemembrance',
        'CA05_02': 'Orange: Positive/Negative\n(lower=more positive)',
        'CA03_05': 'Yellow: Intense/Neutral\n(lower=more intense)'
    }
    
    # Color mapping for comparison types
    comparison_colors = {
        'Europe vs Asia': '#2E86AB',
        'Europe vs Rest of World': '#A23B72',
        'Europe vs America': '#F18F01',
        'Asia vs America': '#C73E1D',
        'Africa vs America': '#6A994E'
    }
    
    y_pos = 0
    y_labels = []
    y_ticks = []
    
    for idx, row in sig_fdr.iterrows():
        var = row['Variable']
        comparison = row['Comparison']
        p_adj = row['P_Value_Adjusted']
        
        # Get group data
        if var.startswith('M') and var.endswith('_correct'):
            # Remembrance variable - calculate proportions
            # Map from "M6_RichtigFalsch_correct" to "M6_RichtigFalsch"
            base_var = var.replace('_correct', '')
            if base_var not in df_clean.columns:
                # Variable should be in format M6_RichtigFalsch
                continue
            if base_var in df_clean.columns:
                # Split by comparison groups
                if 'Europe vs Asia' in comparison:
                    group1_data = df_clean[df_clean['DD04'] == 1][base_var] == 1
                    group2_data = df_clean[df_clean['DD04'] == 4][base_var] == 1
                    g1_name, g2_name = 'Europe', 'Asia'
                elif 'Europe vs Rest of World' in comparison:
                    group1_data = df_clean[df_clean['DD04'] == 1][base_var] == 1
                    group2_data = df_clean[df_clean['DD04'] != 1][base_var] == 1
                    g1_name, g2_name = 'Europe', 'Rest of World'
                elif 'Asia vs America' in comparison:
                    group1_data = df_clean[df_clean['DD04'] == 4][base_var] == 1
                    group2_data = df_clean[df_clean['DD04'] == 2][base_var] == 1
                    g1_name, g2_name = 'Asia', 'America'
                else:
                    continue
                
                g1_prop = group1_data.sum() / group1_data.notna().sum() * 100
                g2_prop = group2_data.sum() / group2_data.notna().sum() * 100
                g1_n = group1_data.notna().sum()
                g2_n = group2_data.notna().sum()
                
                plot_data.append({
                    'y_pos': y_pos,
                    'var_label': var_labels.get(var, var),
                    'comparison': comparison,
                    'g1_name': g1_name,
                    'g2_name': g2_name,
                    'g1_value': g1_prop,
                    'g2_value': g2_prop,
                    'g1_n': g1_n,
                    'g2_n': g2_n,
                    'p_adj': p_adj,
                    'effect_size': row.get('Effect_Size', np.nan),
                    'is_proportion': True,
                    'color': comparison_colors.get(comparison, '#666666')
                })
                
        elif var.startswith('CA'):
            # Color association - use means from comparison data
            g1_mean = row.get('Group1_Mean', np.nan)
            g2_mean = row.get('Group2_Mean', np.nan)
            g1_n = row.get('Group1_N', np.nan)
            g2_n = row.get('Group2_N', np.nan)
            
            if 'Europe vs Asia' in comparison:
                g1_name, g2_name = 'Europe', 'Asia'
            elif 'Europe vs America' in comparison:
                g1_name, g2_name = 'Europe', 'America'
            elif 'Africa vs America' in comparison:
                g1_name, g2_name = 'Africa', 'America'
            else:
                continue
            
            if not np.isnan(g1_mean) and not np.isnan(g2_mean):
                plot_data.append({
                    'y_pos': y_pos,
                    'var_label': var_labels.get(var, var),
                    'comparison': comparison,
                    'g1_name': g1_name,
                    'g2_name': g2_name,
                    'g1_value': g1_mean,
                    'g2_value': g2_mean,
                    'g1_n': int(g1_n) if not np.isnan(g1_n) else 0,
                    'g2_n': int(g2_n) if not np.isnan(g2_n) else 0,
                    'p_adj': p_adj,
                    'effect_size': row.get('Effect_Size', np.nan),
                    'is_proportion': False,
                    'color': comparison_colors.get(comparison, '#666666')
                })
        
        if len(plot_data) > 0 and plot_data[-1]['y_pos'] == y_pos:
            y_labels.append(f"{plot_data[-1]['var_label']}\n({comparison})")
            y_ticks.append(y_pos)
            y_pos += 1
    
    if len(plot_data) == 0:
        plt.close(fig)
        return
    
    # Create grouped bar chart
    x = np.arange(len(plot_data))
    width = 0.35
    
    g1_values = [d['g1_value'] for d in plot_data]
    g2_values = [d['g2_value'] for d in plot_data]
    colors = [d['color'] for d in plot_data]
    
    bars1 = ax_main.barh(x - width/2, g1_values, width, label='Group 1', alpha=0.8)
    bars2 = ax_main.barh(x + width/2, g2_values, width, label='Group 2', alpha=0.8)
    
    # Color bars by comparison type
    for i, (bar1, bar2, color) in enumerate(zip(bars1, bars2, colors)):
        bar1.set_color(color)
        bar2.set_color(color)
        bar2.set_edgecolor('white')
        bar2.set_linewidth(1.5)
    
    # Set labels and title
    ax_main.set_yticks(x)
    ax_main.set_yticklabels([f"{d['var_label']}\n({d['comparison']})" for d in plot_data], fontsize=9)
    if plot_data[0]['is_proportion']:
        xlabel = 'Value (%)'
    else:
        xlabel = 'Mean Rating (1-5 scale, lower=left pole, higher=right pole)'
    ax_main.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax_main.set_title('Significant Results After FDR Correction (α = 0.1)', 
                     fontsize=14, fontweight='bold', pad=20)
    ax_main.grid(axis='x', alpha=0.3, linestyle='--')
    ax_main.legend(['Group 1', 'Group 2'], loc='lower right', fontsize=10)
    
    # Add group labels and values on bars
    for i, d in enumerate(plot_data):
        # Group 1 label and value
        ax_main.text(d['g1_value'] + 1, i - width/2, 
                    f"{d['g1_name']}\n({d['g1_value']:.1f}%)" if d['is_proportion'] else f"{d['g1_name']}\n({d['g1_value']:.2f})",
                    va='center', fontsize=8, fontweight='bold')
        # Group 2 label and value
        ax_main.text(d['g2_value'] + 1, i + width/2,
                    f"{d['g2_name']}\n({d['g2_value']:.1f}%)" if d['is_proportion'] else f"{d['g2_name']}\n({d['g2_value']:.2f})",
                    va='center', fontsize=8, fontweight='bold')
    
    # Add text summary with p-values and effect sizes
    summary_text = "Summary Statistics:\n\n"
    for d in plot_data:
        p_str = f"{d['p_adj']:.4f}" if not np.isnan(d['p_adj']) else "N/A"
        eff_str = f"{d['effect_size']:.3f}" if not np.isnan(d['effect_size']) else "N/A"
        n_str = f"n₁={d['g1_n']}, n₂={d['g2_n']}"
        summary_text += f"{d['var_label']} ({d['comparison']}): p_adj={p_str}, effect={eff_str}, {n_str}\n"
    
    ax_text.text(0.05, 0.95, summary_text, transform=ax_text.transAxes,
                fontsize=8, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_posterior_distributions(samples, var_name, hdi_lower=None, hdi_upper=None, 
                                 save_path=None, title_suffix=""):
    """
    Plot posterior distribution with HDI.
    
    Parameters:
    -----------
    samples : array-like
        Posterior samples
    var_name : str
        Name of variable
    hdi_lower : float, optional
        Lower bound of HDI
    hdi_upper : float, optional
        Upper bound of HDI
    save_path : str, optional
        Path to save figure
    title_suffix : str
        Additional text for title
    """
    if not BAYESIAN_AVAILABLE or samples is None or len(samples) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot posterior distribution
    ax.hist(samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add HDI shading
    if hdi_lower is not None and hdi_upper is not None:
        ax.axvspan(hdi_lower, hdi_upper, alpha=0.2, color='red', label='95% HDI')
        ax.axvline(hdi_lower, color='red', linestyle='--', linewidth=2)
        ax.axvline(hdi_upper, color='red', linestyle='--', linewidth=2)
    
    # Add mean
    mean_val = np.mean(samples)
    ax.axvline(mean_val, color='black', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.3f}')
    
    ax.set_xlabel(var_name, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Posterior Distribution: {var_name}{title_suffix}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        return fig, ax


def plot_bayesian_effect_sizes(bayesian_results_df, save_path, max_comparisons=50, focus_significant=True):
    """
    Create forest plot of Bayesian effect sizes with credible intervals.
    
    Parameters:
    -----------
    bayesian_results_df : DataFrame
        DataFrame with Bayesian results
    save_path : str
        Path to save figure
    max_comparisons : int
        Maximum number of comparisons to plot (default: 50)
    focus_significant : bool
        If True, prioritize significant comparisons (default: True)
    """
    if not BAYESIAN_AVAILABLE or bayesian_results_df is None or len(bayesian_results_df) == 0:
        return
    
    # Prepare data
    plot_data = []
    
    for idx, row in bayesian_results_df.iterrows():
        var = row.get('Variable', 'Unknown')
        comparison = row.get('Comparison', 'Unknown')
        
        # Get effect size information - check both possible column names
        if 'Effect_Size_Mean' in row and pd.notna(row.get('Effect_Size_Mean')):
            effect_mean = row['Effect_Size_Mean']
            effect_lower = row.get('Effect_Size_HDI_Lower')
            effect_upper = row.get('Effect_Size_HDI_Upper')
            if pd.isna(effect_lower):
                effect_lower = effect_mean - 0.5
            if pd.isna(effect_upper):
                effect_upper = effect_mean + 0.5
            effect_type = "Cohen's d"
        elif 'Cohens_D_Posterior_Mean' in row and pd.notna(row.get('Cohens_D_Posterior_Mean')):
            effect_mean = row['Cohens_D_Posterior_Mean']
            effect_lower = row.get('Cohens_D_HDI_Lower', effect_mean - 0.5)
            effect_upper = row.get('Cohens_D_HDI_Upper', effect_mean + 0.5)
            effect_type = "Cohen's d"
        elif 'Odds_Ratio_Mean' in row and pd.notna(row.get('Odds_Ratio_Mean')):
            effect_mean = row['Odds_Ratio_Mean']
            effect_lower = row.get('Odds_Ratio_HDI_Lower')
            effect_upper = row.get('Odds_Ratio_HDI_Upper')
            if pd.isna(effect_lower) or effect_lower <= 0:
                effect_lower = effect_mean * 0.5 if effect_mean > 0 else 0.1
            if pd.isna(effect_upper) or effect_upper <= 0:
                effect_upper = effect_mean * 1.5 if effect_mean > 0 else 10
            effect_type = "Odds Ratio"
        elif 'Odds_Ratio_Posterior_Mean' in row and pd.notna(row.get('Odds_Ratio_Posterior_Mean')):
            effect_mean = row['Odds_Ratio_Posterior_Mean']
            effect_lower = row.get('Odds_Ratio_HDI_Lower', effect_mean * 0.5)
            effect_upper = row.get('Odds_Ratio_HDI_Upper', effect_mean * 1.5)
            effect_type = "Odds Ratio"
        else:
            continue
        
        # Filter out extreme outliers (likely due to very small sample sizes or convergence issues)
        if effect_type == "Cohen's d":
            # Filter out extreme values (|d| > 10 is unrealistic)
            if abs(effect_mean) > 10 or abs(effect_lower) > 10 or abs(effect_upper) > 10:
                continue
            effect_magnitude = abs(effect_mean)
        else:
            # Filter out extreme odds ratios
            if effect_mean > 100 or effect_mean < 0.01 or effect_upper > 1000:
                continue
            effect_magnitude = abs(np.log(effect_mean)) if effect_mean > 0 else 0
        
        plot_data.append({
            'label': f"{var}\n({comparison})",
            'effect_mean': effect_mean,
            'effect_lower': effect_lower,
            'effect_upper': effect_upper,
            'effect_type': effect_type,
            'effect_magnitude': effect_magnitude
        })
    
    if len(plot_data) == 0:
        return
    
    # Sort by effect magnitude and limit number
    plot_data.sort(key=lambda x: x['effect_magnitude'], reverse=True)
    if len(plot_data) > max_comparisons:
        plot_data = plot_data[:max_comparisons]
    
    # Create forest plot
    fig, ax = plt.subplots(figsize=(14, max(8, len(plot_data) * 0.4)))
    
    y_positions = np.arange(len(plot_data))
    
    # Determine x-axis limits
    all_effects = [d['effect_mean'] for d in plot_data]
    all_lowers = [d['effect_lower'] for d in plot_data]
    all_uppers = [d['effect_upper'] for d in plot_data]
    
    x_min = min(min(all_lowers), min(all_effects)) * 1.1
    x_max = max(max(all_uppers), max(all_effects)) * 1.1
    
    for i, data in enumerate(plot_data):
        # Plot point estimate
        ax.scatter(data['effect_mean'], i, s=100, zorder=3, color='steelblue')
        
        # Plot HDI
        ax.plot([data['effect_lower'], data['effect_upper']], [i, i], 
               'k-', linewidth=2, zorder=2)
        ax.plot([data['effect_lower'], data['effect_lower']], [i-0.15, i+0.15], 
               'k-', linewidth=2, zorder=2)
        ax.plot([data['effect_upper'], data['effect_upper']], [i-0.15, i+0.15], 
               'k-', linewidth=2, zorder=2)
    
    # Add reference line at 0 (or 1 for odds ratio)
    if plot_data[0]['effect_type'] == "Cohen's d":
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='No effect')
    else:
        ax.axvline(1, color='red', linestyle='--', alpha=0.5, label='No effect')
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels([d['label'] for d in plot_data], fontsize=8)
    ax.set_xlabel(f"Effect Size ({plot_data[0]['effect_type']})", fontsize=12, fontweight='bold')
    ax.set_title(f'Bayesian Effect Sizes with 95% Credible Intervals\n(Top {len(plot_data)} by magnitude)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(x_min, x_max)
    ax.grid(axis='x', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_frequentist_vs_bayesian(comparisons_df, bayesian_df, save_path):
    """
    Compare frequentist and Bayesian results side-by-side.
    
    Parameters:
    -----------
    comparisons_df : DataFrame
        Frequentist comparison results
    bayesian_df : DataFrame
        Bayesian comparison results
    save_path : str
        Path to save figure
    """
    if comparisons_df is None or bayesian_df is None:
        return
    
    # Check which columns are available in bayesian_df
    available_cols = ['Variable', 'Comparison']
    if 'Effect_Size_Mean' in bayesian_df.columns:
        available_cols.extend(['Effect_Size_Mean', 'Effect_Size_HDI_Lower', 'Effect_Size_HDI_Upper'])
    elif 'Cohens_D_Posterior_Mean' in bayesian_df.columns:
        available_cols.extend(['Cohens_D_Posterior_Mean', 'Cohens_D_HDI_Lower', 'Cohens_D_HDI_Upper'])
    else:
        return  # No effect size columns available
    
    # Merge on Variable and Comparison
    merged = comparisons_df.merge(
        bayesian_df[available_cols],
        on=['Variable', 'Comparison'],
        how='inner',
        suffixes=('_freq', '_bayes')
    )
    
    if len(merged) == 0:
        return
    
    # Determine column names
    if 'Effect_Size_Mean' in merged.columns:
        bayes_effect_col = 'Effect_Size_Mean'
        bayes_lower_col = 'Effect_Size_HDI_Lower'
        bayes_upper_col = 'Effect_Size_HDI_Upper'
    else:
        bayes_effect_col = 'Cohens_D_Posterior_Mean'
        bayes_lower_col = 'Cohens_D_HDI_Lower'
        bayes_upper_col = 'Cohens_D_HDI_Upper'
    
    # Filter to only continuous variables (those with effect sizes)
    # Check if Effect_Size_Name column exists
    if 'Effect_Size_Name' in merged.columns:
        merged_continuous = merged[merged['Effect_Size_Name'].isin(["Cohen's d", "Rank-biserial r"])].copy()
    else:
        # If column doesn't exist, filter by having both Effect_Size and Bayesian effect size
        merged_continuous = merged[
            merged['Effect_Size'].notna() & merged[bayes_effect_col].notna()
        ].copy()
    
    if len(merged_continuous) == 0:
        plt.close(fig)
        return
    
    # Filter out extreme outliers
    merged_continuous = merged_continuous[
        (merged_continuous[bayes_effect_col].abs() <= 10) &
        (merged_continuous['Effect_Size'].abs() <= 10)
    ].copy()
    
    if len(merged_continuous) == 0:
        plt.close(fig)
        return
    
    # Limit to top 30 by absolute effect size for readability
    if len(merged_continuous) > 30:
        merged_continuous['abs_effect'] = merged_continuous['Effect_Size'].abs()
        merged_continuous = merged_continuous.nlargest(30, 'abs_effect').copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(merged_continuous) * 0.3)))
    
    # Plot 1: Effect sizes
    ax1 = axes[0]
    plot_data_1 = []
    for idx, (_, row) in enumerate(merged_continuous.iterrows()):
        if pd.notna(row.get('Effect_Size')) and pd.notna(row.get(bayes_effect_col)):
            freq_effect = row['Effect_Size']
            bayes_effect = row[bayes_effect_col]
            plot_data_1.append({
                'idx': idx,
                'freq': freq_effect,
                'bayes': bayes_effect,
                'bayes_lower': row.get(bayes_lower_col),
                'bayes_upper': row.get(bayes_upper_col),
                'label': f"{row['Variable']}\n({row['Comparison']})"
            })
    
    if len(plot_data_1) == 0:
        plt.close(fig)
        return
    
    y_pos = np.arange(len(plot_data_1))
    for i, data in enumerate(plot_data_1):
        # Frequentist
        ax1.scatter(data['freq'], i, s=100, color='blue', alpha=0.7, label='Frequentist' if i == 0 else '', zorder=3)
        # Bayesian
        ax1.scatter(data['bayes'], i, s=100, color='red', alpha=0.7, marker='s', 
                   label='Bayesian' if i == 0 else '', zorder=3)
        # HDI
        if pd.notna(data['bayes_lower']) and pd.notna(data['bayes_upper']):
            ax1.plot([data['bayes_lower'], data['bayes_upper']], [i, i], 
                    'r-', alpha=0.5, linewidth=2, zorder=2)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([d['label'] for d in plot_data_1], fontsize=8)
    ax1.set_xlabel("Effect Size (Cohen's d)", fontsize=12)
    ax1.set_title('Frequentist vs Bayesian Effect Sizes', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Confidence/Credible Intervals
    ax2 = axes[1]
    plot_data_2 = []
    for idx, (_, row) in enumerate(merged_continuous.iterrows()):
        if pd.notna(row.get('CI_Lower')) and pd.notna(row.get(bayes_lower_col)):
            plot_data_2.append({
                'idx': idx,
                'freq_lower': row.get('CI_Lower', 0),
                'freq_upper': row.get('CI_Upper', 0),
                'bayes_lower': row.get(bayes_lower_col, 0),
                'bayes_upper': row.get(bayes_upper_col, 0),
                'label': f"{row['Variable']}\n({row['Comparison']})"
            })
    
    if len(plot_data_2) > 0:
        y_pos_2 = np.arange(len(plot_data_2))
        for i, data in enumerate(plot_data_2):
            # Frequentist CI width
            freq_width = data['freq_upper'] - data['freq_lower']
            # Bayesian HDI width
            bayes_width = data['bayes_upper'] - data['bayes_lower']
            
            ax2.barh(i*2, freq_width, left=data['freq_lower'], alpha=0.7, color='blue', 
                    height=0.6, label='Frequentist CI' if i == 0 else '')
            ax2.barh(i*2+1, bayes_width, left=data['bayes_lower'], alpha=0.7, color='red',
                    height=0.6, label='Bayesian HDI' if i == 0 else '')
        
        ax2.set_yticks([i*2+0.5 for i in range(len(plot_data_2))])
        ax2.set_yticklabels([d['label'] for d in plot_data_2], fontsize=8)
        ax2.set_xlabel("Effect Size", fontsize=12)
        ax2.set_title('Confidence Intervals vs Credible Intervals', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("PICTOS SURVEY DATA ANALYSIS")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data...")
    codebook_path = "codebook_dangersymbols98158_2025-11-27_16-42.csv"
    data_path = "Results-dataset.csv"
    
    codebook = load_codebook(codebook_path)
    df = load_data(data_path)
    print(f"Loaded {len(df)} cases")
    
    # Preprocess
    print("Preprocessing data...")
    df_clean = preprocess_data(df, codebook)
    print(f"After preprocessing: {len(df_clean)} cases")
    print()
    
    # General statistics
    print("Calculating general statistics...")
    
    # Recognition rates
    recognition_df = calculate_recognition_rates(df_clean)
    recognition_df.to_csv(RESULTS_DIR / "recognition_rates.csv", index=False)
    print(f"Symbol recognition rates calculated for {len(recognition_df)} symbols")
    
    # Remembrance rates
    remembrance_df = calculate_remembrance_rates(df_clean)
    remembrance_df.to_csv(RESULTS_DIR / "remembrance_rates.csv", index=False)
    print(f"Symbol remembrance rates calculated for {len(remembrance_df)} symbols")
    
    # Color associations
    color_df = calculate_color_associations(df_clean)
    color_df.to_csv(RESULTS_DIR / "color_associations.csv", index=False)
    print(f"Color associations calculated for {len(color_df)} color-dimension combinations")
    
    # Demographics
    demo_stats = calculate_demographics(df_clean)
    print("Demographic statistics calculated")
    print()
    
    # Cross-comparisons
    print("Performing cross-comparisons...")
    
    europe_comparisons = compare_europe_vs_rest(df_clean)
    if europe_comparisons is not None and len(europe_comparisons) > 0:
        europe_comparisons = apply_multiple_comparisons_correction(europe_comparisons)
        print(f"Europe vs Rest of World: {len(europe_comparisons)} comparisons")
    
    international_student_comparisons = compare_international_vs_noninternational_student(df_clean)
    if international_student_comparisons is not None and len(international_student_comparisons) > 0:
        international_student_comparisons = apply_multiple_comparisons_correction(international_student_comparisons)
        print(f"International Student vs Non-International Student: {len(international_student_comparisons)} comparisons")
    
    gender_comparisons = compare_gender(df_clean)
    if gender_comparisons is not None and len(gender_comparisons) > 0:
        gender_comparisons = apply_multiple_comparisons_correction(gender_comparisons)
        print(f"Gender comparisons: {len(gender_comparisons)} comparisons")
    
    # Regional pairwise comparisons
    regional_comparisons_list = []
    
    # All pairwise regional comparisons
    regional_pairs = [
        (1, 4, 'Europe vs Asia'),
        (1, 6, 'Europe vs Africa'),
        (1, 2, 'Europe vs America'),
        (4, 6, 'Asia vs Africa'),
        (4, 2, 'Asia vs America'),
        (6, 2, 'Africa vs America')
    ]
    
    for region1_code, region2_code, comparison_name in regional_pairs:
        comparisons = compare_regions(df_clean, region1_code, region2_code, comparison_name)
        if comparisons is not None and len(comparisons) > 0:
            comparisons = apply_multiple_comparisons_correction(comparisons)
            regional_comparisons_list.append(comparisons)
            print(f"{comparison_name}: {len(comparisons)} comparisons")
    
    print()
    
    # Combine all comparisons
    all_comparisons = []
    if europe_comparisons is not None and len(europe_comparisons) > 0:
        all_comparisons.append(europe_comparisons)
    if international_student_comparisons is not None and len(international_student_comparisons) > 0:
        all_comparisons.append(international_student_comparisons)
    if gender_comparisons is not None and len(gender_comparisons) > 0:
        all_comparisons.append(gender_comparisons)
    # Add all regional pairwise comparisons
    for reg_comp in regional_comparisons_list:
        if reg_comp is not None and len(reg_comp) > 0:
            all_comparisons.append(reg_comp)
    
    if all_comparisons:
        comparisons_combined = pd.concat(all_comparisons, ignore_index=True)
        comparisons_combined.to_csv(RESULTS_DIR / "cross_comparisons.csv", index=False)
        print(f"All comparisons saved ({len(comparisons_combined)} total)")
    print()
    
    # Bayesian Analysis
    if BAYESIAN_AVAILABLE:
        print("Running Bayesian analyses...")
        
        # Check if Bayesian results already exist
        bayesian_file = BAYESIAN_DIR / "bayesian_comparisons.csv"
        if bayesian_file.exists():
            print(f"  - Found existing Bayesian results at {bayesian_file}")
            print("  - Skipping Bayesian calculations (delete file to recalculate)")
            bayesian_comparisons = None
        else:
            # Run Bayesian analysis on all comparisons
            if all_comparisons:
                print("  - Running Bayesian comparisons (this may take a while)...")
                bayesian_comparisons = run_bayesian_comparisons(df_clean, comparisons_combined, use_ordinal=True)
                if bayesian_comparisons is not None and len(bayesian_comparisons) > 0:
                    # Save comprehensive Bayesian results
                    bayesian_summary = []
                    for idx, row in bayesian_comparisons.iterrows():
                        summary_row = {
                            'Variable': row.get('Variable'),
                            'Comparison': row.get('Comparison'),
                            'Method': row.get('Method')
                        }
                        
                        # Add continuous/ordinal variable results
                        if 'Cohens_D_Posterior_Mean' in row:
                            summary_row.update({
                                'Effect_Size_Mean': row.get('Cohens_D_Posterior_Mean'),
                                'Effect_Size_HDI_Lower': row.get('Cohens_D_HDI_Lower'),
                                'Effect_Size_HDI_Upper': row.get('Cohens_D_HDI_Upper'),
                                'Mean_Diff_Mean': row.get('Mean_Difference_Posterior_Mean'),
                                'Mean_Diff_HDI_Lower': row.get('Mean_Difference_HDI_Lower'),
                                'Mean_Diff_HDI_Upper': row.get('Mean_Difference_HDI_Upper'),
                                'Prob_Effect_Positive': row.get('Prob_Effect_Positive'),
                                'Prob_Large_Effect': row.get('Prob_Large_Effect')
                            })
                        # Add ordinal model results (Mean_Diff_Posterior_Mean from ordinal model)
                        elif 'Mean_Diff_Posterior_Mean' in row:
                            summary_row.update({
                                'Effect_Size_Mean': row.get('Cohens_D_Posterior_Mean'),  # May be None, will be calculated
                                'Effect_Size_HDI_Lower': row.get('Cohens_D_HDI_Lower'),
                                'Effect_Size_HDI_Upper': row.get('Cohens_D_HDI_Upper'),
                                'Mean_Diff_Mean': row.get('Mean_Diff_Posterior_Mean'),
                                'Mean_Diff_HDI_Lower': row.get('Mean_Diff_HDI_Lower'),
                                'Mean_Diff_HDI_Upper': row.get('Mean_Diff_HDI_Upper'),
                                'Prob_Effect_Positive': row.get('Prob_Effect_Positive'),
                                'Prob_Large_Effect': row.get('Prob_Large_Diff')
                            })
                        
                        # Add binary variable results
                        if 'Odds_Ratio_Posterior_Mean' in row:
                            summary_row.update({
                                'P1_Mean': row.get('P1_Posterior_Mean'),
                                'P1_HDI_Lower': row.get('P1_HDI_Lower'),
                                'P1_HDI_Upper': row.get('P1_HDI_Upper'),
                                'P2_Mean': row.get('P2_Posterior_Mean'),
                                'P2_HDI_Lower': row.get('P2_HDI_Lower'),
                                'P2_HDI_Upper': row.get('P2_HDI_Upper'),
                                'Odds_Ratio_Mean': row.get('Odds_Ratio_Posterior_Mean'),
                                'Odds_Ratio_HDI_Lower': row.get('Odds_Ratio_HDI_Lower'),
                                'Odds_Ratio_HDI_Upper': row.get('Odds_Ratio_HDI_Upper'),
                                'Prob_P1_Greater': row.get('Prob_P1_Greater')
                            })
                        
                        bayesian_summary.append(summary_row)
                
                    bayesian_summary_df = pd.DataFrame(bayesian_summary)
                    bayesian_summary_df.to_csv(BAYESIAN_DIR / "bayesian_comparisons.csv", index=False)
                    print(f"  - Bayesian comparisons saved ({len(bayesian_summary_df)} results)")
                    
                    # Focus on effect sizes
                    effect_sizes_df = bayesian_summary_df.copy()
                    effect_sizes_df.to_csv(BAYESIAN_DIR / "bayesian_effect_sizes.csv", index=False)
                    print(f"  - Bayesian effect sizes saved")
                else:
                    print("  - No Bayesian results generated")
        
        # Detailed analysis of FDR-significant results
        fdr_bayesian_file = BAYESIAN_DIR / "fdr_significant_bayesian.csv"
        if fdr_bayesian_file.exists():
            print(f"  - Found existing FDR-significant Bayesian results")
            print("  - Skipping FDR-significant analysis (delete file to recalculate)")
        elif all_comparisons and 'Significant_Adjusted' in comparisons_combined.columns:
            print("  - Running Bayesian analysis on FDR-significant results...")
            fdr_bayesian = bayesian_analysis_fdr_significant(comparisons_combined, df_clean)
            if fdr_bayesian is not None and len(fdr_bayesian) > 0:
                # Remove Trace objects before saving
                fdr_bayesian_clean = fdr_bayesian.drop(columns=['Trace'], errors='ignore')
                fdr_bayesian_clean.to_csv(BAYESIAN_DIR / "fdr_significant_bayesian.csv", index=False)
                print(f"  - FDR-significant Bayesian analysis saved ({len(fdr_bayesian_clean)} results)")
        print()
    else:
        print("Bayesian analysis skipped (PyMC/ArviZ not available)")
        print()
    
    # Visualizations
    print("Generating visualizations...")
    
    if len(recognition_df) > 0:
        plot_recognition_rates(recognition_df, FIGURES_DIR / "recognition_rates.png")
        print("  - Recognition rates plot saved")
    
    if len(remembrance_df) > 0:
        plot_remembrance_rates(remembrance_df, FIGURES_DIR / "remembrance_rates.png")
        print("  - Remembrance rates plot saved")
    
    if len(color_df) > 0:
        plot_color_associations_heatmap(color_df, FIGURES_DIR / "color_associations_heatmap.png")
        print("  - Color associations heatmap saved")
        plot_color_distributions(df_clean, FIGURES_DIR / "color_distributions.png")
        print("  - Color distributions plot saved")
    
    if all_comparisons:
        plot_comparison_summary(comparisons_combined, FIGURES_DIR / "comparison_summary.png")
        print("  - Comparison summary plot saved")
        plot_fdr_significant_results(comparisons_combined, df_clean, FIGURES_DIR / "fdr_significant_results.png")
        print("  - FDR-corrected significant results plot saved")
    
    # Bayesian visualizations
    if BAYESIAN_AVAILABLE and all_comparisons:
        print("Generating Bayesian visualizations...")
        
        # Load Bayesian results if available
        bayesian_file = BAYESIAN_DIR / "bayesian_comparisons.csv"
        if bayesian_file.exists():
            bayesian_df = pd.read_csv(bayesian_file)
            
            # Effect size forest plot
            plot_bayesian_effect_sizes(bayesian_df, BAYESIAN_FIGURES_DIR / "forest_plot_effect_sizes.png")
            print("  - Bayesian effect size forest plot saved")
            
            # Frequentist vs Bayesian comparison
            plot_frequentist_vs_bayesian(comparisons_combined, bayesian_df, 
                                       BAYESIAN_FIGURES_DIR / "frequentist_vs_bayesian.png")
            print("  - Frequentist vs Bayesian comparison plot saved")
        
        # Detailed posterior plots for FDR-significant results
        fdr_bayesian_file = BAYESIAN_DIR / "fdr_significant_bayesian.csv"
        if fdr_bayesian_file.exists():
            fdr_bayesian_df = pd.read_csv(fdr_bayesian_file)
            
            # Create posterior distribution plots for each FDR-significant result
            # (Note: We can't save trace objects in CSV, so we'll create plots from summary stats)
            print("  - FDR-significant Bayesian results available")
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print()
    
    # Print summary statistics
    print("SUMMARY STATISTICS")
    print("-" * 80)
    print(f"\nTotal cases: {len(df_clean)}")
    
    if 'gender_label' in df_clean.columns:
        print(f"\nGender distribution:")
        for gender, count in demo_stats.get('gender', {}).items():
            pct = demo_stats.get('gender_pct', {}).get(gender, 0)
            print(f"  {gender}: {count} ({pct:.1f}%)")
    
    if 'international_student_status' in df_clean.columns:
        print(f"\nInternational student status:")
        for status, count in demo_stats.get('international_student', {}).items():
            pct = demo_stats.get('international_student_pct', {}).get(status, 0)
            print(f"  {status}: {count} ({pct:.1f}%)")
    
    if 'region_group' in df_clean.columns:
        print(f"\nRegion distribution:")
        for region, count in demo_stats.get('region', {}).items():
            pct = demo_stats.get('region_pct', {}).get(region, 0)
            print(f"  {region}: {count} ({pct:.1f}%)")
    
    if 'age' in demo_stats:
        age_stats = demo_stats['age']
        print(f"\nAge statistics:")
        print(f"  Mean: {age_stats['mean']:.1f} years")
        print(f"  Median: {age_stats['median']:.1f} years")
        print(f"  Range: {age_stats['min']:.0f} - {age_stats['max']:.0f} years")
        print(f"  N: {age_stats['n']}")
    
    if len(recognition_df) > 0:
        print(f"\nSymbol Recognition:")
        print(f"  Mean recognition rate: {recognition_df['Recognition_Rate'].mean():.1f}%")
        print(f"  Range: {recognition_df['Recognition_Rate'].min():.1f}% - {recognition_df['Recognition_Rate'].max():.1f}%")
    
    if len(remembrance_df) > 0:
        print(f"\nSymbol Remembrance:")
        print(f"  Mean remembrance rate: {remembrance_df['Remembrance_Rate'].mean():.1f}%")
        print(f"  Range: {remembrance_df['Remembrance_Rate'].min():.1f}% - {remembrance_df['Remembrance_Rate'].max():.1f}%")
    
    if all_comparisons:
        sig_count = comparisons_combined['Significant'].sum()
        print(f"\nCross-Comparisons:")
        print(f"  Total comparisons: {len(comparisons_combined)}")
        print(f"  Significant (p < 0.05): {sig_count}")
        if 'Significant_Adjusted' in comparisons_combined.columns:
            sig_adj_count = comparisons_combined['Significant_Adjusted'].sum()
            print(f"  Significant (after FDR correction): {sig_adj_count}")


if __name__ == "__main__":
    main()


# %%
# Todo not ready implemented .... Perhaps to combine this with the other classes into one script

# %%
import scipy.stats as stats
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import os

sys.path.append(os.path.abspath(os.path.join('../..')))

from lie_logit_lens_statistics import AnalysisResult

# %%
@dataclass
class StatisticalTest:
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]

class HypothesisTester:
    def __init__(self, truth_results: List[AnalysisResult], lie_results: List[AnalysisResult]):
        """Initialize with analysis results from both conditions."""
        self.truth_results = truth_results
        self.lie_results = lie_results
        self.num_layers = len(truth_results[0].metrics_per_layer)
        
    def extract_metric_arrays(self, metric_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract metric values as numpy arrays for both conditions."""
        truth_values = np.array([[getattr(layer_metrics, metric_name) 
                                for layer_metrics in result.metrics_per_layer]
                               for result in self.truth_results])
        
        lie_values = np.array([[getattr(layer_metrics, metric_name) 
                              for layer_metrics in result.metrics_per_layer]
                              for result in self.lie_results])
        
        return truth_values, lie_values
    
    def test_h1_entropy_difference(self) -> Dict[int, StatisticalTest]:
        """
        Test H1: Entropy is higher in lying condition.
        Performs layer-wise Mann-Whitney U tests.
        """
        truth_entropy, lie_entropy = self.extract_metric_arrays('entropy')
        results = {}
        
        for layer in range(self.num_layers):
            # Perform Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(
                truth_entropy[:, layer],
                lie_entropy[:, layer],
                alternative='less'  # H1: truth < lie
            )
            
            # Calculate effect size (Cliff's delta)
            effect_size = 2 * statistic / (len(truth_entropy) * len(lie_entropy)) - 1
            
            # Bootstrap confidence interval for median difference
            ci = self._bootstrap_ci(
                lie_entropy[:, layer],
                truth_entropy[:, layer],
                np.median,
                n_bootstrap=1000
            )
            
            results[layer] = StatisticalTest(
                test_name="Mann-Whitney U",
                statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=ci
            )
            
        return results
    
    def test_h2_kl_divergence_timing(self) -> StatisticalTest:
        """
        Test H2: KL divergence drops later in lying condition.
        Analyzes the layer index where significant drop occurs.
        """
        truth_kl, lie_kl = self.extract_metric_arrays('kl_divergence')
        
        # Find layer indices where KL divergence starts dropping
        def find_drop_layer(kl_values):
            # Use rolling mean to smooth
            smoothed = np.apply_along_axis(
                lambda x: np.convolve(x, np.ones(3)/3, mode='valid'),
                axis=1,
                arr=kl_values
            )
            # Find where derivative becomes negative
            drops = np.diff(smoothed) < 0
            return np.argmax(drops, axis=1)
        
        truth_drops = find_drop_layer(truth_kl)
        lie_drops = find_drop_layer(lie_kl)
        
        # Compare timing distributions
        statistic, p_value = stats.ttest_ind(truth_drops, lie_drops)
        effect_size = (np.mean(lie_drops) - np.mean(truth_drops)) / np.sqrt(
            (np.var(truth_drops) + np.var(lie_drops)) / 2
        )
        
        ci = self._bootstrap_ci(lie_drops, truth_drops, np.mean, n_bootstrap=1000)
        
        return StatisticalTest(
            test_name="Independent t-test",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci
        )
    
    def test_h3_token_probability_timing(self) -> StatisticalTest:
        """
        Test H3: Target token probability rises earlier in truth condition.
        Analyzes the layer index where probability starts rising significantly.
        """
        truth_prob, lie_prob = self.extract_metric_arrays('target_probability')
        
        # Find layer indices where probability starts rising
        def find_rise_layer(prob_values, threshold=0.1):
            rises = prob_values > threshold
            return np.argmax(rises, axis=1)
        
        truth_rises = find_rise_layer(truth_prob)
        lie_rises = find_rise_layer(lie_prob)
        
        statistic, p_value = stats.ttest_ind(truth_rises, lie_rises)
        effect_size = (np.mean(lie_rises) - np.mean(truth_rises)) / np.sqrt(
            (np.var(truth_rises) + np.var(lie_rises)) / 2
        )
        
        ci = self._bootstrap_ci(truth_rises, lie_rises, np.mean, n_bootstrap=1000)
        
        return StatisticalTest(
            test_name="Independent t-test",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci
        )
    
    def test_h4_entropy_layer_correlation(self) -> Tuple[StatisticalTest, StatisticalTest]:
        """
        Test H4: Entropy difference increases with layer depth.
        Uses regression analysis to test for increasing trend.
        """
        truth_entropy, lie_entropy = self.extract_metric_arrays('entropy')
        
        # Calculate entropy differences
        entropy_diff = np.median(lie_entropy, axis=0) - np.median(truth_entropy, axis=0)
        layers = np.arange(self.num_layers).reshape(-1, 1)
        
        # Fit regression
        reg = LinearRegression().fit(layers, entropy_diff)
        
        # Calculate p-value for slope
        from scipy import stats
        slope = reg.coef_[0]
        n = self.num_layers
        
        # Calculate standard error of slope
        y_pred = reg.predict(layers)
        mse = np.sum((entropy_diff - y_pred) ** 2) / (n - 2)
        std_err = np.sqrt(mse / np.sum((layers - np.mean(layers)) ** 2))
        
        t_stat = slope / std_err
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        regression_test = StatisticalTest(
            test_name="Linear Regression",
            statistic=t_stat,
            p_value=p_value,
            effect_size=reg.score(layers, entropy_diff),  # R²
            confidence_interval=(slope - 1.96 * std_err, slope + 1.96 * std_err)
        )
        
        return regression_test
    
    def _bootstrap_ci(self, 
                     x: np.ndarray, 
                     y: np.ndarray, 
                     statistic: callable,
                     n_bootstrap: int = 1000,
                     confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for difference in statistic."""
        boot_stats = []
        for _ in range(n_bootstrap):
            x_boot = np.random.choice(x, size=len(x), replace=True)
            y_boot = np.random.choice(y, size=len(y), replace=True)
            boot_stats.append(statistic(x_boot) - statistic(y_boot))
            
        return np.percentile(boot_stats, [(1-confidence)*50, (1+confidence)*50])

# Usage example:
def run_hypothesis_tests(truth_results: List[AnalysisResult], 
                        lie_results: List[AnalysisResult],
                        alpha: float = 0.05):
    """Run all hypothesis tests and print results."""
    tester = HypothesisTester(truth_results, lie_results)
    
    print("Running pre-registered hypothesis tests...")
    
    # Test H1: Entropy difference
    h1_results = tester.test_h1_entropy_difference()
    print("\nH1: Entropy differences by layer:")
    for layer, result in h1_results.items():
        if result.p_value < alpha:
            print(f"Layer {layer}: Significant difference (p={result.p_value:.4f}, "
                  f"effect size={result.effect_size:.2f})")
            
    # Test H2: KL divergence timing
    h2_result = tester.test_h2_kl_divergence_timing()
    print("\nH2: KL divergence timing:")
    print(f"p-value: {h2_result.p_value:.4f}")
    print(f"Effect size: {h2_result.effect_size:.2f}")
    print(f"CI: {h2_result.confidence_interval}")
    
    # Test H3: Token probability timing
    h3_result = tester.test_h3_token_probability_timing()
    print("\nH3: Token probability timing:")
    print(f"p-value: {h3_result.p_value:.4f}")
    print(f"Effect size: {h3_result.effect_size:.2f}")
    print(f"CI: {h3_result.confidence_interval}")
    
    # Test H4: Entropy layer correlation
    h4_result = tester.test_h4_entropy_layer_correlation()
    print("\nH4: Entropy layer correlation:")
    print(f"p-value: {h4_result.p_value:.4f}")
    print(f"R²: {h4_result.effect_size:.2f}")
    print(f"Slope CI: {h4_result.confidence_interval}")
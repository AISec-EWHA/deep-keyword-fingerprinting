import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class Histogram:
    def __init__(self):
        self.iat_distribution = defaultdict(int)

    def random_iat_from_distribution(self, percentile):
        """Sample an iat value from the updated distribution, considering the specified percentile."""
        total_iats = sum(self.iat_distribution.values())
        if total_iats == 0:
            return 1
        
        iat_probs = [(iat, count / total_iats) for iat, count in self.iat_distribution.items()]
        iats, probs = zip(*iat_probs)
        
        sorted_iats = sorted(iats)
        threshold_idx = max(1, int(len(sorted_iats) * (percentile / 100.0))) - 1
        threshold = sorted_iats[threshold_idx]
        
        filtered_iats = [iat for iat in iats if iat <= threshold]
        if not filtered_iats:
            return 1
        
        filtered_probs = [prob for iat, prob in iat_probs if iat <= threshold]
        filtered_total = sum(filtered_probs)
        filtered_probs = [prob / filtered_total for prob in filtered_probs]
        
        return np.random.choice(filtered_iats, p=filtered_probs)

    def update_iat_distribution(self, index, trace):
        """Update the iat distribution with the latest inter-arrival time."""
        if index > 0:
            prev_packet = trace[index - 1]
            curr_packet = trace[index]
            if curr_packet.direction == prev_packet.direction:
                iat = curr_packet.timestamp - prev_packet.timestamp
                self.iat_distribution[iat] += 1

    def reset_distribution(self):
        """Reset the iat distribution."""
        self.iat_distribution = defaultdict(int)


class Histogram_0:
    def __init__(self, mean=0.008305, std_dev=0.046445, num_samples=1000, bin_size=50):
        self.mean = mean
        self.std_dev = std_dev
        self.num_samples = num_samples
        self.bin_size = bin_size
        self.histogram = self.create_histogram()
    
    def create_histogram(self):
        """Generate histogram based on normal distribution."""
        samples = np.random.normal(self.mean, self.std_dev, self.num_samples)
        
        samples = samples[samples > 0]

        counts, bins = np.histogram(samples, bins=self.create_bins())
        histogram_dict = dict(zip(bins, counts))
        histogram_dict = self.adjust_histogram(histogram_dict, bins)
        return histogram_dict
    
    def create_bins(self):
        """Create bins for histogram based on the range of samples."""
        min_bin = 0
        max_bin = np.max(np.random.normal(self.mean, self.std_dev, self.num_samples))
        bin_edges = np.linspace(min_bin, max_bin, self.bin_size)
        return bin_edges
    
    def adjust_histogram(self, histogram_dict, bins):
        """Adjust histogram dictionary to ensure it has the right structure."""
        bin_edges = list(bins)
        histogram_dict = dict(histogram_dict)
        # Ensure all bins are represented, including edges
        for edge in bin_edges:
            if edge not in histogram_dict:
                histogram_dict[edge] = 0
        return dict(sorted(histogram_dict.items()))
    
    def random_iat_from_distribution(self, percentile):
        """Sample an IAT value from the updated distribution, considering the specified percentile."""
        samples = np.random.normal(self.mean, self.std_dev, self.num_samples)
        samples = samples[samples > 0]  # Discard non-positive samples if needed
        if len(samples) == 0:
            return 1  # Default fallback if no valid samples
        
        # Compute the value corresponding to the specified percentile
        percentile_value = np.percentile(samples, percentile)
        return percentile_value

class Histogram_normal:
    def __init__(self, mean=0.008305, std_dev=0.046445, num_samples=1000, bin_size=50):
        self.mean = mean
        self.std_dev = std_dev
        self.num_samples = num_samples
        self.bin_size = bin_size
        self.histogram = self.create_histogram()
    
    def create_histogram(self):
        """Generate histogram based on normal distribution."""
        samples = np.random.normal(self.mean, self.std_dev, self.num_samples)
        
        samples = samples[samples > 0]

        counts, bins = np.histogram(samples, bins=self.create_bins())
        histogram_dict = dict(zip(bins, counts))
        histogram_dict = self.adjust_histogram(histogram_dict, bins)
        return histogram_dict
    
    def create_bins(self):
        """Create bins for histogram based on the range of samples."""
        min_bin = 0
        max_bin = np.max(np.random.normal(self.mean, self.std_dev, self.num_samples))
        bin_edges = np.linspace(min_bin, max_bin, self.bin_size)
        return bin_edges
    
    def adjust_histogram(self, histogram_dict, bins):
        """Adjust histogram dictionary to ensure it has the right structure."""
        bin_edges = list(bins)
        histogram_dict = dict(histogram_dict)
        for edge in bin_edges:
            if edge not in histogram_dict:
                histogram_dict[edge] = 0
        return dict(sorted(histogram_dict.items()))
    
    def random_iat_from_distribution(self, percentile):
        """Sample an IAT value from the updated distribution, considering the specified percentile."""
        samples = np.random.normal(self.mean, self.std_dev, self.num_samples)
        samples = samples[samples > 0]
        if len(samples) == 0:
            return 1
        
        percentile_value = np.percentile(samples, percentile)
        return percentile_value
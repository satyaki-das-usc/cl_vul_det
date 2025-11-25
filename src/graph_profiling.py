import torch
import numpy as np

from typing import cast
from omegaconf import DictConfig, OmegaConf
from os.path import join

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from collections import defaultdict
import matplotlib.pyplot as plt

from src.torch_data.datamodules import SliceDataModule
from src.vocabulary import Vocabulary

class GraphMemoryProfiler:
    """Profile memory usage of individual graphs and batches."""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.graph_stats = []
        
    def compute_graph_metrics(self):
        """Compute memory-related metrics for each graph."""
        for idx, data in enumerate(self.dataset):
            stats = {
                'idx': idx,
                'num_nodes': data.num_nodes,
                'num_edges': data.num_edges if hasattr(data, 'num_edges') else data.edge_index.size(1),
                'node_features': data.x.numel() if hasattr(data, 'x') else 0,
                'edge_features': data.edge_attr.numel() if hasattr(data, 'edge_attr') else 0,
            }
            
            # Estimate memory footprint (in MB)
            total_elements = stats['node_features'] + stats['edge_features']
            stats['estimated_memory_mb'] = (total_elements * 4) / (1024 ** 2)  # 4 bytes per float32
            
            self.graph_stats.append(stats)
        
        return self.graph_stats
    
    def identify_outliers(self, metric='estimated_memory_mb', threshold_percentile=95):
        """Identify graphs that are unusually large."""
        if not self.graph_stats:
            self.compute_graph_metrics()
        
        values = [s[metric] for s in self.graph_stats]
        threshold = np.percentile(values, threshold_percentile)
        
        outliers = [s for s in self.graph_stats if s[metric] > threshold]
        print(f"\nFound {len(outliers)} outliers above {threshold_percentile}th percentile:")
        print(f"Threshold: {threshold:.4f}")
        
        for outlier in sorted(outliers, key=lambda x: x[metric], reverse=True)[:10]:
            print(f"  Graph {outlier['idx']}: {outlier['num_nodes']} nodes, "
                  f"{outlier['num_edges']} edges, {outlier['estimated_memory_mb']:.2f} MB")
        
        return outliers
    
    def plot_distribution(self):
        """Visualize the distribution of graph sizes."""
        if not self.graph_stats:
            self.compute_graph_metrics()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['num_nodes', 'num_edges', 'estimated_memory_mb']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            values = [s[metric] for s in self.graph_stats]
            ax.hist(values, bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {metric.replace("_", " ").title()}')
            ax.axvline(np.percentile(values, 95), color='r', linestyle='--', 
                      label='95th percentile')
            ax.legend()
        
        # Scatter plot: nodes vs edges
        ax = axes[1, 1]
        nodes = [s['num_nodes'] for s in self.graph_stats]
        edges = [s['num_edges'] for s in self.graph_stats]
        ax.scatter(nodes, edges, alpha=0.5)
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Number of Edges')
        ax.set_title('Nodes vs Edges')
        
        plt.tight_layout()
        plt.savefig('graph_distribution.png', dpi=150)
        print("\nSaved distribution plot to 'graph_distribution.png'")
        return fig


class BatchMemoryMonitor:
    """Monitor actual GPU memory during training."""
    
    def __init__(self):
        self.batch_memory = []
        self.peak_memory = []
        
    def log_batch(self, batch_idx, epoch):
        """Log memory usage for current batch."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            
            self.batch_memory.append({
                'epoch': epoch,
                'batch': batch_idx,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'peak_gb': peak
            })
            
            return allocated, reserved, peak
        return None, None, None
    
    def find_problematic_batches(self, threshold_gb=None):
        """Identify batches with high memory usage."""
        if not self.batch_memory:
            print("No memory data logged yet.")
            return []
        
        if threshold_gb is None:
            # Use 90th percentile as threshold
            peak_values = [b['peak_gb'] for b in self.batch_memory]
            threshold_gb = np.percentile(peak_values, 90)
        
        problematic = [b for b in self.batch_memory if b['peak_gb'] > threshold_gb]
        
        print(f"\nFound {len(problematic)} batches exceeding {threshold_gb:.2f} GB:")
        for batch in sorted(problematic, key=lambda x: x['peak_gb'], reverse=True)[:10]:
            print(f"  Epoch {batch['epoch']}, Batch {batch['batch']}: "
                  f"{batch['peak_gb']:.2f} GB peak")
        
        return problematic
    
    def plot_memory_timeline(self):
        """Plot memory usage over time."""
        if not self.batch_memory:
            print("No memory data to plot.")
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        batches = list(range(len(self.batch_memory)))
        allocated = [b['allocated_gb'] for b in self.batch_memory]
        reserved = [b['reserved_gb'] for b in self.batch_memory]
        peak = [b['peak_gb'] for b in self.batch_memory]
        
        ax.plot(batches, allocated, label='Allocated', alpha=0.7)
        ax.plot(batches, reserved, label='Reserved', alpha=0.7)
        ax.plot(batches, peak, label='Peak', alpha=0.7, linestyle='--')
        
        ax.set_xlabel('Batch Index')
        ax.set_ylabel('Memory (GB)')
        ax.set_title('GPU Memory Usage Over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('memory_timeline.png', dpi=150)
        print("\nSaved memory timeline to 'memory_timeline.png'")
        return fig


def create_balanced_sampler(dataset, graph_stats, max_memory_per_batch=1.0):
    """Create a custom sampler to avoid large graphs in same batch."""
    from torch.utils.data import Sampler
    
    class BalancedGraphSampler(Sampler):
        def __init__(self, dataset, graph_stats, max_memory):
            self.dataset = dataset
            self.graph_stats = sorted(graph_stats, 
                                     key=lambda x: x['estimated_memory_mb'])
            self.max_memory = max_memory
            
        def __iter__(self):
            # Sort graphs by size
            small_graphs = [s['idx'] for s in self.graph_stats 
                          if s['estimated_memory_mb'] < self.max_memory / 2]
            large_graphs = [s['idx'] for s in self.graph_stats 
                          if s['estimated_memory_mb'] >= self.max_memory / 2]
            
            # Shuffle each group
            np.random.shuffle(small_graphs)
            np.random.shuffle(large_graphs)
            
            # Interleave to distribute large graphs across batches
            indices = []
            for i in range(max(len(small_graphs), len(large_graphs))):
                if i < len(small_graphs):
                    indices.append(small_graphs[i])
                if i < len(large_graphs):
                    indices.append(large_graphs[i])
            
            return iter(indices)
        
        def __len__(self):
            return len(self.dataset)
    
    return BalancedGraphSampler(dataset, graph_stats, max_memory_per_batch)


# Example usage
if __name__ == "__main__":
    # Assuming you have a dataset of Data objects
    # dataset = [Data(...), Data(...), ...]
    
    config = cast(DictConfig, OmegaConf.load("configs/dwk.yaml"))

    dataset_root = join(config.data_folder, config.dataset.name)
    
    vocab = Vocabulary.from_w2v(join(dataset_root, "w2v.wv"))
    vocab_size = vocab.get_vocab_size()
    data_module = SliceDataModule(config, vocab, config.hyper_parameters.batch_sizes[0], train_sampler=None, use_temp_data=False)
    # 1. Profile your graphs
    profiler = GraphMemoryProfiler(data_module.get_train_dataset())
    stats = profiler.compute_graph_metrics()
    outliers = profiler.identify_outliers(threshold_percentile=95)
    profiler.plot_distribution()
    
    # 2. Monitor memory during training
    monitor = BatchMemoryMonitor()
    
    # In your training loop:
    # for epoch in range(num_epochs):
    #     for batch_idx, batch in enumerate(train_loader):
    #         # Before forward pass
    #         if torch.cuda.is_available():
    #             torch.cuda.reset_peak_memory_stats()
    #         
    #         # Your training code here
    #         output = model(batch)
    #         loss = criterion(output, batch.y)
    #         loss.backward()
    #         optimizer.step()
    #         
    #         # Log memory after backward
    #         monitor.log_batch(batch_idx, epoch)
    
    # 3. Analyze problematic batches
    # problematic = monitor.find_problematic_batches()
    # monitor.plot_memory_timeline()
    
    # 4. Create balanced sampler if needed
    # sampler = create_balanced_sampler(dataset, stats, max_memory_per_batch=500)
    # loader = DataLoader(dataset, batch_size=32, sampler=sampler)

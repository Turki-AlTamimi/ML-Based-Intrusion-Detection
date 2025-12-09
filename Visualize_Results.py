# --------------------------------------------------------------------------------------------------------------
#
# Filename: Visualize_Results.py
# Author: Advanced IDS Visualization Tool
# Purpose: Generate comprehensive visual statistics from IDS classifier results
#
# Usage: python Visualize_Results.py -results <results_file.txt>
#
# --------------------------------------------------------------------------------------------------------------

import sys
import argparse
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

class ResultsVisualizer:
    def __init__(self, results_file):
        self.results_file = results_file
        self.classifiers = []
        self.classes = ['Normal', 'Blackhole', 'TCP-SYN', 'PortScan', 'Diversion', 'Overflow']
        self.metrics_data = {}
        
    def parse_results(self):
        """Parse the results file and extract metrics for each classifier"""
        with open(self.results_file, 'r') as f:
            content = f.read()
        
        # Split by classifier sections
        sections = re.split(r'Classifying with (.+?) \. \. \. \. \.', content)
        
        for i in range(1, len(sections), 2):
            classifier_name = sections[i].strip()
            classifier_data = sections[i + 1]
            
            self.classifiers.append(classifier_name)
            self.metrics_data[classifier_name] = self.extract_metrics(classifier_data)
    
    def extract_metrics(self, data):
        """Extract TPR, FPR, Accuracy, F1, Precision for each class"""
        metrics = {
            'TPR': [], 'FPR': [], 'Accuracy': [], 
            'F1 Score': [], 'Precision': []
        }
        
        # Extract per-class metrics
        for class_name in self.classes:
            class_section = re.search(
                rf'Reporting results for Class {class_name}\n(.+?)(?=Reporting results for Class|\nAverage TPR|$)', 
                data, re.DOTALL
            )
            
            if class_section:
                section_text = class_section.group(1)
                metrics['TPR'].append(self._extract_value(section_text, 'TPR'))
                metrics['FPR'].append(self._extract_value(section_text, 'FPR'))
                metrics['Accuracy'].append(self._extract_value(section_text, 'Accuracy'))
                metrics['F1 Score'].append(self._extract_value(section_text, 'F1 Score'))
                metrics['Precision'].append(self._extract_value(section_text, 'Precision'))
        
        # Extract average metrics
        metrics['Avg_TPR'] = self._extract_value(data, 'Average TPR')
        metrics['Avg_FPR'] = self._extract_value(data, 'Average FPR')
        metrics['Avg_Accuracy'] = self._extract_value(data, 'Average Accuracy')
        metrics['Avg_F1'] = self._extract_value(data, 'Average F1 Score')
        metrics['Avg_Precision'] = self._extract_value(data, 'Average Precision')
        
        return metrics
    
    def _extract_value(self, text, metric_name):
        """Extract numeric value for a given metric"""
        pattern = rf'{metric_name}\s*=\s*([\d.]+)'
        match = re.search(pattern, text)
        return float(match.group(1)) if match else 0.0
    
    def plot_classifier_comparison(self):
        """Compare average metrics across all classifiers"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Classifier Performance Comparison - Average Metrics', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        metrics_to_plot = [
            ('Avg_TPR', 'Average TPR (True Positive Rate)', 'Greens'),
            ('Avg_FPR', 'Average FPR (False Positive Rate)', 'Reds'),
            ('Avg_Accuracy', 'Average Accuracy', 'Blues'),
            ('Avg_F1', 'Average F1 Score', 'Purples'),
            ('Avg_Precision', 'Average Precision', 'Oranges')
        ]
        
        for idx, (metric, title, color_palette) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            values = [self.metrics_data[clf][metric] for clf in self.classifiers]
            
            # Use matplotlib colormaps instead of seaborn palettes
            cmap = plt.cm.get_cmap(color_palette)
            colors = [cmap(0.3 + 0.5 * i / len(self.classifiers)) for i in range(len(self.classifiers))]
            
            bars = ax.bar(range(len(self.classifiers)), values, color=colors)
            ax.set_xlabel('Classifier', fontweight='bold')
            ax.set_ylabel(title.split(' - ')[0], fontweight='bold')
            ax.set_title(title, fontweight='bold', pad=10)
            ax.set_xticks(range(len(self.classifiers)))
            ax.set_xticklabels(self.classifiers, rotation=45, ha='right', fontsize=9)
            ax.set_ylim([0, 1.1])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Hide the last subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('classifier_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: classifier_comparison.png")
        plt.close()
    
    def plot_heatmaps(self):
        """Create heatmaps for all metrics across classifiers and classes"""
        metrics = ['TPR', 'FPR', 'Accuracy', 'F1 Score', 'Precision']
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Performance Heatmaps - All Metrics by Classifier and Attack Class', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            # Create data matrix
            data_matrix = np.array([
                self.metrics_data[clf][metric] for clf in self.classifiers
            ])
            
            # Create DataFrame for better labeling
            df = pd.DataFrame(data_matrix, 
                            index=self.classifiers, 
                            columns=self.classes)
            
            # Choose colormap based on metric
            cmap = 'RdYlGn' if metric != 'FPR' else 'RdYlGn_r'
            
            sns.heatmap(df, annot=True, fmt='.3f', cmap=cmap, 
                       cbar_kws={'label': metric}, ax=ax, 
                       vmin=0, vmax=1, linewidths=0.5)
            ax.set_title(f'{metric} by Classifier and Class', 
                        fontweight='bold', pad=10)
            ax.set_xlabel('Attack Class', fontweight='bold')
            ax.set_ylabel('Classifier', fontweight='bold')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Hide the last subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('performance_heatmaps.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: performance_heatmaps.png")
        plt.close()
    
    def plot_radar_charts(self):
        """Create radar charts for top 4 classifiers"""
        # Get top 4 classifiers by average F1 score
        clf_scores = [(clf, self.metrics_data[clf]['Avg_F1']) 
                      for clf in self.classifiers]
        clf_scores.sort(key=lambda x: x[1], reverse=True)
        top_classifiers = [clf[0] for clf in clf_scores[:4]]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 16), 
                                subplot_kw=dict(projection='polar'))
        fig.suptitle('Radar Charts - Per-Class Performance (Top 4 Classifiers)', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        categories = self.classes
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        for idx, clf in enumerate(top_classifiers):
            ax = axes[idx // 2, idx % 2]
            
            # Plot TPR, F1, and Precision
            metrics_to_plot = {
                'TPR': self.metrics_data[clf]['TPR'],
                'F1 Score': self.metrics_data[clf]['F1 Score'],
                'Precision': self.metrics_data[clf]['Precision']
            }
            
            for metric_name, values in metrics_to_plot.items():
                values_plot = values + values[:1]
                ax.plot(angles, values_plot, 'o-', linewidth=2, label=metric_name)
                ax.fill(angles, values_plot, alpha=0.15)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=10)
            ax.set_ylim(0, 1)
            ax.set_title(clf, fontweight='bold', size=12, pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('radar_charts_top4.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: radar_charts_top4.png")
        plt.close()
    
    def plot_class_difficulty(self):
        """Analyze which attack classes are hardest to detect"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle('Attack Class Detection Difficulty Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Calculate average metrics per class across all classifiers
        avg_metrics_per_class = {class_name: [] for class_name in self.classes}
        
        for class_idx, class_name in enumerate(self.classes):
            for clf in self.classifiers:
                avg_metrics_per_class[class_name].append(
                    self.metrics_data[clf]['F1 Score'][class_idx]
                )
        
        class_avg_f1 = {k: np.mean(v) for k, v in avg_metrics_per_class.items()}
        class_std_f1 = {k: np.std(v) for k, v in avg_metrics_per_class.items()}
        
        # Plot 1: Average F1 Score by Class
        ax1 = axes[0]
        classes_sorted = sorted(class_avg_f1.items(), key=lambda x: x[1])
        classes_names = [x[0] for x in classes_sorted]
        f1_scores = [x[1] for x in classes_sorted]
        std_scores = [class_std_f1[x[0]] for x in classes_sorted]
        
        colors = ['#d62728' if f1 < 0.5 else '#ff7f0e' if f1 < 0.7 else '#2ca02c' 
                  for f1 in f1_scores]
        
        bars = ax1.barh(classes_names, f1_scores, color=colors, alpha=0.7)
        ax1.errorbar(f1_scores, classes_names, xerr=std_scores, 
                    fmt='none', ecolor='black', capsize=5, alpha=0.5)
        ax1.set_xlabel('Average F1 Score', fontweight='bold')
        ax1.set_ylabel('Attack Class', fontweight='bold')
        ax1.set_title('Detection Difficulty by Class\n(Lower = Harder to Detect)', 
                     fontweight='bold')
        ax1.set_xlim([0, 1])
        ax1.grid(axis='x', alpha=0.3)
        
        for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
            ax1.text(f1 + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{f1:.3f}', va='center', fontweight='bold')
        
        # Plot 2: Box plot of F1 scores per class
        ax2 = axes[1]
        data_for_boxplot = [avg_metrics_per_class[class_name] 
                           for class_name in self.classes]
        bp = ax2.boxplot(data_for_boxplot, labels=self.classes, patch_artist=True)
        
        for patch, median in zip(bp['boxes'], bp['medians']):
            patch.set_facecolor('#3498db')
            patch.set_alpha(0.6)
        
        ax2.set_ylabel('F1 Score', fontweight='bold')
        ax2.set_xlabel('Attack Class', fontweight='bold')
        ax2.set_title('F1 Score Distribution Across Classifiers', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('class_difficulty_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: class_difficulty_analysis.png")
        plt.close()
    
    def plot_tpr_fpr_tradeoff(self):
        """Plot TPR vs FPR tradeoff for each classifier"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = sns.color_palette("husl", len(self.classifiers))
        
        for idx, clf in enumerate(self.classifiers):
            avg_tpr = self.metrics_data[clf]['Avg_TPR']
            avg_fpr = self.metrics_data[clf]['Avg_FPR']
            
            ax.scatter(avg_fpr, avg_tpr, s=300, alpha=0.6, 
                      color=colors[idx], edgecolors='black', linewidth=2,
                      label=clf)
            ax.annotate(clf, (avg_fpr, avg_tpr), 
                       xytext=(10, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')
        
        # Add diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
        
        # Shade the ideal region
        ax.fill_between([0, 0.2], [0.8, 0.8], [1, 1], 
                        alpha=0.1, color='green', label='Ideal Region')
        
        ax.set_xlabel('Average False Positive Rate (FPR)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Average True Positive Rate (TPR)', fontweight='bold', fontsize=12)
        ax.set_title('TPR vs FPR Tradeoff - Classifier Comparison\n(Closer to top-left is better)', 
                    fontweight='bold', fontsize=14, pad=15)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        
        plt.tight_layout()
        plt.savefig('tpr_fpr_tradeoff.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: tpr_fpr_tradeoff.png")
        plt.close()
    
    def create_summary_report(self):
        """Create a summary statistics report"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.3, top=0.94, bottom=0.05)
        
        fig.suptitle('IDS Classifier Performance Summary Report', 
                     fontsize=18, fontweight='bold', y=0.97)
        
        # Best classifier by each metric
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        metrics_list = ['Avg_TPR', 'Avg_FPR', 'Avg_Accuracy', 'Avg_F1', 'Avg_Precision']
        metric_names = ['TPR', 'FPR', 'Accuracy', 'F1 Score', 'Precision']
        
        summary_text = "BEST CLASSIFIERS BY METRIC\n" + "="*80 + "\n\n"
        
        for metric, name in zip(metrics_list, metric_names):
            if metric == 'Avg_FPR':
                best_clf = min(self.classifiers, 
                              key=lambda x: self.metrics_data[x][metric])
                best_val = self.metrics_data[best_clf][metric]
                summary_text += f"Lowest {name:12s}: {best_clf:30s} = {best_val:.4f}\n"
            else:
                best_clf = max(self.classifiers, 
                              key=lambda x: self.metrics_data[x][metric])
                best_val = self.metrics_data[best_clf][metric]
                summary_text += f"Best {name:12s}: {best_clf:30s} = {best_val:.4f}\n"
        
        ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Top 3 classifiers overall
        ax2 = fig.add_subplot(gs[1, :])
        ax2.axis('off')
        
        overall_scores = []
        for clf in self.classifiers:
            score = (
                self.metrics_data[clf]['Avg_F1'] * 0.4 +
                self.metrics_data[clf]['Avg_TPR'] * 0.3 +
                self.metrics_data[clf]['Avg_Accuracy'] * 0.2 +
                (1 - self.metrics_data[clf]['Avg_FPR']) * 0.1
            )
            overall_scores.append((clf, score))
        
        overall_scores.sort(key=lambda x: x[1], reverse=True)
        
        ranking_text = "\nOVERALL RANKING (Weighted Score)\n" + "="*80 + "\n\n"
        for rank, (clf, score) in enumerate(overall_scores[:5], 1):
            metrics = self.metrics_data[clf]
            ranking_text += f"{rank}. {clf:30s} (Score: {score:.4f})\n"
            ranking_text += f"   TPR: {metrics['Avg_TPR']:.3f} | "
            ranking_text += f"FPR: {metrics['Avg_FPR']:.3f} | "
            ranking_text += f"F1: {metrics['Avg_F1']:.3f} | "
            ranking_text += f"Acc: {metrics['Avg_Accuracy']:.3f}\n\n"
        
        ax2.text(0.05, 0.95, ranking_text, transform=ax2.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # Hardest to detect classes
        ax3 = fig.add_subplot(gs[2, :])
        ax3.axis('off')
        
        avg_f1_per_class = {}
        for class_idx, class_name in enumerate(self.classes):
            scores = [self.metrics_data[clf]['F1 Score'][class_idx] 
                     for clf in self.classifiers]
            avg_f1_per_class[class_name] = np.mean(scores)
        
        sorted_classes = sorted(avg_f1_per_class.items(), key=lambda x: x[1])
        
        difficulty_text = "\nATTACK DETECTION DIFFICULTY\n" + "="*80 + "\n\n"
        difficulty_text += "Hardest to Detect:\n"
        for class_name, f1 in sorted_classes[:3]:
            difficulty_text += f"  • {class_name:15s}: Avg F1 = {f1:.3f}\n"
        
        difficulty_text += "\nEasiest to Detect:\n"
        for class_name, f1 in reversed(sorted_classes[-3:]):
            difficulty_text += f"  • {class_name:15s}: Avg F1 = {f1:.3f}\n"
        
        ax3.text(0.05, 0.95, difficulty_text, transform=ax3.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        plt.savefig('summary_report.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: summary_report.png")
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "="*60)
        print("IDS Results Visualization Tool")
        print("="*60)
        print(f"\nParsing results from: {self.results_file}")
        
        self.parse_results()
        print(f"Found {len(self.classifiers)} classifiers")
        print(f"Classes analyzed: {', '.join(self.classes)}\n")
        
        print("Generating visualizations...")
        print("-" * 60)
        
        self.plot_classifier_comparison()
        self.plot_heatmaps()
        self.plot_radar_charts()
        self.plot_class_difficulty()
        self.plot_tpr_fpr_tradeoff()
        self.create_summary_report()
        
        print("-" * 60)
        print("\n✓ All visualizations generated successfully!")
        print("\nGenerated files:")
        print("  1. classifier_comparison.png")
        print("  2. performance_heatmaps.png")
        print("  3. radar_charts_top4.png")
        print("  4. class_difficulty_analysis.png")
        print("  5. tpr_fpr_tradeoff.png")
        print("  6. summary_report.png")
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize IDS classifier performance results'
    )
    parser.add_argument('-results', type=str, required=True,
                       help='Path to results text file (e.g., data.csv_50.results.txt)')
    
    args = parser.parse_args()
    
    try:
        visualizer = ResultsVisualizer(args.results)
        visualizer.generate_all_visualizations()
    except FileNotFoundError:
        print(f"Error: File '{args.results}' not found!")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
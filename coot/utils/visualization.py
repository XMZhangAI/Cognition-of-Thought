"""
Visualization utilities for COOT analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Any
import pandas as pd


def plot_attention_timeline(attention_data: Dict, 
                          save_path: Optional[str] = None,
                          figsize: tuple = (12, 8)) -> plt.Figure:
    """
    Plot attention timeline showing sharpness scores over generation steps
    
    Args:
        attention_data: Dictionary with 'steps', 'sharpness_scores', etc.
        save_path: Optional path to save the plot
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if not attention_data or 'steps' not in attention_data:
        raise ValueError("Invalid attention data provided")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    steps = attention_data['steps']
    sharpness_scores = attention_data['sharpness_scores']
    max_weights = attention_data.get('max_weights', [])
    entropies = attention_data.get('entropies', [])
    
    # Plot sharpness scores
    ax1.plot(steps, sharpness_scores, 'b-', marker='o', markersize=4, linewidth=2)
    ax1.axhline(y=0.7, color='r', linestyle='--', alpha=0.7, label='Rollback Threshold')
    ax1.set_ylabel('Sharpness Score')
    ax1.set_title('Attention Analysis Timeline')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Plot max weights if available
    if max_weights:
        ax2.plot(steps, max_weights, 'g-', marker='s', markersize=3, linewidth=2)
        ax2.set_ylabel('Max Attention Weight')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
    
    # Plot entropy if available
    if entropies:
        ax3.plot(steps, entropies, 'm-', marker='^', markersize=3, linewidth=2)
        ax3.set_ylabel('Attention Entropy')
        ax3.set_xlabel('Generation Step')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_intervention_analysis(intervention_traces: List[Dict],
                             save_path: Optional[str] = None,
                             figsize: tuple = (15, 10)) -> plt.Figure:
    """
    Plot comprehensive intervention analysis
    
    Args:
        intervention_traces: List of intervention trace dictionaries
        save_path: Optional path to save the plot
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if not intervention_traces:
        print("No intervention traces to plot")
        return None
    
    fig = plt.figure(figsize=figsize)
    
    # Create subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Intervention timeline
    ax1 = fig.add_subplot(gs[0, :])
    steps = [trace['step'] for trace in intervention_traces]
    success = [1 if trace['success'] else 0 for trace in intervention_traces]
    
    colors = ['green' if s else 'red' for s in success]
    ax1.scatter(steps, range(len(steps)), c=colors, s=100, alpha=0.7)
    ax1.set_xlabel('Generation Step')
    ax1.set_ylabel('Intervention #')
    ax1.set_title('Intervention Timeline (Green=Success, Red=Failure)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Violation type distribution
    ax2 = fig.add_subplot(gs[1, 0])
    violation_counts = {'Safety': 0, 'Altruism': 0, 'Egoism': 0}
    
    for trace in intervention_traces:
        state = trace['trigger_state']
        if state[0] == -1:  # Safety violation
            violation_counts['Safety'] += 1
        if state[1] == -1:  # Altruism violation
            violation_counts['Altruism'] += 1
        if state[2] == -1:  # Egoism violation
            violation_counts['Egoism'] += 1
    
    ax2.pie(violation_counts.values(), labels=violation_counts.keys(), 
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Violation Types')
    
    # 3. Success rate by intervention type
    ax3 = fig.add_subplot(gs[1, 1])
    intervention_types = {}
    
    for trace in intervention_traces:
        int_type = trace['intervention_type']
        if int_type not in intervention_types:
            intervention_types[int_type] = {'success': 0, 'total': 0}
        
        intervention_types[int_type]['total'] += 1
        if trace['success']:
            intervention_types[int_type]['success'] += 1
    
    types = list(intervention_types.keys())
    success_rates = [intervention_types[t]['success'] / intervention_types[t]['total'] 
                    for t in types]
    
    ax3.bar(types, success_rates, color=['skyblue', 'lightcoral'])
    ax3.set_ylabel('Success Rate')
    ax3.set_title('Success Rate by Type')
    ax3.set_ylim(0, 1)
    
    # 4. Rollback distance distribution
    ax4 = fig.add_subplot(gs[1, 2])
    rollback_distances = [trace['step'] - trace['rollback_step'] for trace in intervention_traces]
    
    ax4.hist(rollback_distances, bins=10, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_xlabel('Rollback Distance')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Rollback Distance Distribution')
    
    # 5. Tokens regenerated distribution
    ax5 = fig.add_subplot(gs[2, 0])
    tokens_regenerated = [trace['tokens_regenerated'] for trace in intervention_traces]
    
    ax5.hist(tokens_regenerated, bins=10, alpha=0.7, color='purple', edgecolor='black')
    ax5.set_xlabel('Tokens Regenerated')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Tokens Regenerated Distribution')
    
    # 6. Intervention effectiveness over time
    ax6 = fig.add_subplot(gs[2, 1:])
    
    # Moving average of success rate
    window_size = max(3, len(intervention_traces) // 10)
    success_series = pd.Series(success)
    moving_avg = success_series.rolling(window=window_size, min_periods=1).mean()
    
    ax6.plot(steps, moving_avg, 'b-', linewidth=2, label=f'Moving Avg (window={window_size})')
    ax6.scatter(steps, success, alpha=0.5, c=colors, s=30)
    ax6.set_xlabel('Generation Step')
    ax6.set_ylabel('Success Rate')
    ax6.set_title('Intervention Effectiveness Over Time')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 1.1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_state_distribution(state_statistics: Dict,
                          save_path: Optional[str] = None,
                          figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Plot distribution of cognitive states
    
    Args:
        state_statistics: State statistics dictionary
        save_path: Optional path to save the plot
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Violation counts
    violation_counts = state_statistics.get('violation_counts', {})
    
    if violation_counts:
        ax1.bar(violation_counts.keys(), violation_counts.values(), 
               color=['red', 'orange', 'yellow'], alpha=0.7)
        ax1.set_ylabel('Count')
        ax1.set_title('Violations by Law Type')
        ax1.tick_params(axis='x', rotation=45)
    
    # Overall statistics
    total_states = state_statistics.get('total_states', 0)
    interventions = state_statistics.get('interventions', 0)
    intervention_rate = state_statistics.get('intervention_rate', 0)
    
    stats_data = {
        'Total States': total_states,
        'Interventions': interventions,
        'Safe States': total_states - interventions
    }
    
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    ax2.pie(stats_data.values(), labels=stats_data.keys(), 
           autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title(f'State Distribution\n(Intervention Rate: {intervention_rate:.1%})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_comparative_analysis(standard_results: List[str],
                            coot_results: List[tuple],
                            save_path: Optional[str] = None,
                            figsize: tuple = (12, 8)) -> plt.Figure:
    """
    Plot comparative analysis between standard and COOT generation
    
    Args:
        standard_results: List of standard generation results
        coot_results: List of (response, traces) tuples from COOT
        save_path: Optional path to save the plot
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Response length comparison
    standard_lengths = [len(response.split()) for response in standard_results]
    coot_lengths = [len(response.split()) for response, _ in coot_results]
    
    ax1.hist([standard_lengths, coot_lengths], bins=10, alpha=0.7, 
            label=['Standard', 'COOT'], color=['blue', 'red'])
    ax1.set_xlabel('Response Length (words)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Response Length Distribution')
    ax1.legend()
    
    # 2. Intervention frequency
    intervention_counts = [len(traces['intervention_traces']) for _, traces in coot_results]
    
    ax2.hist(intervention_counts, bins=max(1, max(intervention_counts) + 1), 
            alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Number of Interventions')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Intervention Frequency Distribution')
    
    # 3. Average metrics comparison
    avg_standard_length = np.mean(standard_lengths)
    avg_coot_length = np.mean(coot_lengths)
    avg_interventions = np.mean(intervention_counts)
    
    metrics = ['Avg Length\n(Standard)', 'Avg Length\n(COOT)', 'Avg Interventions\n(COOT)']
    values = [avg_standard_length, avg_coot_length, avg_interventions]
    colors = ['blue', 'red', 'orange']
    
    ax3.bar(metrics, values, color=colors, alpha=0.7)
    ax3.set_ylabel('Value')
    ax3.set_title('Average Metrics Comparison')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Safety intervention rate over prompts
    prompt_numbers = list(range(1, len(coot_results) + 1))
    cumulative_interventions = np.cumsum(intervention_counts)
    intervention_rates = cumulative_interventions / prompt_numbers
    
    ax4.plot(prompt_numbers, intervention_rates, 'g-', marker='o', linewidth=2)
    ax4.set_xlabel('Prompt Number')
    ax4.set_ylabel('Cumulative Intervention Rate')
    ax4.set_title('Intervention Rate Trend')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_intervention_heatmap(intervention_traces: List[Dict],
                              save_path: Optional[str] = None,
                              figsize: tuple = (10, 8)) -> plt.Figure:
    """
    Create a heatmap showing intervention patterns
    
    Args:
        intervention_traces: List of intervention trace dictionaries
        save_path: Optional path to save the plot
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if not intervention_traces:
        print("No intervention traces to plot")
        return None
    
    # Create matrix: steps vs violation types
    steps = [trace['step'] for trace in intervention_traces]
    max_step = max(steps) if steps else 1
    
    # Create heatmap data
    heatmap_data = np.zeros((3, max_step + 1))  # 3 laws x steps
    
    for trace in intervention_traces:
        step = trace['step']
        state = trace['trigger_state']
        
        if state[0] == -1:  # Safety violation
            heatmap_data[0, step] += 1
        if state[1] == -1:  # Altruism violation
            heatmap_data[1, step] += 1
        if state[2] == -1:  # Egoism violation
            heatmap_data[2, step] += 1
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(heatmap_data, cmap='Reds', aspect='auto', interpolation='nearest')
    
    # Set labels
    ax.set_xlabel('Generation Step')
    ax.set_ylabel('Law Type')
    ax.set_title('Intervention Heatmap by Step and Law Type')
    
    # Set ticks
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Safety', 'Altruism', 'Egoism'])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Intervention Count')
    
    # Add text annotations
    for i in range(3):
        for j in range(max_step + 1):
            if heatmap_data[i, j] > 0:
                text = ax.text(j, i, int(heatmap_data[i, j]),
                             ha="center", va="center", color="white", fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

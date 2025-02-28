import json
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

def load_trajectories(file_name):
    trajectories = []
    with open(file_name, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                trajectory = eval(line)
                trajectories.append(trajectory)
    return trajectories

def extract_sa_sequences(trajectories, sequence_length=10):
    sa_sequences = []
    for trajectory in trajectories:
        for i in range(0, len(trajectory) - sequence_length + 1, sequence_length):
            sequence = trajectory[i:i + sequence_length]
            sa_sequences.append(sequence)
    return sa_sequences

def preprocess_sequences(sequences):
    """
    Convert sequences into numerical format suitable for clustering.
    Each state-action pair should be represented as a numerical feature.
    """
    numeric_sequences = []
    for seq in sequences:
        numeric_seq = []
        for state, action in seq:
            numeric_seq.append([state, action])
        numeric_sequences.append(numeric_seq)

    return np.array(numeric_sequences)

def cluster_sequences(sequences):
    sequences = preprocess_sequences(sequences)
    sequences = TimeSeriesScalerMeanVariance().fit_transform(sequences)  
    
    clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=5)
    labels = clusterer.fit_predict(sequences.reshape(len(sequences), -1))
    
    return labels, sequences

def visualize_clusters(labels, sequences):
    unique_clusters = set(labels)
    state_colors = {}
    grid_width = 19
    grid_height = 23
    
    colormap = plt.get_cmap("tab10")
    cluster_colors = {cluster: colormap(i % 10) for i, cluster in enumerate(unique_clusters) if cluster != -1}
    
    for label, sequence in zip(labels, sequences):
        if label == -1:
            continue 
        
        for state_action in sequence:
            state = int(state_action[0])
            if state not in state_colors:
                state_colors[state] = []
            state_colors[state].append(cluster_colors[label])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    for state, colors in state_colors.items():
        x = state % grid_width  # Column index
        y = state // grid_width  # Row index
        if len(colors) == 1:
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color=colors[0], alpha=0.7))
        else:
            blended_color = np.mean(colors, axis=0)
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color=blended_color, alpha=0.7))
    
    ax.set_xlim(0, grid_width)
    ax.set_ylim(grid_height, 0)  # Invert y-axis to match grid numbering
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Clustered State Sequences Visualization")
    plt.savefig("clustered_states.png")
    plt.show()

def main():
    trajectories = load_trajectories("base_trajectories.txt")
    sa_sequences = extract_sa_sequences(trajectories)
    
    labels, sequences = cluster_sequences(sa_sequences)
    visualize_clusters(labels, sa_sequences)
    
    print("Cluster labels:", labels)

if __name__ == "__main__":
    main()

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

# Path to your TensorBoard log folder
log_dir = "optuna_logs/db_score_100_epochs/version_0/"

# Load the TensorBoard event data
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

# Print all available metric tags
print("Available tags:", ea.Tags())

# Collect all scalar metrics into DataFrames
dfs = []
for tag in ea.Tags()['scalars']:
    # Extract (step, value) tuples
    values = [(s.step, s.value) for s in ea.Scalars(tag)]
    df = pd.DataFrame(values, columns=["step", tag])

    # If multiple values per step, average them
    df = df.groupby("step").mean().reset_index()

    # Use step as index for easy alignment
    dfs.append(df.set_index("step"))

# Combine all metrics into one DataFrame aligned by step
metrics_df = pd.concat(dfs, axis=1).reset_index()

# Save to CSV
metrics_df.to_csv("metrics_scripts/metrics_100_epochs.csv", index=False)
print(" Metrics exported to metrics_100_epochs.csv")

# Show a quick preview
print(metrics_df.head())

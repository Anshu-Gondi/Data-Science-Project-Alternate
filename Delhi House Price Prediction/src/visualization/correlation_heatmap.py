import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_correlation_heatmap(data_path, save_path=None, figsize=(10, 8)):
    """
    Load a dataset and plot a correlation heatmap.

    Args:
        data_path (str): Path to the CSV file containing the data.
        save_path (str, optional): Path to save the heatmap image. If None, it displays the plot.
        figsize (tuple): Figure size for the heatmap.
    """

    # Load dataset from the given CSV file path
    df = pd.read_csv(data_path)

    # Calculate correlation matrix for numeric columns only
    corr = df.corr(numeric_only=True)

    # Set up the plot figure size
    plt.figure(figsize=figsize)

    # Plot the heatmap using seaborn
    # annot=True to show correlation coefficients on the heatmap
    # cmap='coolwarm' sets the color theme (blue-red gradient)
    # fmt=".2f" formats the annotations to 2 decimal places
    # linewidths=0.5 adds lines between cells for clarity
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

    # Add a title to the heatmap
    plt.title("Feature Correlation Heatmap")

    # If a save path is provided, save the heatmap image to that path
    if save_path:
        # Ensure the directory for the save path exists, create if not
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the figure with tight bounding box to avoid cutting off parts
        plt.savefig(save_path, bbox_inches='tight')

        # Print confirmation message
        print(f"Heatmap saved to: {save_path}")

    # If no save path, just display the heatmap
    else:
        plt.show()

plot_correlation_heatmap(
    data_path='data/processed/final_data.csv',
    save_path='reports/figures/corr_heatmap.png'
)
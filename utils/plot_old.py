# plot.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as sp

def plot_loss(train_losses, val_losses, outdir="results/plots"):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Model Loss")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plt.savefig(os.path.join(outdir, "loss.png"))
    plt.close()

def plot_box_plots_symlog(y_pred, y_test, folder_name):
    # Current order of columns: ["WorstLatency_hls", "IntervalMax_hls", "FF_hls", "LUT_hls", "BRAM_18K_hls", "DSP_hls"]
    # Want this order: BRAM, DSP, FF, LUT, CYCLES, II
    # prediction_labels =  ['BRAM', 'DSP', 'FF', 'LUT', 'CYCLES', 'II']
    # prediction_labels = ["CYCLES", "ff", "lut", "bram", "dsp", "ii"]
    prediction_labels = ["CYCLES", "FF", "LUT", "BRAM", "DSP", "II"] 
    # indices in y_pred/y_test for: BRAM(4), DSP(5), FF(2), LUT(3), CYCLES(0), II(1)
    # plot_order = [4, 5, 2, 3, 0, 1] # Rework this to make more general
    plot_order = [0, 1, 2, 3, 4, 5]  # CYCLES, II, FF, LUT, BRAM, DSP

    prediction_errors = []
    for i in plot_order:
        errors = (y_test[:, i] - y_pred[:, i]) / (y_test[:, i] +1e-5) * 100
        prediction_errors.append(errors)

    plt.rcParams.update({"font.size": 16})
    fig, axis = plt.subplots(1, len(prediction_labels), figsize=(20, 8))
    axis = np.reshape(axis, -1)
    fig.subplots_adjust(hspace=0.1, wspace=0.6)
    iqr_weight = 1.5
    colors = ["pink", "yellow", "lightgreen", "lightblue", "#FFA500", "violet"]
    for idx, label in enumerate(prediction_labels):
        ax = axis[idx]
        errors = prediction_errors[idx]
        bplot = ax.boxplot(
            errors,
            whis=iqr_weight,
            tick_labels=[label.upper()],
            showfliers=True,
            showmeans=True,
            meanline=True,
            vert=True,
            patch_artist=True
        )
        for j, patch in enumerate(bplot["boxes"]):
            patch.set_facecolor(colors[(idx + j) % len(colors)])
        ax.yaxis.grid(True)
        ax.spines.top.set_visible(False)
        ax.xaxis.tick_bottom()
        ax.set_yscale('symlog', linthresh=1)
    median_line = Line2D([0], [0], color="orange", linestyle="--", linewidth=1.5, label="Median")
    mean_line = Line2D([0], [0], color="green", linestyle="--", linewidth=1.5, label="Mean")
    handles = [median_line, mean_line]
    labels = ["Median", "Mean"]
    legends = fig.legend(
        handles,
        labels,
        bbox_to_anchor=[0.95, 1],
        loc="upper right",
        ncol=len(labels) // 2,
    )
    ytext = fig.text(0.02, 0.5, "Relative Percent Error", va="center", rotation="vertical", size=18)
    suptitle = fig.suptitle("Prediction Errors - Boxplots", fontsize=20, y=0.95)

    directory = os.path.join(folder_name, 'plots')
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(
        os.path.join(directory, "box_plot_exemplar_symlog.pdf"),
        dpi=300,
        bbox_extra_artists=(legends, ytext, suptitle),
        bbox_inches="tight",
    )
    plt.savefig(os.path.join(directory, '_box_symlog.pdf'))
    plt.close()


def plot_results_simplified(name, mpl_plots, y_test, y_pred, output_features, folder_name):
    """
    Simplified version of plot_results that doesn't require X_raw_test.
    Creates basic scatter plots without strategy-based grouping.
    """
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'orange']

    if mpl_plots:
        for i, feature in enumerate(output_features):
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test[:, i], y_pred[:, i], s=20, label=feature, color=colors[i % len(colors)], alpha=0.7)
            plt.title('Actual vs Predicted for ' + feature)
            plt.xlabel('Actual Value')
            plt.ylabel('Predicted Value')
            plt.legend()
            vmin = min(np.min(y_test[:, i]), np.min(y_pred[:, i]))
            vmax = max(np.max(y_test[:, i]), np.max(y_pred[:, i]))
            plt.plot([vmin, vmax], [vmin, vmax], 'r--')
            plt.tight_layout()
            directory = os.path.join(folder_name, "plots")
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(os.path.join(directory, feature + '_predicted_vs_true.png'))
            plt.close()

    # Interactive plotly subplots
    n_features = len(output_features)
    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))
    fig = sp.make_subplots(
        rows=n_rows, cols=n_cols,
        vertical_spacing=0.05, horizontal_spacing=0.05,
        x_title='Actual Value',
        y_title='Predicted Value',
        subplot_titles=output_features,
    )

    for i in range(n_features):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        # Calculate feature-specific min/max for this subplot
        feature_min = min(np.min(y_test[:, i]), np.min(y_pred[:, i]))
        feature_max = max(np.max(y_test[:, i]), np.max(y_pred[:, i]))
        
        # Add some padding to the limits (5% on each side)
        padding = (feature_max - feature_min) * 0.05
        axis_min = feature_min - padding
        axis_max = feature_max + padding
        
        # Simple scatter plot without strategy grouping
        scatter = go.Scatter(
            x=y_test[:, i],
            y=y_pred[:, i],
            mode='markers',
            name=f'{output_features[i]}',
            marker=dict(
                color=colors[i % len(colors)],
                size=6,
                opacity=0.7,
            ),
            hovertemplate=
                '<i>Actual</i>: %{x}<br>' +
                '<b>Predicted</b>: %{y}<br><extra></extra>',
        )
        fig.add_trace(scatter, row=row, col=col)

        # Perfect prediction line (feature-specific)
        fig.add_trace(
            go.Scatter(
                x=[feature_min, feature_max],
                y=[feature_min, feature_max],
                mode='lines',
                line=dict(color='black', dash='dash'),
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Set feature-specific axis ranges
        fig.update_xaxes(range=[axis_min, axis_max], row=row, col=col)
        fig.update_yaxes(range=[axis_min, axis_max], row=row, col=col)

    fig.update_layout(
        height=380 * n_rows,
        width=800,
        title='Actual vs Predicted (all outputs)',
        legend=dict(font=dict(size=13))
    )

    directory = os.path.join(folder_name, "plots", "scatterplots")
    if not os.path.exists(directory):
        os.makedirs(directory)
    pio.write_html(fig, file=os.path.join(directory, f"{name}_outputs.html"), auto_open=False)

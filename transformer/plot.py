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
    prediction_labels = ['CYCLES', 'FF', 'LUT', 'BRAM', 'DSP', 'II']
    # indices in y_pred/y_test for: BRAM(4), DSP(5), FF(2), LUT(3), CYCLES(0), II(1)
    plot_order = [4, 5, 2, 3, 0, 1] # Rework this to make more general

    prediction_errors = []
    for i in plot_order:
        # errors = (y_test[:, i] - y_pred[:, i]) / (y_test[:, i]) * 100 # removed +1 in the denominator
        errors = (y_test[:, i] - y_pred[:, i]) / (y_test[:, i] +1) * 100 # +1 as done in the original plotting
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

        # Find the max absolute value for symmetric limits
        max_abs = np.nanmax(np.abs(errors))
        ax.set_ylim(-max_abs, max_abs)
        # Add horizontal zero line
        ax.axhline(0, color='black', linestyle='--', linewidth=1)

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

def plot_results(
    name, mpl_plots, y_test, y_pred, X_raw_test, output_features, folder_name
):
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'orange']

    # Scatter for each output
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

    # 2. Scatterplots
    strat_dict = {0: 'Latency', 1: 'Resource'}
    marker_shapes = {0: 'star', 1: 'square'}
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

    X_raw_test = np.asarray(X_raw_test)
    overall_min = min(np.min(y_test), np.min(y_pred))
    overall_max = max(np.max(y_test), np.max(y_pred))

    for i in range(n_features):
        row = i // n_cols + 1
        col = i % n_cols + 1
        for strategy in [0, 1]:
            mask = X_raw_test[:, strategy + 5] == 1
            sel_points = X_raw_test[mask]
            text_arr = [
                f"{int(point[3])}-bit {int(point[0])}x{int(point[1])}x{int(point[2])} @ RF={int(point[4])} ({strat_dict[strategy]})"
                for point in sel_points
            ]
            scatter = go.Scatter(
                x=y_test[mask, i],
                y=y_pred[mask, i],
                mode='markers',
                name=f'{output_features[i]} - {strat_dict[strategy]}',
                legendgroup=f'{output_features[i]}',
                marker=dict(
                    symbol=marker_shapes[strategy],
                    color=colors[i % len(colors)],
                    size=9,
                    opacity=0.75,
                ),
                hovertemplate=
                    '%{text}<br>' +
                    '<i>Actual</i>: %{x}<br>' +
                    '<b>Predicted</b>: %{y}<br><extra></extra>',
                text=text_arr
            )
            fig.add_trace(scatter, row=row, col=col)

        fig.add_trace(
            go.Scatter(
                x=[overall_min, overall_max],
                y=[overall_min, overall_max],
                mode='lines',
                line=dict(color='black', dash='dash'),
                showlegend=False
            ),
            row=row, col=col
        )

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


def plot_results_simplified(name, mpl_plots, y_test, y_pred, output_features, folder_name, model_types=None):
    """
    Simplified version of plot_results that doesn't require X_raw_test.
    Creates basic scatter plots without strategy-based grouping.
    """
    type_color_map = {'Dense': '#4c72b0', 'Conv1D': '#55a868', 'Conv2D': '#c44e52'}
    colors = ['pink']
    # colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'orange']

    # if mpl_plots:
    #     plt.rcParams.update({"font.size": 24})
    #     for i, feature in enumerate(output_features):
    #         plt.figure(figsize=(8, 6))
    #         if model_types is not None:
    #             unique_types = ['Dense', 'Conv1D', 'Conv2D']
    #             for t in unique_types:
    #                 idxs = [j for j, mt in enumerate(model_types) if mt == t]
    #                 if len(idxs) == 0:
    #                     continue
    #                 plt.scatter(
    #                     y_test[idxs, i],
    #                     y_pred[idxs, i],
    #                     s=10,
    #                     label=t,
    #                     color=type_color_map.get(t, 'gray'),
    #                     alpha=0.75,
    #                     marker='o',   # Use '.' or ',' for tiny fast dots
    #                 )
    #         else:
    #             plt.scatter(
    #                 y_test[:, i], y_pred[:, i],
    #                 s=20, label=feature, color=colors[i % len(colors)], alpha=0.7
    #             )

    #         plt.title('Actual vs Predicted for ' + feature)
    #         plt.xlabel('Actual Value')
    #         plt.ylabel('Predicted Value')
    #         plt.xscale('log')
    #         plt.yscale('log')
    #         plt.legend()
    #         vmin = min(np.min(y_test[:, i]), np.min(y_pred[:, i]))
    #         vmax = max(np.max(y_test[:, i]), np.max(y_pred[:, i]))
    #         plt.plot([vmin, vmax], [vmin, vmax], 'r--')
    #         plt.tight_layout()
    #         directory = os.path.join(folder_name, "plots")
    #         if not os.path.exists(directory):
    #             os.makedirs(directory)
    #         plt.savefig(os.path.join(directory, feature + '_predicted_vs_true.png'))
    #         plt.close()

    if mpl_plots:
        plt.rcParams.update({"font.size": 24})
        # Desired order: BRAM, DSP, FF, LUT, CYCLES, INTERVAL (not II)
        desired_order = [3, 4, 1, 2, 0, 5]  # Indices for your arrays: BRAM, DSP, FF, LUT, CYCLES, II
        label_names = ['BRAM', 'DSP', 'FF', 'LUT', 'CYCLES', 'INTERVAL']

        num_plots = len(desired_order)
        num_y = int(np.sqrt(num_plots))
        num_x = int(np.ceil(np.sqrt(num_plots)))
        fig, axes = plt.subplots(num_y, num_x, figsize=(6 * num_x, 10))
        axes = np.reshape(axes, -1)
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        unique_types = ['Dense', 'Conv1D', 'Conv2D']

        legend_handles = []

        for plot_idx, (array_idx, label) in enumerate(zip(desired_order, label_names)):
            ax_pred = axes[plot_idx]
            actual = []
            predicted = []
            if model_types is not None:
                for t in unique_types:
                    idxs = [j for j, mt in enumerate(model_types) if mt == t]
                    if len(idxs) == 0:
                        continue
                    actual_vals = y_test[idxs, array_idx]
                    pred_vals = y_pred[idxs, array_idx]
                    actual.extend(actual_vals)
                    predicted.extend(pred_vals)
                    scatter = ax_pred.scatter(
                        actual_vals,
                        pred_vals,
                        alpha=0.6,
                        label=t,
                        s=20,
                    )
                    if plot_idx == 0:
                        legend_handles.append(scatter)
            else:
                actual = y_test[:, array_idx]
                predicted = y_pred[:, array_idx]
                scatter = ax_pred.scatter(
                    actual,
                    predicted,
                    alpha=0.7,
                    s=20,
                    label=label,
                )
                if plot_idx == 0:
                    legend_handles.append(scatter)

            # Diagonal perfect prediction line
            if len(actual) > 0 and len(predicted) > 0:
                minval = min(np.min(actual), np.min(predicted))
                maxval = max(np.max(actual), np.max(predicted))
                ax_pred.plot([minval, maxval], [minval, maxval], color="red")
            ax_pred.set_xlabel(f"Actual {label}")
            ax_pred.set_ylabel(f"Predicted {label}")
            ax_pred.set_xscale("log")
            ax_pred.set_yscale("log")

        # Remove unused axes
        for i in range(num_plots, len(axes)):
            fig.delaxes(axes[i])

        legend_title = "Architecture" if model_types is not None else "Feature"
        legend = fig.legend(
            handles=legend_handles,
            title=legend_title,
            loc="upper right",
            bbox_to_anchor=(0.5, 1.08),
            fontsize=18,
            ncol=len(legend_handles),
        )
        title = f"Transformer Prediction Analysis on Test Set"
        suptitle = fig.suptitle(title, fontsize=28, x=0.5, y=1.125)
        directory = os.path.join(folder_name, "plots")
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(
            os.path.join(directory, f"{name}_predicted_vs_true_grid.png"),
            dpi=300,
            bbox_extra_artists=(suptitle, legend),
            bbox_inches="tight",
        )
        plt.close(fig)

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

    overall_min = min(np.min(y_test), np.min(y_pred))
    overall_max = max(np.max(y_test), np.max(y_pred))

    for i in range(n_features):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        # # Simple scatter plot without strategy grouping
        # scatter = go.Scatter(
        #     x=y_test[:, i],
        #     y=y_pred[:, i],
        #     mode='markers',
        #     name=f'{output_features[i]}',
        #     marker=dict(
        #         color=colors[i % len(colors)],
        #         size=6,
        #         opacity=0.7,
        #     ),
        #     hovertemplate=
        #         '<i>Actual</i>: %{x}<br>' +
        #         '<b>Predicted</b>: %{y}<br><extra></extra>',
        # )
        # fig.add_trace(scatter, row=row, col=col)

        if model_types is not None:
            unique_types = ['Dense', 'Conv1D', 'Conv2D']
            for t in unique_types:
                idxs = [j for j, mt in enumerate(model_types) if mt == t]
                if len(idxs) == 0:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=y_test[idxs, i],
                        y=y_pred[idxs, i],
                        mode='markers',
                        name=f'{output_features[i]} - {t}',
                        marker=dict(
                            color=type_color_map.get(t, 'gray'),
                            size=6,
                            opacity=0.7,
                        ),
                        hovertemplate=
                            f'<b>{t}</b><br>' +
                            '<i>Actual</i>: %{x}<br>' +
                            '<b>Predicted</b>: %{y}<br><extra></extra>',
                    ),
                    row=row, col=col
                )
        else:
            fig.add_trace(
                go.Scatter(
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
                ),
                row=row, col=col
            )        

        # Perfect prediction line
        fig.add_trace(
            go.Scatter(
                x=[overall_min, overall_max],
                y=[overall_min, overall_max],
                mode='lines',
                line=dict(color='black', dash='dash'),
                showlegend=False
            ),
            row=row, col=col
        )

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


## To use:
# from plot import plot_loss, plot_box_plots_symlog, plot_results
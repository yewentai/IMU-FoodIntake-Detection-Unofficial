import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch


# Visualize data from DataLoader
def visualize_dataloader(data_loader, window_length):
    for batch_idx, (data, labels) in enumerate(data_loader):
        # Data is shaped as (batch_size, channels, window_length)
        # Labels are shaped as (batch_size,)
        for sample_idx in range(data.shape[0]):
            window = data[sample_idx].numpy()  # Shape: (channels, window_length)
            label = labels[sample_idx].item()

            # Extract accelerometer and gyroscope signals
            acc = window[:3, :]  # First 3 channels
            gyr = window[3:, :]  # Next 3 channels
            t = range(window_length)  # Simulated time axis for window

            # Plot accelerometer and gyroscope signals
            max_acc = max(abs(acc.min()), abs(acc.max()))
            max_gyr = max(abs(gyr.min()), abs(gyr.max()))

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            fig.suptitle(
                f"Sample {batch_idx}-{sample_idx}, Label: {label}", fontsize=16
            )

            ax1.plot(t, acc[0, :], label="Acc X")
            ax1.plot(t, acc[1, :], label="Acc Y")
            ax1.plot(t, acc[2, :], label="Acc Z")
            ax1.set_ylabel("Accelerometer (g)")
            ax1.set_ylim(-max_acc, max_acc)
            ax1.legend()
            ax1.grid()

            ax2.plot(t, gyr[0, :], label="Gyr X")
            ax2.plot(t, gyr[1, :], label="Gyr Y")
            ax2.plot(t, gyr[2, :], label="Gyr Z")
            ax2.set_ylabel("Gyroscope (°/s)")
            ax2.set_xlabel("Time (samples)")
            ax2.set_ylim(-max_gyr, max_gyr)
            ax2.legend()
            ax2.grid()

            # Mark the presence of a "bite" event
            if label == 1:
                ax1.add_patch(
                    patches.Rectangle(
                        (0, -max_acc),
                        window_length,
                        2 * max_acc,
                        edgecolor="red",
                        facecolor="pink",
                        alpha=0.3,
                        label="Bite event",
                    )
                )
                ax2.add_patch(
                    patches.Rectangle(
                        (0, -max_gyr),
                        window_length,
                        2 * max_gyr,
                        edgecolor="red",
                        facecolor="pink",
                        alpha=0.3,
                    )
                )

            plt.legend()
            plt.show()


# Visualize a few samples from the train DataLoader
visualize_dataloader(train_loader, window_length)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

def plot_history(history,save_path=None):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"train history saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False,
                          title='Confusion Matrix', cmap=plt.cm.Blues,
                          figsize=(10, 8), save_path=None):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        thresh = 0.5
    else:
        fmt = 'd'
        thresh = cm.max() / 2.

    plt.figure(figsize=figsize, dpi=100)
    sns.set(font_scale=1.2)
    sns.axes_style("white")

    ax = sns.heatmap(
        cm, annot=True, fmt=fmt, cmap=cmap,
        square=True, linewidths=.5, cbar=False,
        annot_kws={'size': 14, 'color': 'white' if normalize else 'black'},
        xticklabels=classes, yticklabels=classes,
        mask=cm < 0.000
    )

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(length=0, labelsize=12)
    plt.xticks(rotation=45, ha='left')
    plt.yticks(rotation=0)

    plt.colorbar(ax.collections[0], ax=ax, location="right", pad=0.1)

    plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
    plt.ylabel('True Label', fontsize=14, labelpad=10)
    plt.title(title, fontsize=16, pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Confusion matrix saved to {save_path}")

    plt.show()

def visualize_batch(dataloader, num_samples=3):
    features, labels = next(iter(dataloader))
    print(f"Batch feature shape: {features.shape} (batch_size, seq_len, feature_dim)")
    print(f"Batch label shape: {labels.shape}")

    plt.figure(figsize=(15, 5))
    for i in range(min(num_samples, features.shape[0])):
        plt.subplot(1, num_samples, i + 1)
        plt.plot(features[i].numpy())
        plt.title(f"Label: {labels[i].item()}")
        plt.xlabel("Time Step")
        plt.ylabel("Feature Value")
    plt.tight_layout()
    plt.show()

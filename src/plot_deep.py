import matplotlib.pyplot as plt

def plot_loss_values_TV(train_losses,val_losses, add_title = ""):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    title = 'Training and Validation Losses '+add_title
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_real_vs_pred(y_real, y_pred, vars_cibles):
    # plot comparatif des prédictions vs réelles pour le test set OOD
    plt.figure(figsize=(12, 6))
    for i, target in enumerate(vars_cibles):
        plt.subplot(3, 4, i + 1)
        plt.scatter(y_real[:, i], y_pred[:, i], alpha=0.5)
        plt.plot([y_real[:, i].min(), y_real[:, i].max()],
                [y_real[:, i].min(), y_real[:, i].max()],
                'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predictions vs True for {target[:20]}')
        plt.tight_layout()
    plt.show()
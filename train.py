
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformer import F1MLTransformer
from dataset import FSDataset, collate_fn, f1data_train_test_split
from loss import LambdaLoss
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn.metrics import ndcg_score


def print_gradients(model):
    def hook_fn(grad):
        print(grad)

    for name, param in model.named_parameters():
        param.register_hook(hook_fn)

def ndcg_at_k(y_true, y_pred, k=10):
  """
  Calculates NDCG@k.

  Args:
    y_true: Ground truth rankings (shape: batch_size, num_drivers).
    y_pred: Predicted rankings (shape: batch_size, num_drivers).
    k: The number of top-ranked items to consider.

  Returns:
    NDCG@k score.
  """
  # No need to convert y_true to CPU (it's already assumed to be NumPy)
  y_pred = y_pred.cpu().numpy()  # Convert predictions to NumPy
  return ndcg_score(y_true, y_pred, k=k)

def plot_losses(train_losses_batch: list, eval_losses_batch: list, train_losses_epoch: list, eval_losses_epoch: list):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].plot(train_losses_batch, color='blue', label='Train loss per minibatch')
    axs[0, 0].set_title('Train loss per minibatch')
    axs[0, 0].set_xlabel('Minibatch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    axs[0, 1].plot(eval_losses_batch, color='orange', label='Eval loss per minibatch')
    axs[0, 1].set_title('Eval loss per minibatch')
    axs[0, 1].set_xlabel('Minibatch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()

    axs[1, 0].plot(train_losses_epoch, color='green', label='Train loss per epoch')
    axs[1, 0].set_title('Train loss per epoch')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].legend()

    axs[1, 1].plot(eval_losses_epoch, color='red', label='Eval loss per epoch')
    axs[1, 1].set_title('Eval loss per epoch')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig('loss_plots.pdf', format='pdf')
    plt.show()

def training_loop(
        network: torch.nn.Module,
        train_data: torch.utils.data.Dataset,
        eval_data: torch.utils.data.Dataset,
        num_epochs: int,
        learning_rate: float,
        loss_fn: callable = LambdaLoss,
        collate_fn: callable = collate_fn,
        batch_size: int = 4,
        l1_lambda: float = 0.01,
        l2_lambda: float = 0.01,
        show_progress: bool = False,
        writer: SummaryWriter = None,
        pre_optimizer=None,
        pre_lr_scheduler=None,
) -> tuple[list, list, list, list]:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataloader_train = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)
    dataloader_eval = DataLoader(eval_data, batch_size=batch_size, collate_fn=collate_fn)

    minibatch_losses_train = []
    minibatch_losses_eval = []
    epoch_losses_train = []
    epoch_losses_eval = []
    epoch_ndcg_eval = []
    if pre_optimizer is not None:
        optimizer = pre_optimizer
    else:
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    if pre_lr_scheduler is not None:
        scheduler = pre_lr_scheduler
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    loss_fn = loss_fn
    old_loss = 0
    counter = 0
    # best_accuracy = 0
    epoch_ndcg_train = []
    for epoch in range(num_epochs):
        print("-------------------------------------------------")
        print(f"TRAINING: {epoch}")
        print("-------------------------------------------------")

        network.train()
        mbl_train = []
        train_ndcg_k = []
        # batch_num_idx = 0
        if counter == 20:
            break
        for batch in ( tqdm(dataloader_train) if show_progress else dataloader_train):
            # if batch_num_idx == 21 or batch_num_idx == 30:
            #     print('a')
            #     print('\n')
            #     print(batch_num_idx)
            #     print(torch.any(torch.isnan(driver_laps_batch)), torch.all(torch.isfinite(driver_laps_batch)))
            #     print(torch.any(torch.isnan(result_batch)), torch.all(torch.isfinite(result_batch)))
            #     print('\n')
            #
            #
            #     print(result_batch)
            #     print(idx_years, idx_within_years)
            # batch_num_idx += 1
            optimizer.zero_grad()
            driver_laps_batch, result_batch, driver_to_idx_batch, idx_years, idx_within_years = batch
            driver_laps_batch = driver_laps_batch.to(device)
            result_batch = result_batch.to(device)
            preds = network(driver_laps_batch).squeeze()
            loss = loss_fn(preds, result_batch)

            l1_reg = sum(param.abs().sum() for param in network.parameters())
            l2_reg = sum(param.pow(2).sum() for param in network.parameters())
            loss += l1_lambda * l1_reg + l2_lambda * l2_reg

            # print_gradients(network)

            # Calculate NDCG@k
            ndcg_k = ndcg_at_k(result_batch.cpu(), (preds.detach().clone()).cpu(), k=10)
            # Store NDCG@k for evaluation
            train_ndcg_k.append(ndcg_k)

            loss.backward()
            optimizer.step()
            mbl_train.append(loss.item())
            minibatch_losses_train.append(loss.item())
            if writer:
                writer.add_scalar('Loss/Train_Minibatch', loss.item(), len(minibatch_losses_train))

        epoch_losses_train.append(np.mean(mbl_train))
        epoch_ndcg_train.append(np.mean(train_ndcg_k))

        print(f"\nEpoch {epoch}: Train NDCG@10: {epoch_ndcg_train[-1]}")

        if writer:
            writer.add_scalar('Loss/Train_Epoch', np.mean(mbl_train), epoch)
            writer.add_scalar('NDCG@10/Eval_Epoch', epoch_ndcg_train[-1], epoch)

        print("-------------------------------------------------")
        print(f"EVAL: {epoch}")
        print("-------------------------------------------------")

        network.eval()
        mbl_eval = []
        eval_ndcg_k = []
        # correct = 0
        # total = 0
        with torch.no_grad():
            for batch in (tqdm(dataloader_eval) if show_progress else dataloader_eval):
                driver_laps_batch, result_batch, driver_to_idx_batch, _, _ = batch
                driver_laps_batch = driver_laps_batch.to(device)
                result_batch = result_batch.to(device)
                preds = network(driver_laps_batch).squeeze()
                loss = loss_fn(preds, result_batch)

                # Calculate NDCG@k
                ndcg_k = ndcg_at_k(result_batch.cpu(), preds.cpu(), k=10)
                # Store NDCG@k for evaluation
                eval_ndcg_k.append(ndcg_k)

            epoch_ndcg_eval.append(np.mean(eval_ndcg_k))

            print(f"Epoch {epoch}: Eval NDCG@10: {epoch_ndcg_eval[-1]}")

            if writer:
                writer.add_scalar('NDCG@10/Eval_Epoch', epoch_ndcg_eval[-1], epoch)

                # print('\n')
                # print(torch.any(torch.isnan(driver_laps_batch)), torch.all(torch.isfinite(driver_laps_batch)))
                # print(torch.any(torch.isnan(result_batch)), torch.all(torch.isfinite(result_batch)))
                # print('\n')

                mbl_eval.append(loss.item())
                minibatch_losses_eval.append(loss.item())

                if writer:
                    writer.add_scalar('Loss/Eval_Minibatch', loss.item(), len(minibatch_losses_eval))

            epoch_losses_eval.append(np.mean(mbl_eval))
            # accuracy = 100 * correct // total
            # print(f"Accuracy of the network on the test images: {accuracy} %")

        if writer:
            writer.add_scalar('Loss/Eval_Epoch', np.mean(mbl_eval), epoch)

        # if accuracy > best_accuracy:
        #     best_accuracy = accuracy
        #     torch.save(network.state_dict(), 'model.pth')
        #     torch.save(optimizer.state_dict(), 'optimizer.pth')
        #     torch.save(scheduler.state_dict(), 'scheduler.pth')



        if old_loss - np.mean(mbl_eval) < 0.006:
            counter += 1
        else:
            counter = 0
        old_loss = np.mean(mbl_eval)
        scheduler.step()

    return minibatch_losses_train, minibatch_losses_eval, epoch_losses_train, epoch_losses_eval


if __name__ == '__main__':

    TEST_SIZE = 0.05
    BATCH_SIZE = 16
    SEED = 42
    LEARNING_RATE = 0.0012
    NUM_EPOCHS = 15

    # f1data_train_test_split(r'F1Data\f1_preprocessed.csv', r'F1Data\f1_final_result_data.csv', TEST_SIZE)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_dataset = FSDataset(r'F1Data\training_lap_data.csv', r'F1Data\training_results_data.csv')
    test_dataset = FSDataset(r'F1Data\test_lap_data.csv', r'F1Data\test_results_data.csv')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = F1MLTransformer(num_blocks=10, num_heads=8,num_features_list=[106, 128, 128, 128, 128, 128, 128, 128, 128, 128],num_multiplier=4, num_embeddings=106, linear_list=[512, 256, 128, 32])
    net.to(device)
    # net.load_state_dict(torch.load('model.pth'))
    # optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    # optimizer.load_state_dict(torch.load('optimizer.pth'))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # scheduler.load_state_dict(torch.load('scheduler.pth'))
    writer = SummaryWriter()

    minibatch_losses_train, minibatch_losses_eval, epoch_losses_train, epoch_losses_eval = training_loop(
        network = net, train_data = train_dataset, eval_data = test_dataset, num_epochs = NUM_EPOCHS, learning_rate = LEARNING_RATE, loss_fn = LambdaLoss(), collate_fn = lambda batch: collate_fn(batch, pad_flag=True), batch_size = BATCH_SIZE,  l1_lambda=0, l2_lambda=0, show_progress=True,
        writer=writer
    )

    writer.close()

    plot_losses(minibatch_losses_train, minibatch_losses_eval, epoch_losses_train, epoch_losses_eval)





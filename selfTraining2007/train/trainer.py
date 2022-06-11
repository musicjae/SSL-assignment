import torch
import pickle
import wandb
from os import getenv
import pandas as pd

class GnnTrainer(object):
    def __init__(self, model, MetricManager):
        self.model = model
        self.metric_manager = MetricManager(modes=["train", "val"])

    def train(self, data_train, optimizer, criterion, scheduler, args):

        self.data_train = data_train
        for epoch in range(args["epochs"]):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(data_train)

            out = out.reshape((data_train.x.shape[0]))
            loss = criterion(
                out[data_train.train_idx], data_train.y[data_train.train_idx]
            )

            # train data
            target_labels = data_train.y.detach().cpu().numpy()[data_train.train_idx]
            pred_scores = out.detach().cpu().numpy()[data_train.train_idx]
            (
                train_acc,
                train_f1,
                train_f1macro,
                train_aucroc,
                train_recall,
                train_precision,
                train_cm,
            ) = self.metric_manager.store_metrics("train", pred_scores, target_labels)

            ## Training Step
            loss.backward()
            optimizer.step()

            # validation data
            self.model.eval()
            target_labels = data_train.y.detach().cpu().numpy()[data_train.valid_idx]
            pred_scores = out.detach().cpu().numpy()[data_train.valid_idx]
            (
                val_acc,
                val_f1,
                val_f1macro,
                val_aucroc,
                val_recall,
                val_precision,
                val_cm,
            ) = self.metric_manager.store_metrics("val", pred_scores, target_labels)

            wandb.log(
                {
                    "train loss": loss,
                    "val acc": val_acc,
                    "val f1": val_f1,
                    "val f1macro": val_f1macro,
                    "val aucroc": val_aucroc,
                    "val recall": val_recall,
                    "val_precision": val_precision,
                    "val confusion matrix": val_cm,
                }
            )

            if epoch % 5 == 0:
                print(
                    "epoch: {} - loss: {:.4f} - accuracy train: {:.4f} -accuracy valid: {:.4f}  - val roc: {:.4f}  - val f1micro: {:.4f}".format(
                        epoch, loss.item(), train_acc, val_acc, val_aucroc, val_f1
                    )
                )

            if epoch > args["epochs"] - 2:
                model_name = "normal_gcn.pt"
                self.save_model(model_name, path=getenv("SAVE_RESULT_PATH"))
                print(f"Complete to save {model_name}")

    # To predict labels
    def make_pseudo_labels(self, data=None, model_path=None, unclassified_only=True, threshold=0.5):
        self.model.eval()
        self.model.load_state_dict(torch.load(model_path))
        # evaluate model:
        self.model.eval()
        if data is not None:
            self.data_train = data

        out = self.model(self.data_train)
        out = out.reshape((self.data_train.x.shape[0]))

        if unclassified_only:
            pred_scores = out.detach().cpu().numpy()[self.data_train.unclassified_idx]

        else:
            pred_scores = out.detach().cpu().numpy()

        pred_labels = pred_scores > threshold
        print(pred_labels)
        pseudo_labels = []

        for label in pred_labels:
            if label == True:
                pseudo_labels.append(1)

            else:
                pseudo_labels.append(0)

        self.data_train.unclassified_idx = pseudo_labels
        # unclassified_idx = pd.DataFrame(self.data_train.unclassified_idx)
        # unclassified_label = pd.DataFrame(pseudo_labels)
        # result = pd.concat([unclassified_idx,unclassified_label],axis=1)
        # result.columns=[['nodeId','label']]

        return self.data_train

    # To save metrics
    def save_metrics(self, save_name, path="./save/"):
        file_to_store = open(path + save_name, "wb")
        pickle.dump(self.metric_manager, file_to_store)
        file_to_store.close()

    # To save model
    def save_model(self, save_name, path="./save/"):
        torch.save(self.model.state_dict(), path + save_name)

    def test(self, model_path, graph_data):
        self.model.eval()
        self.model.load_state_dict(torch.load(model_path))

        out = self.model(graph_data)
        out = out.reshape((graph_data.x.shape[0]))

        target_labels = graph_data.y.detach().cpu().numpy()[graph_data.test_idx]
        pred_scores = out.detach().cpu().numpy()[graph_data.test_idx]
        (
            test_acc,
            test_f1,
            test_f1macro,
            test_aucroc,
            test_recall,
            test_precision,
            _,
        ) = self.metric_manager.store_metrics("val", pred_scores, target_labels)

        print(
            f"test_acc:{round(test_acc,3)}---test_f1:{round(test_f1,3)}---test_f1macro:{round(test_f1macro,3)}\
             ---test_aucroc:{round(test_aucroc,3)}--- test_recall:{round(test_recall,3)}\
              ---test_precision:{round(test_precision,3)}"
        )

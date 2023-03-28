import torch
from ray.train.trainer import BaseTrainer
from ray.air import session


class MyPytorchTrainer(BaseTrainer):
    def setup(self):
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.1)

    def training_loop(self):
        # You can access any Trainer attributes directly in this method.
        # self.datasets["train"] has already been
        # preprocessed by self.preprocessor
        dataset = self.datasets["train"]

        torch_ds = dataset.iter_torch_batches(dtypes=torch.float)
        loss_fn = torch.nn.MSELoss()

        for epoch_idx in range(10):
            loss = 0
            num_batches = 0
            for batch in torch_ds:
                X, y = torch.unsqueeze(batch["x"], 1), batch["y"]
                # Compute prediction error
                pred = self.model(X)
                batch_loss = loss_fn(pred, y)

                # Backpropagation
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                loss += batch_loss.item()
                num_batches += 1
            loss /= num_batches

            # Use Tune functions to report intermediate
            # results.
            session.report({"loss": loss, "epoch": epoch_idx})

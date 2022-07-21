from .dataset import get_datasets
from models.flow import Flow
from .utils import create_model, argmax_criterion
from survae.utils import sum_except_batch

import wandb
import torch
import numpy as np
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from .visualise import plot_data3d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @torch.jit.script
def remove_mean_with_mask(x, node_mask):
    # assert (x * (1 - node_mask)).abs().sum().item() < 1e-8
    node_mask = node_mask.unsqueeze(2)
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

def center_gravity_zero_gaussian_log_likelihood_with_mask(x, node_mask):
    # assert len(x.size()) == 3
    node_mask = node_mask.unsqueeze(2)
    B, N_embedded, D = x.size()
    # assert_mean_zero_with_mask(x, node_mask)

    # r is invariant to a basis change in the relevant hyperplane, the masked
    # out values will have zero contribution.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    N = node_mask.squeeze(2).sum(1)  # N has shape [B]
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


class FlowExp:
    def __init__(self, config) -> None:
        self.config = config
        self.config['flow'] = "Flow" 
        self.config['model'] = Flow

        if "hidden_dim" not in self.config:
            self.config['hidden_dim'] = 128

        if "base" not in self.config:
            self.config['base'] = "standard"

        self.batch_size = self.config["batch_size"] if "batch_size" in self.config else 128
        self.train_loader, self.test_loader = get_datasets(type="mqm9", batch_size=self.batch_size)
        self.network, self.optimiser, self.scheduler = create_model(self.config)
        self.base = torch.distributions.Normal(loc=0., scale=1.)
        self.total_logged = 0

    def train(self):
        catagorical = torch.randint(0, 5, (1, 29, 1), dtype=torch.float, device=device)
        continuous = torch.randn(1, 29, 3, device=device)

        x = torch.cat([catagorical, continuous], dim=-1)

        self.network(x, mask=torch.ones(1, 29, device=device, dtype=torch.bool))
        print(f"Model Parameters: {sum([p.numel() for p in self.network.parameters()])}")

        images_size = 8
        test_mask = torch.ones(images_size, 29, dtype=torch.bool, device=device)
        mask_size = torch.randint(3, 29, (images_size,))
        
        for idx in range(images_size):
            test_mask[idx, mask_size[idx]:] = False

        categorical = torch.randn(images_size, 29, 5, device=device)
        continuous = remove_mean_with_mask(torch.randn(images_size, 29, 3, device=device), test_mask)

        test_z = torch.cat((categorical * test_mask.unsqueeze(2), continuous), dim=-1)

        scaler = GradScaler()
        with wandb.init(project="molecule-flow-3d", config=self.config, entity="iclac") as run:
            step = 0
            for epoch in range(self.config['epochs']):
                loss_step = 0
                loss_ep_train = 0
                self.network.train()

                report_step = 0

                for idx, batch_data in enumerate(self.train_loader):
                    
                    x = batch_data.x[:, :, 0:1]
                    pos = batch_data.pos

                    input = torch.cat((x, pos), dim=-1).to(device)
                    mask = batch_data.mask.to(device)

                    self.optimiser.zero_grad()
                    
                    with autocast(enabled=self.config['autocast']):
                        z, log_det = self.network(input, mask=mask)

                        log_prob = None

                    if torch.isnan(z).any():
                        step += 1

                        if self.total_logged < 30:
                            torch.save({
                                'epoch': epoch,
                                'step': step,
                                'model_state_dict': self.network.state_dict(),
                                'input': input,
                                'mask': mask,
                                }, f"model_irregularity_{run.id}_{epoch}_{step}.pt")

                            wandb.save(f"model_irregularity_z_{run.id}_{epoch}_{step}.pt")
                            self.total_logged += 1
                        
                        del z, log_det
                        self.optimiser.zero_grad()

                        print(f"NaN detected at step {step} epoch {epoch}")
                        continue

                    if self.config['base'] == "invariant":
                        z = z * mask.unsqueeze(2)

                        categorical, continuous = torch.split(
                            z,
                            [5, 3],
                            dim=-1
                        )

                        zero_mean_continuous = remove_mean_with_mask(continuous, node_mask=mask)
                        log_prob = sum_except_batch(center_gravity_zero_gaussian_log_likelihood_with_mask(zero_mean_continuous, node_mask=mask))
                        log_prob += sum_except_batch(self.base.log_prob(categorical) * mask.unsqueeze(2))
                    else:
                        z = z * mask.unsqueeze(2)
                        log_prob = sum_except_batch(self.base.log_prob(z))

                    loss = argmax_criterion(log_prob, log_det)

                    if (loss > 1e3 and epoch > 5) or torch.isnan(loss):
                        if self.total_logged < 30:
                            torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.network.state_dict(),
                            'input': input,
                            'mask': mask,
                            }, f"model_irregularity_{run.id}_{epoch}_{step}.pt")

                            wandb.save(f"model_irregularity_{run.id}_{epoch}_{step}.pt")

                            self.total_logged += 1


                    loss_step += loss.detach()
                    loss_ep_train += loss.detach()

                    if self.config['autocast']:
                        scaler.scale(loss).backward()
                        scaler.unscale_(self.optimiser)
                        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1)
                        scaler.step(self.optimiser)
                        scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1)
                        self.optimiser.step()
        

                    step += 1
                    report_step += 1

                    if report_step == 10:
                        ll = (loss_step / 10.).item()
                        print(ll)
                        wandb.log({"epoch": epoch, "NLL": ll}, step=step)

                        loss_step = 0
                        report_step = 0

                    if step % 200 == 0:
                        with torch.no_grad():
                            xh, _ = self.network.inverse(test_z, mask=test_mask)
                            xh = xh.to("cpu")
                            
                            atoms_types, pos = torch.split(xh, [1, 3], dim=-1)

                            for idx in range(atoms_types.shape[0]):
                                aty = atoms_types[idx].long().view(-1).numpy()
                                p = pos[idx].view(-1, 3).numpy()
                                
                                plot_data3d(
                                    positions=p,
                                    atom_type=aty,
                                    spheres_3d=False,
                                    save_path=f"{run.id}_{epoch}_{idx}.png"
                                )

                            wandb.log(
                                {
                                    "epoch": epoch,
                                    "image": [
                                        wandb.Image(f"{run.id}_{epoch}_{step}_{idx}.png") for idx in range(atoms_types.shape[0])
                                    ]
                                },
                                step=step
                            )

                if self.scheduler is not None:
                    self.scheduler.step()
                    wandb.log({"Learning Rate/Epoch": self.scheduler.get_last_lr()[0]})
                

                wandb.log({"NLL/Epoch": (loss_ep_train / len(self.train_loader)).item()}, step=epoch)
                if self.config['upload']:
                    if epoch % self.config['upload_interval'] == 0:
                        if self.scheduler is not None:
                            torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.network.state_dict(),
                            'optimizer_state_dict': self.optimiser.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            }, f"model_checkpoint_{run.id}_{epoch}.pt")
                        else:
                            torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.network.state_dict(),
                            'optimizer_state_dict': self.optimiser.state_dict(),
                            }, f"model_checkpoint_{run.id}_{epoch}.pt")
                        
                        wandb.save(f"model_checkpoint_{run.id}_{epoch}.pt")
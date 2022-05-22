from tqdm import tqdm
import torch
from apex import amp
from typing import Optional

def iterate_loader(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_function: torch.optim.Optimizer,
    local_rank: int,
    apex_optimizer: Optional[torch.optim.Optimizer],
    training: bool = False
    ):
    loss, acc, num = 0, 0, 0
    for image, target in tqdm(loader, total=len(loader)):
        if training:
            apex_optimizer.zero_grad()
        image = image.to(local_rank)
        target = target.to(local_rank, dtype=torch.long)
        outputs = model(image)
        predict = torch.argmax(outputs, dim=1)
        batch_loss = loss_function(outputs, target)
        batch_loss /= len(outputs)
        if training:
            # Apex
            with amp.scale_loss(batch_loss, apex_optimizer) as scaled_loss:
                scaled_loss.backward()
            apex_optimizer.step()

        # Calculate loss & acc
        loss += batch_loss.item() * len(image)
        acc += (predict == target).sum().item()
        num += len(image)

    loss = loss / num
    acc = acc / num

    if training:
        curr_lr = apex_optimizer.param_groups[0]['lr']
        return loss, acc, curr_lr
    else:
        return loss, acc
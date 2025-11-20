import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr 
import math

# What happens in the forward pass:

# 1. Model input (what the model receives):
# model_input = input_ids[:-1]  # All tokens EXCEPT the last one
# [150, 2341, 15678, 234, 8901]
# Când  sunt  admi   la   FMI

# 2. Predictions (what the model generates):
# predictions = model(model_input)
# Shape: (5, vocab_size) - 5 predictions, one for each position

# 3. Target (what it should predict):
# targets = labels[1:]  # All tokens EXCEPT the first one
# [2341, 15678, 234, 8901, 2]
# sunt  admi   la   FMI   ?

# 4. Comparison (loss calculation):
# Position 0: input="Când"       → predict="sunt"      vs target="sunt"      ✓
# Position 1: input="sunt"       → predict="admiterile" vs target="admiterile" ✓
# Position 2: input="admiterile" → predict="la"        vs target="la"        ✓
# Position 3: input="la"         → predict="FMI"       vs target="FMI"       ✓
# Position 4: input="FMI"        → predict="?"         vs target="?"         ✓


def train(model, dataloader, optimizer, device, grad_accum_steps=4, epochs=3, max_grad_norm=1.0,scheduler=None):
    model.train()
    is_master = (xr.global_ordinal() == 0)
    world_size = xr.world_size()  



    for epoch in range(epochs):
        if is_master:
            print(f"\n{'='*60}\nEpoch {epoch + 1}/{epochs}\n{'='*60}")

        epoch_loss_sum = 0.0
        steps_count = 0
        optimizer.zero_grad()
        total_batches = len(dataloader)

        for step, batch in enumerate(dataloader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            valid_mask = (labels != -100).any(dim=1)
            if not valid_mask.any():
                xm.mark_step()
                continue
            if valid_mask.sum().item() != labels.size(0):
                input_ids = input_ids[valid_mask]
                attention_mask = attention_mask[valid_mask]
                labels = labels[valid_mask]

            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                loss = outputs.loss / (grad_accum_steps * world_size)

                if not math.isfinite(loss.item()):
                    optimizer.zero_grad(set_to_none=True)
                    xm.mark_step()
                    continue

                loss.backward()
            except RuntimeError as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e):
                    if xr.global_ordinal() == 0:
                        print("[WARN] OOM/RESOURCE_EXHAUSTED on batch; skipping.")
                    optimizer.zero_grad(set_to_none=True)
                    xm.mark_step()
                    continue
                raise

            if step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
                xm.optimizer_step(optimizer, barrier=True)

                if scheduler is not None:
                    scheduler.step()
                    
                optimizer.zero_grad(set_to_none=True)
                xm.mark_step()

            with torch.no_grad():
                step_loss = float(loss.detach() * grad_accum_steps)
                epoch_loss_sum += step_loss
                steps_count += 1

            if step % 100 == 0 and is_master:
                avg_loss = epoch_loss_sum / max(1, steps_count)
                print(f"Step [{step}/{total_batches}] - Loss: {step_loss:.4f} - Avg Loss: {avg_loss:.4f}")

        avg_epoch_loss = epoch_loss_sum / max(1, steps_count)
        if is_master:
            print(f"\nEpoch {epoch + 1} completed - Average Loss: {avg_epoch_loss:.4f}")

    return avg_epoch_loss
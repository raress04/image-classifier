import os
import torch
from datetime import datetime
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Ruleaza un singur pas de antrenare pe intregul loader."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evalueaza modelul pe loader-ul dat.
    Returneaza: loss, accuracy, f1_macro, f1_micro, all_preds, all_labels
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / total
    acc = correct / total
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)

    return avg_loss, acc, f1_macro, f1_micro, all_preds, all_labels


def run_training(model, train_loader, val_loader, criterion, optimizer,
                 num_epochs, device, run_name, scheduler=None):
    """
    Bucla principala de antrenare cu logging in TensorBoard.
    Salveaza cel mai bun model dupa F1-macro.
    """
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_f1_macro": [], "val_f1_micro": [],
    }
    best_f1_macro = 0.0

    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1_macro, va_f1_micro, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # Actualizare history
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_f1_macro"].append(va_f1_macro)
        history["val_f1_micro"].append(va_f1_micro)

        # TensorBoard logging
        writer.add_scalar("Loss/train",    tr_loss,     epoch)
        writer.add_scalar("Loss/val",      va_loss,     epoch)
        writer.add_scalar("Acc/train",     tr_acc,      epoch)
        writer.add_scalar("Acc/val",       va_acc,      epoch)
        writer.add_scalar("F1macro/val",   va_f1_macro, epoch)
        writer.add_scalar("F1micro/val",   va_f1_micro, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        if scheduler is not None:
            scheduler.step()

        # Salvam cel mai bun checkpoint bazat pe F1-macro
        if va_f1_macro > best_f1_macro:
            best_f1_macro = va_f1_macro
            ckpt_path = f"checkpoints/{run_name}_best.pt"
            torch.save(model.state_dict(), ckpt_path)

        print(f"Epoch {epoch+1:02d}/{num_epochs} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val loss={va_loss:.4f} acc={va_acc:.3f} f1_macro={va_f1_macro:.4f}")

    writer.close()
    return history

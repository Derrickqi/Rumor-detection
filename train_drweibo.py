import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from collections import Counter



TRAIN_PT = "data/drweibo_train.pt"
TEST_PT  = "data/drweibo_test.pt"

MODEL_NAME = "bert-base-chinese"   # DRWeibo 中文
NUM_CLASSES = 2

EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01

BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODE = "root"      # "root" / "decay"
DECAY_GAMMA = 0.6


MAX_NODES = 64

USE_CLASS_WEIGHT = True



class RumorTreePTDataset(Dataset):
    def __init__(self, pt_path):
        self.data = torch.load(pt_path, map_location="cpu")
        assert isinstance(self.data, list), "pt file should be a list"
        assert len(self.data) > 0, "pt file is empty"
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch_list):
    return batch_list[0]


def get_label_distribution(dataset):
    labels = []
    for i in range(len(dataset)):
        lab = dataset[i]["label"]
        labels.append(int(lab.item()) if hasattr(lab, "item") else int(lab))
    return Counter(labels)


def build_class_weights(counter, num_classes=2):
    total = sum(counter.values())
    weights = []
    for c in range(num_classes):
        cnt = counter.get(c, 1)
        weights.append(total / (num_classes * cnt))
    return torch.tensor(weights, dtype=torch.float)


def depth_decay_weights(depths, gamma=0.6):
    w = gamma ** depths.float()
    return w / (w.sum() + 1e-9)


class RumorModel(nn.Module):
    def __init__(self, model_name="bert-base-chinese", num_classes=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hid = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hid, num_classes)

    def forward(self, input_ids, attention_mask, depths, mode="root", gamma=0.6):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [N, H]

        if mode == "root":
            tree_vec = cls[0]  # root only

        elif mode == "decay":
            w = depth_decay_weights(depths, gamma=gamma).to(cls.device)
            tree_vec = (cls * w.unsqueeze(-1)).sum(dim=0)

        else:
            raise ValueError("MODE must be root or decay")

        logits = self.classifier(tree_vec.unsqueeze(0))
        return logits


def pack_inputs(sample, device, mode="root", max_nodes=64):
    if mode == "root":
        input_ids = sample["input_ids"][:1]
        attention_mask = sample["attention_mask"][:1]
        depths = sample["depths"][:1]
    else:
        input_ids = sample["input_ids"][:max_nodes]
        attention_mask = sample["attention_mask"][:max_nodes]
        depths = sample["depths"][:max_nodes]

    return (
        input_ids.to(device),
        attention_mask.to(device),
        depths.to(device),
    )


@torch.no_grad()
def evaluate(model, loader, device, mode, gamma, max_nodes=64):
    model.eval()
    y_true, y_pred = [], []

    for sample in loader:
        input_ids, attention_mask, depths = pack_inputs(sample, device, mode, max_nodes)
        label = int(sample["label"].item()) if hasattr(sample["label"], "item") else int(sample["label"])

        logits = model(input_ids, attention_mask, depths, mode=mode, gamma=gamma)
        pred = logits.argmax(dim=-1).item()

        y_true.append(label)
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    rumor_f1 = f1_score(y_true, y_pred, pos_label=1)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\n[Eval] True distribution :", Counter(y_true))
    print("[Eval] Pred distribution :", Counter(y_pred))
    print("[Eval] Confusion Matrix (rows=true, cols=pred):\n", cm)

    return acc, macro_f1, rumor_f1


def train_one_epoch(model, loader, optimizer, scheduler, device, loss_fn, mode, gamma, max_nodes=64):
    model.train()
    total_loss = 0.0
    steps = 0

    for sample in loader:
        input_ids, attention_mask, depths = pack_inputs(sample, device, mode, max_nodes)
        label = sample["label"].to(device).view(1)

        logits = model(input_ids, attention_mask, depths, mode=mode, gamma=gamma)
        loss = loss_fn(logits, label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


def main():
    print("DEVICE =", DEVICE)
    print("TRAIN_PT =", TRAIN_PT)
    print("TEST_PT  =", TEST_PT)
    print("MODEL_NAME =", MODEL_NAME)
    print("MODE =", MODE)
    print("MAX_NODES =", MAX_NODES)
    print("DECAY_GAMMA =", DECAY_GAMMA)

    train_dataset = RumorTreePTDataset(TRAIN_PT)
    test_dataset = RumorTreePTDataset(TEST_PT)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # class weights
    if USE_CLASS_WEIGHT:
        dist = get_label_distribution(train_dataset)
        print("\n[Train] Label distribution:", dist)
        class_weights = build_class_weights(dist, NUM_CLASSES).to(DEVICE)
        print("[Train] Class weights:", class_weights.detach().cpu().tolist())
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    model = RumorModel(MODEL_NAME, NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_macro = -1
    best_info = None

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, DEVICE, loss_fn, MODE, DECAY_GAMMA, MAX_NODES)

        acc, macro_f1, rumor_f1 = evaluate(model, test_loader, DEVICE, MODE, DECAY_GAMMA, MAX_NODES)

        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Test  Acc : {acc:.4f}")
        print(f"Test  Macro-F1: {macro_f1:.4f}")
        print(f"Test  Rumor-F1: {rumor_f1:.4f}")

        if macro_f1 > best_macro:
            best_macro = macro_f1
            best_info = (epoch, acc, macro_f1, rumor_f1)

    if best_info:
        e, acc, mf1, rf1 = best_info
        print("\n✅ BEST RESULT (by Macro-F1)")
        print(f"Epoch={e} | Acc={acc:.4f} | Macro-F1={mf1:.4f} | Rumor-F1={rf1:.4f}")

    print("\n✅ Training Finished.")


if __name__ == "__main__":
    main()

import os
import json
import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer


# =======================
# 你只需要改这里
# =======================
DRWEIBO_DIR = "DRWeibo"          # 你的 DRWeibo 数据目录（里面放很多 json）
OUT_DIR = "data"                # 输出 pt 保存目录
MODEL_NAME = "bert-base-chinese"
MAX_LEN = 128

TRAIN_RATIO = 0.8               # 训练集比例
SEED = 42

SAVE_TRAIN_NAME = "drweibo_train.pt"
SAVE_TEST_NAME  = "drweibo_test.pt"

# 控制树规模（防止太大爆显存）
MAX_NODES = 512
# =======================


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


def list_json_files(folder: str) -> List[str]:
    files = []
    for fn in os.listdir(folder):
        if fn.endswith(".json"):
            files.append(os.path.join(folder, fn))
    files.sort()
    return files


def build_children(comments: List[Dict]) -> Dict[int, List[int]]:
    """
    comment 字段里有：
      comment id: int
      parent: int  (-1 表示挂在 root)
      children: list  (有些文件是空)
    我们统一用 parent 构建树即可。
    """
    children = defaultdict(list)
    for c in comments:
        cid = int(c.get("comment id"))
        parent = int(c.get("parent", -1))
        children[parent].append(cid)
    return children


def compute_depths(children: Dict[int, List[int]]) -> Dict[int, int]:
    """
    root depth=0
    comment id depth = parent depth + 1
    root 用 parent=-1 表示
    """
    depth = { -1: 0 }  # 虚拟root
    q = deque([-1])

    while q:
        u = q.popleft()
        for v in children.get(u, []):
            depth[v] = depth[u] + 1
            q.append(v)

    # 去掉虚拟root
    depth.pop(-1, None)
    return depth


def safe_text(x) -> str:
    if x is None:
        return ""
    x = str(x).strip()
    return x


def encode_tree_sample(data: Dict, tokenizer: AutoTokenizer, max_len=128, max_nodes=512):
    """
    把一个事件 json -> 一个 sample dict
    """
    source = data["source"]
    comments = data.get("comment", [])

    eid = safe_text(source.get("tweet id", ""))
    label = int(source.get("label", 0))  # 你的样例 label=1
    root_text = safe_text(source.get("content", ""))

    # 1) 构建树深度
    children = build_children(comments)
    depth_map = compute_depths(children)

    # 2) 按 BFS 顺序排列节点（root + comments）
    # root 放第一个
    node_texts = [root_text]
    node_depths = [0]

    # 以虚拟root(-1)出发 bfs
    q = deque(children.get(-1, []))
    visited = set()

    while q and len(node_texts) < max_nodes:
        cid = q.popleft()
        if cid in visited:
            continue
        visited.add(cid)

        # 找到 comment 内容
        cobj = None
        # comments 是 list，简单线性找（数据量大可以优化为 dict）
        # 这里我们先转成 dict 方便索引
        # 但为了省内存，我们只在这里做一次映射
        # => 更稳：先构造 dict
        # 下面实现会在外层做映射，不在这里重复

        # 这个函数内部不做线性扫描，外层会传 comment_map
        # 为兼容，这里先略
        q.extend(children.get(cid, []))

    # 上面 BFS 还缺 comment 文本填充，我们改成：先建 map 再 BFS
    comment_map = {int(c["comment id"]): c for c in comments}

    node_texts = [root_text]
    node_depths = [0]

    q = deque(children.get(-1, []))
    visited = set()

    while q and len(node_texts) < max_nodes:
        cid = q.popleft()
        if cid in visited:
            continue
        visited.add(cid)

        cobj = comment_map.get(cid, {})
        ctext = safe_text(cobj.get("content", ""))

        # ✅ 如果 comment 为空，就用特殊占位符（避免空字符串导致 tokenizer 异常短）
        if len(ctext) == 0:
            ctext = "[EMPTY]"

        node_texts.append(ctext)
        node_depths.append(depth_map.get(cid, 1))

        # bfs children
        for nxt in children.get(cid, []):
            if nxt not in visited:
                q.append(nxt)

    # 3) tokenize
    enc = tokenizer(
        node_texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    sample = {
        "eid": eid,
        "input_ids": enc["input_ids"],            # [N, L]
        "attention_mask": enc["attention_mask"],  # [N, L]
        "depths": torch.tensor(node_depths, dtype=torch.long),  # [N]
        "label": torch.tensor(label, dtype=torch.long)
    }
    return sample


def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    files = list_json_files(DRWEIBO_DIR)
    assert len(files) > 0, f"No json files found in {DRWEIBO_DIR}"

    print(f"[DRWeibo] total json files = {len(files)}")
    print("[Tokenizer]", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 1) 随机打乱 + 划分
    random.shuffle(files)
    split = int(len(files) * TRAIN_RATIO)
    train_files = files[:split]
    test_files = files[split:]

    print(f"[Split] train={len(train_files)} test={len(test_files)} (ratio={TRAIN_RATIO})")

    # 2) 构造 pt list
    train_samples = []
    for i, fp in enumerate(train_files, 1):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        sample = encode_tree_sample(
            data=data,
            tokenizer=tokenizer,
            max_len=MAX_LEN,
            max_nodes=MAX_NODES
        )
        train_samples.append(sample)

        if i % 200 == 0:
            print(f"[Train] processed {i}/{len(train_files)}")

    test_samples = []
    for i, fp in enumerate(test_files, 1):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        sample = encode_tree_sample(
            data=data,
            tokenizer=tokenizer,
            max_len=MAX_LEN,
            max_nodes=MAX_NODES
        )
        test_samples.append(sample)

        if i % 200 == 0:
            print(f"[Test] processed {i}/{len(test_files)}")

    # 3) 保存
    train_out = os.path.join(OUT_DIR, SAVE_TRAIN_NAME)
    test_out = os.path.join(OUT_DIR, SAVE_TEST_NAME)

    torch.save(train_samples, train_out)
    torch.save(test_samples, test_out)

    print("\n✅ Saved:")
    print("  Train:", train_out, "| num_samples =", len(train_samples))
    print("  Test :", test_out,  "| num_samples =", len(test_samples))

    # 4) 简单检查一条
    ex = train_samples[0]
    print("\n[Example]")
    print("eid =", ex["eid"])
    print("input_ids shape =", ex["input_ids"].shape)
    print("depths shape =", ex["depths"].shape)
    print("label =", ex["label"].item())


if __name__ == "__main__":
    main()

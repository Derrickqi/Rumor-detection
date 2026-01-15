#生成pt脚本
# import os
# import torch
# import sys
# from collections import defaultdict, deque
# from transformers import AutoTokenizer
#
# # ====== 配置区 ======
# BASE_DIR = "rumor_detection_acl2017"
# OUT_DIR = "data"
# MODEL_NAME = "bert-base-uncased"
# MAX_LEN = 128
# MAX_NODES_PER_TREE = 1000  # 🚀 新增：限制单棵树最大节点数，防止某些异常树撑爆内存
# SPLITS = ["twitter15", "twitter16"]
#
#
# def parse_user_id(triple: str) -> str:
#     """极其鲁棒的 ID 提取"""
#     try:
#         s = triple.strip().replace("[", "").replace("]", "").replace("'", "").replace('"', "")
#         parts = s.split(",")
#         return parts[0].strip()
#     except:
#         return "UNKNOWN"
#
#
# def parse_tree_file_fast(tree_path):
#     """带节点上限的快速解析"""
#     children = defaultdict(list)
#     root = None
#
#     # 增加文件读取异常处理
#     try:
#         with open(tree_path, "r", encoding="utf-8", errors='ignore') as f:
#             for line_idx, line in enumerate(f):
#                 if line_idx > 10000: break  # 防止文件过载
#                 line = line.strip()
#                 if "->" not in line: continue
#
#                 left, right = line.split("->")
#                 p_user = parse_user_id(left)
#                 c_user = parse_user_id(right)
#
#                 if p_user == "ROOT":
#                     root = c_user
#                 else:
#                     children[p_user].append(c_user)
#     except Exception as e:
#         return None, None
#
#     if root is None: return None, None
#
#     # BFS 计算深度
#     depth = {root: 0}
#     q = deque([root])
#     while q:
#         u = q.popleft()
#         if len(depth) > MAX_NODES_PER_TREE: break  # 🚀 截断超大型树
#         for v in children.get(u, []):
#             if v not in depth:
#                 depth[v] = depth[u] + 1
#                 q.append(v)
#
#     return depth, root
#
#
# def load_labels(label_path):
#     labels = {}
#     if not os.path.exists(label_path): return labels
#     with open(label_path, "r", encoding="utf-8") as f:
#         for line in f:
#             if ":" in line:
#                 label, eid = line.strip().split(":")
#                 labels[eid] = label
#     return labels
#
#
# def load_sources(source_path):
#     sources = {}
#     if not os.path.exists(source_path): return sources
#     with open(source_path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split("\t", 1)
#             if len(parts) == 2:
#                 sources[parts[0]] = parts[1]
#     return sources
#
#
# def preprocess_split(split, tokenizer, empty_enc):
#     split_dir = os.path.join(BASE_DIR, split)
#     tree_dir = os.path.join(split_dir, "tree")
#     labels = load_labels(os.path.join(split_dir, "label.txt"))
#     sources = load_sources(os.path.join(split_dir, "source_tweets.txt"))
#
#     event_ids = list(labels.keys())
#     label_map = {"false": 0, "true": 1, "non-rumor": 0, "unverified": 1}
#     samples = []
#
#     print(f"\n🚀 开始处理 {split} | 共 {len(event_ids)} 个事件")
#
#     for i, eid in enumerate(event_ids, start=1):
#         tree_path = os.path.join(tree_dir, f"{eid}.txt")
#         if not os.path.exists(tree_path): continue
#
#         # 1. 快速解析
#         depth_dict, root = parse_tree_file_fast(tree_path)
#         if not depth_dict: continue
#
#         depths_list = [v for k, v in depth_dict.items()]
#         num_nodes = len(depths_list)
#
#         # 2. 文本处理 (使用 context manager 减少内存峰值)
#         with torch.no_grad():
#             root_text = sources.get(eid, "[EMPTY]")
#             root_enc = tokenizer(root_text, padding="max_length", truncation=True, max_length=MAX_LEN,
#                                  return_tensors="pt")
#
#             if num_nodes > 1:
#                 # 🚀 关键：使用 cat + expand，不要 repeat
#                 input_ids = torch.cat([root_enc["input_ids"], empty_enc["input_ids"].expand(num_nodes - 1, -1)], dim=0)
#                 mask = torch.cat([root_enc["attention_mask"], empty_enc["attention_mask"].expand(num_nodes - 1, -1)],
#                                  dim=0)
#             else:
#                 input_ids = root_enc["input_ids"]
#                 mask = root_enc["attention_mask"]
#
#         # 3. 存储
#         samples.append({
#             "eid": eid,
#             "input_ids": input_ids.cpu(),  # 强制放回 CPU 内存
#             "attention_mask": mask.cpu(),
#             "depths": torch.tensor(depths_list, dtype=torch.short),  # 使用 short 减小体积
#             "label": torch.tensor(label_map.get(labels[eid], 0), dtype=torch.long),
#         })
#
#         if i % 50 == 0:
#             print(f"   已完成: {i}/{len(event_ids)} | 当前树节点: {num_nodes}")
#             sys.stdout.flush()  # 强制刷新输出缓冲区，防止显示延迟
#
#     # 4. 保存
#     os.makedirs(OUT_DIR, exist_ok=True)
#     out_path = os.path.join(OUT_DIR, f"{split}.pt")
#     torch.save(samples, out_path)
#     print(f"✅ {split} 保存成功: {len(samples)} 样本")
#
#
# def main():
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     empty_enc = tokenizer("[EMPTY]", padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
#
#     for split in SPLITS:
#         preprocess_split(split, tokenizer, empty_enc)
#
#
# if __name__ == "__main__":
#     main()


# 检测pt脚本
# import torch
# from collections import Counter
# import numpy as np
#
#
# def check_pt(path, name=None, sample_k=200):
#     if name is None:
#         name = path
#
#     print("\n" + "=" * 60)
#     print(f"[CHECK] {name}")
#     print(f"[FILE ] {path}")
#
#     data = torch.load(path, map_location="cpu")
#
#     # 1) 基本信息
#     print(f"[TYPE ] {type(data)}")
#     if not isinstance(data, (list, tuple)):
#         print("[ERROR] The loaded object is not a list/tuple. Stop.")
#         return
#
#     n = len(data)
#     print(f"[NUM  ] num_samples = {n}")
#     if n == 0:
#         print("[FATAL] This pt file is EMPTY (0 samples).")
#         return
#
#     # 2) 抽样检查 keys / shape
#     required = {"input_ids", "attention_mask", "depths", "label"}
#     bad_key = 0
#     bad_shape = 0
#
#     Ns, Ls, max_depths, labels = [], [], [], []
#
#     check_n = min(sample_k, n)
#     for i in range(check_n):
#         s = data[i]
#         if not isinstance(s, dict):
#             bad_key += 1
#             continue
#
#         if not required.issubset(s.keys()):
#             bad_key += 1
#             continue
#
#         ids = s["input_ids"]
#         mask = s["attention_mask"]
#         depths = s["depths"]
#         lab = s["label"]
#
#         # label
#         try:
#             labels.append(int(lab.item()) if hasattr(lab, "item") else int(lab))
#         except:
#             labels.append("UNK")
#
#         # shapes: ids [N,L], mask [N,L], depths [N]
#         if not (torch.is_tensor(ids) and torch.is_tensor(mask) and torch.is_tensor(depths)):
#             bad_shape += 1
#             continue
#
#         if ids.dim() != 2:
#             bad_shape += 1
#             continue
#         if mask.shape != ids.shape:
#             bad_shape += 1
#             continue
#         if depths.dim() != 1 or depths.shape[0] != ids.shape[0]:
#             bad_shape += 1
#             continue
#
#         N, L = ids.shape
#         Ns.append(N)
#         Ls.append(L)
#         max_depths.append(int(depths.max().item()) if depths.numel() > 0 else -1)
#
#     print(f"[KEYS ] required keys = {sorted(list(required))}")
#     print(f"[BAD  ] bad_key_samples(first {check_n})   = {bad_key}")
#     print(f"[BAD  ] bad_shape_samples(first {check_n}) = {bad_shape}")
#
#     # 3) 统计结果
#     if len(Ns) > 0:
#         print("\n[STATS] (based on valid samples in first {})".format(check_n))
#         print(f"  Nodes N: min={min(Ns)}  mean={np.mean(Ns):.2f}  max={max(Ns)}")
#         print(f"  SeqLen L: min={min(Ls)} mean={np.mean(Ls):.2f} max={max(Ls)}")
#         print(f"  MaxDepth: min={min(max_depths)} mean={np.mean(max_depths):.2f} max={max(max_depths)}")
#     else:
#         print("\n[STATS] No valid samples found in the checked subset.")
#         return
#
#     # 4) 标签分布（抽样）
#     print("\n[LABEL] label distribution (first {} samples):".format(check_n))
#     print(" ", Counter(labels))
#
#     # 5) 深度分布（抽样 max_depth）
#     depth_counter = Counter(max_depths)
#     print("\n[DEPTH] max_depth distribution (first {} valid samples):".format(check_n))
#     for k in sorted(depth_counter.keys()):
#         print(f"  depth={k}: {depth_counter[k]}")
#
#     # 6) 打印一个样本示例
#     s0 = data[0]
#     print("\n[EXAMPLE] sample[0] keys:", list(s0.keys()))
#     print("  input_ids shape:", s0["input_ids"].shape)
#     print("  attention_mask shape:", s0["attention_mask"].shape)
#     print("  depths shape:", s0["depths"].shape)
#     print("  label:", int(s0["label"].item()) if hasattr(s0["label"], "item") else s0["label"])
#
#     print("=" * 60)
#
#
# if __name__ == "__main__":
#     # 这里改成你自己的路径
#     check_pt("data/twitter15.pt", name="twitter15.pt")
#     check_pt("data/twitter16.pt", name="twitter16.pt")


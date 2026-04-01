import re
import string
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import torch
from torch.utils.data import Dataset


# --- 1. 预处理函数 ---

def preprocess_text(text):
    """
    文本清洗：
    1. 转小写
    2. 去除 HTML 标签
    3. 去除非字母字符
    4. 去除停用词 (关键优化)
    """
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)  # 去除 HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # 只保留字母
    text = ' '.join(text.split())  # 去除多余空格

    # 去除停用词 (the, is, at 等无意义词)
    tokens = text.split()
    tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS and len(w) > 2]

    return ' '.join(tokens)


def build_vocab(texts, min_freq=2):
    """构建词汇表"""
    word_freq = Counter()
    for text in texts:
        words = text.split()
        word_freq.update(words)

    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_freq.items():
        if freq >= min_freq:
            word_to_idx[word] = len(word_to_idx)
    return word_to_idx


# --- 2. 数据加载主函数 ---

def load_and_preprocess_data():
    categories = ['alt.atheism', 'soc.religion.christian']

    print("正在加载数据...")
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

    # 预处理文本
    print("正在预处理文本...")
    X_train_raw = [preprocess_text(doc) for doc in newsgroups_train.data]
    X_test_raw = [preprocess_text(doc) for doc in newsgroups_test.data]

    # 构建词汇表
    word_to_idx = build_vocab(X_train_raw + X_test_raw, min_freq=2)
    print(f"词汇表大小: {len(word_to_idx)}")

    # 将文本转换为数字序列
    def encode_texts(texts, word_to_idx):
        encoded = []
        for text in texts:
            seq = [word_to_idx.get(word, 1) for word in text.split() if word in word_to_idx]
            if not seq: seq = [1]
            encoded.append(seq)
        return encoded

    X_train_seq = encode_texts(X_train_raw, word_to_idx)
    X_test_seq = encode_texts(X_test_raw, word_to_idx)

    # --- 标签编码修复：确保标签是 [0, 1] ---
    original_train_labels = newsgroups_train.target
    original_test_labels = newsgroups_test.target

    # 获取唯一的标签值并排序，然后映射到 [0, 1]
    unique_labels = sorted(list(set(original_train_labels)))
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

    y_train = [label_map[label] for label in original_train_labels]
    y_test = [label_map[label] for label in original_test_labels]

    # 拆分训练集和验证集
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_seq, y_train, test_size=0.2, random_state=42
    )

    return (X_train_final, y_train_final), (X_val, y_val), (X_test_seq, y_test), word_to_idx

# --- 3. PyTorch Dataset 类 ---

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 直接返回 Tensor
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# --- 4. 修复后的 collate_fn (关键修复) ---

def collate_fn(batch):
    """
    用于 DataLoader，负责将一批数据填充到相同长度。
    修复了之前的嵌套错误。
    """
    texts, labels = zip(*batch)

    # 1. 将列表转换为 Tensor 列表
    texts = [torch.tensor(t, dtype=torch.long) for t in texts]

    # 2. 动态填充 (Padding)
    # batch_first=True 保证输出形状是 (Batch, Seq_Len)
    padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)

    return padded_texts, torch.tensor(labels, dtype=torch.long)
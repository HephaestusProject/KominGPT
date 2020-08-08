"""
    This script was made by Nick at 19/07/20.
    To implement code for data pipeline. (e.g. custom class subclassing torch.utils.data.Dataset)
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BertTokenizer


def build_tokenizer(args):
    return BertTokenizer.from_pretrained(
        "bert-base-multilingual-uncased",
        cache_dir=args.tokenizer_path,
        do_lower_case=args.do_lower_case,
    )


def build_vocab(args):
    return BertTokenizer.from_pretrained(
        "bert-base-multilingual-uncased",
        cache_dir=args.tokenizer_path,
        do_lower_case=args.do_lower_case,
    ).get_vocab()


class TextClassificationDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.args = args
        self.tokenizer = build_tokenizer(args)
        self.vocabulary = build_vocab(args)
        self.token_max_len = self.args.token_max_len
        self.pad_token = self.args.pad_token
        self.cls_token = self.args.cls_token
        self.unk_token = self.args.unk_token
        self.sep_token = self.args.sep_token

    def _tokenize_text(self, text):
        return self.tokenizer.tokenize(text)

    def _token_to_index(self, tokens):
        return [
            self.vocabulary[token]
            if token in self.vocabulary
            else self.vocabulary[self.unk_token]
            for token in tokens
        ]

    def _add_cls(self, tokens):
        return [self.cls_token] + tokens

    def _add_sep(self, tokens):
        return tokens + [self.sep_token]

    def _preprocess(self, text):
        tokens = self._tokenize_text(text)
        indicies = self._token_to_index(tokens)

        return indicies[: self.token_max_len]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (
            self._preprocess(self.dataset[index]["text"]),
            self.dataset[index]["label"],
        )


class TextClassificationCollate(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, batch):
        inputs = pad_sequence(
            [torch.LongTensor(x[0]) for x in batch],
            batch_first=True,
            padding_value=self.args.padding_value,
        )
        mask = (inputs != self.args.padding_value).float()
        labels = torch.LongTensor([x[1] for x in batch])

        return inputs, mask, labels

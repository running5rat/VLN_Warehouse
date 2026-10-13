"""Exact greedy grammar decoding: test high-score tokens before enumerating the vocabulary.

This is batch-one greedy inference, not sampling or beam search. Every accepted token
is checked by the same character grammar. Exhaustive fallback preserves constrained
argmax semantics when no high-score token is legal.
"""

import torch
from transformers import LogitsProcessor


class GreedyGrammarProcessor(LogitsProcessor):
    def __init__(self, tokenizer, parser, prefix_length):
        self.tokenizer = tokenizer
        self.root = parser
        self.prefix_length = prefix_length
        self.prefix = ""
        self.state = parser
        self.calls = 0
        self.tested_tokens = 0
        self.exhaustive_calls = 0

    @staticmethod
    def advance(parser, text):
        for char in text:
            if char not in parser.get_allowed_characters():
                return None
            parser = parser.add_character(char)
        return parser

    def __call__(self, input_ids, scores):
        if len(input_ids) != 1:
            raise ValueError("GreedyGrammarProcessor requires batch=1, num_beams=1")
        suffix = input_ids[0, self.prefix_length :].tolist()
        actual = self.tokenizer.decode(suffix, skip_special_tokens=False)
        if actual != self.prefix:
            self.state = self.advance(self.root, actual)
            if self.state is None:
                raise ValueError("Generated prefix violates grammar")
            self.prefix = actual
        self.calls += 1
        seen = set()
        for limit in (64, 512, scores.shape[-1]):
            limit = min(limit, scores.shape[-1])
            if limit == scores.shape[-1]:
                self.exhaustive_calls += 1
            values, ids = scores[0].topk(limit, sorted=True)
            values, ids = values.tolist(), ids.tolist()
            for value, token in zip(values, ids):
                if token in seen or value == -float("inf"):
                    continue
                seen.add(token)
                self.tested_tokens += 1
                text = self.tokenizer.decode(
                    suffix + [token], skip_special_tokens=False
                )
                if text == actual:
                    continue
                candidate = (
                    self.advance(self.state, text[len(actual) :])
                    if text.startswith(actual)
                    else self.advance(self.root, text)
                )
                if candidate is None:
                    continue
                self.prefix, self.state = text, candidate
                masked = torch.full_like(scores, -float("inf"))
                masked[0, token] = scores[0, token]
                return masked
        raise ValueError("No finite-score token satisfies the output grammar")

    def record(self):
        return dict(
            method="character-grammar-checked greedy argmax; exhaustive fallback",
            calls=self.calls,
            tested_tokens=self.tested_tokens,
            exhaustive_calls=self.exhaustive_calls,
            sampling=False,
            beam_search=False,
        )

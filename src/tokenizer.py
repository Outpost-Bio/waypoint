"""
Taxonomic tokenizer — standalone, no private dependencies.

Each taxonomic string (e.g. "k__Bacteria; p__Firmicutes; g__Lactobacillus")
maps to a single token ID.  When *rank* is set (e.g. "genus"), only that
rank's label is extracted as the token.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

from transformers import PreTrainedTokenizer

TaxonRankType = Literal[
    "species", "genus", "family", "order", "class", "phylum", "kingdom"
]

TAXON_RANK_PREFIXES: dict[TaxonRankType, str] = {
    "species": "s__",
    "genus": "g__",
    "family": "f__",
    "order": "o__",
    "class": "c__",
    "phylum": "p__",
    "kingdom": "k__",
}

TAXON_RANK_ORDER: tuple[TaxonRankType, ...] = (
    "species",
    "genus",
    "family",
    "order",
    "class",
    "phylum",
    "kingdom",
)

PREFIX_TO_RANK = {v: k for k, v in TAXON_RANK_PREFIXES.items()}
RANK_TO_IDX = {r: i for i, r in enumerate(TAXON_RANK_ORDER)}
PREFIX_LEN = 3

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}


def extract_taxon_label(
    taxon: str,
    rank: TaxonRankType,
    fallback_to_higher_rank: bool = False,
) -> str | None:
    """Extract a single rank label (e.g. 'g__Lactobacillus') from a full taxonomy string."""
    target_idx = RANK_TO_IDX[rank]
    fallback_seg: str | None = None
    fallback_idx = len(RANK_TO_IDX)
    for seg in taxon.strip().split(";"):
        seg = seg.strip()
        if len(seg) < PREFIX_LEN:
            continue
        r = PREFIX_TO_RANK.get(seg[:PREFIX_LEN])
        if r is None:
            continue
        rank_idx = RANK_TO_IDX[r]
        if rank_idx == target_idx:
            return seg
        if (
            fallback_to_higher_rank
            and rank_idx > target_idx
            and rank_idx < fallback_idx
        ):
            fallback_seg = seg
            fallback_idx = rank_idx
    return fallback_seg


class TaxonomicTokenizer(PreTrainedTokenizer):
    """
    Tokenizer for taxonomic sequences.  Each taxonomic string maps to a single
    integer token.  A sentence is a sequence of such tokens, separated by
    newlines when encoded from text.

    When *rank* is None, full taxonomic strings are used as tokens.
    When *rank* is set (e.g. "genus"), only that rank's label is extracted.
    *fallback_to_higher_rank* uses the next-higher rank when the target is missing.
    """

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        taxa: list[str] | None = None,
        vocab_file: str | Path | None = None,
        rank: TaxonRankType | None = None,
        fallback_to_higher_rank: bool = False,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        mask_token: str = "<mask>",
        **kwargs,
    ) -> None:
        self._rank = rank
        self._fallback_to_higher_rank = fallback_to_higher_rank

        if vocab is None and taxa is None and vocab_file is None:
            raise ValueError("Provide vocab, taxa, or vocab_file")
        if vocab is None and vocab_file is not None:
            with open(vocab_file, encoding="utf-8") as f:
                vocab = json.load(f)
        elif vocab is None:
            assert taxa is not None
            if rank is None:
                taxa_for_vocab = list(
                    dict.fromkeys(t.strip() for t in taxa if t.strip())
                )
            else:
                taxa_for_vocab = [
                    label
                    for t in taxa
                    if (label := extract_taxon_label(t, rank, fallback_to_higher_rank))
                    is not None
                ]
            vocab = self._build_vocab_from_taxa(
                taxa_for_vocab, pad_token, unk_token, bos_token, eos_token, mask_token
            )

        assert vocab is not None
        self._token_to_id: dict[str, int] = vocab
        self._id_to_token = {v: k for k, v in vocab.items()}

        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            rank=rank,
            fallback_to_higher_rank=fallback_to_higher_rank,
            **kwargs,
        )
        self.add_special_tokens(
            {
                "pad_token": pad_token,
                "unk_token": unk_token,
                "bos_token": bos_token,
                "eos_token": eos_token,
                "mask_token": mask_token,
            }
        )

    @staticmethod
    def _build_vocab_from_taxa(
        taxa: list[str],
        pad_token: str,
        unk_token: str,
        bos_token: str,
        eos_token: str,
        mask_token: str,
    ) -> dict[str, int]:
        special_tokens = {
            pad_token: 0,
            unk_token: 1,
            bos_token: 2,
            eos_token: 3,
            mask_token: 4,
        }
        unique_taxa = sorted(set(taxa))
        return {
            **special_tokens,
            **{t: i + len(special_tokens) for i, t in enumerate(unique_taxa)},
        }

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    @property
    def vocab(self) -> dict[str, int]:
        return self._token_to_id

    def _extract(self, taxon: str) -> str | None:
        if self._rank is None:
            return taxon.strip() or None
        return extract_taxon_label(
            taxon, cast(TaxonRankType, self._rank), self._fallback_to_higher_rank
        )

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        if self._rank is None:
            return lines
        unk = str(self.unk_token) if self.unk_token else "<unk>"
        return [self._extract(line) or unk for line in lines]

    def _convert_token_to_id(self, token: str) -> int:
        if self._rank is not None and token not in self._token_to_id:
            extracted = self._extract(token)
            if extracted is not None:
                token = extracted
        unk = str(self.unk_token) if self.unk_token else "<unk>"
        return self._token_to_id.get(token, self._token_to_id[unk])

    def _convert_id_to_token(self, index: int) -> str:
        unk = str(self.unk_token) if self.unk_token else "<unk>"
        return self._id_to_token.get(index, unk)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "\n".join(tokens)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

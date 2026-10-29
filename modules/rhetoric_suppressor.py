import torch
from exllamav3.generator.filter import Filter
from modules.logging_colors import logger


class RhetoricSuppressionFilter(Filter):
    NEGATION_WORDS = frozenset(
        {
            "don't",
            "doesn't",
            "didn't",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
            "won't",
            "wouldn't",
            "can't",
            "couldn't",
            "cannot",
            "shouldn't",
            "mustn't",
            "needn't",
            "hasn't",
            "haven't",
            "hadn't",
            "not",
            "never",
        }
    )

    MINIMIZER_WORDS = frozenset({"just", "simply", "merely", "only", "barely"})
    CONNECTOR_CHARS = frozenset({";", "\u2014", "\u2013"})

    MINIMIZER_WINDOW = 8
    CONNECTOR_WINDOW = 18

    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, trigger_token=None, prefix_str=None, eos_after_completed=False)
        self.token_count = 0
        self.last_negation_at = -9999
        self._recent_pieces = []
        self._ready = False
        self._minimizer_ids = None
        self._connector_ids = None

    def _lazy_init(self):
        if self._ready:
            return
        self._minimizer_ids = self._find_minimizer_ids()
        self._connector_ids = self._find_connector_ids()
        self._ready = True
        if self._minimizer_ids or self._connector_ids:
            logger.info(
                f"RhetoricSuppressionFilter: "
                f"{len(self._minimizer_ids)} minimizer tokens, "
                f"{len(self._connector_ids)} connector tokens"
            )

    def _find_minimizer_ids(self):
        ids = set()
        id_to_piece = self.tokenizer.get_id_to_piece_list(True)
        for token_id, piece in enumerate(id_to_piece):
            if token_id >= self.vocab_size:
                break
            stripped = piece.replace("\u2581", " ").strip().lower()
            if stripped in self.MINIMIZER_WORDS:
                ids.add(token_id)
        return ids

    def _find_connector_ids(self):
        ids = set()
        id_to_piece = self.tokenizer.get_id_to_piece_list(True)
        for token_id, piece in enumerate(id_to_piece):
            if token_id >= self.vocab_size:
                break
            clean = piece.replace("\u2581", " ").strip()
            non_space = [c for c in clean if c != " "]
            if len(non_space) == 1 and non_space[0] in self.CONNECTOR_CHARS:
                ids.add(token_id)
        return ids

    def reset(self):
        self.token_count = 0
        self.last_negation_at = -9999
        self._recent_pieces = []

    def accept_token(self, token):
        self.token_count += 1
        piece = self.tokenizer.decode(torch.tensor([token], dtype=torch.long))
        self._recent_pieces.append(piece)
        if len(self._recent_pieces) > 3:
            self._recent_pieces.pop(0)

        window = self.token_count - self.last_negation_at
        if window > self.CONNECTOR_WINDOW:
            for n in [1, 2, 3]:
                if len(self._recent_pieces) >= n:
                    candidate = "".join(self._recent_pieces[-n:]).lower()
                    for w in candidate.split():
                        stripped = w.strip(".,!?;:'\"()[]{}")
                        if stripped in self.NEGATION_WORDS:
                            self.last_negation_at = self.token_count
                            return

    def is_completed(self):
        return False

    def use_background_worker(self):
        return False

    def get_next_logit_mask(self):
        self._lazy_init()

        distance = self.token_count - self.last_negation_at
        if distance > self.CONNECTOR_WINDOW:
            return torch.zeros((1, self.vocab_size), dtype=self.logits_dtype)

        mask = torch.zeros((1, self.vocab_size), dtype=self.logits_dtype)
        if distance <= self.MINIMIZER_WINDOW:
            for tid in self._minimizer_ids:
                if 0 <= tid < self.vocab_size:
                    mask[0, tid] = float("-inf")
        for tid in self._connector_ids:
            if 0 <= tid < self.vocab_size:
                mask[0, tid] = float("-inf")
        return mask

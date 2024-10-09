import re
from string import ascii_lowercase

from pyctcdecode import Alphabet, BeamSearchDecoderCTC, build_ctcdecoder

import torch
import kenlm

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, librispeech_vocab_path=None, lm_path=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """
        if librispeech_vocab_path is not None:
            with open(librispeech_vocab_path) as f:
                unigrams = [t.lower() for t in f.read().strip().split("\n")]
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        if lm_path:
            lm_model = kenlm.Model(lm_path)
            self.decoder = build_ctcdecoder(self.vocab, lm_model, unigrams)
        else:
            self.decoder = BeamSearchDecoderCTC(Alphabet(self.vocab, False), None)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded = []
        last_char_ind = self.char2ind[self.EMPTY_TOK]
        for ind in inds:
            if ind == last_char_ind:
                continue
            elif ind != self.char2ind[self.EMPTY_TOK]:
                decoded.append(self.ind2char[ind])
            last_char_ind = ind
        return "".join(decoded)

    def ctc_beam_search_decode(self, inds: torch.Tensor) -> str:
        return self.decoder.decode(inds, beam_width=3)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

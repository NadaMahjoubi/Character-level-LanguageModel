
class CharTokenizer:
    def __init__(self, text=None, stoi=None, itos=None):
        if stoi is not None and itos is not None:
            self.stoi = stoi
            self.itos = {int(k): v for k, v in itos.items()}
        else:
            if text is None:
                raise ValueError("Provide text or a saved vocabulary.")

            chars = sorted(set(text))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for ch, i in self.stoi.items()}

        self.vocab_size = len(self.stoi)

    def encode(self, text):
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:
            raise ValueError(
                f"Character {exc.args[0]!r} is not in the vocabulary."
            ) from exc

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)
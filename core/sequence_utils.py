import torch


class OneHotSeqEncoder:
    def __init__(self):
        self.letter2tensor = {
            'A': torch.Tensor([0.97, 0.01, 0.01, 0.01, ]),
            'T': torch.Tensor([0.01, 0.97, 0.01, 0.01, ]),
            'G': torch.Tensor([0.01, 0.01, 0.97, 0.01, ]),
            'C': torch.Tensor([0.01, 0.01, 0.01, 0.97, ]),
            'N': torch.Tensor([0.01, 0.01, 0.01, 0.01, ]),
        }

        self.idx2letter = {
            0: 'A',
            1: 'T',
            2: 'G',
            3: 'C'
        }

    def __call__(self, sequence: str) -> torch.Tensor:
        nucleotides = torch.cat(
            [self.letter2tensor[nt][None] for nt in sequence], dim=0)
        return nucleotides.view(4, -1)

    def decode(self, sequences: torch.Tensor) -> torch.Tensor:
        result = []
        for seq in sequences:
            result.append(''.join([self.idx2letter[idx.item()] for idx in seq]))
        return result
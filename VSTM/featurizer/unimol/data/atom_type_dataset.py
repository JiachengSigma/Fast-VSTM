from functools import lru_cache
from unicore.data import BaseWrapperDataset


class AtomTypeDataset(BaseWrapperDataset):
    def __init__(
        self,
        raw_dataset,
        dataset,
        smi="smi",
        atoms="atoms",
    ):
        self.raw_dataset = raw_dataset
        self.dataset = dataset
        self.smi = smi
        self.atoms = atoms

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        # for low rdkit version
        if len(self.dataset[index]["atoms"]) != len(self.dataset[index]["coordinates"]):
            min_len = min(
                len(self.dataset[index]["atoms"]),
                len(self.dataset[index]["coordinates"]),
            )
            self.dataset[index]["atoms"] = self.dataset[index]["atoms"][:min_len]
            self.dataset[index]["coordinates"] = self.dataset[index]["coordinates"][
                :min_len
            ]
        return self.dataset[index]

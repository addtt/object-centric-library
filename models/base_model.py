from abc import abstractmethod
from dataclasses import dataclass

from torch import nn

MANDATORY_FIELDS = [
    "loss",  # training loss
    "mask",  # masks for all slots (incl. background if any)
    "slot",  # raw slot reconstructions for all slots (incl. background if any)
    "representation",  # slot representations (only foreground, if applicable)
]


@dataclass(eq=False, repr=False)
class BaseModel(nn.Module):
    name: str
    width: int
    height: int

    # This applies only to object-centric models, but must always be defined.
    num_slots: int

    def __post_init__(self):
        # Run the nn.Module initialization logic before we do anything else. Models
        # should call this post-init at the beginning of their post-init.
        super().__init__()

    @property
    def num_representation_slots(self) -> int:
        """Number of slots used for representation.

        By default, it is equal to the number of slots, but when possible we can
        consider only foreground slots (e.g. in SPACE).
        """
        return self.num_slots

    @property
    @abstractmethod
    def slot_size(self) -> int:
        """Representation size per slot.

        This does not apply to models that are not object-centric, but they should still
        define it in the most sensible possible way.
        """
        ...

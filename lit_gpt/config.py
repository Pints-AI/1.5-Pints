from dataclasses import dataclass
from typing import Type
import lit_gpt.model
from lit_gpt.config_base import ConfigBase

@dataclass
class Config(ConfigBase):

    '''
    Config subclasses ConfigBase to add on things that is not required
    outside of training, so that data processing can be done.
    Independent of additional training modules (that are imported by `lit_gpt.model`).
    For example, modules such as `flash-attention` won't run on non-CUDA devices.
    '''

    @property
    def mlp_class(self) -> Type:
        # `self._mlp_class` cannot be the type to keep the config json serializable
        return getattr(lit_gpt.model, self._mlp_class)

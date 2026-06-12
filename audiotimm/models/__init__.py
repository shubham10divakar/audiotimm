# Import each model module so their _register() calls populate the registry.
# Wave M0
from audiotimm.models import panns   # noqa: F401
from audiotimm.models import yamnet  # noqa: F401  (stub — PyTorch port coming in v0.2)
# Wave M1
from audiotimm.models import ast       # noqa: F401
from audiotimm.models import beats     # noqa: F401
from audiotimm.models import htsat    # noqa: F401
from audiotimm.models import audiomae  # noqa: F401

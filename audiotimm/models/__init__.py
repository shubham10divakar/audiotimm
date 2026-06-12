# Import each model module so their _register() calls populate the registry.
# Add new waves here as they are implemented.
from audiotimm.models import panns  # noqa: F401  Wave M0
from audiotimm.models import yamnet  # noqa: F401  Wave M0 (stub)

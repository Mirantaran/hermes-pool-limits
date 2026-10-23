"""Pool Limits — agent half.

The plugin is UI-only: the Desktop chip lives in ``~/.hermes/desktop-plugins/pool-limits/plugin.js``
and its data route in ``dashboard/plugin_api.py``. Nothing is registered with the agent (no tools, hooks or commands).
"""


def register(ctx) -> None:
    return None

from __future__ import annotations

# Map of known API hosts to UI hosts
_API_TO_UI_HOSTS: dict[str, str] = {
    "open-catalyst-api.metademolab.com": "open-catalyst.metademolab.com",
}


def get_results_ui_url(api_host: str, system_id: str) -> str | None:
    """
    Generates the URL at which results for the input system can be
    visualized.

    Args:
        api_host: The API host on which the system was run.
        system_id: ID of the system being visualized.

    Returns:
        The URL at which the input system can be visualized. None if the
        API host is not recognized.
    """
    if ui_host := _API_TO_UI_HOSTS.get(api_host):
        return f"https://{ui_host}/results/{system_id}"
    return None

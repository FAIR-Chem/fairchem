from unittest import TestCase as UnitTestCase

import requests

from fairchem.demo.ocpapi.client import get_results_ui_url


class TestUI(UnitTestCase):
    """
    Tests that calls to a real server are handled correctly.
    """

    API_HOST: str = "open-catalyst-api.metademolab.com"
    KNOWN_SYSTEM_ID: str = "f9eacd8f-748c-41dd-ae43-f263dd36d735"

    def test_get_results_ui_url(self) -> None:
        # Make sure the UI URL is reachable

        ui_url = get_results_ui_url(self.API_HOST, self.KNOWN_SYSTEM_ID)
        response = requests.head(ui_url)

        self.assertEqual(200, response.status_code)

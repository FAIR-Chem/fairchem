from unittest.mock import patch

from fairchem.core.scripts import download_large_files as dl_large


@patch.object(dl_large, "urlretreive")
def test_download_large_files(url_mock):
    def urlretrieve_mock(x, y):
        if not os.path.exists(os.path.dirname(y)):
            raise ValueError(
                f"The path to {y} does not exist. fairchem directory structure has changed,"
            )

    url_mock.side_effect = urlretrieve_mock
    dl_large.main("ALL")

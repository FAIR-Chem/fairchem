import pandas as pd
import torch
from mendeleev.fetch import fetch_table


class FixedEmbedding:
    def __init__(self, short=False, normalize=True) -> None:

        self.properties_list = [
            "atomic_radius",
            "atomic_volume",
            "atomic_weight",
            "atomic_weight_uncertainty",
            "density",
            "dipole_polarizability",
            "electron_affinity",
            "en_allen",
            "boiling_point",
            "specific_heat",
            "evaporation_heat",
            "fusion_heat",
            "melting_point",
            "thermal_conductivity",
            "heat_of_formation",
            "vdw_radius",
            "metallic_radius",
            "metallic_radius_c12",
            "covalent_radius_pyykko_double",
            "covalent_radius_pyykko_triple",
            "covalent_radius_pyykko",
        ]
        self.short = short
        self.normalize = normalize
        self.dim = None
        self.fixed_embeddings = self.create()

    def create(self):
        """Create a fixed embedding vector for each atom
        containing key properties

        Args:
            short (bool, optional): whether to exclude 'NaN' values columns
            Defaults to False.

        Returns:
            torch.Tensor
        """
        # Load table with all properties of all periodic table elements
        df = fetch_table("elements")
        df = df.set_index("atomic_number")

        # Select only potentially relevant elements
        df = df[self.properties_list]
        df = df.loc[:85, :]

        # Normalize
        if self.normalize:
            df = (df - df.mean()) / df.std()
            # normalized_df=(df-df.min())/(df.max()-df.min())

        # Process 'NaN' values and remove further non-essential columns
        if self.short:
            self.properties_list = df.columns[~df.isnull().any()].tolist()
            df = df[self.properties_list]
        else:
            self.properties_list = df.columns[
                pd.isnull(df).sum() < int(1 / 2 * df.shape[0])
            ].tolist()
            df = df[self.properties_list]
            col_missing_val = df.columns[df.isna().any()].tolist()
            df[col_missing_val] = df[col_missing_val].fillna(
                value=df[col_missing_val].mean()
            )

        self.dim = len(df.columns)

        # Convert to torch tensor and cuda
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return torch.cat(
            [torch.zeros(1, self.dim), torch.from_numpy(df.values).float()]
        ).to(device)

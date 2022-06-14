import pandas as pd
import torch
from mendeleev.fetch import fetch_table


class PhysEmbedding:
    def __init__(self, phys=True, short=False) -> None:

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
        self.phys_embeddings = None
        self.phys_embeds_size = 0
        self.group = None
        self.group_size = 0
        self.period = None
        self.period_size = 0

    def create(self, phys=True):
        """Create an embedding vector for each atom
        containing key physics properties

        Args:
            short (bool, optional): whether to exclude 'NaN' values columns
            Defaults to False.
        """
        # Convert to torch tensor and cuda
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load table with all properties of all periodic table elements
        df = fetch_table("elements")
        df = df.set_index("atomic_number")

        # Fetch group and period data
        df.group_id = df.group_id.fillna(value=19.0)
        self.group_size = df.group_id.unique().shape[0]
        self.group = torch.cat(
            [
                torch.ones(1, dtype=torch.long),
                torch.tensor(df.group_id.loc[:100].values, dtype=torch.long),
            ]
        ).to(device)
        self.period_size = df.period.loc[:100].unique().shape[0]
        self.period = torch.cat(
            [
                torch.ones(1, dtype=torch.long),
                torch.tensor(df.period.loc[:100].values, dtype=torch.long),
            ]
        ).to(device)

        if phys:
            # Select only potentially relevant elements
            df = df[self.properties_list]
            df = df.loc[:85, :]

            # Normalize
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

            self.phys_embeds_size = len(df.columns)

            self.phys_embeddings = torch.cat(
                [
                    torch.zeros(1, self.phys_embeds_size),
                    torch.from_numpy(df.values).float(),
                ]
            ).to(device)

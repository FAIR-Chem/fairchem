import pandas as pd
import torch
from mendeleev.fetch import fetch_table


class PhysEmbedding(torch.nn.ModuleDict):
    def __init__(self, phys=True, pg=False, short=False) -> None:
        """
        Create physicall embeddings meta class with sub-emeddings for each atom

        Args:
            phys (bool, optional): _description_. Defaults to True.
            pg (bool, optional): _description_. Defaults to False.
            short (bool, optional): _description_. Defaults to False.
        """

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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.make_from_mendeleev(phys, pg)

    def to(self, device):
        if self.phys_embeddings is not None:
            self.phys_embeddings = self.phys_embeddings.to(device)
        if self.group is not None:
            self.group = self.group.to(device)
        if self.period is not None:
            self.period = self.period.to(device)
        return self

    def make_from_mendeleev(self, phys=True, pg=True):
        """Create an embedding vector for each atom
        containing key physics properties

        Args:
            phys (bool, optional): whether we include fixed physical embeddings
            Defaults to True.
        """
        # Load table with all properties of all periodic table elements
        df = fetch_table("elements")
        df = df.set_index("atomic_number")

        # Fetch group and period data
        if pg:
            df.group_id = df.group_id.fillna(value=19.0)
            self.group_size = df.group_id.unique().shape[0]
            self["group"] = torch.cat(
                [
                    torch.ones(1, dtype=torch.long),
                    torch.tensor(df.group_id.loc[:100].values, dtype=torch.long),
                ]
            )
            self.period_size = df.period.loc[:100].unique().shape[0]
            self["period"] = torch.cat(
                [
                    torch.ones(1, dtype=torch.long),
                    torch.tensor(df.period.loc[:100].values, dtype=torch.long),
                ]
            )

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

            self["properties"] = torch.cat(
                [
                    torch.zeros(1, self.phys_embeds_size),
                    torch.from_numpy(df.values).float(),
                ]
            )

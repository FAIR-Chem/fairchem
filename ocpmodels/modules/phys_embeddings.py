import pandas as pd
import torch
import torch.nn as nn
from mendeleev.fetch import fetch_ionization_energies, fetch_table


class PhysEmbedding(nn.Module):
    def __init__(self, props=True, props_grad=False, pg=False, short=False) -> None:
        """
        Create physics-aware embeddings meta class with sub-emeddings for each atom

        Args:
            props (bool, optional): Create an embedding of physical
                properties. (default: :obj:`True`)
            props_grad (bool, optional): Learn a physics-aware embedding
                instead of keeping it fixed. (default: :obj:`False`)
            pg (bool, optional): Learn two embeddings based on period and
                group information respectively. (default: :obj:`False`)
            short (bool, optional): Remove all columns containing NaN values.
                (default: :obj:`False`)
        """
        super().__init__()

        self.properties_list = [
            "atomic_radius",
            "atomic_volume",
            "density",
            "dipole_polarizability",
            "electron_affinity",
            "en_allen",
            "vdw_radius",
            "metallic_radius",
            "metallic_radius_c12",
            "covalent_radius_pyykko_double",
            "covalent_radius_pyykko_triple",
            "covalent_radius_pyykko",
            "IE1",
            "IE2",
        ]
        self.group_size = 0
        self.period_size = 0
        self.n_properties = 0

        self.props = props
        self.props_grad = props_grad
        self.pg = pg
        self.short = short

        group = None
        period = None

        # Load table with all properties of all periodic table elements
        df = fetch_table("elements")
        df = df.set_index("atomic_number")

        # Add ionization energy
        ies = fetch_ionization_energies(degree=[1, 2])
        df = pd.concat([df, ies], axis=1)

        # Fetch group and period data
        if pg:
            df.group_id = df.group_id.fillna(value=19.0)
            self.group_size = df.group_id.unique().shape[0]
            group = torch.cat(
                [
                    torch.ones(1, dtype=torch.long),
                    torch.tensor(df.group_id.loc[:100].values, dtype=torch.long),
                ]
            )

            self.period_size = df.period.loc[:100].unique().shape[0]
            period = torch.cat(
                [
                    torch.ones(1, dtype=torch.long),
                    torch.tensor(df.period.loc[:100].values, dtype=torch.long),
                ]
            )

        self.register_buffer("group", group)
        self.register_buffer("period", period)

        # Create an embedding of physical properties
        if props:
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

            self.n_properties = len(df.columns)
            properties = torch.cat(
                [
                    torch.zeros(1, self.n_properties),
                    torch.from_numpy(df.values).float(),
                ]
            )
            if props_grad:
                self.register_parameter("properties", nn.Parameter(properties))
            else:
                self.register_buffer("properties", properties)

    @property
    def device(self):
        if self.props:
            return self.properties.device
        if self.pg:
            return self.group.device
        # raise ValueError("PhysEmb has no device because it has no tensor!")

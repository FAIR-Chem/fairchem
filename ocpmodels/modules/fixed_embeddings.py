import pandas as pd
import torch
from mendeleev.fetch import fetch_table


class FixedEmbedding:
    def __init__(self, short=False, normalize=True) -> None:

        self.properties_list = [
            "atomic_radius",
            "atomic_volume",
            "boiling_point",
            "density",
            "dipole_polarizability",
            "electron_affinity",
            "specific_heat",
            "evaporation_heat",
            "fusion_heat",
            "lattice_constant",
            "melting_point",
            "period",
            "thermal_conductivity",
            "vdw_radius",
            "covalent_radius_cordero",
            "covalent_radius_pyykko",
            "en_pauling",
            "en_allen",
            "proton_affinity",
            "gas_basicity",
            "heat_of_formation",
            "c6",
            "covalent_radius_bragg",
            "vdw_radius_bondi",
            "vdw_radius_truhlar",
            "vdw_radius_rt",
            "vdw_radius_batsanov",
            "vdw_radius_dreiding",
            "vdw_radius_uff",
            "vdw_radius_mm3",
            "en_ghosh",
            "vdw_radius_alvarez",
            "c6_gb",
            "atomic_weight",
            "atomic_weight_uncertainty",
            "atomic_radius_rahm",
            "metallic_radius",
            "metallic_radius_c12",
            "covalent_radius_pyykko_double",
            "covalent_radius_pyykko_triple",
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

        # Select only potentially relevant elements
        df = df[self.properties_list]
        df = df.iloc[:100, :]

        # Process 'NaN' values and remove further non-essential columns
        if self.short:
            self.properties_list = df.columns[~df.isnull().any()].tolist()
            df = df[self.properties_list]
        else:
            self.properties_list = df.columns[
                pd.isnull(df).sum() < 25
            ].tolist()
            df = df[self.properties_list]
            col_missing_val = df.columns[df.isna().any()].tolist()
            df[col_missing_val] = df[col_missing_val].fillna(
                value=df[col_missing_val].mean()
            )

        # Normalize
        if self.normalize:
            period_col = df["period"]
            df = (df - df.mean()) / df.std()
            df["period"] = period_col
            # normalized_df=(df-df.min())/(df.max()-df.min())

        # One hot encode 'period' variable
        y = pd.get_dummies(df.period, prefix="Period_")
        df = pd.merge(
            left=df,
            right=y,
            left_index=True,
            right_index=True,
        )
        df = df.drop("period", axis=1)

        self.dim = len(df.columns)

        # Convert to torch tensor and cuda
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return torch.from_numpy(df.values).float().to(device)

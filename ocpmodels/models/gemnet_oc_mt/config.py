from typing_extensions import override

from ocpmodels.common.typed_config import Singleton, TypedConfig


class BackboneConfig(Singleton, TypedConfig):
    num_targets: int = 1
    num_spherical: int
    num_radial: int
    num_blocks: int
    emb_size_atom: int
    emb_size_edge: int
    emb_size_trip_in: int
    emb_size_trip_out: int
    emb_size_quad_in: int
    emb_size_quad_out: int
    emb_size_aint_in: int
    emb_size_aint_out: int
    emb_size_rbf: int
    emb_size_cbf: int
    emb_size_sbf: int
    num_before_skip: int
    num_after_skip: int
    num_concat: int
    num_atom: int
    num_output_afteratom: int
    num_atom_emb_layers: int = 0
    num_global_out_layers: int = 2
    regress_forces: bool = True
    regress_energy: bool = True
    direct_forces: bool = False
    use_pbc: bool = True
    scale_backprop_forces: bool = False
    rbf: dict = {"name": "gaussian"}
    rbf_spherical: dict | None = None
    envelope: dict = {"name": "polynomial", "exponent": 5}
    cbf: dict = {"name": "spherical_harmonics"}
    sbf: dict = {"name": "spherical_harmonics"}
    extensive: bool = True
    forces_coupled: bool = False
    activation: str = "scaled_silu"
    quad_interaction: bool = False
    atom_edge_interaction: bool = False
    edge_atom_interaction: bool = False
    atom_interaction: bool = False
    scale_basis: bool = False
    qint_tags: list = [0, 1, 2]
    num_elements: int = 120
    otf_graph: bool = False
    ln: bool | str
    dropout: float | None
    scale_file: str | None = None

    replace_scale_factors_with_ln: bool = False

    absolute_rbf_cutoff: float | None = None
    learnable_rbf: bool = False
    learnable_rbf_stds: bool = False

    unique_basis_per_layer: bool = False
    old_gaussian_implementation: bool = False

    edge_dropout: float | None

    @classmethod
    def base(cls):
        return cls(
            num_targets=1,
            num_spherical=7,
            num_radial=128,
            num_blocks=4,
            emb_size_atom=256,
            emb_size_edge=512,
            emb_size_trip_in=64,
            emb_size_trip_out=64,
            emb_size_quad_in=32,
            emb_size_quad_out=32,
            emb_size_aint_in=64,
            emb_size_aint_out=64,
            emb_size_rbf=16,
            emb_size_cbf=16,
            emb_size_sbf=32,
            num_before_skip=2,
            num_after_skip=2,
            num_concat=1,
            num_atom=3,
            num_output_afteratom=3,
            num_atom_emb_layers=2,
            num_global_out_layers=2,
            regress_forces=True,
            regress_energy=True,
            direct_forces=True,
            use_pbc=True,
            scale_backprop_forces=False,
            rbf={"name": "gaussian"},
            rbf_spherical=None,
            envelope={"name": "polynomial", "exponent": 5},
            cbf={"name": "spherical_harmonics"},
            sbf={"name": "legendre_outer"},
            extensive=True,
            forces_coupled=False,
            activation="scaled_silu",
            quad_interaction=True,
            atom_edge_interaction=True,
            edge_atom_interaction=True,
            atom_interaction=True,
            scale_basis=False,
            qint_tags=[1, 2],
            num_elements=120,
            otf_graph=False,
            scale_file=None,
            absolute_rbf_cutoff=12.0,
            ln=False,
            dropout=None,
            edge_dropout=None,
        )

    @classmethod
    def download_base(cls, **kwargs):
        """Load GemNetOCBackbone config from github"""
        import requests
        import yaml

        # read config from url
        response = requests.get(
            "https://raw.githubusercontent.com/Open-Catalyst-Project/ocp/main/configs/s2ef/all/gemnet/gemnet-oc.yml"
        )
        config_dict = yaml.safe_load(response.text)

        model_config: dict = {**config_dict["model"]}
        _ = model_config.pop("name", None)
        _ = model_config.pop("scale_file", None)

        for key in list(model_config.keys()):
            if any(
                [
                    key.startswith(prefix)
                    for prefix in ["cutoff", "max_neighbors"]
                ]
            ):
                _ = model_config.pop(key)

        model_config.update(kwargs)
        config = cls.from_dict(model_config)
        return config

    @classmethod
    def large(cls):
        return cls(
            **{
                "num_targets": 1,
                "num_spherical": 7,
                "num_radial": 128,
                "num_blocks": 6,
                "emb_size_atom": 256,
                "emb_size_edge": 1024,
                "emb_size_trip_in": 64,
                "emb_size_trip_out": 128,
                "emb_size_quad_in": 64,
                "emb_size_quad_out": 32,
                "emb_size_aint_in": 64,
                "emb_size_aint_out": 64,
                "emb_size_rbf": 32,
                "emb_size_cbf": 16,
                "emb_size_sbf": 64,
                "num_before_skip": 2,
                "num_after_skip": 2,
                "num_concat": 4,
                "num_atom": 3,
                "num_output_afteratom": 3,
                "num_atom_emb_layers": 2,
                "num_global_out_layers": 2,
                "regress_forces": True,
                "regress_energy": True,
                "direct_forces": True,
                "use_pbc": True,
                "scale_backprop_forces": False,
                "rbf": {"name": "gaussian"},
                "rbf_spherical": None,
                "envelope": {"name": "polynomial", "exponent": 5},
                "cbf": {"name": "spherical_harmonics"},
                "sbf": {"name": "legendre_outer"},
                "extensive": True,
                "forces_coupled": False,
                "activation": "scaled_silu",
                "quad_interaction": True,
                "atom_edge_interaction": True,
                "edge_atom_interaction": True,
                "atom_interaction": True,
                "scale_basis": False,
                "qint_tags": [1, 2],
                "num_elements": 120,
                "otf_graph": False,
                "scale_file": None,
                "learnable_rbf": False,
                "learnable_rbf_stds": False,
                "unique_basis_per_layer": False,
                "old_gaussian_implementation": False,
            },
            absolute_rbf_cutoff=12.0,
            ln=False,
            dropout=None,
            edge_dropout=None,
        )

    @classmethod
    def download_large(cls, **kwargs):
        """Load GemNetOCBackbone config from github"""
        import requests
        import yaml

        # read config from url
        response = requests.get(
            "https://raw.githubusercontent.com/Open-Catalyst-Project/ocp/main/configs/s2ef/all/gemnet/gemnet-oc-large.yml"
        )
        config_dict = yaml.safe_load(response.text)

        model_config: dict = {**config_dict["model"]}
        _ = model_config.pop("name", None)
        _ = model_config.pop("scale_file", None)

        for key in list(model_config.keys()):
            if any(
                [
                    key.startswith(prefix)
                    for prefix in ["cutoff", "max_neighbors"]
                ]
            ):
                _ = model_config.pop(key)

        model_config.update(kwargs)
        config = cls.from_dict(model_config)
        return config

    @override
    def __post_init__(self):
        pass


class BasesConfig(TypedConfig):
    emb_size_rbf: int
    emb_size_cbf: int
    emb_size_sbf: int
    num_spherical: int
    num_radial: int
    rbf: dict = {"name": "gaussian"}
    rbf_spherical: dict | None = None
    envelope: dict = {"name": "polynomial", "exponent": 5}
    cbf: dict = {"name": "spherical_harmonics"}
    sbf: dict = {"name": "spherical_harmonics"}
    scale_basis: bool = False
    absolute_rbf_cutoff: float | None = None

    num_blocks: int
    quad_interaction: bool
    atom_edge_interaction: bool
    edge_atom_interaction: bool
    atom_interaction: bool

    emb_size_atom: int
    emb_size_edge: int
    activation: str

    learnable: bool = False
    learnable_rbf_stds: bool = False

    unique_per_layer: bool = False

    @classmethod
    def from_backbone_config(cls, backbone_config: BackboneConfig):
        return cls(
            emb_size_rbf=backbone_config.emb_size_rbf,
            emb_size_cbf=backbone_config.emb_size_cbf,
            emb_size_sbf=backbone_config.emb_size_sbf,
            num_spherical=backbone_config.num_spherical,
            num_radial=backbone_config.num_radial,
            rbf=backbone_config.rbf,
            rbf_spherical=backbone_config.rbf_spherical,
            envelope=backbone_config.envelope,
            cbf=backbone_config.cbf,
            sbf=backbone_config.sbf,
            scale_basis=backbone_config.scale_basis,
            absolute_rbf_cutoff=backbone_config.absolute_rbf_cutoff,
            num_blocks=backbone_config.num_blocks,
            quad_interaction=backbone_config.quad_interaction,
            atom_edge_interaction=backbone_config.atom_edge_interaction,
            edge_atom_interaction=backbone_config.edge_atom_interaction,
            atom_interaction=backbone_config.atom_interaction,
            emb_size_atom=backbone_config.emb_size_atom,
            emb_size_edge=backbone_config.emb_size_edge,
            activation=backbone_config.activation,
            learnable=backbone_config.learnable_rbf,
            learnable_rbf_stds=backbone_config.learnable_rbf_stds,
            unique_per_layer=backbone_config.unique_basis_per_layer,
        )

    @override
    def __post_init__(self):
        if not self.rbf_spherical:
            self.rbf_spherical = self.rbf.copy()

        if self.learnable:
            self.rbf["trainable"] = True
            self.rbf_spherical["trainable"] = True

        if self.learnable_rbf_stds:
            self.rbf["trainable_stds"] = True
            self.rbf_spherical["trainable_stds"] = True

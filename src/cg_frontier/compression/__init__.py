"""Learned material compression primitives."""

from .material import (
    Core4Targets,
    DecodedMaterial,
    MaterialDecoder,
    decode_material,
    initialize_decoder_from_tiny,
    load_core4_targets,
    material_loss,
    reconstruct_normal,
)
from .hybrid import (
    AuxMaterialDecoder,
    HybridInitialization,
    decode_auxiliary,
    deterministic_pca_initialization,
    export_hybrid_textures,
    pack_hybrid_textures,
    render_hybrid_material,
    sample_and_decode_hybrid,
)
from .hybrid_factorization import (
    CausalHybridInitialization,
    FactorizationSpec,
    FactorizedAuxDecoder,
    SPECS,
    candidate_aux_channels,
    decoder_for_candidate,
    deterministic_causal_initialization,
    direct_semantic_material,
    gradient_conflict_report,
)

__all__ = [
    "Core4Targets",
    "DecodedMaterial",
    "MaterialDecoder",
    "decode_material",
    "initialize_decoder_from_tiny",
    "load_core4_targets",
    "material_loss",
    "reconstruct_normal",
    "AuxMaterialDecoder",
    "HybridInitialization",
    "decode_auxiliary",
    "deterministic_pca_initialization",
    "export_hybrid_textures",
    "pack_hybrid_textures",
    "render_hybrid_material",
    "sample_and_decode_hybrid",
]

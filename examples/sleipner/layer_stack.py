from __future__ import annotations

from co2ipsimulator.model import LayerKind, LayerProps

from .config import PAPER_DENSITY_KG_M3, PAPER_LAYER_NAMES, PAPER_PTH_KPA, KPA_TO_PA

_DENS = {name: rho for name, rho in zip(PAPER_LAYER_NAMES, PAPER_DENSITY_KG_M3)}

SAND_POROSITY = 0.36
SHALE_POROSITY = 0.0
SAND_PTH_PA = 100.0


def sleipner_layer_stack() -> list[LayerProps]:
    stack: list[LayerProps] = [
        LayerProps(
            name="L9",
            kind=LayerKind.SAND,
            top_surface="TopSW",
            base_surface="ThickShale",
            density_co2=_DENS["L9"],
            pth_pa=SAND_PTH_PA,
            porosity=SAND_POROSITY,
        ),
        LayerProps(
            name="Shale_thick",
            kind=LayerKind.SHALE,
            top_surface="ThickShale",
            base_surface="TopUtsiraFm",
            density_co2=_DENS["L8"],
            pth_pa=PAPER_PTH_KPA[7] * KPA_TO_PA,
            porosity=SHALE_POROSITY,
        ),
        LayerProps(
            name="L8",
            kind=LayerKind.SAND,
            top_surface="TopUtsiraFm",
            base_surface="Reflector7",
            density_co2=_DENS["L8"],
            pth_pa=SAND_PTH_PA,
            porosity=SAND_POROSITY,
        ),
    ]

    for n in range(7, 0, -1):
        stack.append(
            LayerProps(
                name=f"Shale_{n}",
                kind=LayerKind.SHALE,
                top_surface=f"Reflector{n}",
                base_surface=f"Base_Reflector{n}",
                density_co2=_DENS[f"L{n}"],
                pth_pa=PAPER_PTH_KPA[n - 1] * KPA_TO_PA,
                porosity=SHALE_POROSITY,
            )
        )
        base_below = "BaseUtsiraFm" if n == 1 else f"Reflector{n - 1}"
        stack.append(
            LayerProps(
                name=f"L{n}",
                kind=LayerKind.SAND,
                top_surface=f"Base_Reflector{n}",
                base_surface=base_below,
                density_co2=_DENS[f"L{n}"],
                pth_pa=SAND_PTH_PA,
                porosity=SAND_POROSITY,
            )
        )

    return stack

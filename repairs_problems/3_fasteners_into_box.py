from repairs_components.geometry.fasteners import Fastener
from repairs_components.training_utils.env_setup import EnvSetup
from repairs_components.training_utils.sim_state_global import RepairsSimState
from build123d import *
import genesis as gs
import numpy as np
from repairs_components.geometry.b123d_utils import fastener_hole


class FastenersIntoBox(EnvSetup):
    "Simplest env, only for basic debug."

    def starting_state(self, scene: gs.Scene):
        with BuildPart() as box_with_holes:
            with Locations((0, 0, 5)):
                box = Box(10, 10, 10)

            with Locations(box.faces().filter_by(Axis.Z).sort_by(Axis.Z)[-1]):
                with Locations((1, 1, 0)):
                    fastener_hole(2, 2)
        with Locations((30, 30, 30)):
            fastener = Fastener().bd_geometry()

        box_with_holes.part.label = "box@solid"
        return Compound(children=[box_with_holes.part])

    def desired_state(
        self, scene: gs.Scene
    ) -> tuple[Compound, RepairsSimState, dict[str, np.ndarray]]:
        with BuildPart() as box_with_holes:
            with Locations((0, 0, 5)):  # pos unchanged
                box = Box(10, 10, 10)

            with Locations(box.faces().filter_by(Axis.Z).sort_by(Axis.Z)[-1]):
                with Locations((1, 1, 0)):
                    fastener_hole(2, 2)

        fastener = Fastener().bd_geometry()

        box_with_holes.part.label = "box@solid"
        return (Compound(children=[box_with_holes.part]),)

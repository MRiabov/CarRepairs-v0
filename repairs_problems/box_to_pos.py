from repairs_components.training_utils.env_setup import EnvSetup
from repairs_components.training_utils.sim_state_global import RepairsSimState
from build123d import *
import genesis as gs
import numpy as np
from genesis.vis.camera import Camera
from repairs_components.geometry.base_env.tool_stand_plate import tool_stand_plate
from repairs_components.training_utils.translation import (
    translate_compound_to_sim_state,
)


class MoveBox(EnvSetup):
    "Simplest env, only for basic debug."

    def starting_state_geom(
        self, scene: gs.Scene
    ) -> tuple[Compound, RepairsSimState, dict[str, np.ndarray], list[Camera]]:
        with BuildPart() as box:
            with Locations((10, 0, 10)):
                Box(10, 10, 10)

        box.part.label = "box@solid"

        compound = Compound(children=[box.part]).move(Pos(self.BOTTOM_CENTER_OF_PLATE))

        sim_state = translate_compound_to_sim_state(compound)

        return compound, sim_state, {}, []

    def desired_state_def(
        self, scene: gs.Scene
    ) -> tuple[Compound, RepairsSimState, dict[str, np.ndarray], list[Camera]]:
        with BuildPart() as box:
            with Locations((-10, 0, 10)):
                Box(10, 10, 10)

        box.part.label = "box@solid"
        compound = Compound(children=[box.part]).move(Pos(self.BOTTOM_CENTER_OF_PLATE))
        sim_state = translate_compound_to_sim_state(compound)
        return compound, sim_state, {}, []


# the better workflow would be to create desired state, then create many environments for different tasks from the desired state. wouldn't it?
# it would lessen the need for recompilation, too.

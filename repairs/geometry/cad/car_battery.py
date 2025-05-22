from build123d import *
import ocp_vscode
import numpy as np

# Dimensions
battery_dims = (120, 180, 110)  # mm
terminal_radius = 5
terminal_height = 30
grip_dims = (battery_dims[0], battery_dims[1] + 7 * 2, 10)

# Positions
terminal_offset_x = battery_dims[0] * 0.25
terminal_offset_y = battery_dims[1] / 2 - terminal_radius - 20
grip_offset_y = battery_dims[1] / 2 + grip_dims[1] / 2
grip_z = battery_dims[2] * 0.7

with BuildPart() as battery_body:
    base = Box(*battery_dims)
    top_face = battery_body.faces().filter_by(Axis.Z).sort_by(Axis.Z)[-1]

    with BuildSketch(top_face) as terminal_sketch:
        with Locations(
            (terminal_offset_x, terminal_offset_y, 0),
            (terminal_offset_x, -terminal_offset_y, 0),
        ):
            Circle(terminal_radius)
    extrude(terminal_sketch.sketch, amount=terminal_height)

    with BuildSketch(top_face) as text_sketch:
        with Locations((terminal_offset_x - 30, terminal_offset_y, 0)):
            Text("+", font_size=30, rotation=90)
        with Locations((terminal_offset_x - 30, -terminal_offset_y, 0)):
            Text("-", font_size=30, rotation=90)
    extrude(text_sketch.sketch, amount=2)

    with Locations((0, 0, (battery_dims[2] - grip_dims[2]) / 2)):
        Box(*grip_dims)

    top_face = battery_body.part.faces().filter_by(Axis.Z).sort_by(Axis.Z)[0]

export_gltf(battery_body.part, "repairs/geometry/cad/exports/battery.gltf")
ocp_vscode.show(battery_body.part)

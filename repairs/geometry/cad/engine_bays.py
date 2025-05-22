from build123d import *

import ocp_vscode

with BuildPart() as bay_1:
    box = Box(1000, 1000, 600)
    offset(
        amount=-50, openings=box.faces().sort_by(Axis.Z)[-1]
    )  # leaving floor on purpose.
    fillet(
        bay_1.part.edges().filter_by(Axis.Z).sort_by(Axis.X)[-4:], radius=300
    )  # acceptable.
    with BuildSketch(
        box.faces().filter_by(Axis.X).sort_by(Axis.X).last
    ) as wedge_sketch:
        with BuildLine() as wedge_line:
            Polyline(
                [
                    (500.0, 500.0, 300.0),  # top right
                    (-500.0, 500.0, 300.0),  # top left
                    (500.0, 500.0, 100.0),  # 200 lower than top right
                ],
                close=True,
            )
        wedge_face = make_face(wedge_line.line)
    extrude(wedge_face, amount=1000, mode=Mode.SUBTRACT)


# --- Optional: Show in Viewer if run directly ---
export_gltf(bay_1.part, "repairs/geometry/cad/exports/engine_bay_1.gltf")
ocp_vscode.show(bay_1.part)

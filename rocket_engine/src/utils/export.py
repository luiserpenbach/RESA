import numpy as np
import ezdxf
from rocket_engine.src.geometry.nozzle import NozzleGeometryData


def export_contour_to_dxf(geo: NozzleGeometryData, filename: str):
    """
    Exports the engine contour (Inner Wall) to a DXF file.
    The output is a 2D Polyline in the XY plane.

    Args:
        geo: NozzleGeometryData object containing x_full and y_full.
        filename: Output filename (e.g., "output/contour.dxf").
    """
    # Create a new DXF document
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    # Prepare points (X, Y)
    # Convert from meters (code) to millimeters (CAD standard usually)
    x_mm = geo.x_full * 1000.0
    y_mm = geo.y_full * 1000.0

    # Create Inner Wall (Top Half)
    points_top = list(zip(x_mm, y_mm))
    msp.add_lwpolyline(points_top, dxfattribs={'layer': 'INNER_WALL', 'color': 1})  # Red

    # Create Inner Wall (Bottom Half - Mirror)
    points_bot = list(zip(x_mm, -y_mm))
    msp.add_lwpolyline(points_bot, dxfattribs={'layer': 'INNER_WALL', 'color': 1})

    # Optional: Draw Throat Line
    # Find throat index (min Y)
    idx_t = np.argmin(y_mm)
    xt = x_mm[idx_t]
    yt = y_mm[idx_t]
    msp.add_line((xt, yt), (xt, -yt), dxfattribs={'layer': 'THROAT_LINE', 'color': 3})  # Green

    # Save
    doc.saveas(filename)
    print(f"Exported Contour to DXF: {filename}")
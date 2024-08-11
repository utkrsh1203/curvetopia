import numpy as np
import matplotlib.pyplot as plt
import svgwrite
import cairosvg


def read_csv(csv_path):
    """
    Read the CSV file and process it into a list of shapes.
    Each shape is a list of paths, where each path is a numpy array of points.
    """
    np_path_XYs = np.genfromtxt(csv_path, delimiter=",")
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs


def plot(path_XYs):
    """
    Visualize the shapes using Matplotlib.
    """
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(path_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect("equal")
    plt.show()


def polylines2svg(paths_XYs, svg_path):
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)

    # Create a new SVG drawing
    dwg = svgwrite.Drawing(svg_path, profile="tiny", shape_rendering="crispEdges")
    group = dwg.g()

    for i, path in enumerate(paths_XYs):
        path_data = []
        c = colours[i % len(colours)]
        for XY in path:
            path_data.append(("M", (XY[0, 0], XY[0, 1])))
            for j in range(1, len(XY)):
                path_data.append(("L", (XY[j, 0], XY[j, 1])))
            if not np.allclose(XY[0], XY[-1]):
                path_data.append(("Z", None))
        group.add(dwg.path(d=path_data, fill=c, stroke="none", stroke_width=2))

    dwg.add(group)
    dwg.save()

    png_path = svg_path.replace(".svg", ".png")
    fact = max(1, 1024 // min(H, W))
    cairosvg.svg2png(
        url=svg_path,
        write_to=png_path,
        parent_width=W,
        parent_height=H,
        output_width=fact * W,
        output_height=fact * H,
        background_color="white",
    )
    return


colours = ["red", "green", "blue", "cyan", "magenta", "yellow", "black"]
# Example usage
csv_path1 = "./problems/frag2.csv"  # Replace with the actual path to your CSV file
csv_path2 = "./problems/frag2_sol.csv"  # Replace with the actual path to your CSV file
path_XYs1 = read_csv(csv_path1)
path_XYs2 = read_csv(csv_path2)
plot(path_XYs1)
plot(path_XYs2)

# colours = ["r", "g", "b", "c", "m", "y", "k"]  # List of colors for different shapes
paths_XYs = [
    [np.array([[10, 10], [100, 100], [200, 50]]), np.array([[200, 200], [300, 300]])],
    [np.array([[50, 50], [150, 150], [250, 100]]), np.array([[100, 200], [200, 250]])],
]

# plot(paths_XYs)
svg_path = "output.svg"

# polylines2svg(path_XYs, svg_path)

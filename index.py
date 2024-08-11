import numpy as np


def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=",")
    path_XYs = []

    # Group points by path
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)

    return path_XYs


import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import leastsq
import svgwrite


def detect_straight_lines(points):
    model = LinearRegression()
    points = np.array(points)
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    return model, mse


def fit_circle(points):
    def calc_R(xc, yc):
        return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = np.mean(points, axis=0)
    center, _ = leastsq(f_2, center_estimate)
    xc, yc = center
    R = calc_R(xc, yc).mean()
    mse = np.mean((calc_R(xc, yc) - R) ** 2)
    return xc, yc, R, mse


def fit_ellipse(points):
    def ellipse_cost(params):
        a, b, x0, y0, theta = params
        x = points[:, 0]
        y = points[:, 1]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_prime = (x - x0) * cos_theta + (y - y0) * sin_theta
        y_prime = -(x - x0) * sin_theta + (y - y0) * cos_theta
        return (x_prime**2 / a**2) + (y_prime**2 / b**2) - 1

    x_mean, y_mean = np.mean(points, axis=0)
    major_axis = np.max(points[:, 0]) - np.min(points[:, 0])
    minor_axis = np.max(points[:, 1]) - np.min(points[:, 1])
    initial_params = [major_axis / 2, minor_axis / 2, x_mean, y_mean, 0]
    params, _ = leastsq(ellipse_cost, initial_params)
    a, b, x0, y0, theta = params
    mse = np.mean(ellipse_cost(params) ** 2)
    return a, b, x0, y0, theta, mse


def fit_cubic_bezier(points):
    # Fit a cubic Bezier curve to points (simplified)
    if len(points) < 4:
        return None
    p0, p3 = points[0], points[-1]
    p1 = p0 + (points[1] - p0) * 0.5
    p2 = p3 + (points[-2] - p3) * 0.5
    return [p0, p1, p2, p3]


def process_polylines(paths_XYs):
    bezier_paths = []
    for path in paths_XYs:
        for polyline in path:
            model, mse_line = detect_straight_lines(polyline)
            xc, yc, R, mse_circle = fit_circle(np.array(polyline))
            a, b, x0, y0, theta, mse_ellipse = fit_ellipse(np.array(polyline))

            if mse_line < 1e-3:  # Threshold for line detection
                bezier_curve = fit_cubic_bezier(polyline)
                if bezier_curve:
                    bezier_paths.append(bezier_curve)
            elif mse_circle < 1e-3:  # Threshold for circle detection
                circle_points = np.array(
                    [
                        [xc + R * np.cos(t), yc + R * np.sin(t)]
                        for t in np.linspace(0, 2 * np.pi, 100)
                    ]
                )
                bezier_curve = fit_cubic_bezier(circle_points)
                if bezier_curve:
                    bezier_paths.append(bezier_curve)
            elif mse_ellipse < 1e-3:  # Threshold for ellipse detection
                ellipse_points = np.array(
                    [
                        [
                            x0
                            + a * np.cos(t) * np.cos(theta)
                            - b * np.sin(t) * np.sin(theta),
                            y0
                            + a * np.cos(t) * np.sin(theta)
                            + b * np.sin(t) * np.cos(theta),
                        ]
                        for t in np.linspace(0, 2 * np.pi, 100)
                    ]
                )
                bezier_curve = fit_cubic_bezier(ellipse_points)
                if bezier_curve:
                    bezier_paths.append(bezier_curve)
    return bezier_paths


def bezier2svg(bezier_paths, svg_path):
    dwg = svgwrite.Drawing(svg_path, profile="tiny")
    for bezier in bezier_paths:
        if bezier is None:
            continue
        path_data = f"M {bezier[0][0]},{bezier[0][1]} "
        path_data += f"C {bezier[1][0]},{bezier[1][1]} {bezier[2][0]},{bezier[2][1]} {bezier[3][0]},{bezier[3][1]}"
        dwg.add(dwg.path(d=path_data, stroke="black", fill="none"))
    dwg.save()


# Example usage
csv_path = "./problems/frag0.csv"
paths_XYs = read_csv(csv_path)
bezier_paths = process_polylines(paths_XYs)
svg_path = "output.svg"
bezier2svg(bezier_paths, svg_path)

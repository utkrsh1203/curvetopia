import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import svgwrite


# Regularization Functions
def regularize_to_line(points):
    p0, p3 = points[0], points[-1]
    return np.array([p0, p3])  # Making all points collinear


def regularize_to_circle_or_ellipse(points):
    centroid = np.mean(points, axis=0)
    shifted_points = points - centroid
    cov_matrix = np.cov(shifted_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    regularized_points = np.dot(shifted_points, eigenvectors) / np.sqrt(eigenvalues)
    regularized_points *= np.sqrt(eigenvalues).max()
    regularized_points += centroid
    return regularized_points


def regularize_to_rectangle(points):
    centroid = np.mean(points, axis=0)
    vectors = np.diff(points, axis=0, append=[points[0]])
    aligned_vectors = []
    for vector in vectors:
        angle = np.arctan2(vector[1], vector[0])
        aligned_angle = np.round(angle / (np.pi / 2)) * (np.pi / 2)
        aligned_vector = np.array(
            [np.cos(aligned_angle), np.sin(aligned_angle)]
        ) * np.linalg.norm(vector)
        aligned_vectors.append(aligned_vector)
    regularized_points = np.array([centroid])
    for vector in aligned_vectors:
        regularized_points = np.vstack(
            [regularized_points, regularized_points[-1] + vector]
        )
    return regularized_points[:-1]


def regularize_to_polygon(points):
    centroid = np.mean(points, axis=0)
    n = len(points)
    angle_increment = 2 * np.pi / n
    radius = np.mean(np.linalg.norm(points - centroid, axis=1))
    regularized_points = []
    for i in range(n):
        angle = i * angle_increment
        x = centroid[0] + radius * np.cos(angle)
        y = centroid[1] + radius * np.sin(angle)
        regularized_points.append([x, y])
    return np.array(regularized_points)


def regularize_to_star(points):
    centroid = np.mean(points, axis=0)
    n = len(points)
    if n % 2 != 0:
        raise ValueError("Star shapes must have an even number of points")
    angle_increment = 2 * np.pi / n
    outer_points = points[::2]
    inner_points = points[1::2]
    outer_radius = np.mean(np.linalg.norm(outer_points - centroid, axis=1))
    inner_radius = np.mean(np.linalg.norm(inner_points - centroid, axis=1))
    regularized_points = []
    for i in range(n):
        angle = i * angle_increment
        if i % 2 == 0:
            x = centroid[0] + outer_radius * np.cos(angle)
            y = centroid[1] + outer_radius * np.sin(angle)
        else:
            x = centroid[0] + inner_radius * np.cos(angle)
            y = centroid[1] + inner_radius * np.sin(angle)
        regularized_points.append([x, y])
    return np.array(regularized_points)


# Utility Functions
def load_csv(file_path):
    return pd.read_csv(file_path, header=None).values


def save_svg(paths, file_name):
    dwg = svgwrite.Drawing(file_name, profile="tiny")
    for path in paths:
        path_data = ["M {} {}".format(path[0, 0], path[0, 1])]
        for point in path[1:]:
            path_data.append("L {} {}".format(point[0], point[1]))
        dwg.add(dwg.path(d=" ".join(path_data), fill="none", stroke="black"))
    dwg.save()


def preprocess_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def detect_shape(points):
    def is_straight_line(points):
        # Check if all points lie on a straight line
        p0, p1 = points[0], points[-1]
        vec = p1 - p0
        norm_vec = vec / np.linalg.norm(vec)
        for point in points[1:-1]:
            vec_to_point = point - p0
            if not np.allclose(
                np.dot(vec_to_point, norm_vec), np.linalg.norm(vec_to_point), atol=1e-2
            ):
                return False
        return True

    def is_circle_or_ellipse(points):
        # Check if points form a circle or ellipse
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        std_dev = np.std(distances)
        return std_dev < 0.05 * np.mean(distances)  # Tolerance can be adjusted

    def is_rectangle(points):
        hull = ConvexHull(points)
        if len(hull.vertices) != 4:
            return False
        vectors = np.diff(
            points[hull.vertices], axis=0, append=[points[hull.vertices[0]]]
        )
        angles = [
            np.arccos(
                np.dot(vectors[i], vectors[(i + 1) % 4])
                / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[(i + 1) % 4]))
            )
            for i in range(4)
        ]
        return all(np.isclose(angle, np.pi / 2, atol=1e-2) for angle in angles)

    def is_regular_polygon(points):
        hull = ConvexHull(points)
        n = len(hull.vertices)
        if n < 3:
            return False
        angles = []
        vectors = np.diff(
            points[hull.vertices], axis=0, append=[points[hull.vertices[0]]]
        )
        for i in range(n):
            angle = np.arccos(
                np.dot(vectors[i], vectors[(i + 1) % n])
                / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[(i + 1) % n]))
            )
            angles.append(angle)
        return all(np.isclose(angle, 2 * np.pi / n, atol=1e-2) for angle in angles)

    def is_star(points):
        hull = ConvexHull(points)
        if len(hull.vertices) < 6 or len(hull.vertices) % 2 != 0:
            return False
        return True

    if is_straight_line(points):
        return "line"
    elif is_circle_or_ellipse(points):
        return "circle_or_ellipse"
    elif is_rectangle(points):
        return "rectangle"
    elif is_regular_polygon(points):
        return "polygon"
    elif is_star(points):
        return "star"
    else:
        return "unknown"


def regularize_shape(points, shape_type):
    if shape_type == "line":
        return regularize_to_line(points)
    elif shape_type == "circle_or_ellipse":
        return regularize_to_circle_or_ellipse(points)
    elif shape_type == "rectangle":
        return regularize_to_rectangle(points)
    elif shape_type == "polygon":
        return regularize_to_polygon(points)
    elif shape_type == "star":
        return regularize_to_star(points)
    else:
        return points


# Main Pipeline
def process_image(input_csv, output_svg):
    data = load_csv(input_csv)
    preprocessed_data = preprocess_data(data)
    shape_type = detect_shape(preprocessed_data)
    regularized_shape = regularize_shape(preprocessed_data, shape_type)
    save_svg([regularized_shape], output_svg)


# Example Usage
input_csv = "./problems/frag0.csv"
output_svg = "output.svg"
process_image(input_csv, output_svg)

"""Shared CARLA Garage route labels for training and anchor generation."""
import numpy as np
import transfuser_utils as t_u


def prepare_route(route, count=20):
  """Use the same input horizon and padding before smoothing/augmentation."""
  route = np.asarray(route, dtype=np.float64)
  if route.ndim != 2 or route.shape[1] != 2 or not len(route):
    raise ValueError('Expected a nonempty route with shape [N, 2]')
  if not np.isfinite(route).all() or count < 1:
    raise ValueError('Route must be finite and count must be positive')
  route = route[:count]
  if len(route) < count:
    route = np.vstack((route, np.tile(route[-1], (count - len(route), 1))))
  return route


def smooth_route(route, count):
  _, indices = np.unique(route, return_index=True, axis=0)
  # We need to remove the sorting of unique, because this algorithm assumes the order of the path is kept
  route = route[np.sort(indices)]
  if len(route) < 2:
    raise ValueError('Route smoothing requires two distinct points')
  interpolated_route_points = iterative_line_interpolation(route, count)

  return interpolated_route_points

def iterative_line_interpolation(route, count):
  interpolated_route_points = []

  # this value is actually not used anymore, it is overwritten in the loop
  min_distance = 2.4
  target_first_distance = 2.5
  last_interpolated_point = np.array([0.0, 0.0])
  current_route_index = 0
  current_point = route[current_route_index]
  last_point = np.array([0.0, 0.0])
  first_iteration = True

  while len(interpolated_route_points) < count:
    # First point should be target_first_distance away from the vehicle.
    if not first_iteration:
      current_route_index += 1
      last_point = current_point

    if current_route_index < route.shape[0]:
      current_point = route[current_route_index]
      intersection = t_u.circle_line_segment_intersection(
          circle_center=last_interpolated_point,
          circle_radius=min_distance if not first_iteration else target_first_distance,
          pt1=last_interpolated_point,
          pt2=current_point,
          full_line=True)

    else:  # We hit the end of the input route. We extrapolate the last 2 points
      current_point = route[-1]
      last_point = route[-2]
      intersection = t_u.circle_line_segment_intersection(circle_center=last_interpolated_point,
                                                          circle_radius=min_distance,
                                                          pt1=last_point,
                                                          pt2=current_point,
                                                          full_line=True)

    # 3 cases: 0 intersection, 1 intersection, 2 intersection
    if len(intersection) > 1:  # 2 intersections
      # Take the one that is closer to current point
      point_1 = np.array(intersection[0])
      point_2 = np.array(intersection[1])
      direction = current_point - last_point
      dot_p1_to_last = np.dot(point_1, direction)
      dot_p2_to_last = np.dot(point_2, direction)

      if dot_p1_to_last > dot_p2_to_last:
        intersection_point = point_1
      else:
        intersection_point = point_2
      add_point = True
    elif len(intersection) == 1:  # 1 Intersections
      intersection_point = np.array(intersection[0])
      add_point = True
    else:  # 0 Intersection
      add_point = False
      raise Exception("No intersection found. This should never occur.")

    if add_point:
      last_interpolated_point = intersection_point
      interpolated_route_points.append(intersection_point)
      min_distance = 1.0  # After the first point we want each point to be 1 m away from the last.

    first_iteration = False

  interpolated_route_points = np.array(interpolated_route_points)
  return interpolated_route_points


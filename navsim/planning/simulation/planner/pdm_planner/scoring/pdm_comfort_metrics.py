from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
# from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
#     state_array_to_center_state_array,
# )
# TODO: Refactor & add to config

# (1) ego_jerk_metric,
max_abs_mag_jerk = 8.37  # [m/s^3]

# (2) ego_lat_acceleration_metric
max_abs_lat_accel = 4.89  # [m/s^2]

# (3) ego_lon_acceleration_metric
max_lon_accel = 2.40  # [m/s^2]
min_lon_accel = -4.05

# (4) ego_yaw_acceleration_metric
max_abs_yaw_accel = 1.93  # [rad/s^2]

# (5) ego_lon_jerk_metric
max_abs_lon_jerk = 4.13  # [m/s^3]

# (6) ego_yaw_rate_metric
max_abs_yaw_rate = 0.95  # [rad/s]
# (1) ego_jerk_metric,
MAX_ABS_MAG_JERK: float = 8.37  # [m/s^3]

# (2) ego_lat_acceleration_metric
MAX_ABS_LAT_ACCEL: float = 4.89  # [m/s^2]

# (3) ego_lon_acceleration_metric
MAX_LON_ACCEL: float = 2.40  # [m/s^2]
MIN_LON_ACCEL: float = -4.05

# (4) ego_yaw_acceleration_metric
MAX_ABS_YAW_ACCEL: float = 1.93  # [rad/s^2]

# (5) ego_lon_jerk_metric
MAX_ABS_LON_JERK: float = 4.13  # [m/s^3]

# (6) ego_yaw_rate_metric
MAX_ABS_YAW_RATE: float = 0.95  # [rad/s]


# Extended Comfort thresholds
acceleration_threshold: float = 0.7  # [m/s^2]
jerk_threshold: float = 0.5  # [m/s^3]
yaw_rate_threshold: float = 0.1  # [rad/s]
yaw_accel_threshold: float = 0.1  # [rad/s^2]

def _extract_ego_acceleration(
    states: npt.NDArray[np.float64],
    acceleration_coordinate: str,
    decimals: int = 8,
    poly_order: int = 2,
    window_length: int = 8,
) -> npt.NDArray[np.float32]:
    """
    Extract acceleration of ego pose in simulation history over batch-dim
    :param states: array representation of ego state values
    :param acceleration_coordinate: string of axis to extract
    :param decimals: decimal precision, defaults to 8
    :param poly_order: polynomial order, defaults to 2
    :param window_length: window size for extraction, defaults to 8
    :raises ValueError: when coordinate not available
    :return: array containing acceleration values
    """

    n_batch, n_time, n_states = states.shape
    if acceleration_coordinate == "x":
        acceleration: npt.NDArray[np.float64] = states[..., StateIndex.ACCELERATION_X]

    elif acceleration_coordinate == "y":
        acceleration: npt.NDArray[np.float64] = states[..., StateIndex.ACCELERATION_Y]

    elif acceleration_coordinate == "magnitude":
        acceleration: npt.NDArray[np.float64] = np.hypot(
            states[..., StateIndex.ACCELERATION_X],
            states[..., StateIndex.ACCELERATION_Y],
        )
    else:
        raise ValueError(
            f"acceleration_coordinate option: {acceleration_coordinate} not available. "
            f"Available options are: x, y or magnitude"
        )

    acceleration = savgol_filter(
        acceleration,
        polyorder=poly_order,
        window_length=min(window_length, n_time),
        axis=-1,
    )
    acceleration = np.round(acceleration, decimals=decimals)
    return acceleration


def _extract_ego_jerk(
    states: npt.NDArray[np.float64],
    acceleration_coordinate: str,
    time_steps_s: npt.NDArray[np.float64],
    decimals: int = 8,
    deriv_order: int = 1,
    poly_order: int = 2,
    window_length: int = 15,
) -> npt.NDArray[np.float32]:
    """
    Extract jerk of ego pose in simulation history over batch-dim
    :param states: array representation of ego state values
    :param acceleration_coordinate: string of axis to extract
    :param time_steps_s: time steps [s] of time dim
    :param decimals: decimal precision, defaults to 8
    :param deriv_order: order of derivative, defaults to 1
    :param poly_order: polynomial order, defaults to 2
    :param window_length: window size for extraction, defaults to 15
    :return: array containing jerk values
    """
    n_batch, n_time, n_states = states.shape
    ego_acceleration = _extract_ego_acceleration(states, acceleration_coordinate=acceleration_coordinate)
    jerk = _approximate_derivatives(
        ego_acceleration,
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
        window_length=min(window_length, n_time),
    )
    jerk = np.round(jerk, decimals=decimals)
    return jerk


def _extract_ego_yaw_rate(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
    deriv_order: int = 1,
    poly_order: int = 2,
    decimals: int = 8,
    window_length: int = 15,
) -> npt.NDArray[np.float32]:
    """
    Extract yaw-rate of simulation history over batch-dim
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :param deriv_order: order of derivative, defaults to 1
    :param poly_order: polynomial order, defaults to 2
    :param decimals:  decimal precision, defaults to 8
    :param window_length: window size for extraction, defaults to 15
    :return: array containing ego's yaw rate
    """
    ego_headings = states[..., StateIndex.HEADING]
    ego_yaw_rate = _approximate_derivatives(
        _phase_unwrap(ego_headings),
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
    )  # convert to seconds
    ego_yaw_rate = np.round(ego_yaw_rate, decimals=decimals)
    return ego_yaw_rate


def _phase_unwrap(headings: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Returns an array of heading angles equal mod 2 pi to the input heading angles,
    and such that the difference between successive output angles is less than or
    equal to pi radians in absolute value
    :param headings: An array of headings (radians)
    :return The phase-unwrapped equivalent headings.
    """
    # There are some jumps in the heading (e.g. from -np.pi to +np.pi) which causes approximation of yaw to be very large.
    # We want unwrapped[j] = headings[j] - 2*pi*adjustments[j] for some integer-valued adjustments making the absolute value of
    # unwrapped[j+1] - unwrapped[j] at most pi:
    # -pi <= headings[j+1] - headings[j] - 2*pi*(adjustments[j+1] - adjustments[j]) <= pi
    # -1/2 <= (headings[j+1] - headings[j])/(2*pi) - (adjustments[j+1] - adjustments[j]) <= 1/2
    # So adjustments[j+1] - adjustments[j] = round((headings[j+1] - headings[j]) / (2*pi)).
    two_pi = 2.0 * np.pi
    adjustments = np.zeros_like(headings)
    adjustments[..., 1:] = np.cumsum(np.round(np.diff(headings, axis=-1) / two_pi), axis=-1)
    unwrapped = headings - two_pi * adjustments
    return unwrapped


def _approximate_derivatives(
    y: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
    window_length: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
    axis: int = -1,
) -> npt.NDArray[np.float32]:
    """
    Given two equal-length sequences y and x, compute an approximation to the n-th
    derivative of some function interpolating the (x, y) data points, and return its
    values at the x's.  We assume the x's are increasing and equally-spaced.
    :param y: The dependent variable (say of length n)
    :param x: The independent variable (must have the same length n).  Must be strictly
        increasing and equally-spaced.
    :param window_length: The order (default 5) of the Savitsky-Golay filter used.
        (Ignored if the x's are not equally-spaced.)  Must be odd and at least 3
    :param poly_order: The degree (default 2) of the filter polynomial used.  Must
        be less than the window_length
    :param deriv_order: The order of derivative to compute (default 1)
    :param axis: The axis of the array x along which the filter is to be applied. Default is -1.
    :return Derivatives.
    """
    window_length = min(window_length, len(x))

    if not (poly_order < window_length):
        raise ValueError(f"{poly_order} < {window_length} does not hold!")

    dx = np.diff(x, axis=-1)
    if not (dx > 0).all():
        raise RuntimeError("dx is not monotonically increasing!")

    dx = dx.mean()
    derivative: npt.NDArray[np.float32] = savgol_filter(
        y,
        polyorder=poly_order,
        window_length=window_length,
        deriv=deriv_order,
        delta=dx,
        axis=axis,
    )
    return derivative


def _within_bound(
    metric: npt.NDArray[np.float64],
    min_bound: Optional[float] = None,
    max_bound: Optional[float] = None,
) -> npt.NDArray[np.bool_]:
    """
    Determines wether values in batch-dim are within bounds.
    :param metric: metric values
    :param min_bound: minimum bound, defaults to None
    :param max_bound: maximum bound, defaults to None
    :return: array of booleans wether metric values are within bounds
    """
    min_bound = min_bound if min_bound else float(-np.inf)
    max_bound = max_bound if max_bound else float(np.inf)
    metric_values = np.array(metric)
    metric_within_bound = (metric_values > min_bound) & (metric_values < max_bound)
    return np.all(metric_within_bound, axis=-1)


def _compute_lon_acceleration(
    states: npt.NDArray[np.float64], time_steps_s: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """
    Compute longitudinal acceleration over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :return: longitudinal acceleration within bound
    """
    n_batch, n_time, n_states = states.shape
    lon_acceleration = _extract_ego_acceleration(states, acceleration_coordinate="x", window_length=n_time)
    return _within_bound(lon_acceleration, min_bound=min_lon_accel, max_bound=max_lon_accel)


def _compute_lat_acceleration(
    states: npt.NDArray[np.float64], time_steps_s: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """
    Compute lateral acceleration over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :return: lateral acceleration within bound
    """
    n_batch, n_time, n_states = states.shape
    lat_acceleration = _extract_ego_acceleration(states, acceleration_coordinate="y", window_length=n_time)
    return _within_bound(lat_acceleration, min_bound=-max_abs_lat_accel, max_bound=max_abs_lat_accel)


def _compute_jerk_metric(
    states: npt.NDArray[np.float64], time_steps_s: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """
    Compute absolute jerk over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :return: absolute jerk within bound
    """
    n_batch, n_time, n_states = states.shape
    jerk_metric = _extract_ego_jerk(
        states,
        acceleration_coordinate="magnitude",
        time_steps_s=time_steps_s,
        window_length=n_time,
    )
    return _within_bound(jerk_metric, min_bound=-max_abs_mag_jerk, max_bound=max_abs_mag_jerk)


def _compute_lon_jerk_metric(
    states: npt.NDArray[np.float64], time_steps_s: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """
    Compute longitudinal jerk over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :return: longitudinal jerk within bound
    """
    n_batch, n_time, n_states = states.shape
    lon_jerk_metric = _extract_ego_jerk(
        states,
        acceleration_coordinate="x",
        time_steps_s=time_steps_s,
        window_length=n_time,
    )
    return _within_bound(lon_jerk_metric, min_bound=-max_abs_lon_jerk, max_bound=max_abs_lon_jerk)


def _compute_yaw_accel(states: npt.NDArray[np.float64], time_steps_s: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """
    Compute acceleration of yaw-angle over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :return: acceleration of yaw-angle within bound
    """
    n_batch, n_time, n_states = states.shape
    yaw_accel_metric = _extract_ego_yaw_rate(states, time_steps_s, deriv_order=2, poly_order=3, window_length=n_time)
    return _within_bound(yaw_accel_metric, min_bound=-max_abs_yaw_accel, max_bound=max_abs_yaw_accel)


def _compute_yaw_rate(states: npt.NDArray[np.float64], time_steps_s: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """
    Compute velocity of yaw-angle over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :return: velocity of yaw-angle within bound
    """
    n_batch, n_time, n_states = states.shape
    yaw_rate_metric = _extract_ego_yaw_rate(states, time_steps_s, window_length=n_time)
    return _within_bound(yaw_rate_metric, min_bound=-max_abs_yaw_rate, max_bound=max_abs_yaw_rate)


def ego_is_comfortable(states: npt.NDArray[np.float64], time_point_s: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """
    Accumulates all within-bound comfortability metrics
    :param states: array representation of ego state values
    :param time_point_s: time steps [s] of time dim
    :return: _description_
    """
    n_batch, n_time, n_states = states.shape
    assert n_time == len(time_point_s)
    assert n_states == StateIndex.size()

    comfort_metric_functions = [
        _compute_lon_acceleration,
        _compute_lat_acceleration,
        _compute_jerk_metric,
        _compute_lon_jerk_metric,
        _compute_yaw_accel,
        _compute_yaw_rate,
    ]
    results: npt.NDArray[np.bool_] = np.zeros((n_batch, len(comfort_metric_functions)), dtype=np.bool_)
    for idx, metric_function in enumerate(comfort_metric_functions):
        results[:, idx] = metric_function(states, time_point_s)

    return results

def extract_features(
    states: npt.NDArray[np.float64],
    time_point_s: npt.NDArray[np.float64],
    vehicle_parameters: VehicleParameters = get_pacifica_parameters(),
) -> dict:
    """
    Extract features needed for Extended Comfort evaluation.
    :param states: Array of ego states (n_batch, n_time, n_features).
    :param time_point_s: Array of time steps in seconds.
    :param vehicle_parameters: parameters of vehicle
    :return: A dictionary of features.
    """
    return {
        "acceleration": _extract_ego_acceleration(states, "magnitude", vehicle_parameters=vehicle_parameters),
        "jerk": _extract_ego_jerk(states, "magnitude", time_point_s, vehicle_parameters=vehicle_parameters),
        "yaw_rate": _extract_ego_yaw_rate(states, time_point_s),
        "yaw_accel": _extract_ego_yaw_rate(states, time_point_s, deriv_order=2),
    }
def calculate_rms(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute the Root Mean Square (RMS) of the given values along the time axis.

    :param values: Array containing values (n_batch, n_time).
    :return: RMS value per batch (n_batch,).
    """
    squared_values = values**2  # Square the differences
    mean_squared = np.mean(squared_values, axis=1)  # Compute mean along time axis
    rms_values = np.sqrt(mean_squared)  # Square root to get RMS
    return rms_values

def ego_is_two_frame_extended_comfort(
    states_1: npt.NDArray[np.float64],
    states_2: npt.NDArray[np.float64],
    time_point_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """
    Evaluate whether the differences between two consecutive trajectories satisfy the Extended Comfort metric.
    :param states_1: First trajectory (n_batch, n_time, n_features).
    :param states_2: Second trajectory (n_batch, n_time, n_features).
    :param time_point_s: Array of time steps in seconds.
    :return: Boolean array indicating whether the difference between trajectories meets the criteria.
    """
    assert states_1.shape == states_2.shape, "Both trajectories must have the same shape"

    # Extract features for both trajectories
    features_1 = extract_features(states_1, time_point_s)
    features_2 = extract_features(states_2, time_point_s)

    # Compute differences between corresponding time steps
    diff_acceleration = features_1["acceleration"] - features_2["acceleration"]
    diff_jerk = features_1["jerk"] - features_2["jerk"]
    diff_yaw_rate = features_1["yaw_rate"] - features_2["yaw_rate"]
    diff_yaw_accel = features_1["yaw_accel"] - features_2["yaw_accel"]

    # Calculate RMS differences
    rms_acceleration = calculate_rms(diff_acceleration)
    rms_jerk = calculate_rms(diff_jerk)
    rms_yaw_rate = calculate_rms(diff_yaw_rate)
    rms_yaw_accel = calculate_rms(diff_yaw_accel)

    # Compare RMS differences against thresholds
    meets_acceleration = rms_acceleration <= acceleration_threshold
    meets_jerk = rms_jerk <= jerk_threshold
    meets_yaw_rate = rms_yaw_rate <= yaw_rate_threshold
    meets_yaw_accel = rms_yaw_accel <= yaw_accel_threshold

    # Combine all criteria
    return np.logical_and.reduce([meets_acceleration, meets_jerk, meets_yaw_rate, meets_yaw_accel])



import numpy as np
import pytest
from data_pipeline import future_chunk, training_statistics, to_environment_action, quaternion_to_axis_angle


def test_chunk_does_not_cross_episode_and_marks_padding():
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    chunk, mask = future_chunk(actions, 1, 4)
    np.testing.assert_array_equal(chunk[:2], actions[1:])
    np.testing.assert_array_equal(chunk[2:], np.repeat(actions[-1:], 2, axis=0))
    assert mask.tolist() == [False, False, True, True]


def test_statistics_exclude_held_out_episode():
    episodes = {0: {'state': np.zeros((2, 8)), 'action': np.ones((2, 7))},
                1: {'state': np.full((2, 8), 100), 'action': np.full((2, 7), 100)}}
    stats = training_statistics(episodes, [0])
    np.testing.assert_array_equal(stats['observation.state']['mean'], np.zeros(8))
    np.testing.assert_array_equal(stats['action']['mean'], np.ones(7))
    assert np.isfinite(stats['action']['std']).all()
    assert min(stats['action']['std']) > 0


@pytest.mark.parametrize('gripper,expected', [(0, 1), (1, -1), (.25, .5)])
def test_rlds_gripper_conversion(gripper, expected):
    action = np.array([.1, -.2, 0, .3, 0, 0, gripper], dtype=np.float32)
    converted = to_environment_action(action)
    np.testing.assert_allclose(converted[:6], action[:6])
    assert converted[-1] == expected
    assert action[-1] == gripper


def test_quaternion_identity_and_half_turn():
    np.testing.assert_allclose(quaternion_to_axis_angle([0, 0, 0, 1]), [0, 0, 0])
    np.testing.assert_allclose(quaternion_to_axis_angle([0, 0, 1, 0]), [0, 0, np.pi])


def test_quaternion_preserves_dataset_angles_greater_than_pi():
    angle = 3.18
    quat = [np.sin(angle/2), 0, 0, np.cos(angle/2)]
    np.testing.assert_allclose(quaternion_to_axis_angle(quat), [angle, 0, 0], atol=1e-6)

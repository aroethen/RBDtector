import shutil
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal
import pandas as pd
import sys

sys.path.append('../RBDtector')
from app_logic.PSG_controller import PSGController
from util import settings

test_path = './data/Test 1'
reference_path = './reference/Test 1'


def test_normal_settings():
    # run test
    PSGController.run_rbd_detection(test_path, test_path)

    # read test results
    normal_test_path = list(Path(test_path, 'RBDtector output').glob('RBDtector_results*.xlsx'))[0]
    normal_test = pd.read_excel(normal_test_path, header=[0, 1], skiprows=[2])

    normal_test_channels_path = list(Path(test_path, 'RBDtector output').glob('Channel_combinations*.xlsx'))[0]
    normal_test_channels = pd.read_excel(normal_test_channels_path, header=[0])

    # read reference results
    normal_ref_path = Path(reference_path, 'RBDtector_results_normal.xlsx')
    normal_reference = pd.read_excel(normal_ref_path, header=[0, 1], skiprows=[2])

    normal_ref_channels_path = Path(reference_path, 'Channel_combinations_normal.xlsx')
    normal_reference_channels = pd.read_excel(normal_ref_channels_path, header=[0])

    assert_frame_equal(normal_test, normal_reference)
    assert_frame_equal(normal_test_channels, normal_reference_channels)


def test_with_flow_settings(monkeypatch):
    monkeypatch.setattr(settings, 'FLOW', True)
    PSGController.run_rbd_detection(test_path, test_path)
    with_flow_test_path = list(Path(test_path, 'RBDtector output').glob('RBDtector_results*.xlsx'))[0]
    with_flow_test = pd.read_excel(with_flow_test_path, header=[0, 1], skiprows=[2])

    with_flow_test_channels_path = list(Path(test_path, 'RBDtector output').glob('Channel_combinations*.xlsx'))[0]
    with_flow_test_channels = pd.read_excel(with_flow_test_channels_path, header=[0])

    with_flow_ref_path = Path(reference_path, 'RBDtector_results_FlowTrue.xlsx')
    with_flow_reference = pd.read_excel(with_flow_ref_path, header=[0, 1], skiprows=[2])

    with_flow_ref_channels_path = Path(reference_path, 'Channel_combinations_FlowTrue.xlsx')
    with_flow_reference_channels = pd.read_excel(with_flow_ref_channels_path, header=[0])

    assert_frame_equal(with_flow_test, with_flow_reference)
    assert_frame_equal(with_flow_test_channels, with_flow_reference_channels)


def test_no_snore_settings(monkeypatch):
    monkeypatch.setattr(settings, 'SNORE', False)
    PSGController.run_rbd_detection(test_path, test_path)
    no_snore_test_path = list(Path(test_path, 'RBDtector output').glob('RBDtector_results*.xlsx'))[0]
    no_snore_test = pd.read_excel(no_snore_test_path, header=[0, 1], skiprows=[2])

    no_snore_test_channels_path = list(Path(test_path, 'RBDtector output').glob('Channel_combinations*.xlsx'))[0]
    no_snore_test_channels = pd.read_excel(no_snore_test_channels_path, header=[0])

    no_snore_ref_path = Path(reference_path, 'RBDtector_results_SnoreFalse.xlsx')
    no_snore_reference = pd.read_excel(no_snore_ref_path, header=[0, 1], skiprows=[2])

    no_snore_ref_channels_path = Path(reference_path, 'Channel_combinations_SnoreFalse.xlsx')
    no_snore_reference_channels = pd.read_excel(no_snore_ref_channels_path, header=[0])

    assert_frame_equal(no_snore_test, no_snore_reference)
    assert_frame_equal(no_snore_test_channels, no_snore_reference_channels)

def test_3_channels_settings(monkeypatch):
    monkeypatch.setattr(settings, 'SIGNALS_TO_EVALUATE', ['EMG', 'PLM r', 'AUX'])
    monkeypatch.setattr(settings, 'LEGS', 1)
    monkeypatch.setattr(settings, 'ARMS', 2)

    # run test
    PSGController.run_rbd_detection(test_path, test_path)

    # read in results
    less_channels_test_path = list(Path(test_path, 'RBDtector output').glob('RBDtector_results*.xlsx'))[0]
    less_channels_test = pd.read_excel(less_channels_test_path, header=[0, 1], skiprows=[2])

    less_channels_channels_path = list(Path(test_path, 'RBDtector output').glob('Channel_combinations*.xlsx'))[0]
    less_channels_test_channels = pd.read_excel(less_channels_channels_path, header=[0])

    # read in reference results
    less_channels_ref_path = Path(reference_path, 'RBDtector_results_EMG-PLM r-AUX.xlsx')
    less_channels_reference = pd.read_excel(less_channels_ref_path, header=[0, 1], skiprows=[2])

    less_channels_ref_channels_path = Path(reference_path, 'Channel_combinations_EMG-PLM r-AUX.xlsx')
    less_channels_reference_channels = pd.read_excel(less_channels_ref_channels_path, header=[0])

    # test for equality
    assert_frame_equal(less_channels_test, less_channels_reference)
    assert_frame_equal(less_channels_test_channels, less_channels_reference_channels)


def test_human_artifacts_settings(monkeypatch):
    monkeypatch.setattr(settings, 'HUMAN_ARTIFACTS', True)

    # run test
    PSGController.run_rbd_detection(test_path, test_path)

    # read in results
    human_artifacts_test_path = list(Path(test_path, 'RBDtector output').glob('RBDtector_results*.xlsx'))[0]
    human_artifacts_test = pd.read_excel(human_artifacts_test_path, header=[0, 1], skiprows=[2])

    human_artifacts_channels_path = list(Path(test_path, 'RBDtector output').glob('Channel_combinations*.xlsx'))[0]
    human_artifacts_test_channels = pd.read_excel(human_artifacts_channels_path, header=[0])

    # read in reference results
    human_artifacts_ref_path = Path(reference_path, 'RBDtector_results_w-human-artifacts.xlsx')
    human_artifacts_reference = pd.read_excel(human_artifacts_ref_path, header=[0, 1], skiprows=[2])

    human_artifacts_ref_channels_path = Path(reference_path, 'Channel_combinations_w-human-artifacts.xlsx')
    human_artifacts_reference_channels = pd.read_excel(human_artifacts_ref_channels_path, header=[0])

    # test for equality
    assert_frame_equal(human_artifacts_test, human_artifacts_reference)
    assert_frame_equal(human_artifacts_test_channels, human_artifacts_reference_channels)


def test_human_baseline_settings(monkeypatch):
    monkeypatch.setattr(settings, 'HUMAN_BASELINE', True)

    # run test
    PSGController.run_rbd_detection(test_path, test_path)

    # read in results
    human_baseline_test_path = list(Path(test_path, 'RBDtector output').glob('RBDtector_results*.xlsx'))[0]
    human_baseline_test = pd.read_excel(human_baseline_test_path, header=[0, 1], skiprows=[2])

    human_baseline_channels_path = list(Path(test_path, 'RBDtector output').glob('Channel_combinations*.xlsx'))[0]
    human_baseline_test_channels = pd.read_excel(human_baseline_channels_path, header=[0])

    # read in reference results
    human_baseline_ref_path = Path(reference_path, 'RBDtector_results_human-baseline.xlsx')
    human_baseline_reference = pd.read_excel(human_baseline_ref_path, header=[0, 1], skiprows=[2])

    human_baseline_ref_channels_path = Path(reference_path, 'Channel_combinations_human-baseline.xlsx')
    human_baseline_reference_channels = pd.read_excel(human_baseline_ref_channels_path, header=[0])

    # test for equality
    assert_frame_equal(human_baseline_test, human_baseline_reference)
    assert_frame_equal(human_baseline_test_channels, human_baseline_reference_channels)


def test_all_defaults_changed_settings(monkeypatch):
    
    monkeypatch.setattr(settings, 'FLOW', True)
    monkeypatch.setattr(settings, 'SIGNALS_TO_EVALUATE', ['EMG', 'PLM r', 'AUX'])
    monkeypatch.setattr(settings, 'LEGS', 1)
    monkeypatch.setattr(settings, 'ARMS', 2)
    monkeypatch.setattr(settings, 'HUMAN_ARTIFACTS', True)
    monkeypatch.setattr(settings, 'HUMAN_BASELINE', True)
    monkeypatch.setattr(settings, 'SNORE', False)

    # run test
    PSGController.run_rbd_detection(test_path, test_path)

    # read in results
    all_defaults_changed_test_path = list(Path(test_path, 'RBDtector output').glob('RBDtector_results*.xlsx'))[0]
    all_defaults_changed_test = pd.read_excel(all_defaults_changed_test_path, header=[0, 1], skiprows=[2])

    all_defaults_changed_channels_path = list(Path(test_path, 'RBDtector output').glob('Channel_combinations*.xlsx'))[0]
    all_defaults_changed_test_channels = pd.read_excel(all_defaults_changed_channels_path, header=[0])

    # read in reference results
    all_defaults_changed_ref_path = Path(reference_path, 'RBDtector_results_all-defaults-changed.xlsx')
    all_defaults_changed_reference = pd.read_excel(all_defaults_changed_ref_path, header=[0, 1], skiprows=[2])

    all_defaults_changed_ref_channels_path = Path(reference_path, 'Channel_combinations_all-defaults-changed.xlsx')
    all_defaults_changed_reference_channels = pd.read_excel(all_defaults_changed_ref_channels_path, header=[0])

    # test for equality
    assert_frame_equal(all_defaults_changed_test, all_defaults_changed_reference)
    assert_frame_equal(all_defaults_changed_test_channels, all_defaults_changed_reference_channels)


@pytest.fixture(autouse=True)
def delete_tested_files(request):
    # Code that runs before test
    assert Path(test_path).exists()
    assert Path(reference_path).exists()
    assert not Path(test_path, 'RBDtector output').exists()

    # Code that runs after test
    def remove_test_result_dir():
        if Path('./data/Test 1/RBDtector output').exists():
            shutil.rmtree('./data/Test 1/RBDtector output')
    request.addfinalizer(remove_test_result_dir)

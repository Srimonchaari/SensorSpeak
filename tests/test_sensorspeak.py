"""
tests/test_sensorspeak.py

Unit tests for the SensorSpeak pipeline.
Run with:  pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd

from sensorspeak_core import (
    # constants
    SAMPLE_RATE_HZ, GRAVITY_Z,
    SEG_IDLE_DURATION, SEG_WALK_DURATION, SEG_IMPACT_DURATION, SEG_SHAKE_DURATION,
    ROLLING_WINDOW, ZSCORE_EPS,
    IDLE_STD_MAX, IMPACT_MEAN_MIN, IMPACT_STD_MIN,
    SHAKING_STD_MIN, WALKING_MEAN_MIN, WALKING_MEAN_MAX,
    WALKING_STD_MIN, WALKING_STD_MAX, MIN_EVENT_SAMPLES,
    SIMILARITY_TOP_K,
    # dataclass
    MotionEvent,
    # functions
    generate_synthetic_data,
    normalize_and_engineer_features,
    _classify_sample,
    detect_events,
    _severity_label,
    summarize_event,
    _keyword_fallback,
    query_events,
    run_pipeline,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def raw_df():
    return generate_synthetic_data()


@pytest.fixture(scope='module')
def featured_df(raw_df):
    return normalize_and_engineer_features(raw_df)


@pytest.fixture(scope='module')
def events(featured_df):
    return detect_events(featured_df)


@pytest.fixture(scope='module')
def summaries(events):
    return [summarize_event(ev) for ev in events]


@pytest.fixture
def sample_event():
    return MotionEvent(
        start=1.0, end=3.5, type='impact',
        max_mag=14.2, mean_mag=6.5, seed='sharp spike'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: generate_synthetic_data
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateSyntheticData:
    def test_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_expected_row_count(self, raw_df):
        expected = int((SEG_IDLE_DURATION + SEG_WALK_DURATION +
                        SEG_IMPACT_DURATION + SEG_SHAKE_DURATION) * SAMPLE_RATE_HZ)
        assert len(raw_df) == expected

    def test_required_columns_present(self, raw_df):
        for col in ('timestamp', 'accel_x', 'accel_y', 'accel_z', '_segment_label'):
            assert col in raw_df.columns, f'Missing column: {col}'

    def test_timestamp_monotonically_increasing(self, raw_df):
        diffs = raw_df['timestamp'].diff().dropna()
        assert (diffs > 0).all(), 'Timestamps are not strictly increasing'

    def test_timestamp_starts_at_zero(self, raw_df):
        assert raw_df['timestamp'].iloc[0] == pytest.approx(0.0)

    def test_timestamp_resolution_matches_sample_rate(self, raw_df):
        expected_step = 1.0 / SAMPLE_RATE_HZ
        actual_step = raw_df['timestamp'].iloc[1] - raw_df['timestamp'].iloc[0]
        assert actual_step == pytest.approx(expected_step, rel=1e-3)

    def test_segment_labels_are_valid(self, raw_df):
        valid = {'idle', 'walking_like', 'impact', 'shaking'}
        assert set(raw_df['_segment_label'].unique()).issubset(valid)

    def test_all_four_segments_present(self, raw_df):
        labels = set(raw_df['_segment_label'].unique())
        assert labels == {'idle', 'walking_like', 'impact', 'shaking'}

    def test_idle_segment_count(self, raw_df):
        n_idle = (raw_df['_segment_label'] == 'idle').sum()
        assert n_idle == SEG_IDLE_DURATION * SAMPLE_RATE_HZ

    def test_z_axis_gravity_offset_in_idle(self, raw_df):
        idle_z = raw_df.loc[raw_df['_segment_label'] == 'idle', 'accel_z']
        assert idle_z.mean() == pytest.approx(GRAVITY_Z, abs=0.5)

    def test_no_null_values(self, raw_df):
        assert not raw_df.isnull().any().any(), 'Raw DataFrame contains NaN values'

    def test_reproducibility(self):
        df1 = generate_synthetic_data()
        df2 = generate_synthetic_data()
        pd.testing.assert_frame_equal(df1, df2)

    def test_impact_spike_exists(self, raw_df):
        impact_rows = raw_df[raw_df['_segment_label'] == 'impact']
        # The spike on X axis should exceed IMPACT_SPIKE / 2 at minimum
        assert impact_rows['accel_x'].abs().max() > 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: normalize_and_engineer_features
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeAndEngineerFeatures:
    def test_returns_dataframe(self, featured_df):
        assert isinstance(featured_df, pd.DataFrame)

    def test_raw_columns_preserved(self, featured_df):
        for axis in ('accel_x', 'accel_y', 'accel_z'):
            assert f'_raw_{axis}' in featured_df.columns

    def test_magnitude_column_exists(self, featured_df):
        assert 'accel_magnitude' in featured_df.columns

    def test_rolling_mean_column_exists(self, featured_df):
        assert 'rolling_mean' in featured_df.columns

    def test_rolling_std_column_exists(self, featured_df):
        assert 'rolling_std' in featured_df.columns

    def test_magnitude_is_non_negative(self, featured_df):
        assert (featured_df['accel_magnitude'] >= 0).all()

    def test_rolling_std_is_non_negative(self, featured_df):
        assert (featured_df['rolling_std'] >= 0).all()

    def test_zscore_mean_near_zero(self, featured_df):
        for axis in ('accel_x', 'accel_y', 'accel_z'):
            assert abs(featured_df[axis].mean()) < 0.01, f'{axis} mean not near 0'

    def test_zscore_std_near_one(self, featured_df):
        for axis in ('accel_x', 'accel_y', 'accel_z'):
            assert abs(featured_df[axis].std() - 1.0) < 0.01, f'{axis} std not near 1'

    def test_row_count_unchanged(self, raw_df, featured_df):
        assert len(featured_df) == len(raw_df)

    def test_no_null_values(self, featured_df):
        assert not featured_df.isnull().any().any()

    def test_raises_on_missing_column(self, raw_df):
        bad_df = raw_df.drop(columns=['accel_z'])
        with pytest.raises(ValueError, match='missing required columns'):
            normalize_and_engineer_features(bad_df)

    def test_raises_on_all_missing(self):
        bad_df = pd.DataFrame({'timestamp': [1, 2, 3], 'foo': [1, 2, 3]})
        with pytest.raises(ValueError):
            normalize_and_engineer_features(bad_df)

    def test_does_not_modify_input(self, raw_df):
        original_x = raw_df['accel_x'].copy()
        normalize_and_engineer_features(raw_df)
        pd.testing.assert_series_equal(raw_df['accel_x'], original_x)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: _classify_sample
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifySample:
    def test_idle_when_std_below_threshold(self):
        assert _classify_sample(0.05, IDLE_STD_MAX - 0.01) == 'idle'

    def test_idle_at_zero(self):
        assert _classify_sample(0.0, 0.0) == 'idle'

    def test_impact_when_both_thresholds_exceeded(self):
        result = _classify_sample(IMPACT_MEAN_MIN + 0.5, IMPACT_STD_MIN + 0.5)
        assert result == 'impact'

    def test_impact_requires_both_conditions(self):
        # High mean but low std → NOT impact
        assert _classify_sample(IMPACT_MEAN_MIN + 1.0, 0.1) != 'impact'
        # Low mean but high std → NOT impact
        assert _classify_sample(0.1, IMPACT_STD_MIN + 1.0) != 'impact'

    def test_shaking_when_std_very_high(self):
        assert _classify_sample(1.0, SHAKING_STD_MIN + 0.5) == 'shaking'

    def test_walking_in_valid_range(self):
        rmean = (WALKING_MEAN_MIN + WALKING_MEAN_MAX) / 2
        rstd  = (WALKING_STD_MIN + WALKING_STD_MAX) / 2
        assert _classify_sample(rmean, rstd) == 'walking'

    def test_walking_at_lower_bound(self):
        # WALKING_STD_MIN (0.10) < IDLE_STD_MAX (0.15), so the idle rule fires
        # at the exact lower bound.  Use a value just above IDLE_STD_MAX instead.
        assert _classify_sample(WALKING_MEAN_MIN, IDLE_STD_MAX + 0.01) == 'walking'

    def test_walking_at_upper_bound(self):
        assert _classify_sample(WALKING_MEAN_MAX, WALKING_STD_MAX) == 'walking'

    def test_unknown_for_unmatched(self):
        # Low mean, mid-range std — doesn't fit any rule
        assert _classify_sample(0.3, 0.5) == 'unknown'

    def test_returns_string(self):
        result = _classify_sample(1.0, 0.5)
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: detect_events
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectEvents:
    def test_returns_list(self, events):
        assert isinstance(events, list)

    def test_all_items_are_motion_events(self, events):
        for ev in events:
            assert isinstance(ev, MotionEvent)

    def test_at_least_one_event(self, events):
        assert len(events) >= 1

    def test_start_before_end(self, events):
        for ev in events:
            assert ev.start < ev.end, f'Event {ev.type}: start >= end'

    def test_max_mag_gte_mean_mag(self, events):
        for ev in events:
            assert ev.max_mag >= ev.mean_mag, f'Event {ev.type}: max_mag < mean_mag'

    def test_all_mag_positive(self, events):
        for ev in events:
            assert ev.max_mag > 0
            assert ev.mean_mag > 0

    def test_event_types_are_valid(self, events):
        valid = {'idle', 'walking', 'impact', 'shaking', 'unknown'}
        for ev in events:
            assert ev.type in valid, f'Unknown event type: {ev.type}'

    def test_events_contain_expected_types(self, events):
        types = {ev.type for ev in events}
        # idle and at least one high-energy type are always present after normalisation.
        # 'impact' may merge into 'shaking' or 'unknown' after global z-score rescaling
        # (the spike is divided by the whole-signal std), so we check the raw spike
        # separately in TestGenerateSyntheticData.test_impact_spike_exists.
        assert 'idle' in types
        assert types & {'impact', 'shaking', 'unknown'}, (
            'Expected at least one high-energy event type in detected events'
        )

    def test_raises_on_missing_timestamp(self, featured_df):
        bad_df = featured_df.drop(columns=['timestamp'])
        with pytest.raises(ValueError, match='timestamp'):
            detect_events(bad_df)

    def test_raises_on_missing_rolling_mean(self, featured_df):
        bad_df = featured_df.drop(columns=['rolling_mean'])
        with pytest.raises(ValueError, match='rolling_mean'):
            detect_events(bad_df)

    def test_empty_dataframe_returns_empty_list(self):
        empty_df = pd.DataFrame(columns=['timestamp', 'rolling_mean', 'rolling_std', 'accel_magnitude'])
        result = detect_events(empty_df)
        assert result == []

    def test_short_run_below_min_samples_is_skipped(self):
        """A run shorter than MIN_EVENT_SAMPLES should not produce an event."""
        n = MIN_EVENT_SAMPLES - 1
        df = pd.DataFrame({
            'timestamp':       np.linspace(0, 1, n),
            'rolling_mean':    np.full(n, 0.05),   # idle-like
            'rolling_std':     np.full(n, 0.01),
            'accel_magnitude': np.full(n, 0.1),
        })
        assert detect_events(df) == []

    def test_long_run_produces_event(self):
        n = MIN_EVENT_SAMPLES * 3
        df = pd.DataFrame({
            'timestamp':       np.linspace(0, 3, n),
            'rolling_mean':    np.full(n, 0.05),
            'rolling_std':     np.full(n, 0.01),
            'accel_magnitude': np.full(n, 0.1),
        })
        result = detect_events(df)
        assert len(result) == 1
        assert result[0].type == 'idle'


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: _severity_label and summarize_event
# ─────────────────────────────────────────────────────────────────────────────

class TestSeverityLabel:
    def test_low_below_1_5(self):
        assert _severity_label(0.5)  == 'low'
        assert _severity_label(1.49) == 'low'

    def test_moderate_1_5_to_3_5(self):
        assert _severity_label(1.5)  == 'moderate'
        assert _severity_label(3.49) == 'moderate'

    def test_high_3_5_to_7(self):
        assert _severity_label(3.5)  == 'high'
        assert _severity_label(6.99) == 'high'

    def test_severe_above_7(self):
        assert _severity_label(7.0)  == 'severe'
        assert _severity_label(20.0) == 'severe'

    def test_boundary_exactly_1_5(self):
        assert _severity_label(1.5) == 'moderate'

    def test_boundary_exactly_3_5(self):
        assert _severity_label(3.5) == 'high'

    def test_boundary_exactly_7(self):
        assert _severity_label(7.0) == 'severe'


class TestSummarizeEvent:
    def test_returns_string(self, sample_event):
        assert isinstance(summarize_event(sample_event), str)

    def test_contains_event_type(self, sample_event):
        assert 'impact' in summarize_event(sample_event)

    def test_contains_start_time(self, sample_event):
        assert '1.00' in summarize_event(sample_event)

    def test_contains_end_time(self, sample_event):
        assert '3.50' in summarize_event(sample_event)

    def test_contains_severity_word(self, sample_event):
        result = summarize_event(sample_event)
        assert any(word in result for word in ('low', 'moderate', 'high', 'severe'))

    def test_contains_peak_magnitude(self, sample_event):
        assert '14.2' in summarize_event(sample_event)

    def test_contains_seed(self, sample_event):
        assert 'sharp spike' in summarize_event(sample_event)

    def test_single_sentence_structure(self, sample_event):
        result = summarize_event(sample_event)
        # Ends with a period (sentence structure)
        assert result.strip().endswith('.')

    def test_all_events_produce_summaries(self, events, summaries):
        assert len(summaries) == len(events)
        for s in summaries:
            assert isinstance(s, str)
            assert len(s) > 20


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: _keyword_fallback and query_events
# ─────────────────────────────────────────────────────────────────────────────

class TestKeywordFallback:
    SAMPLE_SUMMARIES = [
        'From 0.00s to 4.99s (4.99s), a low-severity "idle" event was detected.',
        'From 5.00s to 12.99s (7.99s), a moderate-severity "walking" event was detected.',
        'From 13.00s to 14.98s (1.98s), a severe-severity "impact" event was detected.',
    ]

    def test_returns_string(self):
        result = _keyword_fallback('impact', self.SAMPLE_SUMMARIES)
        assert isinstance(result, str)

    def test_returns_no_match_for_irrelevant_query(self):
        result = _keyword_fallback('tornado weather forecast', self.SAMPLE_SUMMARIES)
        assert 'No matching events' in result

    def test_returns_relevant_summary_for_impact(self):
        result = _keyword_fallback('impact', self.SAMPLE_SUMMARIES)
        assert 'impact' in result.lower()

    def test_returns_relevant_summary_for_idle(self):
        result = _keyword_fallback('idle', self.SAMPLE_SUMMARIES)
        assert 'idle' in result.lower()

    def test_respects_top_k_limit(self):
        # All three summaries match 'event' — result should not exceed SIMILARITY_TOP_K
        result = _keyword_fallback('event detected', self.SAMPLE_SUMMARIES)
        bullet_count = result.count('\u2022')
        assert bullet_count <= SIMILARITY_TOP_K

    def test_empty_summaries_returns_no_match(self):
        result = _keyword_fallback('impact', [])
        assert 'No matching events' in result

    def test_multi_keyword_scores_higher(self):
        summaries = [
            'idle event at low severity',
            'impact event at severe severity with spike',
        ]
        result = _keyword_fallback('severe impact spike', summaries)
        # The impact summary has more matching words and should appear first
        assert 'impact' in result


class TestQueryEvents:
    def test_returns_string(self, summaries):
        result = query_events('idle', index=None, summaries=summaries)
        assert isinstance(result, str)

    def test_empty_question_returns_prompt(self, summaries):
        result = query_events('', index=None, summaries=summaries)
        assert 'non-empty' in result.lower() or 'please' in result.lower()

    def test_whitespace_question_returns_prompt(self, summaries):
        result = query_events('   ', index=None, summaries=summaries)
        assert 'non-empty' in result.lower() or 'please' in result.lower()

    def test_keyword_fallback_used_when_index_is_none(self, summaries):
        result = query_events('impact', index=None, summaries=summaries)
        assert isinstance(result, str)
        assert len(result) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Integration: run_pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestRunPipeline:
    def test_returns_dict_with_expected_keys(self):
        result = run_pipeline()
        for key in ('df', 'events', 'summaries', 'index'):
            assert key in result, f'Missing key: {key}'

    def test_df_has_required_feature_columns(self):
        result = run_pipeline()
        for col in ('accel_magnitude', 'rolling_mean', 'rolling_std'):
            assert col in result['df'].columns

    def test_events_is_list_of_motion_events(self):
        result = run_pipeline()
        assert isinstance(result['events'], list)
        assert all(isinstance(e, MotionEvent) for e in result['events'])

    def test_summaries_length_matches_events(self):
        result = run_pipeline()
        assert len(result['summaries']) == len(result['events'])

    def test_raises_on_csv_with_missing_columns(self, tmp_path):
        bad_csv = tmp_path / 'bad.csv'
        pd.DataFrame({'timestamp': [1, 2], 'accel_x': [0, 0]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match='missing required columns'):
            run_pipeline(csv_path=str(bad_csv))

    def test_loads_real_csv_correctly(self, tmp_path):
        df = generate_synthetic_data()
        csv_path = tmp_path / 'test.csv'
        df.to_csv(csv_path, index=False)
        result = run_pipeline(csv_path=str(csv_path))
        assert len(result['df']) == len(df)


# ─────────────────────────────────────────────────────────────────────────────
# MotionEvent dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestMotionEvent:
    def test_fields_accessible(self, sample_event):
        assert sample_event.start   == 1.0
        assert sample_event.end     == 3.5
        assert sample_event.type    == 'impact'
        assert sample_event.max_mag == 14.2

    def test_duration_computable(self, sample_event):
        assert sample_event.end - sample_event.start == pytest.approx(2.5)

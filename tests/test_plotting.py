"""plot() options and spectrogram(): the figures used throughout docs/

The existing plot() behavior is covered in test_utils.py and must not change.
These tests cover what the documentation relies on: saving to a file, stacked
panels, labels, looking at a window of time that starts after zero, and a
spectrogram that shows how frequency content changes.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import sympy as sp

from daw.effects import pan
from daw.filters import low_pass_filter
from daw.stereo import StereoExpr
from daw.waves.basic import sawtooth, sine, square, trigger
from daw.waves.time import BPM, TIME
from daw.waves.utils import compile, linspace, plot, spectrogram

SR = 1000
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def shown(monkeypatch):
    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
    return calls


class TestPlotToFile:
    def test_writes_a_png_and_does_not_open_a_window(self, shown, tmp_path):
        out = tmp_path / "wave.png"
        plot(sine(5), duration=1.0, sample_rate=SR, output_file=str(out))
        assert out.read_bytes().startswith(PNG_MAGIC)
        assert shown == []

    def test_accepts_a_path_object(self, shown, tmp_path):
        out = tmp_path / "wave.png"
        plot(sine(5), duration=1.0, sample_rate=SR, output_file=out)
        assert out.exists()

    def test_creates_missing_directories(self, shown, tmp_path):
        out = tmp_path / "a" / "b" / "wave.png"
        plot(sine(5), duration=1.0, sample_rate=SR, output_file=str(out))
        assert out.exists()

    def test_closes_the_figure_afterwards(self, shown, tmp_path):
        plot(sine(5), duration=1.0, sample_rate=SR, output_file=str(tmp_path / "w.png"))
        assert plt.get_fignums() == []

    def test_the_image_is_not_blank_and_is_reasonably_sized(self, shown, tmp_path):
        out = tmp_path / "wave.png"
        plot(sine(5), duration=1.0, sample_rate=SR, output_file=str(out))
        image = plt.imread(out)
        assert image.shape[1] > 400 and image.shape[0] > 150
        assert image[..., :3].std() > 0.05

    def test_a_constant_signal_is_drawn_as_a_flat_line(self, shown):
        # compile() returns a bare number for constants; plotting must cope
        plot(sp.Float(0.5), duration=0.1, sample_rate=SR)
        y = plt.gcf().axes[0].lines[0].get_ydata()
        np.testing.assert_allclose(y, 0.5)
        assert len(y) == 100

    def test_without_an_output_file_it_still_shows(self, shown):
        plot(sine(5), duration=1.0, sample_rate=SR)
        assert shown == [1]

    def test_the_same_figure_is_what_gets_saved(self, shown, tmp_path, monkeypatch):
        saved = []
        original = plt.Figure.savefig

        def spy(fig, *args, **kwargs):
            saved.append(fig)
            return original(fig, *args, **kwargs)

        monkeypatch.setattr(plt.Figure, "savefig", spy)
        plot(
            [sine(5), square(3)],
            1.0,
            sample_rate=SR,
            output_file=str(tmp_path / "w.png"),
        )
        assert len(saved) == 1
        assert len(saved[0].axes[0].lines) == 2


class TestLabelsAndTitle:
    def test_labels_become_the_legend(self, shown):
        plot([sine(5), square(3)], 1.0, sample_rate=SR, labels=["sine", "square"])
        legend = plt.gcf().axes[0].get_legend()
        assert [t.get_text() for t in legend.get_texts()] == ["sine", "square"]

    def test_no_legend_without_labels(self, shown):
        plot([sine(5), square(3)], 1.0, sample_rate=SR)
        assert plt.gcf().axes[0].get_legend() is None

    def test_a_single_expression_can_be_labelled(self, shown):
        plot(sine(5), 1.0, sample_rate=SR, labels=["only"])
        assert plt.gcf().axes[0].get_legend().get_texts()[0].get_text() == "only"

    def test_a_custom_title(self, shown):
        plot(sine(5), 1.0, sample_rate=SR, title="My signal")
        assert plt.gcf().axes[0].get_title() == "My signal"

    def test_the_default_title_is_unchanged(self, shown):
        plot(sine(5), 1.0, sample_rate=SR)
        assert plt.gcf().axes[0].get_title() == "Signal Plot"

    def test_wrong_number_of_labels_is_rejected(self, shown):
        with pytest.raises(ValueError, match="labels"):
            plot([sine(5), square(3)], 1.0, sample_rate=SR, labels=["one"])

    def test_an_empty_list_is_rejected(self, shown):
        with pytest.raises(ValueError, match="at least one"):
            plot([], 1.0, sample_rate=SR)


class TestStacked:
    SIGNALS = [sine(5), 0.1 * square(3), 3 * sawtooth(2)]

    def test_one_panel_per_signal(self, shown):
        plot(self.SIGNALS, 1.0, sample_rate=SR, stacked=True)
        axes = plt.gcf().axes
        assert len(axes) == 3
        assert all(len(ax.lines) == 1 for ax in axes)

    def test_each_panel_shows_its_own_signal(self, shown):
        plot(self.SIGNALS, 1.0, sample_rate=SR, stacked=True)
        for ax, expr in zip(plt.gcf().axes, self.SIGNALS, strict=True):
            np.testing.assert_allclose(
                ax.lines[0].get_ydata(), compile(expr, 1.0, SR), atol=1e-12
            )

    def test_panels_have_their_own_vertical_scale(self, shown):
        plot(self.SIGNALS, 1.0, sample_rate=SR, stacked=True)
        quiet, loud = plt.gcf().axes[1], plt.gcf().axes[2]
        assert quiet.get_ylim()[1] < 0.5
        assert loud.get_ylim()[1] > 2.5

    def test_panels_share_the_time_axis(self, shown):
        plot(self.SIGNALS, 1.0, sample_rate=SR, stacked=True)
        axes = plt.gcf().axes
        axes[0].set_xlim(0.2, 0.4)
        assert all(ax.get_xlim() == (0.2, 0.4) for ax in axes)

    def test_only_the_bottom_panel_has_a_time_label(self, shown):
        plot(self.SIGNALS, 1.0, sample_rate=SR, stacked=True)
        labels = [ax.get_xlabel() for ax in plt.gcf().axes]
        assert labels == ["", "", "Time [s]"]

    def test_labels_title_each_panel_on_the_left(self, shown):
        plot(self.SIGNALS, 1.0, sample_rate=SR, stacked=True, labels=["a", "b", "c"])
        axes = plt.gcf().axes
        assert [ax.get_title(loc="left") for ax in axes] == ["a", "b", "c"]

    def test_long_labels_do_not_squeeze_the_vertical_axis_label(self, shown):
        plot(
            self.SIGNALS,
            1.0,
            sample_rate=SR,
            stacked=True,
            labels=["a very long description of the first signal", "b", "c"],
        )
        assert {ax.get_ylabel() for ax in plt.gcf().axes} == {"Amplitude"}

    def test_without_labels_panels_have_no_left_title(self, shown):
        plot(self.SIGNALS, 1.0, sample_rate=SR, stacked=True)
        assert {ax.get_title(loc="left") for ax in plt.gcf().axes} == {""}
        assert {ax.get_ylabel() for ax in plt.gcf().axes} == {"Amplitude"}

    def test_the_figure_grows_with_the_number_of_panels(self, shown):
        plot(self.SIGNALS[:1], 1.0, sample_rate=SR, stacked=True)
        short = plt.gcf().get_figheight()
        plt.close("all")
        plot(self.SIGNALS, 1.0, sample_rate=SR, stacked=True)
        assert plt.gcf().get_figheight() > short

    def test_title_goes_on_the_top_panel_and_labels_do_not_replace_it(self, shown):
        plot(
            self.SIGNALS,
            1.0,
            sample_rate=SR,
            stacked=True,
            title="Three",
            labels=["a", "b", "c"],
        )
        axes = plt.gcf().axes
        assert axes[0].get_title() == "Three"
        assert axes[0].get_title(loc="left") == "a"
        assert axes[1].get_title() == ""

    def test_beats_mode_labels_the_bottom_axis_in_beats(self, shown):
        plot(self.SIGNALS, 4, bpm=BPM(120), sample_rate=SR, stacked=True)
        assert plt.gcf().axes[-1].get_xlabel() == "Beat [b]"

    def test_can_be_saved(self, shown, tmp_path):
        out = tmp_path / "stack.png"
        plot(self.SIGNALS, 1.0, sample_rate=SR, stacked=True, output_file=str(out))
        assert out.read_bytes().startswith(PNG_MAGIC)


class TestStart:
    def test_shows_a_window_that_begins_later(self, shown):
        plot(sine(5), duration=0.5, start=0.25, sample_rate=SR)
        line = plt.gcf().axes[0].lines[0]
        x = line.get_xdata()
        assert x[0] == pytest.approx(0.25)
        assert x[-1] == pytest.approx(0.75, abs=2 / SR)
        assert len(x) == 500

    def test_the_window_is_the_same_samples_as_the_full_signal(self, shown):
        plot(sine(5), duration=0.5, start=0.25, sample_rate=SR)
        y = plt.gcf().axes[0].lines[0].get_ydata()
        np.testing.assert_allclose(y, compile(sine(5), 0.75, SR)[250:], atol=1e-12)

    def test_a_filter_is_not_restarted_at_the_window(self, shown):
        # The filter runs from time zero, so its state at the window start is
        # whatever it had built up, not rest
        expr = low_pass_filter(square(5), 3)
        plot(expr, duration=0.5, start=0.5, sample_rate=SR)
        y = plt.gcf().axes[0].lines[0].get_ydata()
        full = compile(expr, 1.0, SR)
        np.testing.assert_allclose(y, full[500:], atol=1e-12)
        assert abs(y[0]) > 0.05

    def test_stacked_panels_use_the_window_too(self, shown):
        plot([sine(5), square(5)], 0.5, start=0.25, sample_rate=SR, stacked=True)
        for ax in plt.gcf().axes:
            assert ax.lines[0].get_xdata()[0] == pytest.approx(0.25)

    def test_start_is_in_beats_when_bpm_is_given(self, shown):
        bpm = BPM(120)  # half a second per beat
        plot(sine(5), duration=2, start=1, bpm=bpm, sample_rate=SR)
        line = plt.gcf().axes[0].lines[0]
        assert line.get_xdata()[0] == pytest.approx(1, abs=0.01)
        assert len(line.get_xdata()) == 1000
        np.testing.assert_allclose(
            line.get_ydata(), compile(sine(5), 1.5, SR)[500:], atol=1e-12
        )

    def test_a_negative_start_is_rejected(self, shown):
        with pytest.raises(ValueError, match="start"):
            plot(sine(5), 1.0, start=-1, sample_rate=SR)

    def test_the_default_start_is_zero(self, shown):
        plot(sine(5), 1.0, sample_rate=SR)
        np.testing.assert_allclose(
            plt.gcf().axes[0].lines[0].get_xdata(), linspace(1.0, SR)
        )


def image_peak_frequency(fig, column):
    """Frequency of the loudest bin in one time column of the spectrogram"""
    image = fig.axes[0].images[0]
    left, right, bottom, top = image.get_extent()
    data = image.get_array()
    row = int(np.argmax(data[:, column]))
    # Row 0 is the lowest frequency
    return bottom + (row + 0.5) * (top - bottom) / data.shape[0]


class TestSpectrogram:
    SR = 8000

    def test_a_pure_tone_peaks_at_its_frequency_throughout(self, shown):
        spectrogram(sine(1000), duration=1.0, sample_rate=self.SR)
        fig = plt.gcf()
        columns = fig.axes[0].images[0].get_array().shape[1]
        for column in (0, columns // 2, columns - 1):
            assert image_peak_frequency(fig, column) == pytest.approx(1000, abs=15)

    def test_two_tones_make_two_bands(self, shown):
        spectrogram(sine(500) + sine(2000), duration=1.0, sample_rate=self.SR)
        data = plt.gcf().axes[0].images[0].get_array()
        column = data[:, data.shape[1] // 2]
        top_two = np.sort(np.argsort(column)[-6:])
        left, right, bottom, top = plt.gcf().axes[0].images[0].get_extent()
        freqs = bottom + (top_two + 0.5) * (top - bottom) / data.shape[0]
        assert np.any(np.abs(freqs - 500) < 30)
        assert np.any(np.abs(freqs - 2000) < 30)

    def test_a_rising_sweep_climbs(self, shown):
        # slow enough that a 0.26 s analysis window sees a nearly steady tone
        spectrogram(sine(1000 + 400 * TIME), duration=1.0, sample_rate=self.SR)
        fig = plt.gcf()
        columns = fig.axes[0].images[0].get_array().shape[1]
        early = image_peak_frequency(fig, 2)
        middle = image_peak_frequency(fig, columns // 2)
        late = image_peak_frequency(fig, columns - 3)
        assert early < middle < late
        # the instantaneous frequency is 1000 + 400 t, so 1200 Hz at 0.5 s
        assert middle == pytest.approx(1200, abs=60)

    def test_the_axes_are_labelled(self, shown):
        spectrogram(sine(440), duration=0.5, sample_rate=self.SR)
        ax = plt.gcf().axes[0]
        assert ax.get_xlabel() == "Time [s]"
        assert ax.get_ylabel() == "Frequency [Hz]"
        assert ax.get_title() == "Spectrogram"

    def test_has_a_colour_bar(self, shown):
        spectrogram(sine(440), duration=0.5, sample_rate=self.SR)
        assert len(plt.gcf().axes) == 2

    def test_the_colour_scale_spans_ninety_decibels(self, shown):
        spectrogram(sine(440), duration=0.5, sample_rate=self.SR)
        low, high = plt.gcf().axes[0].images[0].get_clim()
        assert high - low == pytest.approx(90)

    def test_defaults_to_showing_up_to_the_nyquist_frequency(self, shown):
        spectrogram(sine(440), duration=0.5, sample_rate=self.SR)
        assert plt.gcf().axes[0].get_ylim() == pytest.approx((0, self.SR / 2))

    def test_max_frequency_limits_the_vertical_axis(self, shown):
        spectrogram(sine(440), duration=0.5, sample_rate=self.SR, max_frequency=1500)
        assert plt.gcf().axes[0].get_ylim() == pytest.approx((0, 1500))

    def test_a_custom_title(self, shown):
        spectrogram(sine(440), duration=0.5, sample_rate=self.SR, title="Hello")
        assert plt.gcf().axes[0].get_title() == "Hello"

    def test_the_time_axis_covers_the_duration(self, shown):
        spectrogram(sine(440), duration=1.0, sample_rate=self.SR)
        left, right = plt.gcf().axes[0].get_xlim()
        assert left == pytest.approx(0, abs=0.1)
        assert right == pytest.approx(1.0, abs=0.1)

    def test_start_shows_a_later_window(self, shown):
        # 1000 Hz then 3000 Hz, a second each; look at the second half only
        expr = sine(1000) * (1 - trigger(1.0)) + sine(3000) * trigger(1.0)
        spectrogram(expr, duration=1.0, start=1.0, sample_rate=self.SR)
        fig = plt.gcf()
        left, right = fig.axes[0].get_xlim()
        assert left == pytest.approx(1.0, abs=0.1)
        assert right == pytest.approx(2.0, abs=0.1)
        assert image_peak_frequency(fig, 3) == pytest.approx(3000, abs=20)

    def test_a_filter_is_not_restarted_at_the_window(self, shown):
        # A 20 Hz filter on a 200 Hz tone has a long start-up transient. A
        # filter restarted at the window would show it as extra low-frequency
        # energy in the first column; one that ran from time zero does not
        expr = low_pass_filter(sine(200), 20)
        spectrogram(expr, duration=0.5, start=0.5, sample_rate=self.SR)
        data = plt.gcf().axes[0].images[0].get_array()
        low_band = data[:10, :]  # the first rows are the lowest frequencies
        assert abs(low_band[:, 0].mean() - low_band[:, -1].mean()) < 6

    def test_negative_start_and_non_positive_duration_are_rejected(self, shown):
        with pytest.raises(ValueError, match="start"):
            spectrogram(sine(440), duration=1.0, start=-1, sample_rate=self.SR)
        with pytest.raises(ValueError, match="duration"):
            spectrogram(sine(440), duration=0.0, sample_rate=self.SR)

    def test_a_window_too_short_to_analyse_is_rejected(self, shown):
        with pytest.raises(ValueError, match="too short"):
            spectrogram(sine(440), duration=0.01, sample_rate=self.SR)

    def test_writes_a_png_instead_of_showing(self, shown, tmp_path):
        out = tmp_path / "sub" / "spec.png"
        spectrogram(sine(440), duration=0.5, sample_rate=self.SR, output_file=out)
        assert out.read_bytes().startswith(PNG_MAGIC)
        assert shown == []
        assert plt.get_fignums() == []

    def test_shows_when_no_file_is_given(self, shown):
        spectrogram(sine(440), duration=0.5, sample_rate=self.SR)
        assert shown == [1]

    def test_a_constant_signal_does_not_crash(self, shown):
        # silence has no peak to normalise against
        spectrogram(sine(440) * 0, duration=0.5, sample_rate=self.SR)


class TestPlotStereo:
    PAIR = StereoExpr(sine(5), 0.5 * square(3))

    def test_a_stereo_signal_is_two_lines_in_one_panel(self, shown):
        plot(self.PAIR, 1.0, sample_rate=SR)
        (ax,) = plt.gcf().axes
        assert len(ax.lines) == 2

    def test_the_lines_are_the_left_and_the_right_samples(self, shown):
        plot(self.PAIR, 1.0, sample_rate=SR)
        left, right = plt.gcf().axes[0].lines
        np.testing.assert_allclose(
            left.get_ydata(), compile(self.PAIR.left, 1.0, SR), atol=1e-12
        )
        np.testing.assert_allclose(
            right.get_ydata(), compile(self.PAIR.right, 1.0, SR), atol=1e-12
        )

    def test_both_lines_share_the_time_axis(self, shown):
        plot(self.PAIR, 1.0, sample_rate=SR)
        left, right = plt.gcf().axes[0].lines
        np.testing.assert_array_equal(left.get_xdata(), right.get_xdata())
        np.testing.assert_allclose(left.get_xdata(), linspace(1.0, SR))

    def test_an_unlabelled_stereo_signal_is_named_left_and_right(self, shown):
        plot(self.PAIR, 1.0, sample_rate=SR)
        texts = [t.get_text() for t in plt.gcf().axes[0].get_legend().get_texts()]
        assert texts == ["left", "right"]

    def test_a_label_names_both_sides(self, shown):
        plot(self.PAIR, 1.0, sample_rate=SR, labels=["pair"])
        texts = [t.get_text() for t in plt.gcf().axes[0].get_legend().get_texts()]
        assert texts == ["pair (left)", "pair (right)"]

    def test_a_list_can_mix_mono_and_stereo_with_one_label_per_signal(self, shown):
        plot([sine(5), self.PAIR], 1.0, sample_rate=SR, labels=["mono", "pair"])
        ax = plt.gcf().axes[0]
        assert len(ax.lines) == 3
        texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert texts == ["mono", "pair (left)", "pair (right)"]

    def test_a_label_for_each_side_is_too_many(self, shown):
        with pytest.raises(ValueError, match="one name per signal"):
            plot(self.PAIR, 1.0, sample_rate=SR, labels=["l", "r"])

    def test_a_mono_signal_next_to_a_stereo_one_stays_unlabelled_by_default(
        self, shown
    ):
        plot([sine(5), self.PAIR], 1.0, sample_rate=SR)
        texts = [t.get_text() for t in plt.gcf().axes[0].get_legend().get_texts()]
        assert texts == ["left", "right"]

    def test_mono_plots_have_no_legend_without_labels(self, shown):
        plot(sine(5), 1.0, sample_rate=SR)
        assert plt.gcf().axes[0].get_legend() is None

    def test_stacked_gives_each_side_its_own_panel(self, shown):
        plot(self.PAIR, 1.0, sample_rate=SR, stacked=True)
        axes = plt.gcf().axes
        assert len(axes) == 2
        np.testing.assert_allclose(
            axes[0].lines[0].get_ydata(), compile(self.PAIR.left, 1.0, SR), atol=1e-12
        )
        np.testing.assert_allclose(
            axes[1].lines[0].get_ydata(), compile(self.PAIR.right, 1.0, SR), atol=1e-12
        )

    def test_stacked_panels_are_titled_left_and_right(self, shown):
        plot(self.PAIR, 1.0, sample_rate=SR, stacked=True)
        axes = plt.gcf().axes
        assert axes[0].get_title(loc="left") == "left"
        assert axes[1].get_title(loc="left") == "right"

    def test_stacked_panels_use_the_label(self, shown):
        plot(self.PAIR, 1.0, sample_rate=SR, stacked=True, labels=["pair"])
        axes = plt.gcf().axes
        assert axes[0].get_title(loc="left") == "pair (left)"
        assert axes[1].get_title(loc="left") == "pair (right)"

    def test_stacked_mixed_list_has_a_panel_per_line(self, shown):
        plot([sine(5), self.PAIR], 1.0, sample_rate=SR, stacked=True)
        assert len(plt.gcf().axes) == 3

    def test_start_shows_the_same_window_of_both_sides(self, shown):
        plot(self.PAIR, 0.5, sample_rate=SR, start=0.25)
        left, right = plt.gcf().axes[0].lines
        np.testing.assert_allclose(
            left.get_ydata(), compile(self.PAIR.left, 0.75, SR)[250:], atol=1e-12
        )
        np.testing.assert_allclose(
            right.get_ydata(), compile(self.PAIR.right, 0.75, SR)[250:], atol=1e-12
        )

    def test_it_works_in_beats(self, shown):
        plot(self.PAIR, 2.0, BPM(120), sample_rate=SR)
        ax = plt.gcf().axes[0]
        assert ax.get_xlabel() == "Beat [b]"
        assert len(ax.lines) == 2
        assert ax.lines[0].get_xdata()[-1] == pytest.approx(2.0, abs=0.01)

    def test_a_panned_signal_is_louder_on_the_side_it_is_panned_to(self, shown):
        plot(pan(sine(5), -0.5), 1.0, sample_rate=SR)
        left, right = plt.gcf().axes[0].lines
        assert np.abs(left.get_ydata()).max() > 2 * np.abs(right.get_ydata()).max()

    def test_a_constant_side_is_drawn_flat(self, shown):
        plot(StereoExpr(sine(5), 0.25), 0.1, sample_rate=SR)
        _, right = plt.gcf().axes[0].lines
        np.testing.assert_allclose(right.get_ydata(), 0.25)
        assert len(right.get_ydata()) == 100

    def test_it_can_be_saved(self, shown, tmp_path):
        out = tmp_path / "pair.png"
        plot(self.PAIR, 1.0, sample_rate=SR, output_file=out)
        assert out.read_bytes().startswith(PNG_MAGIC)
        assert shown == []


class TestSpectrogramStereo:
    SR = 8000
    PAIR = StereoExpr(sine(500), sine(2000))

    def peak(self, **options):
        spectrogram(self.PAIR, duration=1.0, sample_rate=self.SR, **options)
        fig = plt.gcf()
        columns = fig.axes[0].images[0].get_array().shape[1]
        return image_peak_frequency(fig, columns // 2)

    def test_the_left_channel_shows_the_left_tone(self, shown):
        assert self.peak(channel="left") == pytest.approx(500, abs=15)

    def test_the_right_channel_shows_the_right_tone(self, shown):
        assert self.peak(channel="right") == pytest.approx(2000, abs=30)

    def test_the_mix_shows_both_tones(self, shown):
        spectrogram(self.PAIR, duration=1.0, sample_rate=self.SR)
        data = plt.gcf().axes[0].images[0].get_array()
        column = data[:, data.shape[1] // 2]
        top = np.sort(np.argsort(column)[-6:])
        _, _, bottom, high = plt.gcf().axes[0].images[0].get_extent()
        freqs = bottom + (top + 0.5) * (high - bottom) / data.shape[0]
        assert np.any(np.abs(freqs - 500) < 30)
        assert np.any(np.abs(freqs - 2000) < 30)

    def test_mix_is_the_default(self, shown):
        spectrogram(self.PAIR, duration=1.0, sample_rate=self.SR)
        default = plt.gcf().axes[0].images[0].get_array().copy()
        plt.close("all")
        spectrogram(self.PAIR, duration=1.0, sample_rate=self.SR, channel="mix")
        np.testing.assert_array_equal(default, plt.gcf().axes[0].images[0].get_array())

    def test_a_channel_matches_a_mono_signal_of_that_side(self, shown):
        spectrogram(self.PAIR, 1.0, self.SR, channel="right")
        stereo_image = plt.gcf().axes[0].images[0].get_array().copy()
        plt.close("all")
        spectrogram(self.PAIR.right, 1.0, self.SR)
        np.testing.assert_array_equal(
            stereo_image, plt.gcf().axes[0].images[0].get_array()
        )

    def test_the_mix_of_identical_sides_is_that_side(self, shown):
        spectrogram(StereoExpr(sine(500), sine(500)), 1.0, self.SR)
        stereo_image = plt.gcf().axes[0].images[0].get_array().copy()
        plt.close("all")
        spectrogram(sine(500), 1.0, self.SR)
        np.testing.assert_allclose(
            stereo_image, plt.gcf().axes[0].images[0].get_array(), atol=1e-9
        )

    def test_start_applies_to_the_chosen_channel(self, shown):
        spectrogram(self.PAIR, 0.5, self.SR, start=0.25, channel="left")
        left, _, _, _ = plt.gcf().axes[0].images[0].get_extent()
        assert left == pytest.approx(0.25)

    def test_an_unknown_channel_is_rejected(self, shown):
        with pytest.raises(ValueError, match="'left', 'right' or 'mix'"):
            spectrogram(self.PAIR, 1.0, self.SR, channel="middle")

    def test_a_channel_does_not_apply_to_a_mono_signal(self, shown):
        with pytest.raises(ValueError, match="only applies to a stereo"):
            spectrogram(sine(500), 1.0, self.SR, channel="left")

    def test_mono_signals_still_work_with_the_default_channel(self, shown):
        spectrogram(sine(500), 1.0, self.SR, channel="mix")
        assert plt.get_fignums() == [1]

    def test_a_short_stereo_window_is_rejected(self, shown):
        with pytest.raises(ValueError, match="too short"):
            spectrogram(self.PAIR, duration=0.01, sample_rate=self.SR)

    def test_silent_stereo_does_not_crash(self, shown):
        spectrogram(StereoExpr(0, 0), 0.5, self.SR)

    def test_writes_a_png(self, shown, tmp_path):
        out = tmp_path / "pair.png"
        spectrogram(self.PAIR, 0.5, self.SR, channel="left", output_file=out)
        assert out.read_bytes().startswith(PNG_MAGIC)

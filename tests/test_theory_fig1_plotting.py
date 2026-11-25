"""Display sampling must not change estimates or depend on plot/input order."""

import unittest
from unittest.mock import Mock

from plot_scripts.theory_fig1.plot_coefficients import PlotOptions, draw_points, subsample_points


class PlotSamplingTests(unittest.TestCase):
    def points(self):
        return [dict(rule="nwta", condition="shared", seed=i%3, t=10, nd=100,
                     fork_steps=10, kind="drift", age=0, r=i, s=i, age_averaged=False,
                     observed=float(i), predicted=float(i), empirical_se=0.1, theory_se=0.2)
                for i in range(100)]

    def test_selection_is_repeatable_order_independent_and_value_independent(self):
        points = self.points()
        options = PlotOptions(max_points=12, sample_seed=3)
        chosen = subsample_points(points, options)
        self.assertEqual(len(chosen), 12)
        self.assertEqual(chosen, subsample_points(points[::-1], options))
        changed_values = [dict(p, observed=9999, predicted=-1) for p in points]
        self.assertEqual([p["r"] for p in chosen], [p["r"] for p in subsample_points(changed_values, options)])
        self.assertNotEqual(chosen, subsample_points(points, PlotOptions(max_points=12, sample_seed=4)))
        self.assertEqual(len(points), 100)
        self.assertEqual(subsample_points(points, PlotOptions()), points)
        diagonal = [dict(p, kind="diagonal") for p in points]
        self.assertEqual([p["r"] for p in chosen], [p["r"] for p in subsample_points(diagonal, options)])

    def test_error_bar_and_size_controls_preserve_point_estimates(self):
        ax = Mock()
        options = PlotOptions(max_points=12, point_size=1.5, error_bars=False)
        draw_points(ax, self.points(), {10: "s"}, options=options)
        kwargs = ax.errorbar.call_args.kwargs
        self.assertIsNone(kwargs["xerr"])
        self.assertIsNone(kwargs["yerr"])
        self.assertEqual(kwargs["markersize"], 1.5)
        self.assertEqual(len(ax.errorbar.call_args.args[0]), 12)


if __name__ == "__main__":
    unittest.main()

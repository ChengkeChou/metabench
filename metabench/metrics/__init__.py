"""Metrics: coverage, width, bias, failure and boundary rates."""

from .core import (bias, coverage, dispersion, mcse_halfwidth, mcse_proportion,
                   mean_width, rates, summarise)

__all__ = ["bias", "coverage", "dispersion", "mcse_halfwidth",
           "mcse_proportion", "mean_width", "rates", "summarise"]

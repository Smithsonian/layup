"""Tests for the (A, D) tangent-plane basis vectors attached to each
astrometric observation.

These vectors must point along the canonical equatorial-tangent
directions:

    A = ∂ρ̂/∂α  (unit-normalized, direction of increasing RA)
      = (-sin α,  cos α,  0)
    D = ∂ρ̂/∂δ  (direction of increasing Dec, already unit length)
      = (-sin δ cos α,  -sin δ sin α,  cos δ)

A previous implementation used a different (and unnormalized) basis
that was rotated about ρ̂ by an object-position-dependent angle. That
silently biased fits whenever per-observation RA/Dec uncertainties
were not equal. This test pins the basis to the canonical convention.
"""

import numpy as np
import pytest

from layup.routines import Observation


def _canonical_basis(ra_rad, dec_rad):
    rho = np.array(
        [
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        ]
    )
    a = np.array([-np.sin(ra_rad), np.cos(ra_rad), 0.0])
    d = np.array(
        [
            -np.sin(dec_rad) * np.cos(ra_rad),
            -np.sin(dec_rad) * np.sin(ra_rad),
            np.cos(dec_rad),
        ]
    )
    return rho, a, d


@pytest.mark.parametrize(
    "ra_deg,dec_deg",
    [
        (0.0, 0.0),
        (45.0, 30.0),
        (135.0, -20.0),
        (10.0, 60.0),
        (270.0, -45.0),
        (180.0, 5.0),
    ],
)
def test_a_d_vectors_are_canonical(ra_deg, dec_deg):
    """Observation.from_astrometry must populate a_vec/d_vec with the
    canonical equatorial-tangent vectors at ρ̂."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    obs = Observation.from_astrometry(ra, dec, 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    rho_canon, a_canon, d_canon = _canonical_basis(ra, dec)

    np.testing.assert_allclose(np.asarray(obs.rho_hat), rho_canon, atol=1e-12)
    np.testing.assert_allclose(np.asarray(obs.a_vec), a_canon, atol=1e-12)
    np.testing.assert_allclose(np.asarray(obs.d_vec), d_canon, atol=1e-12)


@pytest.mark.parametrize(
    "ra_deg,dec_deg",
    [
        (45.0, 30.0),
        (135.0, -20.0),
        (270.0, -45.0),
    ],
)
def test_a_d_vectors_form_orthonormal_tangent_frame(ra_deg, dec_deg):
    """{a_vec, d_vec} must be unit vectors, mutually orthogonal, and
    perpendicular to ρ̂."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    obs = Observation.from_astrometry(ra, dec, 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    rho = np.asarray(obs.rho_hat)
    a = np.asarray(obs.a_vec)
    d = np.asarray(obs.d_vec)

    assert abs(np.linalg.norm(a) - 1.0) < 1e-12
    assert abs(np.linalg.norm(d) - 1.0) < 1e-12
    assert abs(np.dot(a, d)) < 1e-12
    assert abs(np.dot(rho, a)) < 1e-12
    assert abs(np.dot(rho, d)) < 1e-12

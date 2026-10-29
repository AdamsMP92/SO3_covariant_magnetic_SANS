#!/usr/bin/env python3
"""Check the affine vortex pair by full SO(3) quadrature, without rendering.

Use periodic azimuth/roll grids and Gauss--Legendre polar nodes for normalized
Haar measure. Directly average rotated Fourier amplitudes, then compare with
the closed spectra and magnetic channel contractions. Requires only NumPy.
"""
import json
import numpy as np


def cross_matrix(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotations():
    z, wz = np.polynomial.legendre.leggauss(12)
    a, z, g = np.meshgrid(2*np.pi*np.arange(12)/12, z,
                          2*np.pi*np.arange(24)/24, indexing="ij")
    weights = np.broadcast_to(wz[None, :, None]/(2*12*24), a.shape).ravel()
    ca, sa = np.cos(a).ravel(), np.sin(a).ravel()
    cg, sg = np.cos(g).ravel(), np.sin(g).ravel()
    z = z.ravel()
    s = np.sqrt(1-z*z)
    rotation = np.stack((ca*z*cg-sa*sg, -ca*z*sg-sa*cg, ca*s,
                         sa*z*cg+ca*sg, -sa*z*sg+ca*cg, sa*s,
                         -s*cg, s*sg, z), axis=-1).reshape(-1, 3, 3)
    assert np.max(np.abs(rotation @ rotation.transpose(0, 2, 1)-np.eye(3))) < 1e-14
    assert abs(weights.sum()-1) < 1e-14
    return rotation, weights


def main():
    rotation, weights = rotations()
    kappa = 7.0  # m=R=1; spectra in Gamma_0 units, channels in K*Gamma_0.
    errors = dict(tensor=0.0, longitudinal_difference=0.0,
                  projected_tensor=0.0, channels=0.0, first_moment=0.0)
    realization_difference = 0.0
    for angle in (0.0, np.pi/9, np.pi/6):
        c, s = np.cos(angle), np.sin(angle)
        shear = np.sqrt(5/3)*kappa*s
        models = (
            (np.array([s, 0.0, c]), kappa*cross_matrix([0, 0, 1])),
            (np.array([0.0, 0.0, 1.0]),
             np.array([[0, -kappa*c+shear, 0], [kappa*c+shear, 0, 0], [0, 0, 0]])),
        )
        assert s/kappa < 1.0 and kappa*c > shear
        for q in ((0, 0, 0.3), (0, 2, 1), (1, -2, 2.5), (0, 0, 4.8), (0, 3, 0)):
            q = np.asarray(q, dtype=float)
            x = np.linalg.norm(q)
            n = q/x
            f = 3*(np.sin(x)-x*np.cos(x))/x**3
            fp = 3*((x*x-3)*np.sin(x)+3*x*np.cos(x))/x**4
            pl = np.outer(n, n)
            pt = np.eye(3)-pl
            a = (f*f+kappa*kappa*fp*fp)/3
            b = f*f/3
            chiral = 2*kappa*c*f*fp/3
            delta = 4*kappa*kappa*s*s*fp*fp/9
            tensors, projected, channels, single = [], [], [], []
            for j, (m, matrix) in enumerate(models):
                mr = np.einsum('nij,j->ni', rotation, m)
                lr = np.einsum('nai,ij,nbj->nab', rotation, matrix, rotation)
                amplitude = f*mr+1j*fp*np.einsum('nij,j->ni', lr, n)
                tensor = np.einsum('n,ni,nj->ij', weights, amplitude, amplitude.conj())
                expected = a*pt+(b+j*delta)*pl-1j*chiral*cross_matrix(n)
                errors['tensor'] = max(errors['tensor'], float(np.max(np.abs(tensor-expected))))
                errors['first_moment'] = max(errors['first_moment'],
                    float(np.max(np.abs(np.einsum('n,ni->i', weights, amplitude)))))
                g = pt @ tensor @ pt
                sf = float(np.real(g[0, 0]+g[1, 1]))
                chi = float(np.real(-1j*(g[0, 1]-g[1, 0])))
                nsf = float(np.real(g[2, 2]))
                channels.append(np.array([sf+chi, sf-chi, nsf, nsf,
                                          sf+chi+nsf, sf-chi+nsf, sf+nsf]))
                tensors.append(tensor)
                projected.append(g)
                single.append(pt @ (f*m+1j*fp*matrix@n))
            errors['longitudinal_difference'] = max(errors['longitudinal_difference'],
                float(np.max(np.abs(tensors[1]-tensors[0]-delta*pl))))
            errors['projected_tensor'] = max(errors['projected_tensor'],
                float(np.max(np.abs(projected[1]-projected[0]))))
            errors['channels'] = max(errors['channels'],
                float(np.max(np.abs(channels[1]-channels[0]))))
            realization_difference = max(realization_difference,
                float(np.linalg.norm(single[1]-single[0])))
    assert max(errors.values()) < 1e-11, errors
    assert realization_difference > 0.1
    print(json.dumps({'SO3_nodes': len(weights), 'angles_degrees': [0, 20, 30],
                      'max_absolute_errors': errors,
                      'max_single_orientation_amplitude_difference': realization_difference}, indent=2))


if __name__ == '__main__':
    main()

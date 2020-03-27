# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantum channels that are commonly used in the literature."""

from typing import Iterable, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np

from cirq import protocols, value
from cirq.ops import (raw_types, common_gates, pauli_gates, gate_features,
                      identity)

if TYPE_CHECKING:
    import cirq


@value.value_equality
class OverX(gate_features.SingleQubitGate):
    """Overrotates along the X axis, with a coherence defined by kappa and an angle of eps.

    -Dripto
    """

    def __init__(self, eps: float, kappa: float) -> None:

        self._kappa = kappa
        self._eps = eps

    def _channel_(self) -> Iterable[np.ndarray]:
        k0 = np.sqrt(self._kappa)
        k1 = np.sqrt(1. - self._kappa)
        sine = np.sin(self._eps/2)
        cose = np.cos(self._eps/2)

        return (
            k0 * np.array([[cose, -1j*sine], [-1j*sine, cose]]),
            k1 * cose * np.array([[1., 0.], [0., 1.]]),
            k1 * sine * np.array([[0., 1.], [1., 0.]]),
        )

    def _has_channel_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._eps, self._kappa

    def __repr__(self) -> str:
        return 'cirq.OverX(eps={!r},kappa={!r})'.format(
            self._eps, self._kappa
        )

    def __str__(self) -> str:
        return 'cirq.OverX(eps={!r},kappa={!r})'.format(
            self._eps, self._kappa
        )

    def _circuit_diagram_info_(self,
                               args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return 'OX({},{})'.format(f, f).format(self._eps, self._kappa)
        return 'OX({!r},{!r})'.format(self._eps, self._kappa)

    @property
    def eps(self) -> float:
        """The strength of the overrotation."""
        return self._eps

    @property
    def kappa(self) -> float:
        """The proportion of the error that is coherent."""
        return self._kappa

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['eps', 'kappa'])


@value.value_equality
class OverY(gate_features.SingleQubitGate):
    """Overrotates along the Y axis, with a coherence defined by kappa and a strength defined by eps.

    -Dripto
    """

    def __init__(self, eps: float, kappa: float) -> None:

        self._kappa = kappa
        self._eps = eps

    def _channel_(self) -> Iterable[np.ndarray]:
        k0 = np.sqrt(self._kappa)
        k1 = np.sqrt(1. - self._kappa)
        sine = np.sin(self._eps/2)
        cose = np.cos(self._eps/2)


        return (
            k0 * np.array([[cose, -1*sine], [sine, cose]]),
            k1 * cose * np.array([[1., 0.], [0., 1.]]),
            k1 * sine * np.array([[0., -1j], [1j, 0.]]),
        )

    def _has_channel_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._eps, self._kappa

    def __repr__(self) -> str:
        return 'cirq.OverY(eps={!r},kappa={!r})'.format(
            self._eps, self._kappa
        )

    def __str__(self) -> str:
        return 'cirq.OverY(eps={!r},kappa={!r})'.format(
            self._eps, self._kappa
        )

    def _circuit_diagram_info_(self,
                               args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return 'OY({},{})'.format(f, f).format(self._eps, self._kappa)
        return 'OY({!r},{!r})'.format(self._eps, self._kappa)

    @property
    def eps(self) -> float:
        """The strength of the overrotation."""
        return self._eps

    @property
    def kappa(self) -> float:
        """The proportion of the error that is coherent."""
        return self._kappa

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['eps', 'kappa'])

@value.value_equality
class OverZ(gate_features.SingleQubitGate):
    """Overrotates along the Z axis, with a coherence defined by kappa and a strength defined by eps.

    -Dripto
    """

    def __init__(self, eps: float, kappa: float) -> None:

        self._kappa = kappa
        self._eps = eps

    def _channel_(self) -> Iterable[np.ndarray]:
        k0 = np.sqrt(self._kappa)
        k1 = np.sqrt(1. - self._kappa)
        sine = np.sin(self._eps/2)
        cose = np.cos(self._eps/2)
        # sine = np.sqrt(self._eps)
        # cose = np.sqrt(1-self._eps)


        return (
            k0 * np.array([[cose - 1j*sine, 0.], [0., cose + 1j*sine]]),
            k1 * cose * np.array([[1., 0.], [0., 1.]]),
            k1 * sine * np.array([[0., -1j], [1j, 0.]]),
        )

    def _has_channel_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._eps, self._kappa

    def __repr__(self) -> str:
        return 'cirq.OverZ(eps={!r},kappa={!r})'.format(
            self._eps, self._kappa
        )

    def __str__(self) -> str:
        return 'cirq.OverZ(eps={!r},kappa={!r})'.format(
            self._eps, self._kappa
        )

    def _circuit_diagram_info_(self,
                               args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return 'OZ({},{})'.format(f, f).format(self._eps, self._kappa)
        return 'OZ({!r},{!r})'.format(self._eps, self._kappa)

    @property
    def eps(self) -> float:
        """The strength of the overrotation."""
        return self._eps

    @property
    def kappa(self) -> float:
        """The proportion of the error that is coherent."""
        return self._kappa

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['eps', 'kappa'])


@value.value_equality
class OverCNOT(gate_features.TwoQubitGate):
    """Overrotates along the CNOT axis, with a coherence defined by kappa and a strength defined by eps.

    -Dripto
    """

    def __init__(self, eps: float, kappa: float) -> None:

        self._kappa = kappa
        self._eps = eps

    def _channel_(self) -> Iterable[np.ndarray]:
        k0 = np.sqrt(self._kappa)
        k1 = np.sqrt(1. - self._kappa)
        sine = np.sin(self._eps)
        cose = np.cos(self._eps)


        return (
            k0 * np.array([[cose - 1j*sine, 0., 0., 0.], [0., cose - 1j*sine, 0., 0.], [0., 0., cose, -1j*sine], [0., 0., -1j*sine, cose]]),
            k1 * cose * np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]),
            k1 * sine * np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.], [0., 0., 1., 0.]]),
        )

    def _has_channel_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._eps, self._kappa

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'OCNOT'
        return 'OCNOT(eps={!r}, kappa={!r})'.format(self._eps, self._kappa)

    def __repr__(self):
        return (
            'cirq.OCNOT(eps={!r}, '
            'kappa={!r})'
        ).format(self._eps, self._kappa)

    def on(self, *args: 'cirq.Qid',
           **kwargs: 'cirq.Qid') -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))

    def _circuit_diagram_info_(self,
                               args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return 'OCNOT({},{})'.format(f, f).format(self._eps, self._kappa)
        return 'OCNOT({!r},{!r})'.format(self._eps, self._kappa)

    @property
    def eps(self) -> float:
        """The strength of the overrotation."""
        return self._eps

    @property
    def kappa(self) -> float:
        """The proportion of the error that is coherent."""
        return self._kappa

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['eps', 'kappa'])


@value.value_equality
class OverCZ(gate_features.TwoQubitGate):
    """Overrotates along the CZ axis, with a coherence defined by kappa and a strength defined by eps.

    -Dripto
    """

    def __init__(self, eps: float, kappa: float) -> None:

        self._kappa = kappa
        self._eps = eps

    def _channel_(self) -> Iterable[np.ndarray]:
        k0 = np.sqrt(self._kappa)
        k1 = np.sqrt(1. - self._kappa)
        sine = np.sin(self._eps)
        cose = np.cos(self._eps)


        return (
            k0 * np.array([[cose - 1j*sine, 0., 0., 0.], [0., cose - 1j*sine, 0., 0.], [0., 0., cose - 1j*sine, 0.], [0., 0., 0., cose + 1j*sine]]),
            k1 * cose * np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]),
            k1 * sine * np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., -1.]]),
        )

    def _has_channel_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._eps, self._kappa

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'OCZ'
        return 'OCZ(eps={!r}, kappa={!r})'.format(self._eps, self._kappa)

    def __repr__(self):
        return (
            'cirq.OCZ(eps={!r}, '
            'kappa={!r})'
        ).format(self._eps, self._kappa)

    def on(self, *args: 'cirq.Qid',
           **kwargs: 'cirq.Qid') -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))

    def _circuit_diagram_info_(self,
                               args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return 'OCZ({},{})'.format(f, f).format(self._eps, self._kappa)
        return 'OCZ({!r},{!r})'.format(self._eps, self._kappa)

    @property
    def eps(self) -> float:
        """The strength of the overrotation."""
        return self._eps

    @property
    def kappa(self) -> float:
        """The proportion of the error that is coherent."""
        return self._kappa

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['eps', 'kappa'])

@value.value_equality
class OverXX(gate_features.TwoQubitGate):
    """Overrotates along the XX axis, with a coherence defined by kappa and a strength defined by eps.

    -Dripto
    """

    def __init__(self, eps: float, kappa: float) -> None:

        self._kappa = kappa
        self._eps = eps

    def _channel_(self) -> Iterable[np.ndarray]:
        k0 = np.sqrt(self._kappa)
        k1 = np.sqrt(1. - self._kappa)
        sine = np.sin(self._eps)
        cose = np.cos(self._eps)


        return (
            k0 * np.array([[cose , 0., 0., -1j*sine], [0., cose, -1j*sine, 0.], [0., -1j*sine, cose, 0.], [-1j*sine, 0., 0., cose]]),
            k1 * cose * np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]),
            k1 * sine * np.array([[0., 0., 0., 1.], [0., 0., 1., 0.], [0., 1., 0., 0.], [1., 0., 0., 0.]]),
        )

    def _has_channel_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._eps, self._kappa

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'OXX'
        return 'OXX(eps={!r}, kappa={!r})'.format(self._eps, self._kappa)

    def __repr__(self):
        return (
            'cirq.OXX(eps={!r}, '
            'kappa={!r})'
        ).format(self._eps, self._kappa)

    def on(self, *args: 'cirq.Qid',
           **kwargs: 'cirq.Qid') -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))

    def _circuit_diagram_info_(self,
                               args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return 'OXX({},{})'.format(f, f).format(self._eps, self._kappa)
        return 'OXX({!r},{!r})'.format(self._eps, self._kappa)

    @property
    def eps(self) -> float:
        """The strength of the overrotation."""
        return self._eps

    @property
    def kappa(self) -> float:
        """The proportion of the error that is coherent."""
        return self._kappa

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['eps', 'kappa'])


@value.value_equality
class ZX(gate_features.TwoQubitGate):
    """Overrotates along the XX axis, with a coherence defined by kappa and a strength defined by eps.

    -Dripto
    """

    def __init__(self, p: float) -> None:

        self._p = p

    def _channel_(self) -> Iterable[np.ndarray]:
        p0 = np.sqrt(self._p)
        p1 = np.sqrt(1. - self._p)


        return (
            p0 * np.array([[0. , 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., -1.], [0., 0., -1., 0.]]),
            p1 * np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]),
            )

    def _has_channel_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._eps, self._kappa

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'OXX'
        return 'OXX(eps={!r}, kappa={!r})'.format(self._eps, self._kappa)

    def __repr__(self):
        return (
            'cirq.OXX(eps={!r}, '
            'kappa={!r})'
        ).format(self._eps, self._kappa)

    def on(self, *args: 'cirq.Qid',
           **kwargs: 'cirq.Qid') -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))

    def _circuit_diagram_info_(self,
                               args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return 'OXX({},{})'.format(f, f).format(self._eps, self._kappa)
        return 'OXX({!r},{!r})'.format(self._eps, self._kappa)

    @property
    def eps(self) -> float:
        """The strength of the overrotation."""
        return self._eps

    @property
    def kappa(self) -> float:
        """The proportion of the error that is coherent."""
        return self._kappa

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['eps', 'kappa'])

# -*- coding: utf-8 -*-

# © Copyright Charles Vanwynsberghe (2021)

# Pyworld3 is a computer program whose purpose is to run configurable
# simulations of the World3 model as described in the book "Dynamics
# of Growth in a Finite World".

# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".

# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.

# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import os
import json

import numpy as np

from .specials import TableFunction, clip
from .utils import (build_execution_order, compile_steps,
                     run_init_loops, step)


class Resource:
    """
    Nonrenewable Resource sector. Can be run independantly from other sectors
    with exogenous inputs. The initial code is defined p.405.

    Examples
    --------
    Running the nonerenewable resource sector alone requires artificial
    (exogenous) inputs which should be provided by the other sectors. Start
    from the following example:

    >>> rsc = Resource()
    >>> rsc.set_resource_table_functions()
    >>> rsc.init_resource_variables()
    >>> rsc.init_resource_constants()
    >>> rsc.set_resource_delay_functions()
    >>> rsc.init_exogenous_inputs()
    >>> rsc.run_resource()

    Parameters
    ----------
    year_min : float, optional
        start year of the simulation [year]. The default is 1900.
    year_max : float, optional
        end year of the simulation [year]. The default is 2100.
    dt : float, optional
        time step of the simulation [year]. The default is 1.
    pyear : float, optional
        implementation date of new policies [year]. The default is 1975.
    verbose : bool, optional
        print information for debugging. The default is False.

    Attributes
    ----------
    nri : float, optional
        nonrenewable resources initial [resource units]. The default is 1e12.
    nruf1 : float, optional
        nruf value before time=pyear []. The default is 1.
    nruf2 : float, optional
        nruf value after time=pyear []. The default is 1.
    nr : numpy.ndarray
        nonrenewable resources [resource units]. It is a state variable.
    nrfr : numpy.ndarray
        nonrenewable resource fraction remaining [].
    nruf : numpy.ndarray
        nonrenewable resource usage factor [].
    nrur : numpy.ndarray
        nonrenewable resource usage rate [resource units/year].
    pcrum : numpy.ndarray
        per capita resource usage multiplier [resource units/person-year].
    fcaor : numpy.ndarray
        fraction of capital allocated to obtaining resources [].
    fcaor1 : numpy.ndarray
        fcaor value before time=pyear [].
    fcaor2 : numpy.ndarray
        fcaor value after time=pyear [].

    """

    def __init__(self, year_min=1900, year_max=2100, dt=1, pyear=1975,
                 verbose=False):
        self.pyear = pyear
        self.dt = dt
        self.year_min = year_min
        self.year_max = year_max
        self.verbose = verbose
        self.length = self.year_max - self.year_min
        self.n = int(self.length / self.dt)
        self.time = np.arange(self.year_min, self.year_max, self.dt)

    def init_resource_constants(self, nri=1e12, nruf1=1, nruf2=1):
        """
        Initialize the constant parameters of the resource sector. Constants
        and their unit are documented above at the class level.

        """
        self.nri = nri
        self.nruf1 = nruf1
        self.nruf2 = nruf2

    def init_resource_variables(self):
        """
        Initialize the state and rate variables of the resource sector
        (memory allocation). Variables and their unit are documented above at
        the class level.

        """
        self.nr = np.full((self.n,), np.nan)
        self.nrfr = np.full((self.n,), np.nan)
        self.nruf = np.full((self.n,), np.nan)
        self.nrur = np.full((self.n,), np.nan)
        self.pcrum = np.full((self.n,), np.nan)
        self.fcaor = np.full((self.n,), np.nan)
        self.fcaor1 = np.full((self.n,), np.nan)
        self.fcaor2 = np.full((self.n,), np.nan)

    def set_resource_delay_functions(self, method="euler"):
        """
        Set the linear smoothing and delay functions of the 1st or the 3rd
        order, for the resource sector. One should call
        `self.set_resource_delay_functions` after calling
        `self.init_resource_constants`.

        Parameters
        ----------
        method : str, optional
            Numerical integration method: "euler" or "odeint". The default is
            "euler".

        """
        pass

    def set_resource_table_functions(self, json_file=None):
        """
        Set the nonlinear functions of the resource sector, based on a json
        file. By default, the `functions_table_world3.json` file from pyworld3
        is used.

        Parameters
        ----------
        json_file : file, optional
            json file containing all tables. The default is None.

        """
        if json_file is None:
            json_file = "./functions_table_world3.json"
            json_file = os.path.join(os.path.dirname(__file__), json_file)
        with open(json_file) as fjson:
            tables = json.load(fjson)

        func_names = ["PCRUM", "FCAOR1", "FCAOR2"]

        for func_name in func_names:
            for table in tables:
                if table["y.name"] == func_name:
                    func = TableFunction(table["x.values"], table["y.values"])
                    setattr(self, func_name.lower()+"_f", func)

    def init_exogenous_inputs(self):
        """
        Initialize all the necessary constants and variables to run the
        resource sector alone. These exogenous parameters are outputs from
        the 4 other remaining sectors in a full simulation of World3.

        """
        # constants
        self.popi = 1.65e9
        self.gc = 0.012
        self.pop2 = 4e9
        self.zpgt = 2500
        self.ici = 2.1e11
        self.fioaa = 0.12
        self.fioas = 0.12
        self.fioac = 0.43
        self.alic = 14
        self.icor = 3
        # variables
        self.pop = np.full((self.n,), np.nan)
        self.pop1 = np.full((self.n,), np.nan)
        self.ic = np.full((self.n,), np.nan)
        self.icir = np.full((self.n,), np.nan)
        self.icdr = np.full((self.n,), np.nan)
        self.io = np.full((self.n,), np.nan)
        self.iopc = np.full((self.n,), np.nan)

    @step(reads={"fcaor": 'k', "ic": ['j', 'k'], "icdr": 'jk', "icir": 'jk', "io": 'k', "pop": 'k', "pop1": 'k', "time": 'k'}, writes={"ic": 'k', "icdr": 'kl', "icir": 'kl', "io": 'k', "iopc": 'k', "pop": 'k', "pop1": 'k'})
    def loopk_exogenous(self, k):
        """
        Run a sorted sequence to update one loop of the exogenous parameters.
        The updates follow the @step dependency graph order.

        """
        j = k - 1
        kl = k
        jk = j

        self.ic[k] = self.ic[j] + self.dt * (self.icir[jk] - self.icdr[jk])

        self.pop1[k] = self.popi * np.exp(self.gc * (self.time[k] - 1900))
        self.pop[k] = clip(self.pop2, self.pop1[k], self.time[k], self.zpgt)
        self.io[k] = self.ic[k] * (1 - self.fcaor[k]) / self.icor
        self.iopc[k] = self.io[k] / self.pop[k]
        self.icir[kl] = self.io[k] * (1 - self.fioaa - self.fioas - self.fioac)
        self.icdr[kl] = self.ic[k] / self.alic

    def loop0_exogenous(self):
        """
        Run a sequence to initialize the exogenous parameters (loop with k=0).

        """
        self.ic[0] = self.ici
        self.pop1[0] = self.popi * np.exp(self.gc * (self.time[0] - 1900))
        self.pop[0] = clip(self.pop2, self.pop1[0], self.time[0], self.zpgt)
        self.io[0] = self.ic[0] * (1 - self.fcaor[0]) / self.icor
        self.iopc[0] = self.io[0] / self.pop[0]
        self.icir[0] = self.io[0] * (1 - self.fioaa - self.fioas - self.fioac)
        self.icdr[0] = self.ic[0] / self.alic

    def loop0_resource(self, alone=False):
        """
        Set the initial conditions of the sector at k=0; the
        _update_* methods are then run by run_init_loops in the
        k=0 execution order generated from the @step graph.

        Parameters
        ----------
        alone : boolean, optional
            if True, run the sector alone with exogenous inputs. The default
            is False.

        """
        self.nr[0] = self.nri
        if alone:
            self.loop0_exogenous()

    def run_resource(self):
        """
        Run the resource sector alone with exogenous inputs. Initialization
        (k=0) reschedules until all reads are known; for k >= 1 the updates
        (including the exogenous ones) follow the topological order of the
        @step dependency graph.

        """
        run_init_loops(self, [lambda: self.loop0_resource(alone=True)])
        steps = compile_steps(self, build_execution_order(type(self)))
        for k_ in range(1, self.n):
            if self.verbose:
                print("go loop", k_)
            j = k_ - 1
            for fn in steps:
                fn(k_, j, j, k_)

    @step(reads={"nr": 'j', "nrur": 'jk'}, writes={"nr": 'k'})
    def _update_state_nr(self, k, j, jk):
        """
        State variable, requires previous step only
        """
        self.nr[k] = self.nr[j] - self.dt * self.nrur[jk]

    @step(reads={"nr": 'k'}, writes={"nrfr": 'k'})
    def _update_nrfr(self, k):
        """
        From step k requires: NR
        """
        self.nrfr[k] = self.nr[k] / self.nri

    @step(reads={"fcaor1": 'k', "fcaor2": 'k', "nrfr": 'k', "time": 'k'}, writes={"fcaor": 'k', "fcaor1": 'k', "fcaor2": 'k'})
    def _update_fcaor(self, k):
        """
        From step k requires: NRFR
        """
        self.fcaor1[k] = self.fcaor1_f(self.nrfr[k])
        self.fcaor2[k] = self.fcaor2_f(self.nrfr[k])
        self.fcaor[k] = clip(self.fcaor2[k], self.fcaor1[k], self.time[k],
                             self.pyear)

    @step(reads={"time": 'k'}, writes={"nruf": 'k'})
    def _update_nruf(self, k):
        """
        From step k requires: nothing
        """
        self.nruf[k] = clip(self.nruf2, self.nruf1, self.time[k], self.pyear)

    @step(reads={"iopc": 'k'}, writes={"pcrum": 'k'})
    def _update_pcrum(self, k):
        """
        From step k requires: IOPC
        """
        self.pcrum[k] = self.pcrum_f(self.iopc[k])

    @step(reads={"nruf": 'k', "pcrum": 'k', "pop": 'k'}, writes={"nrur": 'kl'})
    def _update_nrur(self, k, kl):
        """
        From step k requires: POP PCRUM NRUF
        """
        self.nrur[kl] = self.pop[k] * self.pcrum[k] * self.nruf[k]

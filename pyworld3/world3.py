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

from numpy import arange

from .population import Population
from .capital import Capital
from .agriculture import Agriculture
from .pollution import Pollution
from .resource import Resource
from .utils import build_execution_order, compile_steps, run_init_loops


class World3(Population, Capital, Agriculture, Pollution, Resource):
    """
    The World3 model as it is described in the technical book [1]_. World3 is
    structured in 5 main sectors and contains 12 state variables. The figures
    in the first prints of the Limits to Growth [2]_ result from an older
    version of the model, with slighly different numerical parameters and
    some missing dynamical phenomena.

    See the details about all variables (state and rate and constant types) in
    the documentation of each sector.

    Examples
    --------
    The creation and the initialization of a World3 instance should respect
    the order of the following example:

    >>> world3 = World3()                    # choose the time limits and step.
    >>> world3.init_world3_constants()       # choose the model constants.
    >>> world3.init_world3_variables()       # initialize all variables.
    >>> world3.set_world3_table_functions()  # get tables from a json file.
    >>> world3.set_world3_delay_functions()  # initialize delay functions.
    >>> world3.run_world3()

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
    iphst : float, optional
        implementation date of new policy on health service time [year].
        The default is 1940.
    verbose : bool, optional
        print information for debugging. The default is False.

    References
    ----------
    .. [1] Meadows, Dennis L., William W. Behrens, Donella H. Meadows, Roger F.
       Naill, Jørgen Randers, and Erich Zahn. *Dynamics of growth in a finite
       world*. Cambridge, MA: Wright-Allen Press, 1974.

    .. [2] Meadows, Donella H., Dennis L. Meadows, Jorgen Randers, and William
       W. Behrens. *The limits to growth*. New York 102, no. 1972 (1972): 27.

    """

    def __init__(self, year_min=1900, year_max=2100, dt=0.5, pyear=1975,
                 iphst=1940, verbose=False):
        self.iphst = iphst
        self.pyear = pyear
        self.dt = dt
        self.year_min = year_min
        self.year_max = year_max
        self.length = self.year_max - self.year_min
        self.n = int(self.length / self.dt) + 1
        self.time = arange(self.year_min, self.year_max + self.dt, self.dt)
        self.verbose = False

    def init_world3_constants(self, p1i=65e7, p2i=70e7, p3i=19e7, p4i=6e7,
                              dcfsn=4, fcest=4000, hsid=20, ieat=3, len=28,
                              lpd=20, mtfn=12, pet=4000, rlt=30, sad=20,
                              zpgt=4000,
                              ici=2.1e11, sci=1.44e11, iet=4000, iopcd=400,
                              lfpf=0.75, lufdt=2, icor1=3, icor2=3, scor1=1,
                              scor2=1, alic1=14, alic2=14, alsc1=20, alsc2=20,
                              fioac1=0.43, fioac2=0.43,
                              ali=0.9e9, pali=2.3e9, lfh=0.7, palt=3.2e9,
                              pl=0.1, alai1=2, alai2=2, io70=7.9e11, lyf1=1,
                              lyf2=1, sd=0.07, uili=8.2e6, alln=6000, uildt=10,
                              lferti=600, ilf=600, fspd=2, sfpc=230,
                              ppoli=2.5e7, ppol70=1.36e8, ahl70=1.5, amti=1,
                              imti=10, imef=0.1, fipm=0.001, frpm=0.02,
                              ppgf1=1, ppgf2=1, ppgf21=1, pptd1=20, pptd2=20,
                              nri=1e12, nruf1=1, nruf2=1):
        """
        Initialize the constant parameters of the 5 sectors. Constants and
        their unit are defined in the documentation of the corresponding
        sectors.

        """
        self.init_population_constants(p1i, p2i, p3i, p4i, dcfsn, fcest, hsid,
                                       ieat, len, lpd, mtfn, pet, rlt, sad,
                                       zpgt)
        self.init_capital_constants(ici, sci, iet, iopcd, lfpf, lufdt, icor1,
                                    icor2, scor1, scor2, alic1, alic2, alsc1,
                                    alsc2, fioac1, fioac2)
        self.init_agriculture_constants(ali, pali, lfh, palt, pl, alai1, alai2,
                                        io70, lyf1, lyf2, sd, uili, alln,
                                        uildt, lferti, ilf, fspd, sfpc)
        self.init_pollution_constants(ppoli, ppol70, ahl70, amti, imti, imef,
                                      fipm, frpm, ppgf1, ppgf2, ppgf21, pptd1,
                                      pptd2)
        self.init_resource_constants(nri, nruf1, nruf2)

    def init_world3_variables(self):
        """
        Initialize the state and rate variables of the 5 sectors (memory
        allocation). Variables and their unit are defined in the documentation
        of the corresponding sectors.

        """
        self.init_population_variables()
        self.init_capital_variables()
        self.init_agriculture_variables()
        self.init_pollution_variables()
        self.init_resource_variables()

    def set_world3_delay_functions(self, method="euler"):
        """
        Set the linear smoothing and delay functions of the 1st or the 3rd
        order, in the 5 sectors. The effect depends on time constants, defined
        before with the method `self.init_world3_constants`. One should call
        `self.set_world3_delay_functions` after calling
        `self.init_world3_constants`.

        Parameters
        ----------
        method : str, optional
            Numerical integration method: "euler" or "odeint". The default is
            "euler".

        """
        self.set_population_delay_functions(method=method)
        self.set_capital_delay_functions(method=method)
        self.set_agriculture_delay_functions(method=method)
        self.set_pollution_delay_functions(method=method)
        self.set_resource_delay_functions(method=method)

    def set_world3_table_functions(self, json_file=None):
        """
        Set the nonlinear functions of the 5 sectors, based on a json file. By
        default, the `functions_table_world3.json` file from pyworld3 is used.

        Parameters
        ----------
        json_file : file, optional
            json file containing all tables. The default is None.

        """
        self.set_population_table_functions(json_file)
        self.set_capital_table_functions(json_file)
        self.set_agriculture_table_functions(json_file)
        self.set_pollution_table_functions(json_file)
        self.set_resource_table_functions(json_file)

    def run_world3(self):
        """
        Run a simulation of the World3 instance. One should initialize the
        model first (constants, variables, delay & table functions).

        The execution order of the update methods is derived once from the
        dependency graph declared by the @step decorators (variable and time
        index of each read and write), then the simulation runs without
        per-step dependency checks.

        """
        # k = 0: initial conditions and rates at year_min (rescheduling,
        # the model has circular dependencies at k=0)
        run_init_loops(self, [self.loop0_population, self.loop0_capital,
                              self.loop0_agriculture, self.loop0_pollution,
                              self.loop0_resource])

        steps = compile_steps(self, build_execution_order(type(self)))
        for k_ in range(1, self.n):
            if self.verbose:
                print("go loop", k_)
            j = k_ - 1
            for fn in steps:
                fn(k_, j, j, k_)


def hello_world3():
    """
    "Hello world" example with the well-known standard run of World3.

    """
    from .utils import plot_world_variables
    from matplotlib.pyplot import rcParams, show
    params = {'lines.linewidth': '3'}
    rcParams.update(params)

    world3 = World3()
    world3.init_world3_constants()
    world3.init_world3_variables()
    world3.set_world3_table_functions()
    world3.set_world3_delay_functions()
    world3.run_world3()

    plot_world_variables(world3.time,
                         [world3.nrfr, world3.iopc, world3.fpc, world3.pop,
                          world3.ppolx],
                         ["NRFR", "IOPC", "FPC", "POP", "PPOLX"],
                         [[0, 1], [0, 1e3], [0, 1e3], [0, 16e9], [0, 32]],
                         figsize=(7, 5),
                         grid=1,
                         title="World3 standard run")
    show()


if __name__ == "__main__":
    hello_world3()

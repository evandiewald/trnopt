"""The Pyomo plugin: same answers as HiGHS through Pyomo, on small models covering every construct it maps."""

import pytest

pyo = pytest.importorskip("pyomo.environ")
pytest.importorskip("highspy")

import trnopt.pyomo_plugin  # noqa: E402,F401  (registers SolverFactory("trnopt"))
from pyomo.opt import TerminationCondition as TC  # noqa: E402


def highs_obj(m):
    r = pyo.SolverFactory("appsi_highs").solve(m)
    assert r.solver.termination_condition == TC.optimal
    return pyo.value(next(m.component_data_objects(pyo.Objective, active=True)))


def mixed_model(maximize=False):
    """<=, >=, ==, a two-sided row, bounded / free / integer / binary variables, an objective offset."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 4))
    m.y = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(0, 10))
    m.z = pyo.Var(domain=pyo.Integers, bounds=(-2, 6))
    m.b = pyo.Var(domain=pyo.Binary)
    m.w = pyo.Var()                                      # free continuous
    m.c1 = pyo.Constraint(expr=m.x + 2 * m.y + m.b <= 7.5)
    m.c2 = pyo.Constraint(expr=m.x - m.z >= -1)
    m.c3 = pyo.Constraint(expr=m.y + m.z + m.w == 3)
    m.c4 = pyo.Constraint(expr=pyo.inequality(1, m.x + m.y - m.b, 5))
    m.c5 = pyo.Constraint(expr=m.w >= -4)
    expr = 3 * m.x + 5 * m.y - m.z + 2 * m.b - 0.5 * m.w + 10
    m.obj = pyo.Objective(expr=expr if maximize else -expr, sense=pyo.maximize if maximize else pyo.minimize)
    return m


@pytest.mark.parametrize("maximize", [False, True])
def test_mip_matches_highs(maximize):
    ref = highs_obj(mixed_model(maximize))
    m = mixed_model(maximize)
    r = pyo.SolverFactory("trnopt").solve(m)
    assert r.solver.termination_condition == TC.optimal
    assert pyo.value(m.obj) == pytest.approx(ref, abs=1e-6)
    for v in (m.y, m.z, m.b):                            # integer values loaded as integers
        assert pyo.value(v) == round(pyo.value(v))
    # bounds bracket the answer in the model's own sense
    assert r.problem.lower_bound <= pyo.value(m.obj) + 1e-6 and pyo.value(m.obj) <= r.problem.upper_bound + 1e-6


def test_lp_matches_highs():
    def lp():
        m = pyo.ConcreteModel()
        m.x = pyo.Var(range(3), bounds=(0, None))
        m.c1 = pyo.Constraint(expr=m.x[0] + m.x[1] + m.x[2] <= 10)
        m.c2 = pyo.Constraint(expr=m.x[0] - m.x[1] >= -2)
        m.c3 = pyo.Constraint(expr=2 * m.x[1] + m.x[2] == 8)
        m.obj = pyo.Objective(expr=3 * m.x[0] + 2 * m.x[1] + 4 * m.x[2], sense=pyo.maximize)
        return m
    ref = highs_obj(lp())
    m = lp()
    r = pyo.SolverFactory("trnopt").solve(m)
    assert r.solver.termination_condition == TC.optimal
    assert pyo.value(m.obj) == pytest.approx(ref, rel=1e-5)


def test_knapsack_and_options():
    w = [12, 7, 11, 8, 9, 6, 13, 5]
    v = [24, 13, 23, 15, 16, 11, 27, 9]

    def knap():
        m = pyo.ConcreteModel()
        m.x = pyo.Var(range(len(w)), domain=pyo.Binary)
        m.cap = pyo.Constraint(expr=sum(w[i] * m.x[i] for i in range(len(w))) <= 40)
        m.obj = pyo.Objective(expr=sum(v[i] * m.x[i] for i in range(len(w))), sense=pyo.maximize)
        return m
    ref = highs_obj(knap())
    for branching in ("reliability_trnopt", "strong_trnopt", "pseudocost"):
        m = knap()
        opt = pyo.SolverFactory("trnopt")
        opt.options["branching"] = branching
        r = opt.solve(m, timelimit=60)
        assert r.solver.termination_condition == TC.optimal, branching
        assert pyo.value(m.obj) == pytest.approx(ref), branching
    with pytest.raises(ValueError, match="unknown trnopt option"):
        pyo.SolverFactory("trnopt").solve(knap(), options={"bogus": 1})


def test_infeasible_and_unbounded():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(domain=pyo.Integers, bounds=(0, 10))
    m.c1 = pyo.Constraint(expr=m.x >= 4.2)
    m.c2 = pyo.Constraint(expr=m.x <= 4.8)
    m.obj = pyo.Objective(expr=m.x)
    r = pyo.SolverFactory("trnopt").solve(m)
    assert r.solver.termination_condition == TC.infeasible

    u = pyo.ConcreteModel()
    u.x = pyo.Var(bounds=(0, None))
    u.y = pyo.Var(bounds=(0, None))
    u.c = pyo.Constraint(expr=u.x - u.y <= 1)
    u.obj = pyo.Objective(expr=u.x + u.y, sense=pyo.maximize)
    r = pyo.SolverFactory("trnopt").solve(u)
    assert r.solver.termination_condition in (TC.unbounded, TC.infeasibleOrUnbounded)


def test_lp_to_pyomo_round_trip():
    """trnopt LP -> Pyomo model -> trnopt LP keeps the problem: same optimum as HiGHS on the original."""
    from trnopt.generators import set_cover
    from trnopt.pyomo_plugin import lp_to_pyomo, model_to_lp
    from trnopt.reference import solve_highs_mip
    lp = set_cover(60, 120, 0.08, seed=1)
    m = lp_to_pyomo(lp)
    back, _ = model_to_lp(m)
    assert back.K.nnz == lp.K.nnz and back.integer.sum() == lp.integer.sum()
    ref = solve_highs_mip(lp, 60)
    r = pyo.SolverFactory("trnopt").solve(m)
    assert r.solver.termination_condition == TC.optimal
    assert pyo.value(m.obj) == pytest.approx(ref.objective, abs=1e-6)
    assert highs_obj(lp_to_pyomo(lp)) == pytest.approx(ref.objective, abs=1e-6)


@pytest.mark.parametrize("presolve", [False, True])
def test_presolve_same_answer(presolve):
    """HiGHS presolve in front of the B&B changes the work, not the answer (instance it actually reduces)."""
    from trnopt.generators import combinatorial_auction
    from trnopt.pyomo_plugin import lp_to_pyomo
    from trnopt.reference import solve_highs_mip
    lp = combinatorial_auction(40, 200, seed=4)
    ref = solve_highs_mip(lp, 60)
    m = lp_to_pyomo(lp)
    r = pyo.SolverFactory("trnopt").solve(m, options={"presolve": presolve})
    assert r.solver.termination_condition == TC.optimal
    assert pyo.value(m.obj) == pytest.approx(ref.objective, abs=1e-6)
    assert r.problem.lower_bound == pytest.approx(ref.objective, abs=1e-4 * max(1, abs(ref.objective)))

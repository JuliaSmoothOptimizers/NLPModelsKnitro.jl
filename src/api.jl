using NLPModels, NLPModelsModifiers, SolverCore

# Knitro does not accept least-squares problems with constraints other than bounds.
# We must treat those as general NLPs.
# If an NLSModel has constraints other than bounds, we convert it to a FeasibilityFormNLS.
# Because FeasibilityFormNLS <: AbstractNLSModel, we need a trait to dispatch on.
_is_general_nlp(nlp::AbstractNLPModel) = Val{true}()
_is_general_nlp(nls::AbstractNLSModel) = Val{isa(nls, FeasibilityFormNLS)}()

function knitro_statuses(code::Integer)
  if code == 0
    return :first_order
  end
  if code == -100
    return :acceptable
  end
  if -103 ≤ code ≤ -101
    return :stalled #feasible
  end
  if -299 ≤ code ≤ -200
    return :infeasible
  end
  if -301 ≤ code ≤ -300
    return :unbounded
  end
  if code == -400 || code == -410  # -400 = feasible, -410 = infeasible
    return :max_iter
  end
  if code == -401 || code == -411  # -401 = feasible, -411 = infeasible
    return :max_time
  end
  if code == -402 || code == -412  # -402 = feasible, -412 = infeasible
    return :max_eval
  end
  if -600 ≤ code ≤ -500
    return :exception
  end
  return :unknown
end

function Base.finalize(solver::KnitroSolver)
  #  KNITRO.KN_reset_params_to_defaults(solver.kc)
  KNITRO.KN_free(solver.kc)
end

# pass options to KNITRO
# could remove some stuff from both functions below (careful to default x0)
function setparams!(solver::KnitroSolver; kwargs...)
  kc = solver.kc
  KNITRO.KN_reset_params_to_defaults(kc)
  # set primal and dual initial guess
  kwargs = Dict(kwargs)
  if :x0 ∈ keys(kwargs)
    KNITRO.KN_set_var_primal_init_values(kc, kwargs[:x0])
    pop!(kwargs, :x0)
  end
  if :y0 ∈ keys(kwargs)
    KNITRO.KN_set_con_dual_init_values(kc, kwargs[:y0])
    pop!(kwargs, :y0)
  end
  if :z0 ∈ keys(kwargs)
    KNITRO.KN_set_var_dual_init_values(kc, kwargs[:z0])
    pop!(kwargs, :z0)
  end

  # specify that we are able to provide the Hessian without including the objective
  KNITRO.KN_set_param(kc, KNITRO.KN_PARAM_HESSIAN_NO_F, KNITRO.KN_HESSIAN_NO_F_ALLOW)

  # pass options to KNITRO
  for (k, v) in kwargs
    KNITRO.KN_set_param(kc, string(k), v)
  end

  return solver
end

KnitroSolver(nlp::AbstractNLPModel; kwargs...) = KnitroSolver(_is_general_nlp(nlp), nlp; kwargs...)

include("nlp.jl")
include("nls.jl")

function knitro(nlp::AbstractNLPModel; kwargs...)
  val = _is_general_nlp(nlp)
  solver = KnitroSolver(val, nlp; kwargs...)
  stats = knitro!(nlp, solver)
  finalize(solver)
  return stats
end

function knitro!(nlp::AbstractNLPModel, solver::KnitroSolver)
  kc = solver.kc
  t = @timed begin
    nStatus = KNITRO.KN_solve(kc)
  end

  nStatus, obj_val, x, lambda_ = KNITRO.KN_get_solution(kc)
  n = length(x)
  m = length(lambda_) - n
  primal_feas = KNITRO.KN_get_abs_feas_error(kc)
  dual_feas = KNITRO.KN_get_abs_opt_error(kc)
  iter = KNITRO.KN_get_number_iters(kc)
  if KNITRO_VERSION ≥ v"12.0"
    Δt = KNITRO.KN_get_solve_time_cpu(kc)
    real_time = KNITRO.KN_get_solve_time_real(kc)
  else
    Δt = real_time = t[2]
  end

  return GenericExecutionStats(
    knitro_statuses(nStatus),
    nlp,
    solution = x,
    objective = obj_val,
    dual_feas = dual_feas,
    iter = convert(Int, iter),
    primal_feas = primal_feas,
    elapsed_time = Δt,
    multipliers = lambda_[1:m],
    multipliers_L = lambda_[(m + 1):(m + n)],  # don't know how to get those separately
    multipliers_U = eltype(x)[],
    solver_specific = Dict(:internal_msg => nStatus, :real_time => real_time),
  )
end

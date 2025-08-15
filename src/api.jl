using NLPModels, NLPModelsModifiers

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
  KNITRO.KN_set_int_param(kc, KNITRO.KN_PARAM_HESSIAN_NO_F, KNITRO.KN_HESSIAN_NO_F_ALLOW)

  # pass options to KNITRO
  for (k, v) in kwargs
    if v isa Integer
      KNITRO.KN_set_int_param_by_name(kc, string(k), v)
    elseif v isa Cdouble
      KNITRO.KN_set_double_param_by_name(kc, string(k), v)
    elseif v isa AbstractString
      KNITRO.KN_set_char_param_by_name(kc, string(k), v)
    else
      @warn "The option $(string(k)) was ignored."
    end
  end

  return solver
end

KnitroSolver(nlp::AbstractNLPModel; kwargs...) = KnitroSolver(_is_general_nlp(nlp), nlp; kwargs...)

include("nlp.jl")
include("nls.jl")

function knitro(nlp::AbstractNLPModel; kwargs...)
  solver = KnitroSolver(nlp; kwargs...)
  stats = solve!(solver, nlp)
  finalize(solver)
  return stats
end

function knitro(nlp::FeasibilityFormNLS; kwargs...)
  solver = KnitroSolver(nlp; kwargs...)
  stats = solve!(solver, nlp)
  finalize(solver)
  return stats
end

function knitro(nls::AbstractNLSModel; kwargs...)
  if nls.meta.ncon > 0
    @warn "Knitro only treats bound-constrained least-squares problems; converting to feasibility form"
    fnls = FeasibilityFormNLS(nls)
    fstats = knitro(fnls; kwargs...)
    # extract stats for the original problem
    stats = GenericExecutionStats(nls)
    fstats.status_reliable || error("status unreliable")
    set_status!(stats, fstats.status)
    fstats.solution_reliable || error("solution unreliable")
    set_solution!(stats, fstats.solution[1:(nls.meta.nvar)])
    fstats.objective_reliable || error("objective unreliable")
    set_objective!(stats, stats.objective)
    fstats.primal_residual_reliable || error("primal residual unreliable")
    set_primal_residual!(stats, fstats.primal_feas)
    fstats.dual_residual_reliable || error("dual residual unreliable")
    set_dual_residual!(stats, fstats.dual_feas)
    fstats.multipliers_reliable || error("multipliers unreliable")
    set_constraint_multipliers!(stats, fstats.multipliers[(nls.nls_meta.nequ + 1):end])
    if has_bounds(nls)
      fstats.bounds_multipliers_reliable || error("bounds multipliers unreliable")
      set_bounds_multipliers!(
        stats,
        fstats.multipliers_L[1:(nls.meta.nvar)],
        fstats.multipliers_U[1:(nls.meta.nvar)],
      )
    end
    fstats.iter_reliable || error("number of iterations unreliable")
    set_iter!(stats, fstats.iter)
    fstats.time_reliable || error("elapsed time unreliable")
    set_time!(stats, fstats.elapsed_time)
    if fstats.solver_specific_reliable
      for (k, v) ∈ fstats.solver_specific
        set_solver_specific!(stats, k, v)
      end
    else
      error("solver-specific stats unreliable")
    end
    return stats
  end
  solver = KnitroSolver(nls; kwargs...)
  stats = solve!(solver, nls)
  finalize(solver)
  return stats
end

function SolverCore.solve!(
  solver::KnitroSolver,
  nlp::AbstractNLPModel,
  stats::GenericExecutionStats,
)
  kc = solver.kc
  reset!(stats)
  t = @timed begin
    nStatus = KNITRO.KN_solve(kc)
  end

  nStatus, obj_val, x, lambda_ = KNITRO.KN_get_solution(kc)
  n = length(x)
  m = length(lambda_) - n
  pCdouble = Ref{Cdouble}()
  KNITRO.KN_get_abs_feas_error(kc, pCdouble)
  primal_feas = pCdouble[]
  KNITRO.KN_get_abs_opt_error(kc, pCdouble)
  dual_feas = pCdouble[]
  pCint = Ref{Cint}()
  KNITRO.KN_get_number_iters(kc, pCint)
  iter = pCint[]
  if KNITRO.knitro_version() ≥ v"12.0"
    KNITRO.KN_get_solve_time_cpu(kc, pCdouble)
    Δt = pCdouble[]
    KNITRO.KN_get_solve_time_real(kc, pCdouble)
    real_time = pCdouble[]
  else
    Δt = real_time = t[2]
  end

  set_status!(stats, knitro_statuses(nStatus))
  set_solution!(stats, x)
  set_objective!(stats, obj_val)
  set_residuals!(stats, primal_feas, dual_feas)
  set_iter!(stats, convert(Int, iter))
  set_time!(stats, Δt)
  zL = similar(lambda_, has_bounds(nlp) ? nlp.meta.nvar : 0)
  zU = similar(lambda_, has_bounds(nlp) ? nlp.meta.nvar : 0)
  if has_bounds(nlp)
    zL .= max.(lambda_[(m + 1):(m + n)], 0)
    zU .= .-min.(lambda_[(m + 1):(m + n)], 0)
  end
  set_multipliers!(stats, lambda_[1:m], zL, zU)
  set_solver_specific!(stats, :internal_msg, nStatus)
  set_solver_specific!(stats, :real_time, real_time)
  stats
end

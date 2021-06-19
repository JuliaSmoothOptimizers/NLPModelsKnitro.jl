function _knitro(
  ::Val{false},
  nls::AbstractNLSModel;
  callback::Union{Function, Nothing} = nothing,
  kwargs...,
)
  n, m, ne = nls.meta.nvar, nls.meta.ncon, nls.nls_meta.nequ

  if m > 0
    @warn "Knitro only treats bound-constrained least-squares problems; converting to feasibility form"
    return knitro(FeasibilityFormNLS(nls), callback = callback; kwargs...)
  end

  kc = KNITRO.KN_new()
  release = KNITRO.get_release()
  KNITRO.KN_reset_params_to_defaults(kc)
  if nls.meta.minimize
    KNITRO.KN_set_obj_goal(kc, KNITRO.KN_OBJGOAL_MINIMIZE)
  else
    KNITRO.KN_set_obj_goal(kc, KNITRO.KN_OBJGOAL_MAXIMIZE)
  end

  # add variables and bound constraints
  KNITRO.KN_add_vars(kc, n)

  lvarinf = isinf.(nls.meta.lvar)
  if !all(lvarinf)
    lvar = nls.meta.lvar
    if any(lvarinf)
      lvar = copy(nls.meta.lvar)
      lvar[lvarinf] .= -KNITRO.KN_INFINITY
    end
    KNITRO.KN_set_var_lobnds(kc, lvar)
  end

  uvarinf = isinf.(nls.meta.uvar)
  if !all(uvarinf)
    uvar = nls.meta.uvar
    if any(uvarinf)
      uvar = copy(nls.meta.uvar)
      uvar[uvarinf] .= KNITRO.KN_INFINITY
    end
    KNITRO.KN_set_var_upbnds(kc, uvar)
  end

  # set primal and dual initial guess
  kwargs = Dict(kwargs)
  if :x0 ∈ keys(kwargs)
    KNITRO.KN_set_var_primal_init_values(kc, kwargs[:x0])
    pop!(kwargs, :x0)
  else
    KNITRO.KN_set_var_primal_init_values(kc, nls.meta.x0)
  end
  if :z0 ∈ keys(kwargs)
    KNITRO.KN_set_var_dual_init_values(kc, kwargs[:z0])
    pop!(kwargs, :z0)
  end

  # add number of residual functions
  KNITRO.KN_add_rsds(kc, ne)

  jrows, jcols = jac_structure_residual(nls)

  # define callback for residual
  function callbackEvalR(kc, cb, evalRequest, evalResult, userParams)
    if evalRequest.evalRequestCode != KNITRO.KN_RC_EVALR
      @warn "callbackEvalR incorrectly called with eval request code " evalRequest.evalRequestCode
      return -1
    end
    residual!(nls, evalRequest.x, evalResult.rsd)
    return 0
  end

  # define callback for residual Jacobian
  function callbackEvalRJ(kc, cb, evalRequest, evalResult, userParams)
    if evalRequest.evalRequestCode != KNITRO.KN_RC_EVALRJ
      @warn "callbackEvalRJ incorrectly called with eval request code " evalRequest.evalRequestCode
      return -1
    end
    jac_coord_residual!(nls, evalRequest.x, evalResult.rsdJac)
    return 0
  end

  # register callbacks
  cb = KNITRO.KN_add_lsq_eval_callback(kc, callbackEvalR)
  KNITRO.KN_set_cb_rsd_jac(
    kc,
    cb,
    nls.nls_meta.nnzj,
    callbackEvalRJ,
    jacIndexRsds = convert(Vector{Int32}, jrows .- 1),  # indices must be 0-based
    jacIndexVars = convert(Vector{Int32}, jcols .- 1),
  )

  # pass options to KNITRO
  for (k, v) in kwargs
    KNITRO.KN_set_param(kc, string(k), v)
  end

  # set user-defined callback called after each iteration
  callback == nothing || KNITRO.KN_set_newpt_callback(kc, callback)

  t = @timed begin
    nStatus = KNITRO.KN_solve(kc)
  end

  nStatus, obj_val, x, lambda_ = KNITRO.KN_get_solution(kc)
  primal_feas = KNITRO.KN_get_abs_feas_error(kc)
  dual_feas = KNITRO.KN_get_abs_opt_error(kc)
  iter = KNITRO.KN_get_number_iters(kc)
  if KNITRO_VERSION ≥ v"12.0"
    Δt = KNITRO.KN_get_solve_time_cpu(kc)
    real_time = KNITRO.KN_get_solve_time_real(kc)
  else
    Δt = real_time = t[2]
  end

  KNITRO.KN_reset_params_to_defaults(kc)
  KNITRO.KN_free(kc)

  return GenericExecutionStats(
    knitro_statuses(nStatus),
    nls,
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

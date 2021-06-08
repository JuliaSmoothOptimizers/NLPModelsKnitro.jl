function KnitroSolver(
  ::Val{false},
  nls::AbstractNLSModel;
  callback::Union{Function, Nothing} = nothing,
  kwargs...,
) where T
  n, m, ne = nls.meta.nvar, nls.meta.ncon, nls.nls_meta.nequ

  if m > 0
    @warn "Knitro only treats bound-constrained least-squares problems; converting to feasibility form"
    fnls = FeasibilityFormNLS(nls)
    return KnitroSolver(_is_general_nlp(fnls), fnls, callback = callback; kwargs...)
  end

  kc = KNITRO.KN_new()
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

  return KnitroSolver(kc)
end

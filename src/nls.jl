function KnitroSolver(
  ::Val{false},
  nls::AbstractNLSModel;
  callback::Union{Function, Nothing} = nothing,
  kwargs...,
)
  @assert nls.meta.ncon == 0
  n, ne = nls.meta.nvar, nls.nls_meta.nequ

  kc = KNITRO.KN_new()
  KNITRO.KN_reset_params_to_defaults(kc)
  if nls.meta.minimize
    KNITRO.KN_set_obj_goal(kc, KNITRO.KN_OBJGOAL_MINIMIZE)
  else
    KNITRO.KN_set_obj_goal(kc, KNITRO.KN_OBJGOAL_MAXIMIZE)
  end

  # add variables and bound constraints
  KNITRO.KN_add_vars(kc, n, C_NULL)

  lvarinf = isinf.(nls.meta.lvar)
  if !all(lvarinf)
    lvar = nls.meta.lvar
    if any(lvarinf)
      lvar = copy(nls.meta.lvar)
      lvar[lvarinf] .= -KNITRO.KN_INFINITY
    end
    KNITRO.KN_set_var_lobnds_all(kc, lvar)
  end

  uvarinf = isinf.(nls.meta.uvar)
  if !all(uvarinf)
    uvar = nls.meta.uvar
    if any(uvarinf)
      uvar = copy(nls.meta.uvar)
      uvar[uvarinf] .= KNITRO.KN_INFINITY
    end
    KNITRO.KN_set_var_upbnds_all(kc, uvar)
  end

  # set primal and dual initial guess
  kwargs = Dict(kwargs)
  if :x0 ∈ keys(kwargs)
    KNITRO.KN_set_var_primal_init_values_all(kc, kwargs[:x0])
    pop!(kwargs, :x0)
  else
    KNITRO.KN_set_var_primal_init_values_all(kc, nls.meta.x0)
  end
  if :z0 ∈ keys(kwargs)
    KNITRO.KN_set_var_dual_init_values_all(kc, kwargs[:z0])
    pop!(kwargs, :z0)
  end

  # add number of residual functions
  KNITRO.KN_add_rsds(kc, ne, C_NULL)

  jrows, jcols = jac_structure_residual(nls)

  # define callback for residual
  function callbackEvalR(kc, cb, evalRequest, evalResult, userParams)
    if evalRequest.evalRequestCode != KNITRO.KN_RC_EVALR
      @error "callbackEvalR incorrectly called with eval request code " evalRequest.evalRequestCode
      return -1
    end
    residual!(nls, evalRequest.x, evalResult.rsd)
    return 0
  end

  # define callback for residual Jacobian
  function callbackEvalRJ(kc, cb, evalRequest, evalResult, userParams)
    if evalRequest.evalRequestCode != KNITRO.KN_RC_EVALRJ
      @error "callbackEvalRJ incorrectly called with eval request code " evalRequest.evalRequestCode
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
    if v isa Integer
      KNITRO.KN_set_int_param_by_name(kc, string(k), v)
    elseif v isa Cdouble
      KNITRO.KN_set_double_param_by_name(kc, string(k), v)
    else
      @assert v isa AbstractString
      KNITRO.KN_set_char_param_by_name(kc, string(k), v)
    end
  end

  # set user-defined callback called after each iteration
  callback == nothing || KNITRO.KN_set_newpt_callback(kc, callback)

  return KnitroSolver(kc)
end

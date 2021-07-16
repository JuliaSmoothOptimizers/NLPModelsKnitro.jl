function KnitroSolver(
  ::Val{true},
  nlp::AbstractNLPModel;
  callback::Union{Function, Nothing} = nothing,
  kwargs...,
) where {T}
  n, m = nlp.meta.nvar, nlp.meta.ncon

  kc = KNITRO.KN_new()
  KNITRO.KN_reset_params_to_defaults(kc)
  if nlp.meta.minimize
    KNITRO.KN_set_obj_goal(kc, KNITRO.KN_OBJGOAL_MINIMIZE)
  else
    KNITRO.KN_set_obj_goal(kc, KNITRO.KN_OBJGOAL_MAXIMIZE)
  end

  # add variables and bound constraints
  KNITRO.KN_add_vars(kc, n)

  lvarinf = isinf.(nlp.meta.lvar)
  if !all(lvarinf)
    lvar = nlp.meta.lvar
    if any(lvarinf)
      lvar = copy(nlp.meta.lvar)
      lvar[lvarinf] .= -KNITRO.KN_INFINITY
    end
    KNITRO.KN_set_var_lobnds(kc, lvar)
  end

  uvarinf = isinf.(nlp.meta.uvar)
  if !all(uvarinf)
    uvar = nlp.meta.uvar
    if any(uvarinf)
      uvar = copy(nlp.meta.uvar)
      uvar[uvarinf] .= KNITRO.KN_INFINITY
    end
    KNITRO.KN_set_var_upbnds(kc, uvar)
  end

  # add constraints
  KNITRO.KN_add_cons(kc, m)
  lcon = nlp.meta.lcon
  lconinf = isinf.(lcon)
  if any(lconinf)
    lcon = copy(nlp.meta.lcon)
    lcon[lconinf] .= -KNITRO.KN_INFINITY
  end
  ucon = nlp.meta.ucon
  uconinf = isinf.(ucon)
  if any(uconinf)
    ucon = copy(nlp.meta.ucon)
    ucon[uconinf] .= KNITRO.KN_INFINITY
  end
  if nlp.meta.nlin > 0
    jac_lin = jac(nlp, nlp.meta.x0)
    for klin in nlp.meta.lin 
     KNITRO.KN_add_con_linear_struct(kc, Int32(klin - 1), Int32.(0:(n-1)), jac_lin[klin, :])
    end
  end
  KNITRO.KN_set_con_lobnds(kc, lcon)
  KNITRO.KN_set_con_upbnds(kc, ucon)

  # set primal and dual initial guess
  kwargs = Dict(kwargs)
  if :x0 ∈ keys(kwargs)
    KNITRO.KN_set_var_primal_init_values(kc, kwargs[:x0])
    pop!(kwargs, :x0)
  else
    KNITRO.KN_set_var_primal_init_values(kc, nlp.meta.x0)
  end
  if :y0 ∈ keys(kwargs)
    KNITRO.KN_set_con_dual_init_values(kc, kwargs[:y0])
    pop!(kwargs, :y0)
  end
  if :z0 ∈ keys(kwargs)
    KNITRO.KN_set_var_dual_init_values(kc, kwargs[:z0])
    pop!(kwargs, :z0)
  end

  jrows, jcols = m > 0 ? jac_structure(nlp) : (Int[], Int[])
  hrows, hcols = hess_structure(nlp)

  # define evaluation callback
  function evalAll(kc, cb, evalRequest, evalResult, userParams)
    x = evalRequest.x
    evalRequestCode = evalRequest.evalRequestCode

    if evalRequestCode == KNITRO.KN_RC_EVALFC
      evalResult.obj[1] = obj(nlp, x)
      if m > 0
        cons!(nlp, x, evalResult.c)
        evalResult.c[nlp.meta.lin] .= 0
      end
    elseif evalRequestCode == KNITRO.KN_RC_EVALGA
      grad!(nlp, x, evalResult.objGrad)
      m > 0 && jac_coord!(nlp, x, evalResult.jac)
    elseif evalRequestCode == KNITRO.KN_RC_EVALH
      if m > 0
        hess_coord!(
          nlp,
          x,
          view(evalRequest.lambda, 1:m),
          evalResult.hess,
          obj_weight = evalRequest.sigma,
        )
      else
        hess_coord!(nlp, x, evalResult.hess, obj_weight = evalRequest.sigma)
      end
    elseif evalRequestCode == KNITRO.KN_RC_EVALHV
      vec = evalRequest.vec
      if m > 0
        hprod!(
          nlp,
          x,
          view(evalRequest.lambda, 1:m),
          vec,
          evalResult.hessVec,
          obj_weight = evalRequest.sigma,
        )
      else
        hprod!(nlp, x, vec, evalResult.hessVec, obj_weight = evalRequest.sigma)
      end
    elseif evalRequestCode == KNITRO.KN_RC_EVALH_NO_F  # it would be silly to call this on unconstrained problems but better be careful
      if m > 0
        hess_coord!(nlp, x, view(evalRequest.lambda, 1:m), evalResult.hess, obj_weight = 0.0)
      else
        hess_coord!(nlp, x, evalResult.hess, obj_weight = 0.0)
      end
    elseif evalRequestCode == KNITRO.KN_RC_EVALHV_NO_F
      vec = evalRequest.vec
      if m > 0
        hprod!(nlp, x, view(evalRequest.lambda, 1:m), vec, evalResult.hessVec, obj_weight = 0.0)
      else
        hprod!(nlp, x, vec, evalResult.hessVec, obj_weight = 0.0)
      end
    else
      return KNITRO.KN_RC_CALLBACK_ERR
    end
    return 0
  end

  # register callbacks
  cb = KNITRO.KN_add_eval_callback_all(kc, evalAll)
  KNITRO.KN_set_cb_grad(
    kc,
    cb,
    evalAll,
    jacIndexCons = convert(Vector{Int32}, jrows .- 1),  # indices must be 0-based
    jacIndexVars = convert(Vector{Int32}, jcols .- 1),
  )
  KNITRO.KN_set_cb_hess(
    kc,
    cb,
    nlp.meta.nnzh,
    evalAll,
    hessIndexVars1 = convert(Vector{Int32}, hcols .- 1),  # Knitro wants the upper triangle
    hessIndexVars2 = convert(Vector{Int32}, hrows .- 1),
  )

  # specify that we are able to provide the Hessian without including the objective
  KNITRO.KN_set_param(kc, KNITRO.KN_PARAM_HESSIAN_NO_F, KNITRO.KN_HESSIAN_NO_F_ALLOW)

  # pass options to KNITRO
  for (k, v) in kwargs
    KNITRO.KN_set_param(kc, string(k), v)
  end

  # set user-defined callback called after each iteration
  callback == nothing || KNITRO.KN_set_newpt_callback(kc, callback)

  return KnitroSolver(kc)
end

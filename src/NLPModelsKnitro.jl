module NLPModelsKnitro

export knitro

using NLPModels, KNITRO, SolverTools

const knitro_statuses = Dict(0 => :first_order,
                             # 1 => :first_order,
                             # 2 => :infeasible,
                             # 3 => :small_step,
                             #4 => Diverging iterates
                             #5 => User requestep stop
                             #6 => Feasible point found
                             # -1 => :max_iter,
                             #-2 => Restoration failed
                             #-3 => Error in step computation
                             # -4 => :max_time,
                             #-10 => Not enough degress of freedom
                             #-11 => Invalid problem definition
                             #-12 => Invalid option
                             #-13 => Invalid number detected
                             -300 => :unbounded)

"""`output = knitro(nlp)`

Solves the `NLPModel` problem `nlp` using KNITRO.
"""
function knitro(nlp :: AbstractNLPModel;
                callback :: Union{Function,Nothing} = nothing,
                ignore_time :: Bool = false,
                kwargs...)
  n, m = nlp.meta.nvar, nlp.meta.ncon

  kc = KNITRO.KN_new()
  release = KNITRO.get_release()
  KNITRO.KN_reset_params_to_defaults(kc)
  KNITRO.KN_set_obj_goal(kc, KNITRO.KN_OBJGOAL_MINIMIZE)

  # add variables and bound constraints
  KNITRO.KN_add_vars(kc, n)

  lvarinf = isinf.(nlp.meta.lvar)
  if !all(lvarinf)
    lvar = nlp.meta.lvar
    if any(lvarinf)
      lvar = copy(nlp.meta.lvar)
      lvar[lvarinf] .= -KTR_INFINITY
    end
    KNITRO.KN_set_var_lobnds(kc, lvar)
  end

  uvarinf = isinf.(nlp.meta.uvar)
  if !all(uvarinf)
    uvar = nlp.meta.uvar
    if any(uvarinf)
      uvar = copy(nlp.meta.uvar)
      uvar[uvarinf] .= KTR_INFINITY
    end
    KNITRO.KN_set_var_upbnds(kc, uvar)
  end

  # set primal and dual initial guess
  KNITRO.KN_set_var_primal_init_values(kc, nlp.meta.x0)
  KNITRO.KN_set_var_dual_init_values(kc,  nlp.meta.y0)

  # add constraints
  KNITRO.KN_add_cons(kc, m)
  lcon = nlp.meta.lcon
  lconinf = isinf.(lcon)
  if any(lconinf)
    lcon = copy(nlp.meta.lcon)
    lcon[lconinf] .= -KTR_INFBOUND
  end
  ucon = nlp.meta.ucon
  uconinf = isinf.(ucon)
  if any(uconinf)
    ucon = copy(nlp.meta.ucon)
    ucon[uconinf] .= KTR_INFBOUND
  end
  KNITRO.KN_set_con_lobnds(kc, lcon)
  KNITRO.KN_set_con_upbnds(kc, ucon)

  jrows, jcols = jac_structure(nlp)
  hrows, hcols = hess_structure(nlp)

  # define evaluation callback
  function evalAll(kc, cb, evalRequest, evalResult, userParams)
    x = evalRequest.x
    evalRequestCode = evalRequest.evalRequestCode

    if evalRequestCode == KNITRO.KN_RC_EVALFC
      evalResult.obj[1] = obj(nlp, x)
      m > 0 && cons!(nlp, x, evalResult.c)
    elseif evalRequestCode == KNITRO.KN_RC_EVALGA
      grad!(nlp, x, evalResult.objGrad)
      m > 0 && jac_coord!(nlp, x, jrows, jcols, evalResult.jac)
    elseif evalRequestCode == KNITRO.KN_RC_EVALH
      hess_coord!(nlp, x, hrows, hcols, evalResult.hess, obj_weight=evalRequest.sigma, y=evalRequest.lambda)
    elseif evalRequestCode == KNITRO.KN_RC_EVALHV
      vec = evalRequest.vec
      hprod!(nlp, x, vec, evalResult.hessVec, obj_weight=evalRequest.sigma, y=evalRequest.lambda)
    elseif evalRequestCode == KNITRO.KN_RC_EVALH_NO_F
      hess_coord!(nlp, x, hrows, hcols, evalResult.hess, obj_weight=0.0)
    elseif evalRequestCode == KNITRO.KN_RC_EVALHV_NO_F
      vec = evalRequest.vec
      hprod!(nlp, x, vec, evalResult.hessVec, obj_weight=0.0)
    else
        return KNITRO.KN_RC_CALLBACK_ERR
    end
    return 0
  end

  # register callbacks
  cb = KNITRO.KN_add_eval_callback(kc, evalAll)
  KNITRO.KN_set_cb_grad(kc, cb, evalAll,
                        jacIndexCons=convert(Vector{Int32}, jrows .- 1),  # indices must be 0-based
                        jacIndexVars=convert(Vector{Int32}, jcols .- 1))
  KNITRO.KN_set_cb_hess(kc, cb, nlp.meta.nnzh, evalAll,
                        hessIndexVars1=convert(Vector{Int32}, hcols .- 1),  # Knitro wants the upper triangle
                        hessIndexVars2=convert(Vector{Int32}, hrows .- 1))

  # specify that we are able to provide the Hessian without including the objective
  KNITRO.KN_set_param(kc, KNITRO.KN_PARAM_HESSIAN_NO_F, KNITRO.KN_HESSIAN_NO_F_ALLOW)

  print_output = true

  # Options
  for (k,v) in kwargs
    if ignore_time || k != :outlev || v ≥ KNITRO.KN_OUTLEV_SUMMARY
      KNITRO.KN_set_param(kc, string(k), v)
    else
      if v > 0
        @warn("`outlev` should be ≥ $(KNITRO.KN_OUTLEV_SUMMARY) to get the elapsed time, if you don't care about the elapsed time, pass `ignore_time=true`")
      end
      print_output = false
      KNITRO.KN_set_param(kc, "outlev", KNITRO.KN_OUTLEV_SUMMARY)
    end
  end

  # set user-defined callback called after each iteration
  callback == nothing || KNITRO.KN_set_newpt_callback(kc, callback)

  tmpfile = tempname()
  local status
  open(tmpfile, "w") do io
    redirect_stdout(io) do
      nStatus = KNITRO.KN_solve(kc)
    end
  end
  knitro_output = readlines(tmpfile)

  Δt = 0.0
  for line in knitro_output
    if occursin("CPU time", line)
      Δt = Meta.parse(split(split(line, "=")[2], "(")[1])
      break
    end
  end
  if print_output
    println(join(knitro_output, "\n"))
  end

  nStatus, obj_val, x, lambda_ = KNITRO.KN_get_solution(kc)
  primal_feas = KNITRO.KN_get_abs_feas_error(kc)
  dual_feas = KNITRO.KN_get_abs_opt_error(kc)

  KNITRO.KN_reset_params_to_defaults(kc)
  KNITRO.KN_free(kc)

  return GenericExecutionStats(get(knitro_statuses, nStatus, :unknown), nlp, solution=x,
                               objective=obj_val, dual_feas=dual_feas,
                               primal_feas=primal_feas, elapsed_time=Δt,
                               solver_specific=Dict(:multipliers_con => lambda_[1:m],
                                                    :multipliers_L => lambda_[m+1:m+n],  # don't know how to get those separately
                                                    :multipliers_U => [],
                                                    :internal_msg => nStatus)
                              )

end

end # module

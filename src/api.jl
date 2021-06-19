export knitro

using NLPModels, NLPModelsModifiers, SolverCore

# Knitro does not accept least-squares problems with constraints other than bounds.
# We must treat those as general NLPs.
# If an NLSModel has constraints other than bounds, we convert it to a FeasibilityFormNLS.
# Because FeasibilityFormNLS <: AbstractNLSModel, we need a trait to dispatch on.
_is_general_nlp(nlp::AbstractNLPModel) = Val{true}()
_is_general_nlp(nls::AbstractNLSModel) = Val{isa(nls, FeasibilityFormNLS)}()

"""`output = knitro(nlp; kwargs...)`

Solves the `NLPModel` problem `nlp` using KNITRO.

# Optional keyword arguments
* `x0`: a vector of size `nlp.meta.nvar` to specify an initial primal guess
* `y0`: a vector of size `nlp.meta.ncon` to specify an initial dual guess for the general constraints
* `z0`: a vector of size `nlp.meta.nvar` to specify initial multipliers for the bound constraints
* `callback`: a user-defined `Function` called by KNITRO at each iteration.

For more information on callbacks, see https://www.artelys.com/docs/knitro/2_userGuide/callbacks.html and
the docstring of `KNITRO.KN_set_newpt_callback`.

All other keyword arguments will be passed to KNITRO as an option.
See https://www.artelys.com/docs/knitro/3_referenceManual/userOptions.html for the list of options accepted.
"""
knitro(nlp::AbstractNLPModel, args...; kwargs...) =
  _knitro(_is_general_nlp(nlp), nlp, args...; kwargs...)

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

include("nlp.jl")
include("nls.jl")

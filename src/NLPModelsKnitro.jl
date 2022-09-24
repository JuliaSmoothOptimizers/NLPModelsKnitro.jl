module NLPModelsKnitro

using KNITRO

export knitro, KnitroSolver, finalize, setparams!

"""
    output = knitro(nlp; kwargs...)

Solves the `NLPModel` problem `nlp` using KNITRO.

For advanced usage, first define a `KnitroSolver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = KnitroSolver(nlp)
    solve!(solver, nlp; kwargs...)
    solve!(solver, nlp, stats; kwargs...)

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
function knitro end

"""
    KnitroSolver(::Val{Bool}, nlp; kwargs...,)

Returns a `KnitroSolver` structure to solve the problem `nlp` with `knitro!`.

Knitro does not accept least-squares problems with constraints other than bounds. 
If an NLSModel has constraints other than bounds, we convert it to a FeasibilityFormNLS.
The first argument is `Val(false)` if the problem has been converted, and `Val(true)` otherwise.

For the possible `kwargs`, we refer to `knitro`.
"""
mutable struct KnitroSolver
  kc
end

const KNITRO_VERSION = KNITRO.KNITRO_VERSION
if KNITRO_VERSION == v"0.0.0"
  @error "KNITRO is not installed correctly"
else
  include("api.jl")
end

end # module

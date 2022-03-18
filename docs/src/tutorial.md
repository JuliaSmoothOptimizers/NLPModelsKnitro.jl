# Tutorial

NLPModelsKnitro is a thin KNITRO wrapper for NLPModels. In this tutorial we show examples of problems created with NLPModels and solved with KNITRO.
```@contents
Pages = ["tutorial.md"]
```

Calling KNITRO is simple:
```julia
`output = knitro(nlp; kwargs...)`

Solves the `NLPModel` problem `nlp` using KNITRO.

# Optional keyword arguments
* `x0`: a vector of size `nlp.meta.nvar` to specify an initial primal guess
* `y0`: a vector of size `nlp.meta.ncon` to specify an initial dual guess for the general constraints
* `z0`: a vector of size `nlp.meta.nvar` to specify initial multipliers for the bound constraints
* `callback`: a user-defined `Function` called by KNITRO at each iteration.

For more information on callbacks, see [https://www.artelys.com/docs/knitro/2_userGuide/callbacks.html](https://www.artelys.com/docs/knitro/2_userGuide/callbacks.html) and
the docstring of `KNITRO.KN_set_newpt_callback`.

All other keyword arguments will be passed to KNITRO as an option.
See [https://www.artelys.com/docs/knitro/3_referenceManual/userOptions.html](https://www.artelys.com/docs/knitro/3_referenceManual/userOptions.html) for the list of options accepted.
```

## Simple problems

Let's create an NLPModel for the Rosenbrock function
```math
f(x) = (x_1 - 1)^2 + 100 (x_2 - x_1^2)^2
```
and solve it with KNITRO:
```julia
using ADNLPModels, NLPModelsKnitro

nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
stats = knitro(nlp)
print(stats)
```

For comparison, we present the same problem and output using JuMP:
```julia
using JuMP, KNITRO

model = Model(KNITRO.Optimizer)
x0 = [-1.2; 1.0]
@variable(model, x[i=1:2], start=x0[i])
@NLobjective(model, Min, (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2)
optimize!(model)
```

Here is an example with a constrained problem:
```julia
n = 10
x0 = ones(n)
x0[1:2:end] .= -1.2
nlp = ADNLPModel(x -> sum((x[i] - 1)^2 + 100 * (x[i+1] - x[i]^2)^2 for i = 1:n-1), x0,
                 x -> [3 * x[k+1]^3 + 2 * x[k+2] - 5 + sin(x[k+1] - x[k+2]) * sin(x[k+1] + x[k+2]) +
                       4 * x[k+1] - x[k] * exp(x[k] - x[k+1]) - 3 for k = 1:n-2],
                 zeros(n-2), zeros(n-2))
stats = knitro(nlp, outlev=0)
print(stats)
```

## Return value

The return value of `knitro` is a `GenericExecutionStats` from `SolverTools`. It contains basic information on the solution returned by the solver.
In addition to the built-in fields of `GenericExecutionStats`, we store the following subfields in the `solver_specific` field:

- `multipliers_con`: constraints multipliers;
- `multipliers_L`: variables lower-bound multipliers;
- `multipliers_U`: variables upper-bound multipliers;
- `internal_msg`: detailed KNITRO output message.

Here is an example using the constrained problem solve:
```julia
stats.solver_specific[:internal_msg]
```

## Manual input

In this section, we work through an example where we specify the problem and its derivatives manually. For this, we need to implement the following `NLPModel` API methods:
- `obj(nlp, x)`: evaluate the objective value at `x`;
- `grad!(nlp, x, g)`: evaluate the objective gradient at `x`;
- `cons!(nlp, x, c)`: evaluate the vector of constraints, if any;
- `jac_structure!(nlp, rows, cols)`: fill `rows` and `cols` with the spartity structure of the Jacobian, if the problem is constrained;
- `jac_coord!(nlp, x, vals)`: fill `vals` with the Jacobian values corresponding to the sparsity structure returned by `jac_structure!()`;
- `hess_structure!(nlp, rows, cols)`: fill `rows` and `cols` with the spartity structure of the lower triangle of the Hessian of the Lagrangian;
- `hess_coord!(nlp, x, y, vals; obj_weight=1.0)`: fill `vals` with the values of the Hessian of the Lagrangian corresponding to the sparsity structure returned by `hess_structure!()`, where `obj_weight` is the weight assigned to the objective, and `y` is the vector of multipliers.

The model that we implement is a logistic regression modelq. We consider the model ``h(\beta; x) = (1 + e^{-\beta^Tx})^{-1}``, and the loss function
```math
\ell(\beta) = -\sum_{i = 1}^m y_i \ln h(\beta; x_i) + (1 - y_i) \ln(1 - h(\beta; x_i))
```
with regularization ``\lambda \|\beta\|^2 / 2``.

```julia
using DataFrames, LinearAlgebra, NLPModels, NLPModelsKnitro, Random

mutable struct LogisticRegression <: AbstractNLPModel{Float64, Vector{Float64}}
  X :: Matrix
  y :: Vector
  λ :: Real
  meta :: NLPModelMeta{Float64, Vector{Float64}} # required by AbstractNLPModel
  counters :: Counters # required by AbstractNLPModel
end

function LogisticRegression(X, y, λ = 0.0)
  m, n = size(X)
  meta = NLPModelMeta(n, name="LogisticRegression", nnzh=div(n * (n+1), 2) + n) # nnzh is the length of the coordinates vectors
  return LogisticRegression(X, y, λ, meta, Counters())
end

function NLPModels.obj(nlp :: LogisticRegression, β::AbstractVector)
  hβ = 1 ./ (1 .+ exp.(-nlp.X * β))
  return -sum(nlp.y .* log.(hβ .+ 1e-8) .+ (1 .- nlp.y) .* log.(1 .- hβ .+ 1e-8)) + nlp.λ * dot(β, β) / 2
end

function NLPModels.grad!(nlp :: LogisticRegression, β::AbstractVector, g::AbstractVector)
  hβ = 1 ./ (1 .+ exp.(-nlp.X * β))
  g .= nlp.X' * (hβ .- nlp.y) + nlp.λ * β
end

function NLPModels.hess_structure!(nlp :: LogisticRegression, rows::AbstractVector{<: Integer}, cols::AbstractVector{<: Integer})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1 : nlp.meta.nnzh] .= [getindex.(I, 1); 1:n]
  cols[1 : nlp.meta.nnzh] .= [getindex.(I, 2); 1:n]
  return rows, cols
end

function NLPModels.hess_coord!(nlp :: LogisticRegression, β::AbstractVector, vals::AbstractVector; obj_weight=1.0, y=Float64[])
  n, m = nlp.meta.nvar, length(nlp.y)
  hβ = 1 ./ (1 .+ exp.(-nlp.X * β))
  fill!(vals, 0.0)
  for k = 1:m
    hk = hβ[k]
    p = 1
    for j = 1:n, i = j:n
      vals[p] += obj_weight * hk * (1 - hk) * nlp.X[k,i] * nlp.X[k,j]
      p += 1
    end
  end
  vals[nlp.meta.nnzh+1:end] .= nlp.λ * obj_weight
  return vals
end

Random.seed!(0)

# Training set
m = 1000
df = DataFrame(:age => rand(18:60, m), :salary => rand(40:180, m) * 1000)
df[!, :buy] = (df.age .> 40 .+ randn(m) * 5) .| (df.salary .> 120_000 .+ randn(m) * 10_000)

X = [ones(m) df.age df.age.^2 df.salary df.salary.^2 df.age .* df.salary]
y = df.buy

λ = 1.0e-2
nlp = LogisticRegression(X, y, λ)
stats = knitro(nlp, outlev=0)
β = stats.solution

# Test set - same generation method
m = 100
df = DataFrame(:age => rand(18:60, m), :salary => rand(40:180, m) * 1000)
df[!, :buy] = (df.age .> 40 .+ randn(m) * 5) .| (df.salary .> 120_000 .+ randn(m) * 10_000)

X = [ones(m) df.age df.age.^2 df.salary df.salary.^2 df.age .* df.salary]
hβ = 1 ./ (1 .+ exp.(-X * β))
ypred = hβ .> 0.5

acc = count(df.buy .== ypred) / m
println("acc = $acc")
```

```julia
using Plots
gr()

f(a, b) = dot(β, [1.0; a; a^2; b; b^2; a * b])
P = findall(df.buy .== true)
scatter(df.age[P], df.salary[P], c=:blue, m=:square)
P = findall(df.buy .== false)
scatter!(df.age[P], df.salary[P], c=:red, m=:xcross, ms=7)
contour!(range(18, 60, step=0.1), range(40_000, 180_000, step=1.0), f, levels=[0.5])
png("ex3")
```

```
using Plots
gr()

f(a, b) = dot(β, [1.0; a; a^2; b; b^2; a * b])
P = findall(df.buy .== true)
scatter(df.age[P], df.salary[P], c=:blue, m=:square)
P = findall(df.buy .== false)
scatter!(df.age[P], df.salary[P], c=:red, m=:xcross, ms=7)
contour!(range(18, 60, step=0.1), range(40_000, 180_000, step=1.0), f, levels=[0.5])
```

![](assets/ex3.png)

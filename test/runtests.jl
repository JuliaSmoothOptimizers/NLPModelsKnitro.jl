using LinearAlgebra
using Test

using KNITRO

using NLPModels, NLPModelsKnitro

function test_unconstrained()
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  stats = knitro(nlp)
  @test isapprox(stats.solution, [1.0; 1.0], rtol=1e-6)
  @test stats.status == :first_order
end

function test_qp()
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - 3)^2, zeros(2),
                   c=x->[sum(x) - 1.0], lcon=[0.0], ucon=[0.0])
  stats = knitro(nlp)
  @test isapprox(stats.solution, [-1.4; 2.4], rtol=1e-6)
  @test stats.iter == 1
  @test stats.status == :first_order
end

function test_constrained()
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - 3)^2, zeros(2),
                   c=x->[dot(x, x)], lcon=[0.0], ucon=[1.0])
  stats = knitro(nlp)
  @test isapprox(stats.solution, [0.11021046172567574, 0.9939082725775202], rtol=1e-6)
  @test stats.status == :first_order

  # test with a good primal-dual initial guess
  x0 = copy(stats.solution)
  y0 = copy(stats.solver_specific[:multipliers_con])
  z0 = copy(stats.solver_specific[:multipliers_L])
  stats = knitro(nlp, x0=x0, y0=y0, z0=z0)
  @test stats.status == :first_order
  @test isapprox(stats.solution, [0.11021046172567574, 0.9939082725775202], rtol=1e-6)
  @test stats.iter == 2
end

function test_with_params()
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  stats = knitro(nlp, opttol=1e-12, presolve=0)
  @test isapprox(stats.solution, [1.0; 1.0], rtol=1e-6)
  @test stats.status == :first_order
end

function test_with_callback()
  function callback(kc, x, lambda_, userParams)
    if KNITRO.KN_get_number_iters(kc) > 1
      return KNITRO.KN_RC_USER_TERMINATION
    end
    return 0
  end
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  stats = knitro(nlp, opttol=1e-12, callback=callback)
  @test stats.solver_specific[:internal_msg] == KNITRO.KN_RC_USER_TERMINATION
  @test stats.iter == 2
  @test stats.status == :exception
end

function test_unconstrained_nls()
  F_Rosen(x) = [x[1] - 1; 10 * (x[2] - x[1]^2)]
  nls = ADNLSModel(F_Rosen, [-1.2; 1.0], 2)
  stats = knitro(nls)
  @test isapprox(stats.objective, 0, atol=1.0e-6)
  @test isapprox(stats.solution, ones(2), rtol=1e-6)
  @test stats.status == :first_order
end

function test_larger_unconstrained_nls()
  n = 100
  F_larger(x) = [[10 * (x[i+1] - x[i]^2) for i = 1:n-1]; [x[i] - 1 for i = 1:n-1]]
  nls = ADNLSModel(F_larger, 0.9 * ones(n), 2 * (n-1))  # there are local solutions other than ones(n)
  stats = knitro(nls)
  @test isapprox(stats.objective, 0, atol=1.0e-6)
  @test isapprox(stats.solution, ones(n), rtol=1e-6)
  @test stats.status == :first_order

  # test with a good primalinitial guess
  x0 = copy(stats.solution)
  stats = knitro(nls, x0=x0)
  @test isapprox(stats.objective, 0, atol=1.0e-6)
  @test isapprox(stats.solution, ones(n), rtol=1e-6)
  @test stats.status == :first_order
  @test stats.iter == 0
end

function test_constrained_nls()
  n = 3
  F_larger(x) = [[10 * (x[i+1] - x[i]^2) for i = 1:n-1]; [x[i] - 1 for i = 1:n-1]]
  c_quad(x) = [sum(x.^2) - 5; prod(x) - 2]
  nls = ADNLSModel(F_larger, [0.5; 1.0; 1.5], 2 * (n-1), c=c_quad, lcon=zeros(2), ucon=zeros(2))
  stats = knitro(nls, opttol=1e-12)
  # this constrained NLS problem will have been converted to a FeasibilityFormNLS; extract the solution
  x = stats.solution[1:n]
  @test isapprox(x, [1.0647319483656656, 1.21502560462289, 1.5459814546883264], rtol=1e-5)
  @test stats.status == :first_order
end

test_unconstrained()
test_qp()
test_constrained()
test_with_params()
test_with_callback()

test_unconstrained_nls()
test_larger_unconstrained_nls()
test_constrained_nls()

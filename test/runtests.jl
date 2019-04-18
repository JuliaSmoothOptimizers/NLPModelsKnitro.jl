using NLPModelsKnitro, NLPModels, KNITRO, Test

function test_unconstrained()
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  stats = knitro(nlp)
  @test isapprox(stats.solution, [1.0; 1.0], rtol=1e-6)
end

function test_constrained()
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - 3)^2, zeros(2),
                   c=x->[sum(x) - 1.0], lcon=[0.0], ucon=[0.0])
  stats = knitro(nlp)
  @test isapprox(stats.solution, [-1.4; 2.4], rtol=1e-6)
end

function test_with_params()
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  stats = knitro(nlp, opttol=1e-12, presolve=0)
  @test isapprox(stats.solution, [1.0; 1.0], rtol=1e-6)
end

function test_with_callback()
  function callback(kc, x, lambda_, userParams)
    if KNITRO.KN_get_number_iters(userParams) > 1
      return KNITRO.KN_RC_USER_TERMINATION
    end
    return 0
  end
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  stats = knitro(nlp, opttol=1e-12, callback=callback)
  @test stats.solver_specific[:internal_msg] == KNITRO.KN_RC_USER_TERMINATION
end

function test_unconstrained_nls()
  F_Rosen(x) = [x[1] - 1; 10 * (x[2] - x[1]^2)]
  nls = ADNLSModel(F_Rosen, [-1.2; 1.0], 2)
  stats = knitro(nls)
  @test isapprox(stats.objective, 0, atol=1.0e-6)
  @test isapprox(stats.solution, ones(2), rtol=1e-6)
end

function test_larger_unconstrained_nls()
  n = 100
  F_larger(x) = [[10 * (x[i+1] - x[i]^2) for i = 1:n-1]; [x[i] - 1 for i = 1:n-1]]
  nls = ADNLSModel(F_larger, 0.9 * ones(n), 2 * (n-1))  # there are local solutions other than ones(n)
  stats = knitro(nls)
  @test isapprox(stats.objective, 0, atol=1.0e-6)
  @test isapprox(stats.solution, ones(n), rtol=1e-6)
end

function test_constrained_nls()
  n = 3
  F_larger(x) = [[10 * (x[i+1] - x[i]^2) for i = 1:n-1]; [x[i] - 1 for i = 1:n-1]]
  c_quad(x) = [sum(x.^2) - 5; prod(x) - 2]
  nls = ADNLSModel(F_larger, [0.5; 1.0; 1.5], 2 * (n-1), c=c_quad, lcon=zeros(2), ucon=zeros(2))
  stats = knitro(nls, opttol=1e-12)
  # this constrained NLS problem will have been converted to a FeasibilityFormNLS; extract the solution
  x = stats.solution[1:n]
  @test isapprox(x, [1.06473, 1.21503, 1.54598], rtol=1e-5)
end

test_unconstrained()
test_constrained()
test_with_params()
test_with_callback()

test_unconstrained_nls()
test_larger_unconstrained_nls()
test_constrained_nls()

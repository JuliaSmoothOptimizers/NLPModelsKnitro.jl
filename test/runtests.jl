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
  # @test stats.solver_specific[:internal_msg] == KNITRO.KN_RC_USER_TERMINATION  # strangely, this fails!
end

test_unconstrained()
test_constrained()
test_with_params()
test_with_callback()

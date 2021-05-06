module NLPModelsKnitro

using KNITRO

const KNITRO_VERSION = KNITRO.KNITRO_VERSION
if KNITRO_VERSION == v"0.0.0"
  @error "KNITRO is not installed correctly"
else
  include("api.jl")
end

end # module

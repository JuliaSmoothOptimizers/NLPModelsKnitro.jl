using Documenter
using NLPModelsKnitro

makedocs(
  modules = [NLPModelsKnitro],
  doctest = true,
  strict = true,
  format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
    assets = ["assets/style.css"],
  ),
  sitename = "NLPModelsKnitro.jl",
  pages = Any["Home" => "index.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md"]
)

if "deploy" in ARGS
  include("../../faketravis.jl")
end

deploydocs(
  deps = nothing,
  make = nothing,
  repo = "github.com/JuliaSmoothOptimizers/NLPModelsKnitro.jl.git",
  target = "build",
  branch = "gh-pages",
  devbranch = "main",
  push_preview = true,
)

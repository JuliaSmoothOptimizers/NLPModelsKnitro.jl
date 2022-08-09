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
  pages = Any["Home" => "index.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md"],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/NLPModelsKnitro.jl.git",
  devbranch = "main",
  push_preview = true,
)

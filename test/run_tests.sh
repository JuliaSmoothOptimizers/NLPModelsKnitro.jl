#!/bin/bash

# Specify the module to test (e.g "HSL")
julia -E 'using Pkg; module_name = "NLPModelsKnitro"; Pkg.activate("test_env"); Pkg.develop(PackageSpec(url=joinpath("."))); Pkg.test(module_name)' &> test_results.txt

# Create the gist and create comment on PR:
julia test/send_gist_url.jl

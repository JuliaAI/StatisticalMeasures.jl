import StatisticalMeasuresBase as API
using StatisticalMeasures
using Test
using ScientificTypes
import ScientificTypes as ST
using Statistics
using CategoricalArrays
import StableRNGs.StableRNG
using MLUtils
using OrderedCollections
using CategoricalDistributions
using LinearAlgebra
import Distributions

const CM = ConfusionMatrices

# because tests were developed before measures were required to be directly callable and
# before `call` was removed from StatisticalMeasuresBase:
call(m, args...) = m(args...)

srng(n=123) = StableRNG(n)

@testset "tools.jl" begin
    include("tools.jl")
end

@testset "functions.jl" begin
    include("functions.jl")
end

@testset "confusion_matrices.jl" begin
    include("confusion_matrices.jl")
end

@testset "roc.jl" begin
    include("roc.jl")
end

@testset "continuous.jl" begin
    include("continuous.jl")
end

@testset "finite.jl" begin
    include("finite.jl")
end

@testset "probabilistic.jl" begin
    include("probabilistic.jl")
end

@testset "LossFunctionsExt.jl" begin
    include("LossFunctionsExt.jl")
end

@testset "ScientificTypesExt.jl" begin
    include("ScientificTypesExt.jl")
end

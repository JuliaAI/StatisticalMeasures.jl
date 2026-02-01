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
import StatsBase: corspearman, randperm

const CM = ConfusionMatrices

# because tests were developed before measures were required to be directly callable and
# before `call` was removed from StatisticalMeasuresBase:
call(m, args...) = m(args...)

srng(n=123) = StableRNG(n)


test_files = [
    "tools.jl",
    "functions.jl",
    "confusion_matrices.jl",
    "roc.jl",
    "precision_recall.jl",
    "continuous.jl",
    "finite.jl",
    "probabilistic.jl",
    "LossFunctionsExt.jl",
    "ScientificTypesExt.jl",
#    "registry.jl",
]

files = isempty(ARGS) ? test_files : ARGS

for file in files
    quote
        @testset $file begin
            include($file)
        end
    end |> eval
end

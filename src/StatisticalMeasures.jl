module StatisticalMeasures

using Statistics
using MacroTools
using CategoricalArrays
using CategoricalDistributions
using OrderedCollections
using ScientificTypesBase
using StatisticalMeasuresBase
import StatisticalMeasuresBase as API
import StatisticalMeasuresBase.unwrap
import LearnAPI
using LinearAlgebra
using StatsBase
import Distributions
using PrecompileTools
using REPL # needed for `Base.Docs.doc`

const SM = "StatisticalMeasures"
const CatArrOrSub{T, N} =
    Union{
        CategoricalArrays.CategoricalArray{T, N},
        SubArray{T, N, <:CategoricalArrays.CategoricalArray},
    }
const NonMissingCatArrOrSub{T,N,R} = Union{
    CategoricalArrays.CategoricalArray{T, N, R, <:Any, <:Any, Union{}},
    SubArray{
        <:Any,
        N,
        <:CategoricalArrays.CategoricalArray{T, <:Any, R, <:Any, <:Any, Union{}},
    },
}

include("tools.jl")
include("functions.jl")
include("confusion_matrices.jl")
include("roc.jl")
include("precision_recall.jl")
include("docstrings.jl")
include("registry.jl")
include("continuous.jl")
include("finite.jl")
include("probabilistic.jl")
include("precompile.jl")

const MEASURES_FOR_EXPORT = let measures = measures()
    ret = Symbol[]
    for C in keys(measures)
        push!(ret, Symbol(C))
        for alias  in measures[C].aliases
            alias == "precision" && continue
            push!(ret, Symbol(alias))
        end
    end
    ret
end

for m in MEASURES_FOR_EXPORT
    @eval export $m
end

# re-exporting from OrderedCollections:
export LittleDict

# re-exporting from StatisticalMeasuresBase:
export measures,
    measurements,
    aggregate,
    unfussy,
    robust_measure,
    Measure,
    multimeasure,
    supports_missings_measure,
    fussy_measure

export Functions, ConfusionMatrices, NoAvg, MacroAvg, MicroAvg
export roc_curve, precision_recall_curve

end

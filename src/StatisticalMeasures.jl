module StatisticalMeasures 

using Statistics
using MacroTools
import CategoricalArrays
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

export Functions, ConfusionMatrices, NoAvg, MacroAvg, MicroAvg, roc_curve

#tod look out for MLJBase.aggregate called on scalars, which is not supported here.
#todo in mljbase, single(measure, array1, array2)

#todo need a show(::Measure)
#todo `is_measure_type` in MLJBase is not provided here

#todo: following needs adding to section on continuous measures
# _scale(x, w::Arr, i) = x*w[i]
# _scale(x, ::Nothing, i::Any) = x

#todo: _skipinvalid from MLJBase/src/data/data.jl is needed for balanced accuracy, barring
# a refactor of that measure to use `skipinvalid` as provided in this package.

#todo: look for uses of aggregation of dictionaries in MLJBase, which is no longer
# supported, or add support.

end

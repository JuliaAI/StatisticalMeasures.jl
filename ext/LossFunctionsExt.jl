module LossFunctionsExt

import LossFunctions: SupervisedLoss, MarginLoss, DistanceLoss
import StatisticalMeasuresBase as API
import StatisticalMeasuresBase.unwrap
import LearnAPI
using StatisticalMeasuresBase
import ScientificTypesBase: Finite, Infinite
import Distributions.pdf
import CategoricalDistributions.UnivariateFiniteArray
import StatisticalMeasures.NonMissingCatArrOrSub

# these types only used in trying to catch MarginLoss wrappers amenable to performance
# boost given later:
const S{L<:SupervisedLoss} = API.Measure{L}
const Scalar{L<:SupervisedLoss} = Union{
    S{L},
    API.Wrapper{<:S{L}},
    API.Wrapper{<:API.Wrapper{<:S{L}}},
    API.Wrapper{<:API.Wrapper{<:API.Wrapper{<:S{L}}}},
}
const Vec{L<:SupervisedLoss} = API.Multimeasure{<:Scalar{L}}

# recursive unwrapping for nested wraps of functions from LossFunctions.jl (which do not
# subtype `Measure`):
loss(measure::StatisticalMeasuresBase.Wrapper) = loss(API.unwrap(measure))
loss(measure) = measure

# display:
# Base.show(io::IO, mime::MIME"text/plain", measure::LossFunctionType) =
#     show(io, mime, loss(measure))
# Base.show(io::IO, measure::LossFunctionType) = show(io, loss(measure))

# # DISTANCE LOSS TYPE

@trait(
    S{<:DistanceLoss},
    kind_of_proxy=LearnAPI.LiteralTarget(),
    orientation=Loss(),
    external_aggregation_mode = Mean(),
)

# not inherited from scalar:
API.observation_scitype(::Vec{<:DistanceLoss}) = Infinite

(loss::API.Measure{<:DistanceLoss})(ŷ, y) = unwrap(loss)(ŷ, y)


# # MARGIN LOSS TYPE

@trait(
    S{<:MarginLoss},
    kind_of_proxy=LearnAPI.Distribution(),
    orientation=Loss(),
    external_aggregation_mode = Mean(),
)

# not inherited from scalar:
API.observation_scitype(::Vec{<:MarginLoss}) = Finite{2}

# transform from [0, 1] (probabilities) to [-1, +1] (LossFunction.jl scores):
score(p) = 2p - 1

(loss::API.Measure{<:MarginLoss})(ŷ, y) = unwrap(loss)(score(pdf(ŷ, y)), 1)

# To get performant broadasting of `pdf` in case of no `missing` values and
# `UnivariateFiniteArray` distibutions, for loss functions wrapped as multimeasures:
function unweighted(measure::Vec{<:MarginLoss}, ŷ, y)
    probs_of_observed =  broadcast(pdf, ŷ, y)
    loss = LossFunctionsExt.loss(measure)
    loss.(score.(probs_of_observed), 1)
end
(measure::Vec{<:MarginLoss})(
    ŷ::UnivariateFiniteArray,
    y::NonMissingCatArrOrSub,
    weight_args...,
) = aggregate(
    unweighted(measure, ŷ, y),
    weights = API.CompositeWeights(y, weight_args...),
)
API.measurements(
    measure::Vec{<:MarginLoss},
    ŷ::UnivariateFiniteArray,
    y::NonMissingCatArrOrSub,
    weight_args...,
) = API.weighted(
    unweighted(measure, ŷ, y),
    weights = API.CompositeWeights(y, weight_args...),
)

end # module

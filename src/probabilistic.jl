const DOC_DISTRIBUTIONS =
"""
In the case the predictions `ŷ` are continuous probability
distributions, such as `Distributions.Normal`, replace the above sum
with an integral, and interpret `p` as the probablity density
function. In case of discrete distributions over the integers, such as
`Distributions.Poisson`, sum over all integers instead of `C`.
"""
const WITH_L2NORM_CONTINUOUS =
    [@eval(Distributions.$d) for d in [
        :Chisq,
        :Gamma,
        :Beta,
        :Chi,
        :Cauchy,
        :Normal,
        :Uniform,
        :Logistic,
        :Exponential]]

const WITH_L2NORM_COUNT =
    [@eval(Distributions.$d) for d in [
        :Poisson,
        :DiscreteUniform,
        :DiscreteNonParametric]]

const WITH_L2NORM = vcat([UnivariateFinite, ],
                         WITH_L2NORM_CONTINUOUS,
                         WITH_L2NORM_COUNT)

const ERR_L2_NORM = ArgumentError(
    "Distribution not supported by measure. "*
    "Supported distributions are "*
    join(string.(map(s->"`$s`", WITH_L2NORM)), ", ", ", and "))

# for the spherical score:
const ERR_UNSUPPORTED_ALPHA = ArgumentError(
    "Only `alpha = 2` is supported, unless scoring a `Finite` target. ")

# Extra check for L2 norm based proper scoring rules
function l2_check(measure, yhat, y, weight_args...)

    # We attempt to extract the type of distribution from the eltype of `yhat` first:
    D = nonmissing(eltype(yhat))

    # It can happen that this won't actually work (e.g., the type is too abstract; see the
    # `l2_check` tests). So in that case we try to find a non-missing element from which
    # to extract the type directly:
    D <: Distributions.Distribution || D <: UnivariateFinite || begin
        yhat_clean = skipmissing(yhat)
        D = isempty(yhat_clean) ? Nothing : typeof(first(yhat_clean))
    end
    D <: Union{Nothing, WITH_L2NORM...} ||
        throw(ERR_L2_NORM)

    if  measure isa SphericalScoreType
        measure.alpha == 2 || throw(ERR_UNSUPPORTED_ALPHA)
    end

    return nothing
end

# ---------------------------------------------------------
# AreaUnderCurve

struct _AreaUnderCurve  end

function (::_AreaUnderCurve)(ŷ::AbstractArray{<:UnivariateFinite}, y)

    # actually this choice theoretically does not matter:
    positive_class = CategoricalArrays.levels(first(ŷ))|> last
    scores = pdf.(ŷ, positive_class)

    return Functions.auc(scores, y, positive_class)

end

const AreaUnderCurve() = _AreaUnderCurve() |> robust_measure |> fussy_measure

const AreaUnderCurveType = API.FussyMeasure{
    <:API.RobustMeasure{<:_AreaUnderCurve}
}

@fix_show AreaUnderCurve::AreaUnderCurveType

# `AreaUnderCurve` will inherit traits from `_AreaUnderCurve`:
@trait(_AreaUnderCurve,
       consumes_multiple_observations=true,
       observation_scitype = Finite{2},
       kind_of_proxy=LearnAPI.Distribution(),
       orientation=Score(),
       external_aggregation_mode=Mean(),
       human_name = "area under the receiver operator characteritic",
)

register(AreaUnderCurve, "auc", "area_under_curve")

const AreaUnderCurveDoc = docstring(
    "AreaUnderCurve()",
    body=
"""
See the [*Recevier operator
chararacteristic*](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) (ROC)
Wikipedia article for a definition. It is expected that `ŷ` be a vector of distributions
over the binary set of unique elements of `y`; specifically, `ŷ` should have eltype
`<:UnivariateFinite` from the CategoricalDistributions.jl package.

$(Functions.DOC_AUC_REF)

Core implementation: [`Functions.auc`](@ref).

$INVARIANT_LABEL
""",
    scitype = "",
    footer="See also [`roc_curve`](@ref). ",
)

"$AreaUnderCurveDoc"
AreaUnderCurve
"$AreaUnderCurveDoc"
const auc = AreaUnderCurve()
"$AreaUnderCurveDoc"
const area_under_curve = auc


# ---------------------------------------------------------------------
# LogScore

struct LogScoreOnScalars{R <: Real}
    tol::R
end

(measure::LogScoreOnScalars)(ŷ, y) =
    log(clamp(pdf(ŷ, y), measure.tol, 1 - measure.tol))

_LogScore(tol) = multimeasure(supports_missings_measure(LogScoreOnScalars(tol)))

const _LogScoreType = API.Multimeasure{
    <:API.SupportsMissingsMeasure{<:LogScoreOnScalars}
}

# to get performant broadasting of `pdf` in case of no `missing` values and
# `UnivariateFiniteArray` distibutions:
function (measure::_LogScoreType)(
    ŷ::UnivariateFiniteArray,
    y::NonMissingCatArrOrSub,
    weight_args...,
    )
    unweighted = log.(clamp.(broadcast(pdf, ŷ, y), measure.tol, 1 - measure.tol))
    aggregate(unweighted, weights = API.CompositeWeights(y, weight_args...))
end
function API.measurements(
    measure::_LogScoreType,
    ŷ::UnivariateFiniteArray,
    y::NonMissingCatArrOrSub,
    weight_args...,
    )
    unweighted = log.(clamp.(broadcast(pdf, ŷ, y), measure.tol, 1 - measure.tol))
    API.weighted(unweighted, weights = API.CompositeWeights(y, weight_args...))
end

LogScore(tol) = fussy_measure(_LogScore(tol) |> robust_measure, extra_check=l2_check)
LogScore(;eps=eps(), tol=eps) = LogScore(tol)

const LogScoreType = API.FussyMeasure{<:API.RobustMeasure{<:_LogScoreType}}

@fix_show LogScore::LogScoreType

# `LogScoreType` inherits traits from `_LogScoreType`:
@trait(
    _LogScoreType,
    consumes_multiple_observations=true,
    kind_of_proxy=LearnAPI.Distribution(),
    # observation_scitype depends on distribution type
    observation_scitype = Union{Missing,Finite,Infinite},
    supports_weights=true,
    supports_class_weights=true,
    orientation=Score(),
    human_name = "log score",
)

register(LogScore, "log_score")

const LogScoreDoc = docstring(
    "LogScore(; tol=eps())",
    body=
"""
The score is a mean of observational scores. More generally, observational scores are
pre-multiplied by the specified weights before averaging. See below for the form that
probabilistic predictions `ŷ` should take. Raw probabilities are clamped away from `0` and
`1`. Specifically, if `p` is the probability mass/density function evaluated at given
observed ground truth observation `η`, then the score for that example is defined as

    log(clamp(p(η), tol, 1 - tol).

For example, for a binary target with "yes"/"no" labels, if the probabilistic prediction
scores 0.8 for a "yes", then for a corresponding ground truth observation of "no", that
example's contribution to the score is `log(0.2)`.

The predictions `ŷ` should be a vector of `UnivariateFinite` distributions from
CategoricalDistritutions.jl, in the case of `Finite` target `y` (a `CategoricalVector`)
and should otherwise be a supported `Distributions.UnivariateDistribution` such as
`Normal` or `Poisson`.

See also [`LogLoss`](@ref), which differs only in sign.
""",
    scitype=DOC_MULTI,
)

"$LogScoreDoc"
LogScore
"$LogScoreDoc"
const log_score = LogScore()


# ---------------------------------------------------------------------
# LogLoss

struct _LogLossType{R<:Real}
    tol::R
end

(measure::_LogLossType)(ŷ, y, weight_args...) =
    -_LogScore(measure.tol)(ŷ, y, weight_args...)
API.measurements(measure::_LogLossType, ŷ, y, weight_args...) =
    -measurements(_LogScore(measure.tol), ŷ, y, weight_args...)

LogLoss(tol) = fussy_measure(_LogLossType(tol) |> robust_measure, extra_check=l2_check)
LogLoss(;eps=eps(), tol=eps) = LogLoss(tol)

const LogLossType = API.FussyMeasure{<:API.RobustMeasure{<:_LogLossType}}

@fix_show LogLoss::LogLossType

# `LogLossType` inherits traits from `_LogLossType`:
@trait(
    _LogLossType,
    consumes_multiple_observations=true,
    can_report_unaggregated=true,
    kind_of_proxy=LearnAPI.Distribution(),
    # observation_scitype depends on distribution type
    observation_scitype = Union{Missing,Finite,Infinite},
    supports_weights=true,
    supports_class_weights=true,
    orientation=Loss(),
    human_name = "log loss",
)

register(LogLoss, "log_loss", "cross_entropy")

const LogLossDoc = docstring(
    "LogLoss(; tol=eps())",
    body=
"""
For details, see [`LogScore`](@ref), which differs only by a sign.
""",
    scitype=DOC_MULTI,
)

"$LogLossDoc"
LogLoss
"$LogLossDoc"
const log_loss = LogLoss()
"$LogLossDoc"
const cross_entropy = log_loss

# ---------------------------------------------------------------------
# BrierScore

struct BrierScoreOnScalars  end

# finite case:
@inline function (measure::BrierScoreOnScalars)(ŷ::UnivariateFinite, y)
    levels = CategoricalArrays.levels(ŷ)
    pvec = broadcast(pdf, ŷ, levels)
    offset = 1 + sum(pvec.^2)
    return 2 * pdf(ŷ, y) - offset
end

# infinite case:
(measure::BrierScoreOnScalars)(ŷ, y) =
        2*pdf(ŷ, y) - Distributions.pdfsquaredL2norm(ŷ)

_BrierScore() = multimeasure(supports_missings_measure(BrierScoreOnScalars()))

const _BrierScoreType = API.Multimeasure{
    <:API.SupportsMissingsMeasure{<:BrierScoreOnScalars}
}

# to get performant broadasting of `pdf` in case of no `missing` values and
# `UnivariateFiniteArray` distibutions:
function (measure::_BrierScoreType)(
    ŷ::UnivariateFiniteArray,
    y::NonMissingCatArrOrSub,
    weight_args...,
    )
    probs = pdf(ŷ, CategoricalArrays.levels(first(ŷ)))
    offset = 1 .+ vec(sum(probs.^2, dims=2))
    unweighted = 2 .* broadcast(pdf, ŷ, y) .- offset
    aggregate(unweighted, weights = API.CompositeWeights(y, weight_args...))
end
function API.measurements(
    measure::_BrierScoreType,
    ŷ::UnivariateFiniteArray,
    y::NonMissingCatArrOrSub,
    weight_args...,
    )
    probs = pdf(ŷ, CategoricalArrays.levels(first(ŷ)))
    offset = 1 .+ vec(sum(probs.^2, dims=2))
    unweighted = 2 .* broadcast(pdf, ŷ, y) .- offset
    API.weighted(unweighted, weights = API.CompositeWeights(y, weight_args...))
end

BrierScore() = fussy_measure(_BrierScore() |> robust_measure, extra_check=l2_check)

const BrierScoreType = API.FussyMeasure{<:API.RobustMeasure{<:_BrierScoreType}}

@fix_show BrierScore::BrierScoreType

# `BrierScoreType` inherits traits from `_BrierScoreType`:
@trait(
    _BrierScoreType,
    consumes_multiple_observations=true,
    kind_of_proxy=LearnAPI.Distribution(),
    # observation_scitype depends on distribution type
    observation_scitype = Union{Missing,Finite,Infinite},
    supports_weights=true,
    supports_class_weights=true,
    orientation=Score(),
    human_name = "brier score",
)

register(BrierScore, "brier_score", "quadratic_score")

const BrierScoreDoc = docstring(
    "BrierScore()",
    body=
"""
The score is a mean of observational scores. More generally, observational scores are
pre-multiplied by the specified weights before averaging. See below for the form that
probabilistic predictions `ŷ` should take.

Convention as in $PROPER_SCORING_RULES

*Finite case.* If `p(η)` is the predicted probability for a
*single* observation `η`, and `C` all possible classes, then the
corresponding score for that example is given by

``2p(η) - \\left(\\sum_{c ∈ C} p(c)^2\\right) - 1``

*Warning.* `BrierScore()` is a "score" in the sense that bigger is
better (with `0` optimal, and all other values negative). In Brier's
original 1950 paper, and many other places, it has the opposite sign,
despite the name. Moreover, the present implementation does not treat
the binary case as special, so that the score may differ in the binary
case by a factor of two from usage elsewhere.

*Infinite case.* Replacing the sum above with an integral does *not*
lead to the formula adopted here in the case of `Continuous` or
`Count` target `y`. Rather the convention in the paper cited above is
adopted, which means returning a score of

``2p(η) - ∫ p(t)^2 dt``

in the `Continuous` case (`p` the probablity density function) or

``2p(η) - ∑_t p(t)^2``

in the `Count` case (`p` the probablity mass function).

The predictions `ŷ` should be a vector of `UnivariateFinite` distributions from
CategoricalDistritutions.jl, in the case of `Finite` target `y` (a `CategoricalVector`)
and should otherwise be a supported `Distributions.UnivariateDistribution` such as
`Normal` or `Poisson`.

See also [`BrierLoss`](@ref), which differs only in sign.
""",
    scitype=DOC_MULTI,
)

"$BrierScoreDoc"
BrierScore
"$BrierScoreDoc"
const brier_score = BrierScore()
"$BrierScoreDoc"
const quadratic_score = brier_score


# ---------------------------------------------------------------------
# BrierLoss

struct _BrierLossType  end

(measure::_BrierLossType)(ŷ, y, weight_args...) =
    -_BrierScore()(ŷ, y, weight_args...)

API.measurements(measure::_BrierLossType, ŷ, y, weight_args...) =
    -measurements(_BrierScore(), ŷ, y, weight_args...)

BrierLoss() = fussy_measure(_BrierLossType() |> robust_measure, extra_check=l2_check)

const BrierLossType = API.FussyMeasure{<:API.RobustMeasure{<:_BrierLossType}}

@fix_show BrierLoss::BrierLossType

# `BrierLossType` inherits traits from `_BrierLossType`:
@trait(
    _BrierLossType,
    consumes_multiple_observations=true,
    can_report_unaggregated=true,
    kind_of_proxy=LearnAPI.Distribution(),
    # observation_scitype depends on distribution type
    observation_scitype = Union{Missing,Finite,Infinite},
    supports_weights=true,
    supports_class_weights=true,
    orientation=Loss(),
    human_name = "brier loss",
)

register(BrierLoss, "brier_loss", "cross_entropy", "quadratic_loss")

const BrierLossDoc = docstring(
    "BrierLoss()",
    body=
"""
For details, see [`BrierScore`](@ref), which differs only by a sign.
""",
    scitype=DOC_MULTI,
)

"$BrierLossDoc"
BrierLoss
"$BrierLossDoc"
const brier_loss = BrierLoss()
"$BrierLossDoc"
const quadratic_loss = BrierLoss()

# ---------------------------------------------------------------------
# SphericalScore

struct SphericalScoreOnScalars{T<:Real}
    alpha::T
end

# finite case:
@inline function (measure::SphericalScoreOnScalars)(
    ŷ::UnivariateFinite,
    y,
    )
    α = measure.alpha
    levels = CategoricalArrays.levels(ŷ)
    probs = broadcast(pdf, ŷ, levels)
    return (pdf(ŷ, y)/norm(probs, α))^(α - 1)
end

# infinite case (ignores parameter `alpha` as unsupported):
(measure::SphericalScoreOnScalars)(ŷ, y) =
    pdf(ŷ, y)/sqrt(Distributions.pdfsquaredL2norm(ŷ))

_SphericalScore(alpha) = SphericalScoreOnScalars(alpha) |>
    supports_missings_measure |> multimeasure

const _SphericalScoreType = API.Multimeasure{
    <:API.SupportsMissingsMeasure{<:SphericalScoreOnScalars}
}

# to compute the α-norm along last dimension:
_norm(A::AbstractArray{<:Any,N}, α) where N =
    sum(x -> x^α, A, dims=N).^(1/α)

# to get performant broadasting of `pdf` in case of no `missing` values and
# `UnivariateFiniteArray` distibutions:
function unweighted(measure::_SphericalScoreType, ŷ, y)
    α = measure.alpha
    alphanorm(A) = _norm(A, α)
    predicted_probs = pdf(ŷ, CategoricalArrays.levels(first(ŷ)))
    (broadcast(pdf, ŷ, y) ./ alphanorm(predicted_probs)).^(α - 1)
end
(measure::_SphericalScoreType)(
    ŷ::UnivariateFiniteArray,
    y::NonMissingCatArrOrSub,
    weight_args...,
) = aggregate(
    unweighted(measure, ŷ, y),
    weights = API.CompositeWeights(y, weight_args...),
)
API.measurements(
    measure::_SphericalScoreType,
    ŷ::UnivariateFiniteArray,
    y::NonMissingCatArrOrSub,
    weight_args...,
) = API.weighted(
    unweighted(measure, ŷ, y),
    weights = API.CompositeWeights(y, weight_args...),
)

SphericalScore(alpha) =
    fussy_measure(robust_measure(_SphericalScore(alpha)); extra_check=l2_check)
SphericalScore(; α=2, alpha=α) = SphericalScore(alpha)

const SphericalScoreType =
    API.FussyMeasure{<:API.RobustMeasure{<:_SphericalScoreType}}

@fix_show SphericalScore::SphericalScoreType

# `SphericalScoreType` inherits traits from `_SphericalScoreType`:
@trait(
    _SphericalScoreType,
    consumes_multiple_observations=true,
    kind_of_proxy=LearnAPI.Distribution(),
    # observation_scitype depends on distribution type
    observation_scitype = Union{Missing,Finite,Infinite},
    supports_weights=true,
    supports_class_weights=true,
    orientation=Score(),
    human_name = "spherical score",
)

register(SphericalScore, "spherical_score")

const SphericalScoreDoc = docstring(
    "SphericalScore()",
    body=
"""
The score is a mean of observational scores. More generally, observational scores are
pre-multiplied by the specified weights before averaging. See below for the form that
probabilistic predictions `ŷ` should take.

Convention as in $PROPER_SCORING_RULES: If `y` takes on a finite
number of classes `C` and `p(y)` is the predicted probability for a
single observation `y`, then the corresponding score for that
example is given by

``p(y)^α / \\left(\\sum_{η ∈ C} p(η)^α\\right)^{1-α} - 1``

where `α` is the measure parameter `alpha`.

$DOC_DISTRIBUTIONS
""",
    scitype=DOC_MULTI,
)

"$SphericalScoreDoc"
SphericalScore
"$SphericalScoreDoc"
const spherical_score = SphericalScore()

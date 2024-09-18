# -----------------------------------------------------------
# ConfusionMatrix (as a measure constructor)

struct _ConfusionMatrix{L}
    levels::Union{Vector{L},Nothing}
    perm::Union{Vector{Int},Nothing}
    rev::Union{Bool,Nothing}
    checks::Bool
    function _ConfusionMatrix(levels::Nothing, perm, rev, checks)
        perm2 = ConfusionMatrices.permutation(perm, rev, levels)
        rev2 = isnothing(perm2) ? rev : nothing
        new{Nothing}(levels, perm2, rev2, checks)
    end
    function _ConfusionMatrix(levels::Vector{L}, perm, rev, checks) where L
        perm2 = ConfusionMatrices.permutation(perm, rev, levels)
        rev2 = isnothing(perm2) ? rev : nothing
       new{L}(levels, perm2, rev2, checks)
    end
end

(m::_ConfusionMatrix)(ŷ, y) = ConfusionMatrices.confmat(
    ŷ,
    y;
    levels=m.levels,
    perm=m.perm,
    rev=m.rev,
    checks=m.checks,
)

ConfusionMatrix(; levels=nothing, perm=nothing, rev=nothing, checks=true) =
   _ConfusionMatrix(levels, perm, rev, checks) |> robust_measure |> fussy_measure
ConfusionMatrix(levels; kwargs...) = ConfusionMatrix(; levels, kwargs...)

const ConfusionMatrixType = API.FussyMeasure{
    <:API.RobustMeasure{<:_ConfusionMatrix}
}

@fix_show ConfusionMatrix::ConfusionMatrixType

# `ConfusionMatrix` inherits traits from `_ConfusionMatrix`:
@trait(
    _ConfusionMatrix,
    consumes_multiple_observations=true,
    kind_of_proxy=LearnAPI.LiteralTarget(),
    observation_scitype=Union{Missing,Finite},
    orientation=Unoriented(),
    external_aggregation_mode=Sum(),
    human_name="confusion matrix",
)

register(ConfusionMatrix, "confmat", "confusion_matrix")

const ConfusionMatrixDoc = docstring(
    "ConfusionMatrix(; levels=nothing, rev=false, perm=nothing, checks=true)",
    body=
"""
See the $(ConfusionMatrices.DOC_REF).

$(ConfusionMatrices.DOC_ORDER_REQUIREMENTS)

""",
    scitype=DOC_FINITE,
    footer=
"""
$(ConfusionMatrices.DOC_OPTIONS())

$(ConfusionMatrices.DOC_OPTIMISED)

For more on the type of object returned and its interface,
see [`ConfusionMatrices.ConfusionMatrix`](@ref).

# Example

```julia

using StatisticalMeasures

y = ["a", "b", "a", "a", "b", "a", "a", "b", "b", "a"]
ŷ = ["b", "a", "a", "b", "a", "b", "b", "b", "a", "a"]

julia> cm = ConfusionMatrix()(ŷ, y)  # or `confmat((ŷ, y)`.

              ┌───────────────────────────┐
              │       Ground Truth        │
┌─────────────┼─────────────┬─────────────┤
│  Predicted  │      a      │      b      │
├─────────────┼─────────────┼─────────────┤
│      a      │      2      │      3      │
├─────────────┼─────────────┼─────────────┤
│      b      │      4      │      1      │
└─────────────┴─────────────┴─────────────┘

julia> cm("a", "b")
3
```

Core algorithm:  [`ConfusionMatrices.confmat`](@ref).
"""
)

"$ConfusionMatrixDoc"
ConfusionMatrix
"$ConfusionMatrixDoc"
const confmat = ConfusionMatrix()
"$ConfusionMatrixDoc"
const confusion_matrix =  confmat

# -------------------------------------------------------
# MisclassificationRate, MultitargetMisclassificationRate

# define both constructors:
@combination(
    MisclassificationRate() = multimeasure(!=),
    kind_of_proxy=LearnAPI.LiteralTarget(),
    observation_scitype=Finite,
    orientation=Loss(),
)


# ## MisclassificationRate

# type:
const MisclassificationRateType = API.Wrapper{<:API.Wrapper{<:API.Wrapper{
    <:API.Wrapper{<:MisclassificationRateOnScalars}}}}

# make callable on confusion matrices:
(::MisclassificationRateType)(cm::ConfusionMatrices.ConfusionMatrix) =
    1.0 - ConfusionMatrices.accuracy(cm)

register(MisclassificationRate, "misclassification_rate", "mcr")

const MisclassificationRateDoc = docstring(
    "MisclassificationRate()",
    scitype=DOC_FINITE,
    body=
"""
That, is, return the proportion of predictions `ŷᵢ` that are different from the
corresponding ground truth `yᵢ`. More generally, average the specified weights over
incorrectly identified observations. Can also be called on a confusion matrix. See
[`ConfusionMatrix`](@ref).

$INVARIANT_LABEL
""",
    footer="See also [`$ConfusionMatrices.ConfusionMatrix`](@ref) and "*
        "[`ConfusionMatrix`](@ref). ",
)

"$MisclassificationRateDoc"
MisclassificationRate
"$MisclassificationRateDoc"
const misclassification_rate = MisclassificationRate()
"$MisclassificationRateDoc"
const mcr = misclassification_rate


# ## MultitargetMisclassificationRate

register(
    MultitargetMisclassificationRate,
    "multitarget_misclassification_rate",
    "multitarget_mcr",
)

const MultitargetMisclassificationRateDoc = docstring(
    "MultitargetMisclassificationRate()",
    body=DOC_MULTITARGET(MisclassificationRate),
    scitype=DOC_FINITE,
)

"$MultitargetMisclassificationRateDoc"
MultitargetMisclassificationRate
"$MultitargetMisclassificationRateDoc"
const multitarget_misclassification_rate = MultitargetMisclassificationRate()
"$MultitargetMisclassificationRateDoc"
const multitarget_mcr = multitarget_misclassification_rate


# -------------------------------------------------------------
# Accuracy, MultitargetAccuracy

# define both constructors:
@combination(
    Accuracy() = multimeasure(==),
    kind_of_proxy=LearnAPI.LiteralTarget(),
    observation_scitype=Finite,
    orientation=Score(),
)

# ## # Accuracy

# type:
const AccuracyType =
    API.Wrapper{<:API.Wrapper{<:API.Wrapper{<:API.Wrapper{<:AccuracyOnScalars}}}}

# make callable on confusion matrices:
(::AccuracyType)(cm::ConfusionMatrices.ConfusionMatrix) =
    ConfusionMatrices.accuracy(cm)

register(Accuracy, "accuracy")

const AccuracyDoc = docstring(
    "Accuracy()",
    scitype=DOC_FINITE,
    body=
"""
That is, compute the proportion of predictions `ŷᵢ` that agree with the corresponding
ground truth `yᵢ`. More generally, average the specified weights over all correctly
predicted observations.  Can also be called on a confusion matrix. See
[`ConfusionMatrix`](@ref).

$INVARIANT_LABEL
""",
footer="See also [`ConfusionMatrices.ConfusionMatrix`](@ref) and "*
    "[`ConfusionMatrix`](@ref). ",
)


"$AccuracyDoc"
Accuracy
"$AccuracyDoc"
const accuracy = Accuracy()


# ## MultitargetAccuracy

register(MultitargetAccuracy, "multitarget_accuracy")

const MultitargetAccuracyDoc = docstring(
    "MultitargetAccuracy()",
    body=DOC_MULTITARGET(Accuracy),
    scitype=DOC_FINITE,
)

"$MultitargetAccuracyDoc"
MultitargetAccuracy
"$MultitargetAccuracyDoc"
const multitarget_accuracy = MultitargetAccuracy()


# -----------------------------------------------------------
# BalancedAccuracy

struct _BalancedAccuracy
    adjusted::Bool
end
BalancedAccuracy(adjusted) =
    _BalancedAccuracy(adjusted) |> robust_measure |> fussy_measure
BalancedAccuracy(; adjusted=false) = BalancedAccuracy(adjusted)

const BalancedAccuracyType = API.FussyMeasure{
    <:API.RobustMeasure{<:_BalancedAccuracy}
}

nlevels(y) = length(unique(skipmissing(y)))
nlevels(y::CategoricalArrays.CatArrOrSub) = length(CategoricalArrays.levels(y))
spread(y) = length(collect(y))/nlevels(y)
function adjust_bac(score, nlevels)
    chance = 1 / nlevels
    score -= chance
    score /= 1 - chance
end

# make callable on confusion matrices:
function (m::BalancedAccuracyType)(cm::ConfusionMatrices.ConfusionMatrix{N}) where N
    score = ConfusionMatrices.balanced_accuracy(cm)
    return m.adjusted ? adjust_bac(score, N) : score
end
function (m::_BalancedAccuracy)(ŷ, y, weights=nothing)
    # we need `spread` below because of the way `Accuracy` handles class
    # weights (it has aggregation mode `Mean()` but we want `IMean()` here).
    balancing_class_weights = if isnothing(weights)
        ypure = skipmissing(y)
        # use of `LittleDict` here gives an abstract type, but class weights must have
        # `<:Real` values:
        Dict(c => spread(ypure)/sum(ypure .== c) for c in unique(ypure)) |> freeze
    else # following sklearn
        ypure, weights_pure = API.skipinvalid(y, weights, skipnan=false)
        # use of `LittleDict` here gives an abstract type, but class weights must have
        # `<:Real` values:
        Dict(c => spread(y)/sum(weights_pure[ypure .== c]) for c in unique(ypure)) |>
            freeze
    end
    score = Accuracy()(ŷ, y, weights, balancing_class_weights)
    return m.adjusted ? adjust_bac(score, nlevels(y)) : score
end

@fix_show BalancedAccuracy::BalancedAccuracyType

# `BalancedAccuracy` inherits traits from `_BalancedAccuracy`:
@trait(
    _BalancedAccuracy,
    consumes_multiple_observations=true,
    kind_of_proxy=LearnAPI.LiteralTarget(),
    observation_scitype=Union{Missing,Finite},
    supports_weights=true,
    orientation=Score(),
    human_name="balanced accuracy",
)

register(
    BalancedAccuracy,
    "balanced_accuracy",
    "bacc",
    "bac",
    "probability_of_correct_classification",
)

const BalancedAccuracyDoc = docstring(
    "BalancedAccuracy(; adjusted=false)",
    body=
"""
This is a variation of [`Accuracy`](@ref) compensating for class imbalance.
See [https://en.wikipedia.org/wiki/Precision_and_recall#Imbalanced_data](https://en.wikipedia.org/wiki/Precision_and_recall#Imbalanced_data).

Setting `adjusted=true` rescales the score in the way prescribed in [L. Mosley
(2013)](https://lib.dr.iastate.edu/etd/13537/): A balanced approach to the multi-class
imbalance problem. PhD thesis, Iowa State University. In the binary case, the adjusted
balanced accuracy is also known as *Youden’s J statistic*, or *informedness*.

Can also be called on a confusion matrix. See [`ConfusionMatrix`](@ref).

$INVARIANT_LABEL
""",
scitype=DOC_FINITE)

"$BalancedAccuracyDoc"
BalancedAccuracy
"$BalancedAccuracyDoc"
const bac = BalancedAccuracy()
"$BalancedAccuracyDoc"
const balanced_accuracy = bac
"$BalancedAccuracyDoc"
const bacc = bac

# ---------------------------------------------------
# Kappa

# type for unrobust measure without argument checks:
struct _Kappa  end

(::_Kappa)(cm::ConfusionMatrices.ConfusionMatrix) =
    ConfusionMatrices.kappa(cm)
(measure::_Kappa)(yhat, y) =
    measure(ConfusionMatrices.confmat(yhat, y))

# constructor for wrapped measure:
Kappa() = API.fussy_measure(API.robust_measure(_Kappa()))

# supertype for measures so-constructed:
const KappaType = API.FussyMeasure{<:API.RobustMeasure{<:_Kappa}}

# Allow callable on confusion matrices:
(measure::KappaType)(cm::ConfusionMatrices.ConfusionMatrix) =
        ConfusionMatrices.kappa(cm)

# these traits will be inherited by `Kappa`:
@trait(
    _Kappa,
    consumes_multiple_observations=true,
    kind_of_proxy=LearnAPI.LiteralTarget(),
    observation_scitype=Union{Missing,Finite},
    supports_weights=true,
    orientation=Score(),
    human_name = "Cohen's κ",
)

# friendly show method:
@fix_show Kappa::KappaType

register(Kappa, "kappa")

const KappaDoc = docstring(
    "Kappa()",
    scitype=DOC_FINITE,
    body=
"""
For details, see the [Cohen's κ](https://en.wikipedia.org/wiki/Cohen%27s_kappa) Wikipedia
article. Can also be called on confusion matrices. See
[`ConfusionMatrix`](@ref).

$INVARIANT_LABEL
""",
    footer="See also [`$ConfusionMatrices.ConfusionMatrix`](@ref) and "*
        "[`ConfusionMatrix`](@ref). \n\nCore algorithm: [`Functions.kappa`](@ref)",
)

"$KappaDoc"
Kappa
"$KappaDoc"
const kappa = Kappa()

# ------------------------------------------------------------------
# MatthewsCorrelation

struct _MatthewsCorrelation  end

(::_MatthewsCorrelation)(cm::ConfusionMatrices.ConfusionMatrix) =
    ConfusionMatrices.matthews_correlation(cm)
(measure::_MatthewsCorrelation)(yhat, y) =
    measure(ConfusionMatrices.confmat(yhat, y))

# constructor for wrapped measure:
MatthewsCorrelation() = _MatthewsCorrelation() |> API.robust_measure |> API.fussy_measure

# supertype for measures so-constructed:
const MatthewsCorrelationType = API.FussyMeasure{
    <:API.RobustMeasure{<:_MatthewsCorrelation}
}

# Allow callable on confusion matrices:
(measure::MatthewsCorrelationType)(cm::ConfusionMatrices.ConfusionMatrix) =
        ConfusionMatrices.matthews_correlation(cm)

# these traits will be inherited by `MatthewsCorrelation`:
@trait(
    _MatthewsCorrelation,
    consumes_multiple_observations=true,
    kind_of_proxy=LearnAPI.LiteralTarget(),
    observation_scitype=Union{Missing,Finite},
    orientation=Score(),
    human_name = "Matthew's correlation",
)

# friendly show method:
@fix_show MatthewsCorrelation::MatthewsCorrelationType

register(MatthewsCorrelation, "matthews_correlation", "mcc")

const MatthewsCorrelationDoc = docstring(
    "MatthewsCorrelation()",
    scitype=DOC_FINITE,
    body=
"""
See the [Wikipedia *Matthew's Correlation*
page](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient).
Can also be called on confusion matrices.  See
[`ConfusionMatrix`](@ref).

$INVARIANT_LABEL
""",
    footer="See also [`$ConfusionMatrices.ConfusionMatrix`](@ref) and "*
        "[`ConfusionMatrix`](@ref).\n\nCore algorithm: [`Functions.matthews_correlation`](@ref)",
)

"$MatthewsCorrelationDoc"
MatthewsCorrelation
"$MatthewsCorrelationDoc"
const mcc = MatthewsCorrelation()
"$MatthewsCorrelationDoc"
const matthews_correlation = mcc


# ==========================================================================
# DETERMINISTIC BINARY PREDICTIONS - ORDER DEPENDENT

const ERR_NONBINARY_LEVELS = ArgumentError(
    "For a binary measure, the number of levels must be two. "
)

# ## multi-class helpers

# wrap `per_class_vector` as dictionary, getting keys from levels of confusion matrix
# `cm`:
dict(per_class_vector, cm) = LittleDict(
    # need `Tuple` here instead of `freeze`, because of
    # https://github.com/JuliaCollections/OrderedCollections.jl/issues/104
    Tuple(ConfusionMatrices.levels(cm)),
    Tuple(per_class_vector),
)

# transform a dict keyed on classes to a vector, using levels of confusion matrix `cm`:
vector(dict, cm) = [dict[c] for c in ConfusionMatrices.levels(cm)]


# ---------------------------------------------------
# FScore

# type for unrobust measure without argument checks:
struct _FScore{T<:Number,L}
    beta::T
    levels::Union{Vector{L},Nothing}
    rev::Union{Bool,Nothing}
    checks::Bool
    function _FScore(beta::T, levels::Nothing, rev, checks) where T
        isnothing(levels) || length(levels) == 2 || throw(ERR_NONBINARY_LEVELS)
        new{T,Nothing}(beta, levels, rev, checks)
    end
    function _FScore(beta::T, levels::Vector{L}, rev, checks) where {T,L}
        isnothing(levels) || length(levels) == 2 || throw(ERR_NONBINARY_LEVELS)
        new{T,L}(beta, levels, rev, checks)
    end
end

(measure::_FScore)(cm::ConfusionMatrices.ConfusionMatrix) =
    ConfusionMatrices.fscore(cm, measure.beta)
(measure::_FScore)(yhat, y) =
    measure(ConfusionMatrices.confmat(
        yhat,
        y;
        levels=measure.levels,
        rev=measure.rev,
        checks=measure.checks,
    ))

# constructor for wrapped measure:
FScore(beta; levels=nothing, rev=nothing, checks=true) =
    _FScore(beta, levels, rev, checks) |> API.robust_measure |> API.fussy_measure
FScore(; β=1.0, beta=β, kwargs...) = FScore(beta; kwargs...)

# supertype for measures so-constructed:
const FScoreType = API.FussyMeasure{<:API.RobustMeasure{<:_FScore}}

# Allow callable on confusion matrices:
(measure::FScoreType)(cm::ConfusionMatrices.ConfusionMatrix) =
        ConfusionMatrices.fscore(cm, measure.beta)

# these traits will be inherited by `FScore`:
@trait(
    _FScore,
    consumes_multiple_observations=true,
    kind_of_proxy=LearnAPI.LiteralTarget(),
    observation_scitype=Union{Missing,OrderedFactor{2}},
    orientation=Score(),
    human_name = "``F_β`` score",
)

# friendly show method:
@fix_show FScore::FScoreType

register(FScore, "f1score")

const FScoreDoc = docstring(
    "FScore(; beta=1.0, levels=nothing, rev=nothing, checks=true)",
    scitype=DOC_ORDERED_FACTOR_BINARY,
    body=
"""
This is the one-parameter generalization, ``F_β``, of the ``F``-measure or balanced
``F``-score. Choose `beta=β` in the range ``[0,∞]``, using `beta > 1` to emphasize recall
([`TruePositiveRate`](@ref)) over precision ([`PositivePredictiveValue`](@ref)). When
`beta = 1`, the score is the harmonic mean of precision and recall. See the [*F1 score*
Wikipedia page](https://en.wikipedia.org/wiki/F-score) for details.

If ordering classes (levels) on the basis of the eltype of `y`, then the *second* level is
the "positive" class. To reverse roles, specify `rev=true`.

$(ConfusionMatrices.DOC_OPTIMISED)

`FScore` mesaures can also be called on a confusion matrix.  See
[`ConfusionMatrix`](@ref).

$(ConfusionMatrices.DOC_OPTIONS(binary=true))
""",
    footer="See also [`$ConfusionMatrices.ConfusionMatrix`](@ref) and "*
        "[`ConfusionMatrix`](@ref).\n\nCore algorithm: [`Functions.fscore`](@ref) ",
)

"$FScoreDoc"
FScore
"$FScoreDoc"
const f1score = FScore()

# ---------------------------------------------------
# TruePositive and its cousins

const TRUE_POSITIVE_AND_COUSINS =
    (:TruePositive, :TrueNegative, :FalsePositive, :FalseNegative,
     :TruePositiveRate, :TrueNegativeRate, :FalsePositiveRate,
     :FalseNegativeRate, :FalseDiscoveryRate, :PositivePredictiveValue,
     :NegativePredictiveValue)

const ORIENTATION_GIVEN_MEASURE = Dict(
    :TruePositive => Score(),
    :TrueNegative => Score(),
    :FalsePositive => Loss(),
    :FalseNegative => Loss(),
    :TruePositiveRate => Score(),
    :TrueNegativeRate => Score(),
    :FalsePositiveRate => Loss(),
    :FalseNegativeRate => Loss(),
    :FalseDiscoveryRate => Loss(),
    :PositivePredictiveValue => Score(),
    :NegativePredictiveValue => Score(),
)

const MODE_GIVEN_MEASURE = Dict(
    :TruePositive => Sum(),
    :TrueNegative => Sum(),
    :FalsePositive => Sum(),
    :FalseNegative => Sum(),
    :TruePositiveRate => Mean(),
    :TrueNegativeRate => Mean(),
    :FalsePositiveRate => Mean(),
    :FalseNegativeRate => Mean(),
    :FalseDiscoveryRate => Mean(),
    :PositivePredictiveValue => Mean(),
    :NegativePredictiveValue => Mean(),
)

const NAME_GIVEN_MEASURE = Dict(
    :TruePositive => "true positive count",
    :TrueNegative => "true negative count",
    :FalsePositive => "false positive count",
    :FalseNegative => "false negative count",
    :TruePositiveRate => "true positive rate",
    :TrueNegativeRate => "true negative rate",
    :FalsePositiveRate => "false positive rate",
    :FalseNegativeRate => "false negative rate",
    :FalseDiscoveryRate => "false discovery rate",
    :PositivePredictiveValue => "positive predictive value",
    :NegativePredictiveValue => "negative predictive value",
)

const ALIASES_GIVEN_MEASURE = Dict(
    :TruePositive => [:true_positive, :truepositive],
    :TrueNegative => [:true_negative, :truenegative],
    :FalsePositive => [:false_positive, :falsepositive],
    :FalseNegative => [:false_negative, :falsenegative],
    :TruePositiveRate => [
        :true_positive_rate,
        :truepositive_rate,
        :tpr,
        :sensitivity,
        :recall,
        :hit_rate,
    ],
    :TrueNegativeRate => [
        :true_negative_rate,
        :truenegative_rate,
        :tnr,
        :specificity,
        :selectivity,
    ],
    :FalsePositiveRate => [:false_positive_rate, :falsepositive_rate, :fpr, :fallout],
    :FalseNegativeRate => [
        :false_negative_rate,
        :falsenegative_rate,
        :fnr,
        :miss_rate,
    ],
    :FalseDiscoveryRate => [:false_discovery_rate, :falsediscovery_rate, :fdr],
    :PositivePredictiveValue => [
        :positive_predictive_value,
        :ppv,
        :positivepredictive_value,
        :precision,
    ],
    :NegativePredictiveValue => [
        :negative_predictive_value,
        :negativepredictive_value,
        :npv,
    ],
)

for Measure in TRUE_POSITIVE_AND_COUSINS

    let Measure_str = string(Measure),
        _Measure = "_$Measure" |> Symbol,
        MeasureType = "$(Measure)Type" |> Symbol,
        MeasureDoc = "$(Measure)Doc" |> Symbol,
        f = snakecase(Measure),
        f_str = string(f),
        orientation = ORIENTATION_GIVEN_MEASURE[Measure],
        aliases = string.(ALIASES_GIVEN_MEASURE[Measure]),
        mode = MODE_GIVEN_MEASURE[Measure],
        name = NAME_GIVEN_MEASURE[Measure],
        program = quote end

        quote
            # type for unrobust measure without argument checks:
            struct $_Measure{L}
                levels::Union{Vector{L},Nothing}
                rev::Union{Bool,Nothing}
                checks::Bool
                function $_Measure(levels::Nothing, rev, checks)
                    isnothing(levels) || length(levels) == 2 || throw(ERR_NONBINARY_LEVELS)
                    new{Nothing}(levels, rev, checks)
                end
                function $_Measure(levels::Vector{L}, rev, checks) where L
                    isnothing(levels) || length(levels) == 2 || throw(ERR_NONBINARY_LEVELS)
                    new{L}(levels, rev, checks)
                end
            end

            (measure::$_Measure)(cm::ConfusionMatrices.ConfusionMatrix) =
                ConfusionMatrices.$f(cm)
            (measure::$_Measure)(yhat, y) =
                measure(ConfusionMatrices.confmat(
                    yhat,
                    y;
                    levels=measure.levels,
                    rev=measure.rev,
                    checks=measure.checks,
                ))

            # constructor for wrapped measure:
            global $Measure(;levels=nothing, rev=nothing, checks=true) =
                $_Measure(levels, rev, checks) |> API.robust_measure |> API.fussy_measure

            # supertype for measures so-constructed:
            global $MeasureType = API.FussyMeasure{<:API.RobustMeasure{<:$_Measure}}

            # Allow callable on confusion matrices:
            (measure::$MeasureType)(cm::ConfusionMatrices.ConfusionMatrix) =
                API.unwrap(API.unwrap(measure))(cm)

            # these traits will be inherited by `$Measure`:
            @trait(
                $_Measure,
                consumes_multiple_observations=true,
                kind_of_proxy=LearnAPI.LiteralTarget(),
                observation_scitype=Union{Missing,OrderedFactor{2}},
                orientation=$orientation,
                human_name=$name,
                external_aggregation_mode=$mode,
            )

            # friendly show method:
            @fix_show $Measure::$MeasureType

            register($Measure, $(aliases...))

            $MeasureDoc = docstring(
                $Measure_str*"(; levels=nothing, rev=nothing, checks=true)",
                scitype=DOC_ORDERED_FACTOR_BINARY,
                body=
                    """
                    When ordering classes (levels) on the basis of the eltype of `y`, the
                    *second* level is the "positive" class. To reverse roles, specify
                    `rev=true`.

                    $(ConfusionMatrices.DOC_OPTIMISED)

                    `m` can also be called on a confusion matrix. See
                    [`ConfusionMatrix`](@ref).

                    $(ConfusionMatrices.DOC_OPTIONS(binary=true))
                    """,
                footer="See also [`Multiclass$($Measure)`](@ref), "*
                    "[`$ConfusionMatrices.ConfusionMatrix`](@ref) and "*
                    "[`ConfusionMatrix`](@ref).\n\nCore algorithm: [`Functions.$($f_str)`](@ref)",

            )

            "$($MeasureDoc)"
            $Measure

        end |> eval

        # define aliases:
        for alias in Symbol.(aliases)
            push!(
                program.args,
                quote
                    "$($MeasureDoc)"
                    const $alias = ($Measure)()
                end,
            )
        end
        eval(program)
    end
end


# ---------------------------------------------------
# MulticlassTruePositive, MulticlassTrueNegative,
# MulticlassFalsePositive, MulticlassFalseNegative

for Measure in [
    :MulticlassTruePositive,
    :MulticlassTrueNegative,
    :MulticlassFalsePositive,
    :MulticlassFalseNegative,
    ]

    let Measure_str = string(Measure),
        BinaryMeasure_str = split(string(Measure), "Multiclass") |> last,
        BinaryMeasure = Symbol(BinaryMeasure_str),
        _Measure = "_$Measure" |> Symbol,
        MeasureType = "$(Measure)Type" |> Symbol,
        MeasureDoc = "$(Measure)Doc" |> Symbol,
        f = snakecase(Measure),
        f_str = string(f),
        orientation = ORIENTATION_GIVEN_MEASURE[BinaryMeasure],
        aliases = map(ALIASES_GIVEN_MEASURE[BinaryMeasure]) do alias
            "multiclass_$alias"
        end,
        mode = MODE_GIVEN_MEASURE[BinaryMeasure],
        name = "multi-class "*NAME_GIVEN_MEASURE[BinaryMeasure],
        program = quote end

        quote
            # type for unrobust measure without argument checks:
            struct $_Measure{
                R<:Union{Vector,LittleDict}, # return type
                L, # level type
                }
                return_type::Type{R}
                levels::Union{Nothing,Vector{L}}
                perm::Union{Nothing,Vector{Int}}
                rev::Union{Nothing,Bool}
                checks::Bool
                function $_Measure(
                    return_type::Type{R},
                    levels::Nothing,
                    perm,
                    rev,
                    checks,
                    ) where R
                    perm2 = ConfusionMatrices.permutation(perm, rev, levels)
                    rev2 = isnothing(perm2) ? rev : nothing
                    new{R,Nothing}(return_type, levels, perm2, rev2, checks)
                end
                function $_Measure(
                    return_type::Type{R},
                    levels::AbstractVector{L},
                    perm,
                    rev,
                    checks,
                    ) where {R,L}
                    perm2 = ConfusionMatrices.permutation(perm, rev, levels)
                    rev2 = isnothing(perm2) ? rev : nothing
                    new{R,L}(return_type, levels, perm2, rev2, checks)
                end
            end

            (measure::$_Measure{Vector})(
                cm::ConfusionMatrices.ConfusionMatrix,
            ) = ConfusionMatrices.$f(cm)
            (measure::$_Measure{LittleDict})(
                cm::ConfusionMatrices.ConfusionMatrix,
            ) = dict(ConfusionMatrices.$f(cm), cm)
            (measure::$_Measure)(yhat, y) =
                measure(ConfusionMatrices.confmat(
                    yhat,
                    y;
                    levels=measure.levels,
                    perm=measure.perm,
                    rev=measure.rev,
                    checks=measure.checks,
                ))

            # constructor for wrapped measure:
            global $Measure(;
                     return_type=LittleDict,
                     levels=nothing,
                     perm=nothing,
                     rev=nothing,
                     checks=true,
                     ) = $_Measure(
                         return_type,
                         levels,
                         perm,
                         rev,
                         checks,
                     ) |> API.robust_measure |> API.fussy_measure

            # supertype for measures so-constructed:
            $MeasureType = API.FussyMeasure{<:API.RobustMeasure{<:$_Measure}}

            # Allow callable on confusion matrices:
            (measure::$MeasureType)(cm::ConfusionMatrices.ConfusionMatrix) =
                (API.unwrap(API.unwrap(measure)))(cm)

            # these traits will be inherited by `$Measure`:
            @trait(
                $_Measure,
                consumes_multiple_observations=true,
                kind_of_proxy=LearnAPI.LiteralTarget(),
                observation_scitype=Union{Missing,Finite},
                orientation=$orientation,
                human_name=$name,
                external_aggregation_mode=$mode,
            )

            # friendly show method:
            @fix_show $Measure::$MeasureType

            register($Measure, $(aliases...))

            $MeasureDoc = docstring(
                $Measure_str*"(; levels=nothing, more_options...)",
                scitype=DOC_ORDERED_FACTOR_BINARY,
                body=
                    """

                    \n\nThis is a one-versus-rest version of the binary measure
                    [`$($BinaryMeasure_str)`](@ref), returning a dictionary keyed on target
                    class (level), or a vector (see options below), instead of a single
                    number, even on binary data.

                    $(ConfusionMatrices.DOC_OPTIMISED)

                    `m` can also be called on a confusion matrix.  Construct confusion
                    matrices using [`ConfusionMatrix`](@ref).

                    """,
                footer="$(ConfusionMatrices.DOC_OPTIONS(return_type=true))\n\n"*
                    "$(ConfusionMatrices.DOC_OPTIMISED)\n\n"*
                    "See also [`$($BinaryMeasure)`](@ref), "*
                    "[`$ConfusionMatrices.ConfusionMatrix`](@ref) and "*
                    "[`ConfusionMatrix`](@ref).\n\nCore algorithm: [`Functions.$($f_str)`](@ref)",
            )

            "$($MeasureDoc)"
            $Measure

        end |> eval

        # defined aliases:
        for alias in Symbol.(aliases)
            push!(
                program.args,
                quote
                    "$($MeasureDoc)"
                    const $alias = ($Measure)()
                end,
            )
        end
        eval(program)
    end
end


# -------------------------------------------------------------------
# MulticlassTruePositiveRate, MulticlassTrueNegativeRate,
# MulticlassFalsePositiveRate, MulticlassFalseNegativeRate,
# MulticlassFalseDiscoveryRate, MulticlassPositivePredictiveValue,
# MulticlassNegativePredictiveValue

const macro_avg = Functions.MacroAvg()
const micro_avg = Functions.MicroAvg()
const no_avg    = Functions.NoAvg()

const WARN_MICRO_IGNORING_WEIGHTS =
    "Class weights are unsupported in micro-averaging. Treating weights as uniform. "

for Measure in [
    :MulticlassTruePositiveRate,
    :MulticlassTrueNegativeRate,
    :MulticlassFalsePositiveRate,
    :MulticlassFalseNegativeRate,
    :MulticlassFalseDiscoveryRate,
    :MulticlassPositivePredictiveValue,
    :MulticlassNegativePredictiveValue,
    ]

    let Measure_str = string(Measure),
        BinaryMeasure_str = split(string(Measure), "Multiclass") |> last,
        BinaryMeasure = Symbol(BinaryMeasure_str),
        _Measure = "_$Measure" |> Symbol,
        MeasureType = "$(Measure)Type" |> Symbol,
        MeasureDoc = "$(Measure)Doc" |> Symbol,
        f = snakecase(Measure),
        f_str = string(f),
        orientation = ORIENTATION_GIVEN_MEASURE[BinaryMeasure],
        aliases = map(ALIASES_GIVEN_MEASURE[BinaryMeasure]) do alias
            "multiclass_$alias"
        end,
        mode = MODE_GIVEN_MEASURE[BinaryMeasure],
        name = "multi-class "*NAME_GIVEN_MEASURE[BinaryMeasure],
        program = quote end

        quote
            # type for unrobust measure without argument checks:
            struct $_Measure{
                A<:Functions.MulticlassAvg,
                R<:Union{Vector,LittleDict}, # return type
                L, # level type
                }
                average::A
                return_type::Type{R}
                levels::Union{Nothing,Vector{L}}
                perm::Union{Nothing,Vector{Int}}
                rev::Union{Nothing,Bool}
                checks::Bool
                function $_Measure(
                    average::A,
                    return_type::Type{R},
                    levels::Nothing,
                    perm,
                    rev,
                    checks,
                    ) where {A,R}
                    perm2 = ConfusionMatrices.permutation(perm, rev, levels)
                    rev2 = isnothing(perm2) ? rev : nothing
                    new{A,R,Nothing}(average, return_type, levels, perm2, rev2, checks)
                end
                function $_Measure(
                    average::A,
                    return_type::Type{R},
                    levels::AbstractVector{L},
                    perm,
                    rev,
                    checks,
                    ) where {A,R,L}
                    perm2 = ConfusionMatrices.permutation(perm, rev, levels)
                    rev2 = isnothing(perm2) ? rev : nothing
                    new{A,R,L}(average, return_type, levels, perm2, rev2, checks)
                end
            end

            # ## `_call` acts on confusion matrices and returns a vector in `NoAvg()` case

            # no class weights
            _call(
                measure::$_Measure{A},
                cm::ConfusionMatrices.ConfusionMatrix,
            ) where A = ConfusionMatrices.$f(cm, measure.average)

            # ✓ class weights:
            _call(
                measure::$_Measure{A},
                cm::ConfusionMatrices.ConfusionMatrix,
                class_weights,
            ) where A = ConfusionMatrices.$f(cm, measure.average, vector(class_weights, cm))
            @inline function _call(
                measure::$_Measure{Functions.MicroAvg},
                cm::ConfusionMatrices.ConfusionMatrix,
                class_weights,
                )
                @warn WARN_MICRO_IGNORING_WEIGHTS
                ConfusionMatrices.$f(cm, Functions.MicroAvg())
            end

            # ## `call` on confusion matrices

            # Cases excluding `NoAvg()`:
            (measure::$_Measure)(
                cm::ConfusionMatrices.ConfusionMatrix,
            )  = _call(measure, cm)
            (measure::$_Measure)(
                cm::ConfusionMatrices.ConfusionMatrix,
                class_weights::AbstractDict,
            ) =  _call(measure, cm, class_weights)

            # `NoAvg()` with `Vector` return type:
            (measure::$_Measure{NoAvg,Vector})(
                cm::ConfusionMatrices.ConfusionMatrix,
            ) = _call(measure, cm)
            (measure::$_Measure{NoAvg,Vector})(
                cm::ConfusionMatrices.ConfusionMatrix,
                class_weights::AbstractDict,
            ) =  _call(measure, cm, class_weights)

            # `NoAvg()` with `LittleDict` return type:
            (measure::$_Measure{NoAvg,LittleDict})(
                cm::ConfusionMatrices.ConfusionMatrix,
            ) = dict(_call(measure, cm), cm)
            (measure::$_Measure{NoAvg,LittleDict})(
                cm::ConfusionMatrices.ConfusionMatrix,
                class_weights::AbstractDict,
            ) = dict(_call(measure, cm, class_weights), cm)

            # ## `call` on `(ŷ, y)`

            #  no class weights:
            (measure::$_Measure)(yhat, y) =
                measure(ConfusionMatrices.confmat(
                    yhat,
                    y;
                    levels=measure.levels,
                    perm=measure.perm,
                    rev=measure.rev,
                    checks=measure.checks,
                ))
            (measure::$_Measure)(yhat, y, class_weights::AbstractDict) =
                measure(ConfusionMatrices.confmat(
                    yhat,
                    y;
                    levels=measure.levels,
                    perm=measure.perm,
                    rev=measure.rev,
                    checks=measure.checks,
                ), class_weights)

            # constructor for wrapped measure:

            $Measure(;
                     average=Functions.MacroAvg(),
                     return_type=LittleDict,
                     levels=nothing,
                     perm=nothing,
                     rev=nothing,
                     checks=true,
                     ) = $_Measure(
                         average,
                         return_type,
                         levels,
                         perm,
                         rev,
                         checks,
                     ) |> API.robust_measure |> API.fussy_measure

            # supertype for measures so-constructed:
            $MeasureType = API.FussyMeasure{<:API.RobustMeasure{<:$_Measure}}

            # Allow callable on confusion matrices:
            (measure::$MeasureType)(
                cm::ConfusionMatrices.ConfusionMatrix,
                class_weights...) =
                    API.unwrap(API.unwrap(measure))(cm, class_weights...)

            # these traits will be inherited by `$Measure`:
            @trait(
                $_Measure,
                consumes_multiple_observations=true,
                kind_of_proxy=LearnAPI.LiteralTarget(),
                observation_scitype=Union{Missing,Finite},
                supports_class_weights=true,
                orientation=$orientation,
                human_name=$name,
                external_aggregation_mode=$mode,
            )

            # friendly show method:
            @fix_show $Measure::$MeasureType

            register($Measure, $(aliases...))

            const $MeasureDoc = docstring(
                $Measure_str*"(; average=macro_avg, levels=nothing, more_options...)",
                scitype=DOC_ORDERED_FACTOR_BINARY,
                body=
                    """

                    \n\nThis is an averaged one-versus-rest version of the binary
                    [`$($BinaryMeasure_str)`](@ref). Or it can return a dictionary keyed
                    on target class (or a vector); see `average` options below.

                    $(ConfusionMatrices.DOC_OPTIMISED)

                    You can also call `m` on confusion matrices. Construct confusion
                    matrices using [`ConfusionMatrix`](@ref).

                    """,
                footer=
                    "$(ConfusionMatrices.DOC_OPTIONS(return_type=true, average=true))\n\n"*
                    "$(ConfusionMatrices.DOC_OPTIMISED)\n\n"*
                    "See also [`$($BinaryMeasure)`](@ref), "*
                    "[`$ConfusionMatrices.ConfusionMatrix`](@ref) and "*
                    "[`ConfusionMatrix`](@ref).\n\nCore algorithm: [`Functions.$($f_str)`](@ref)",
            )

            "$($MeasureDoc)"
            $Measure

        end |> eval

        # defined aliases:
        for alias in Symbol.(aliases)
            push!(
                program.args,
                quote
                    "$($MeasureDoc)"
                    const $alias = ($Measure)()
                end,
            )
        end
        eval(program)

    end
end

# -------------------------------------------------------------------
# MulticlassFScore

# type for unrobust measure without argument checks:
struct _MulticlassFScore{
    A<:Functions.MulticlassAvg,
    R<:Union{Vector,LittleDict}, # return type
    L, # level type
    T, # type of beta
    }
    beta::T
    average::A
    return_type::Type{R}
    levels::Union{Nothing,Vector{L}}
    perm::Union{Nothing,Vector{Int}}
    rev::Union{Nothing,Bool}
    checks::Bool
    function _MulticlassFScore(
        beta::T,
        average::A,
        return_type::Type{R},
        levels::Nothing,
        perm,
        rev,
        checks,
        ) where {A,R,T}
        perm2 = ConfusionMatrices.permutation(perm, rev, levels)
        rev2 = isnothing(perm2) ? rev : nothing
        new{A,R,Nothing,T}(beta, average, return_type, levels, perm2, rev2, checks)
    end
    function _MulticlassFScore(
        beta::T,
        average::A,
        return_type::Type{R},
        levels::AbstractVector{L},
        perm,
        rev,
        checks,
        ) where {A,R,L,T}
        perm2 = ConfusionMatrices.permutation(perm, rev, levels)
        rev2 = isnothing(perm2) ? rev : nothing
        new{A,R,L,T}(beta, average, return_type, levels, perm2, rev2, checks)
    end
end

# ## `_call` acts on confusion matrices and returns a vector in `NoAvg()` case

# no class weights
_call(
    measure::_MulticlassFScore{A},
    cm::ConfusionMatrices.ConfusionMatrix,
) where A = ConfusionMatrices.multiclass_fscore(cm, measure.beta, measure.average)

# ✓ class weights:
_call(
    measure::_MulticlassFScore{A},
    cm::ConfusionMatrices.ConfusionMatrix,
    class_weights,
) where A = ConfusionMatrices.multiclass_fscore(
    cm,
    measure.beta,
    measure.average,
    vector(class_weights, cm),
)
@inline function _call(
    measure::_MulticlassFScore{Functions.MicroAvg},
    cm::ConfusionMatrices.ConfusionMatrix,
    class_weights,
    )
    @warn WARN_MICRO_IGNORING_WEIGHTS
    ConfusionMatrices.multiclass_fscore(cm, measure.beta, Functions.MicroAvg())
end

# ## `call` on confusion matrices

# Cases excluding `NoAvg()`:
(measure::_MulticlassFScore)(
    cm::ConfusionMatrices.ConfusionMatrix,
)  = _call(measure, cm)
(measure::_MulticlassFScore)(
    cm::ConfusionMatrices.ConfusionMatrix,
    class_weights::AbstractDict,
) =  _call(measure, cm, class_weights)

# `NoAvg()` with `Vector` return type:
(measure::_MulticlassFScore{NoAvg,Vector})(
    cm::ConfusionMatrices.ConfusionMatrix,
) = _call(measure, cm)
(measure::_MulticlassFScore{NoAvg,Vector})(
    cm::ConfusionMatrices.ConfusionMatrix,
    class_weights::AbstractDict,
) =  _call(measure, cm, class_weights)

# `NoAvg()` with `LittleDict` return type:
(measure::_MulticlassFScore{NoAvg,LittleDict})(
    cm::ConfusionMatrices.ConfusionMatrix,
) = dict(_call(measure, cm), cm)
(measure::_MulticlassFScore{NoAvg,LittleDict})(
    cm::ConfusionMatrices.ConfusionMatrix,
    class_weights::AbstractDict,
) = dict(_call(measure, cm, class_weights), cm)

# ## `call` on `(ŷ, y)`

#  no class weights:
(measure::_MulticlassFScore)(yhat, y) =
    measure(ConfusionMatrices.confmat(
        yhat,
        y;
        levels=measure.levels,
        perm=measure.perm,
        rev=measure.rev,
        checks=measure.checks,
    ))
(measure::_MulticlassFScore)(yhat, y, class_weights::AbstractDict) =
    measure(ConfusionMatrices.confmat(
        yhat,
        y;
        levels=measure.levels,
        perm=measure.perm,
        rev=measure.rev,
        checks=measure.checks,
    ), class_weights)

# constructor for wrapped measure:

MulticlassFScore(;
                 β=1.0,
                 beta=β,
                 average=Functions.MacroAvg(),
                 return_type=LittleDict,
                 levels=nothing,
                 perm=nothing,
                 rev=nothing,
                 checks=true,
                 ) = _MulticlassFScore(
                     beta,
                     average,
                     return_type,
                     levels,
                     perm,
                     rev,
                     checks,
                 ) |> API.robust_measure |> API.fussy_measure
MulticlassFScore(beta; kwargs...) = Multiclass(; beta, kwargs...)

# supertype for measures so-constructed:
MulticlassFScoreType = API.FussyMeasure{
    <:API.RobustMeasure{<:_MulticlassFScore}
}

# Allow callable on confusion matrices:
(measure::MulticlassFScoreType)(
    cm::ConfusionMatrices.ConfusionMatrix,
    class_weights...) =
        API.unwrap(API.unwrap(measure))(cm, class_weights...)

# these traits will be inherited by `MulticlassFScore`:
@trait(
    _MulticlassFScore,
    consumes_multiple_observations=true,
    kind_of_proxy=LearnAPI.LiteralTarget(),
    observation_scitype=Union{Missing,Finite},
    supports_class_weights=true,
    orientation=Score(),
    human_name="multi-class ``F_β`` score",
    external_aggregation_mode=Mean(),
)

# friendly show method:
@fix_show MulticlassFScore::MulticlassFScoreType

register(
    MulticlassFScore,
    "macro_f1score",
    "micro_f1score",
    "multiclass_f1score",
)

const MulticlassFScoreDoc = docstring(
    "MulticlassFScore(; average=macro_avg, levels=nothing, more_options...)",
    scitype=DOC_ORDERED_FACTOR_BINARY,
    body=
        """
        \n\nThis is an averaged one-versus-rest version of the binary
        [`FScore`](@ref). Or it can return a dictionary keyed
        on target class (or a vector); see `average` options below.

        $(ConfusionMatrices.DOC_OPTIMISED)

        You can also call `m` on confusion matrices. Construct confusion matrices using
        [`ConfusionMatrix`](@ref).

                        """,
    footer=
        "$(ConfusionMatrices.DOC_OPTIONS(return_type=true, average=true, beta=true))\n\n"*
        "$(ConfusionMatrices.DOC_OPTIMISED)\n\n"*
        "See also [`$(FScore)`](@ref), "*
        "[`$ConfusionMatrices.ConfusionMatrix`](@ref) and "*
        "[`ConfusionMatrix`](@ref).\n\nCore algorithm: "*
        "[`Functions.multiclass_fscore`](@ref)",
)

"$MulticlassFScoreDoc"
MulticlassFScore
const micro_f1score = MulticlassFScore(average=micro_avg)
"$MulticlassFScoreDoc"
const macro_f1score = MulticlassFScore(average=macro_avg)
"$MulticlassFScoreDoc"
const multiclass_f1score = macro_f1score

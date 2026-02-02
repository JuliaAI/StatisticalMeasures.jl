const ERR_NEED_CATEGORICAL_PR = ArgumentError(
    "Was expecting categorical arguments: "*
        "In a call like `precision_recall_curve(ŷ, y)`, `ŷ` must have eltype "*
        "`<:CategoricalDistributions.UnivariateFinite` and `y` must have eltype "*
        "`<:CategoricalArrays.CategoricalArray` . If using raw probabilities, consider "*
        "using `Functions.precision_recall_curve` instead. "
)

const ERR_PR1 = ArgumentError(
    "probabilistic predictions should be for exactly two classes (levels)"
)

const ERR_PR2 = ArgumentError(
    "ground truth observations must have exactly two classes (levels) in the pool"
)

# perform some argument checks and return the ordered levels:
function binary_levels_pr(
    yhat::AbstractArray{<:Union{Missing,UnivariateFinite{<:Finite{2}}}},
    y::CategoricalArrays.CatArrOrSub
    )
    classes = CategoricalArrays.levels(y)
    length(classes) == 2 || throw(ERR_PR2)
    API.check_numobs(yhat, y)
    API.check_pools(yhat, y)
    warn_unordered(classes)
    classes
end
binary_levels_pr(
    yhat::AbstractArray{<:Union{Missing,UnivariateFinite{<:Finite}}},
    y::CategoricalArrays.CatArrOrSub
) = throw(ERR_PR1)
binary_levels_pr(yhat, y) = throw(ERR_NEED_CATEGORICAL_PR)

"""
    precision_recall_curve(ŷ, y) -> false_positive_rates, true_positive_rates, thresholds

$(Functions.DOC_PRECISION_RECALL(
    middle="Here `ŷ` is a vector of `UnivariateFinite` distributions "*
        "(from CategoricalDistributions.jl) over the two "*
        "values taken by the ground truth observations `y`, a `CategoricalVector`. "*
        "The `thresholds`, listed in descending order, are the distinct predicted "*
        "probabilities of the positive class. ",
    footer="Core algorithm: [`Functions.precision_recall_curve`](@ref). "
))
"""
function precision_recall_curve(yhat, y)
    # `binary_levels` also performs argument checks and issues warnings about order:
    positive_class = binary_levels_pr(yhat, y) |> last
    scores = pdf.(yhat, positive_class)
    Functions.precision_recall_curve(scores, y, positive_class)
end

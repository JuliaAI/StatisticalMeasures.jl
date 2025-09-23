const ERR_NEED_CATEGORICAL = ArgumentError(
    "Was expecting categorical arguments: "*
        "In a call like `roc_curve(ŷ, y)`, `ŷ` must have eltype "*
        "`<:CategoricalDistributions.UnivariateFinite` and `y` must have eltype "*
        "`<:CategoricalArrays.CategoricalArray` . If using raw probabilities, consider "*
        "using `Functions.roc_curve` instead. "
)

const ERR_ROC1 = ArgumentError(
    "probabilistic predictions should be for exactly two classes (levels)"
)

const ERR_ROC2 = ArgumentError(
    "ground truth observations must have exactly two classes (levels) in the pool"
)

# perform some argument checks and return the ordered levels:
function binary_levels(
    yhat::AbstractArray{<:Union{Missing,UnivariateFinite{<:Finite{2}}}},
    y::CategoricalArrays.CatArrOrSub
    )
    classes = CategoricalArrays.levels(y)
    length(classes) == 2 || throw(ERR_ROC2)
    API.check_numobs(yhat, y)
    API.check_pools(yhat, y)
    yhat isa AbstractArray{<:UnivariateFinite{<:OrderedFactor}} && warn_unordered(classes)
    classes
end
binary_levels(
    yhat::AbstractArray{<:Union{Missing,UnivariateFinite{<:Finite}}},
    y::CategoricalArrays.CatArrOrSub
) = throw(ERR_ROC1)
binary_levels(yhat, y) = throw(ERR_NEED_CATEGORICAL)


"""
    roc_curve(ŷ, y) -> false_positive_rates, true_positive_rates, thresholds

$(Functions.DOC_ROC(
    middle="Here `ŷ` is a vector of `UnivariateFinite` distributions "*
        "(from CategoricalDistributions.jl) over the two "*
        "values taken by the ground truth observations `y`, a `CategoricalVector`. ",
    footer="Core algorithm: [`Functions.roc_curve`](@ref)"*
        "\n\nSee also [`AreaUnderCurve`](@ref). ",
))
"""
function roc_curve(yhat, y)
    # `binary_levels` also performs argument checks and issues warnings about order:
    positive_class = binary_levels(yhat, y) |> last
    scores = pdf.(yhat, positive_class)
    Functions.roc_curve(scores, y, positive_class)
end

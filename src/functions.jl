module Functions
using LinearAlgebra
import StatisticalMeasuresBase as API
using StatsBase

export NoAvg, MacroAvg, MicroAvg

# # SCALAR FUNCTIONS

absolute_difference(yhat, y) = abs(yhat - y)

pth_power_of_absolute_difference(yhat, y, p=2) = p === 1 ? abs(yhat - y) : abs(yhat - y)^p

absolute_difference_of_logs(yhat, y) = abs(log(y) - log(yhat))

absolute_difference_of_logs(yhat, y, offset) = abs(log(y + offset) - log(yhat + offset))

@inline function normalized_absolute_difference(yhat, y, tol::T) where T
    scale = abs(y)
    scale > tol || return zero(T)
    abs(yhat - y)/scale
end
normalized_absolute_difference(yhat, y) = normalized_absolute_difference(yhat, y, eps())

_softplus(x::T) where T<:Real = x > zero(T) ? x + log1p(exp(-x)) : log1p(exp(x))
log_cosh(x::T) where T<:Real = x + _softplus(-2x) - log(convert(T, 2))
log_cosh_difference(yhat, y) = log_cosh(yhat - y)


# # ROC CURVE

"""
    _idx_unique_sorted(v)

*Private method.*

Return the index of unique elements in `Real` vector `v` under the assumption that the
vector `v` is sorted in decreasing order.

"""
function _idx_unique_sorted(v)
    n    = length(v)
    idx  = ones(Int, n)
    p, h = 1, 1
    cur  = v[1]
    @inbounds while h < n
        h     += 1                  # head position
        cand   = v[h]               # candidate value
        cand   < cur || continue    # is it new? otherwise skip
        p     += 1                  # if new store it
        idx[p] = h
        cur    = cand               # and update the last seen value
    end
    p < n && deleteat!(idx, p+1:n)
    return idx
end

const DOC_ROC(;middle="", footer="") =
"""
Return data for plotting the receiver operator characteristic (ROC curve) for a binary
classification problem.

$middle

If there are `k` unique probabilities, then there are correspondingly `k` thresholds
and `k+1` "bins" over which the false positive and true positive rates are constant.:

- `[0.0 - thresholds[1]]`
- `[thresholds[1] - thresholds[2]]`
- ...
- `[thresholds[k] - 1]`

Consequently, `true_positive_rates` and `false_positive_rates` have length `k+1` if
`thresholds` has length `k`.

To plot the curve using your favorite plotting backend, do something like
`plot(false_positive_rates, true_positive_rates)`.

$footer
"""

"""
    Functions.roc_curve(probs_of_positive, ground_truth_obs, positive_class) ->
        false_positive_rates, true_positive_rates, thresholds

$(DOC_ROC())

Assumes there are no more than two classes but does not check this. Does not check that
`positive_class` is one of the observed classes.

"""
function roc_curve(scores, y, positive_class)
    n = length(y)

    ranking = sortperm(scores, rev=true)

    scores_sort = scores[ranking]
    y_sort_bin  = (y[ranking] .== positive_class)

    idx_unique = _idx_unique_sorted(scores_sort)
    thresholds = scores_sort[idx_unique]

    # detailed computations with example:
    # y = [  1   0   0   1   0   0   1]
    # s = [0.5 0.5 0.2 0.2 0.1 0.1 0.1] thresh are 0.5 0.2, 0.1 // idx [1, 3, 5]
    # ŷ = [  0   0   0   0   0   0   0] (0.5 - 1.0] # no pos pred
    # ŷ = [  1   1   0   0   0   0   0] (0.2 - 0.5] # 2 pos pred
    # ŷ = [  1   1   1   1   0   0   0] (0.1 - 0.2] # 4 pos pred
    # ŷ = [  1   1   1   1   1   1   1] [0.0 - 0.1] # all pos pre

    idx_unique_2 = idx_unique[2:end]   # [3, 5]
    n_ŷ_pos      = idx_unique_2 .- 1   # [2, 4] implicit [0, 2, 4, 7]

    cs   = cumsum(y_sort_bin)          # [1, 1, 1, 2, 2, 2, 3]
    n_tp = cs[n_ŷ_pos]                 # [1, 2] implicit [0, 1, 2, 3]
    n_fp = n_ŷ_pos .- n_tp             # [1, 2] implicit [0, 1, 2, 4]

    # add end points
    P = sum(y_sort_bin) # total number of true positives
    N = n - P           # total number of true negatives

    n_tp = [0, n_tp..., P] # [0, 1, 2, 3]
    n_fp = [0, n_fp..., N] # [0, 1, 2, 4]

    tprs = n_tp ./ P  # [0/3, 1/3, 2/3, 1]
    fprs = n_fp ./ N  # [0/4, 1/4, 2/4, 1]

    return fprs, tprs, thresholds
end

const DOC_AUC_REF =
    "Implementation is based on the Mann-Whitney U statistic.  See the "*
    "[*Whitney U "*
    "test*](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test"*
    "#Area_under_curve_(AUC)_statistic_for_ROC_curves) "*
    "Wikipedia page for details. "

"""
    Functions.auc(probabilities_of_positive, ground_truth_observations, positive_class)

Return the area under the ROC ([receiver operator
characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)). $DOC_AUC_REF

"""
function auc(scores, y, positive_class)
    ranks = StatsBase.tiedrank(scores)
    n = length(y)
    n_neg = 0  # to keep of the number of negative preds
    T = eltype(ranks)
    R_pos = zero(T) # sum of positive ranks
    @inbounds for (i,j) in zip(eachindex(y), eachindex(ranks))
        if y[i] == positive_class
            R_pos += ranks[j]
        else
            n_neg += 1
        end
    end
    n_pos = n - n_neg # number of positive predictions
    U = R_pos - T(0.5)*n_pos*(n_pos + 1) # Mann-Whitney U statistic
    return U / (n_neg * n_pos)
end


# # FUNCTIONS ON MATRICES INTERPRETED AS CONFUSION MATRICES

clean(s) = join(split(last(split(s, ".")), "_"), " ")
function docstring(measure; sig="(m)", name=clean(measure), binary=false, the=false)
    footer = binary ?
        "The first index corresponds to the \"negative\" class, "*
        "the second to the \"positive\" class.\n\n "*
        "Assumes `m` is a 2 x 2 matrix but does not check this. " :
        "Assumes `m` is a square matrix, but does not check this. "
    article = the ? "the" : ""
    extra = sig == "(m)" ? "" :
        """
        Here `average` is one of: `NoAvg()`, `MicroAvg()`, `MacroAvg()`; `weights` is a
        vector of class weights. Usual [weighted
        means](https://en.wikipedia.org/wiki/Weighted_arithmetic_mean), and not means of
        weighted sums, are used. Weights are not supported by the `Micro()` option.
        """
    """
        $measure$sig

    Return $article $name for the the matrix `m`, interpreted as a confusion matrix.
    $extra

    $footer
    """
end

# ## accuracy, kappa, matthew's correlation

"""$(docstring("Functions.accuracy", the=true))"""
accuracy(m) =  sum(diag(m)) / sum(m)

"""$(docstring("Functions.balanced_accuracy", the=true))"""
function balanced_accuracy(m)
    # the number of correctly classified observations per class
    d = diag(m)
    total = 0.0
    for i in eachindex(d)
        # calculate accuracy for each class
        total += d[i] / sum(view(m, :, i))
    end
    # return the averaged accuracy
    return total/length(d)
end

"""$(docstring("Functions.kappa"))"""
function kappa(m)
    C = size(m, 1)
    # relative observed agreement - same as accuracy
    p₀ = sum(diag(m))/sum(m)

    # probability of agreement due to chance - for each class cᵢ, this
    # would be: (#predicted=cᵢ)/(#instances) x (#observed=cᵢ)/(#instances)
    rows_sum = sum!(similar(m, 1, C), m) # 1 x C matrix
    cols_sum = sum!(similar(m, C, 1), m) # C X 1 matrix
    pₑ = first(rows_sum*cols_sum)/sum(rows_sum)^2

    # Kappa calculation
    κ = (p₀ - pₑ)/(1 - pₑ)
    return κ
end

"""$(docstring("Functions.matthews_correlation", name="Matthew's correlation"))"""
function matthews_correlation(m)
    # http://rk.kvl.dk/introduction/index.html
    # NOTE: this is O(C^3), there may be a clever way to
    # speed this up though in general this is only used for low  C
    num = 0
    C = size(m, 1)
    @inbounds for k in 1:C, l in 1:C, j in 1:C
        num += m[k,k] * m[l,j] - m[k,l] * m[j,k]
    end
    den1 = 0
    den2 = 0
    @inbounds for k in 1:C
        a = sum(m[k, :])
        b = sum(m[setdiff(1:C, k), :])
        den1 += a * b
        a = sum(m[:, k])
        b = sum(m[:, setdiff(1:C, k)])
        den2 += a * b
    end
    mcc = num / sqrt(float(den1) * float(den2))

    isnan(mcc) && return 0
    return mcc
end


# ## binary, but NOT invariant under class relabellings

"""$(docstring("Functions.true_positive", name="the true positive count", binary=true))"""
true_positive(m) = m[2,2]

"""$(docstring("Functions.true_negative", name="the true negative count", binary=true))"""
true_negative(m) = m[1,1]

"""$(docstring("Functions.false_positive", name="the false positive count", binary=true))"""
false_positive(m) = m[2,1]

"""$(docstring("Functions.false_negative", name="the false negative count", binary=true))"""
false_negative(m) = m[1,2]

# same as recall"
"""$(docstring("Functions.true_positive_rate", binary=true, the=true))"""
true_positive_rate(m) = true_positive(m) / (true_positive(m) + false_negative(m))

"""$(docstring("Functions.true_negative_rate", binary=true, the=true))"""
true_negative_rate(m) = true_negative(m) / (true_negative(m) + false_positive(m))

"""$(docstring("Functions.false_positive_rate", binary=true, the=true))"""
false_positive_rate(m) = 1 - true_negative_rate(m)

"""$(docstring("Functions.false_negative_rate", binary=true, the=true))"""
false_negative_rate(m) = 1 - true_positive_rate(m)

"""$(docstring("Functions.false_discovery_rate", binary=true, the=true))"""
false_discovery_rate(m) = false_positive(m) / (true_positive(m) + false_positive(m))

"""$(docstring("Functions.negative_predictive_value", binary=true, the=true))"""
negative_predictive_value(m) = true_negative(m) / (true_negative(m) + false_negative(m))

# same as "precision":
"""$(docstring("Functions.positive_predictive_value", binary=true, the=true))"""
positive_predictive_value(m) = 1 - false_discovery_rate(m)

"""
    Functions.fscore(m, β=1.0)

Return the ``F_β`` score of the matrix `m`, interpreted as a confusion matrix. The first
index corresponds to the "negative" class, the second to the "positive".

Assumes `m` is a 2 x 2 matrix but does not check this.

"""
function fscore(m, β=1.0)
    β2   = β^2
    tp = true_positive(m)
    fn = false_negative(m)
    fp = false_positive(m)
    return (1 + β2) * tp / ((1 + β2)*tp + β2*fn + fp)
end


# ## multiclass, derived from one-versus-rest binary metrics

# ### modes of averaging across classes

abstract type MulticlassAvg end
struct MacroAvg <: MulticlassAvg end
struct MicroAvg <: MulticlassAvg end
struct NoAvg <: MulticlassAvg end


# ### helpers to combine atomic functions, like `true_positive` and `false_negative`,
# ### into more complicated ones that depend on a mode of averaging, like
# ### `true_positive_rate`

# In these functions, `a` and `b` are vectors of measurements (e.g., num true positives),
# one for each target class; `weights` is a vector of class weights; and `m` is a matrix,
# to be interpreted as a confusion matrix.
inv_prop_add(::NoAvg, a,  b) = a ./ (a + b)
inv_prop_add(average::NoAvg, a,  b, weights) = weights .* inv_prop_add(average, a, b)
inv_prop_add(::MicroAvg, a, b) = inv_prop_add(NoAvg(), sum(a), sum(b))
inv_prop_add(::MacroAvg, a,  b, weights=nothing) =
    API.aggregate(zip(a, b); weights, skipnan=true) do (x, y)
        inv_prop_add(NoAvg(), x, y)
    end
subtracted_from_one(average::NoAvg, f, m) = 1.0 .- f(m, average)
subtracted_from_one(average::NoAvg, f, m, weights) =
    weights .* subtracted_from_one(average, f, m)
subtracted_from_one(average::MicroAvg, f, m) = 1.0 - f(m, average)
subtracted_from_one(::MacroAvg, f, m, weights=nothing) =
    API.aggregate(subtracted_from_one(NoAvg(), f, m); weights, skipnan=true)


# ### the atomic functions

"""
$(docstring(
    "Functions.multiclass_true_positive",
    name="one-versus-rest true positive counts",
    the=true,
))
"""
multiclass_true_positive(m) = diag(m)

"""
$(docstring(
    "Functions.multiclass_false_positive",
    name="one-versus-rest false positive counts",
    the=true,
))
"""
@inline function multiclass_false_positive(m)
    col_sum = vec(sum(m, dims=2))
    col_sum .-= diag(m)
end

"""
$(docstring(
    "Functions.multiclass_false_negative",
    name="one-versus-rest false negative counts",
    the=true,
))
"""
@inline function multiclass_false_negative(m)
    row_sum = vec(collect(transpose(sum(m, dims=1))))
     row_sum .-= diag(m)
end

"""
$(docstring(
    "Functions.multiclass_true_negative",
    name="one-versus-rest true negative counts",
    the=true,
))
"""
@inline function multiclass_true_negative(m)
    _sum = sum(m, dims=2)
    _sum .= sum(m) .- (_sum .+= sum(m, dims=1)'.- diag(m))
   vec(_sum)
end


# ### the derived functions

# same as recall
"""
$(docstring(
    "Functions.multiclass_true_positive_rate",
    sig="(m, average[, weights])",
    name="one-versus-rest true positive rates",
    the=true,
))
"""
multiclass_true_positive_rate(m, average, weights...) = inv_prop_add(
    average,
    multiclass_true_positive(m),
    multiclass_false_negative(m),
    weights...,
)

"""
$(docstring(
    "Functions.multiclass_true_negative_rate",
    sig="(m, average[, weights])",
    name="one-versus-rest true negative rates",
    the=true,
))
"""
multiclass_true_negative_rate(m, average, weights...) = inv_prop_add(
    average,
    multiclass_true_negative(m),
    multiclass_false_positive(m),
    weights...,
)

"""
$(docstring(
    "Functions.multiclass_false_positive_rate",
    sig="(m, average[, weights])",
    name="one-versus-rest false positive rates",
    the=true,
))
"""
multiclass_false_positive_rate(m, average, weights...) = subtracted_from_one(
    average,
    multiclass_true_negative_rate,
    m,
    weights...,
)

"""
$(docstring(
    "Functions.multiclass_false_negative_rate",
    sig="(m, average[, weights])",
    name="one-versus-rest false negative rates",
    the=true,
))
"""
multiclass_false_negative_rate(m, average, weights...) = subtracted_from_one(
    average,
    multiclass_true_positive_rate,
    m,
    weights...,
)

"""
$(docstring(
    "Functions.multiclass_false_discovery_rate",
    sig="(m, average[, weights])",
    name="one-versus-rest false discovery rates",
    the=true,
))
"""
multiclass_false_discovery_rate(m, average, weights...) = inv_prop_add(
    average,
    multiclass_false_positive(m),
    multiclass_true_positive(m),
    weights...,
)

"""
$(docstring(
    "Functions.multiclass_negative_predictive_value",
    sig="(m, average[, weights])",
    name="one-versus-rest negative predictive values",
    the=true,
))
"""
multiclass_negative_predictive_value(m, average, weights...) = inv_prop_add(
    average,
    multiclass_true_negative(m),
    multiclass_false_negative(m),
    weights...)

# same as precision:
"""
$(docstring(
    "Functions.multiclass_positive_predictive_value",
    sig="(m, average[, weights])",
    name="one-versus-rest positive predictive values",
    the=true,
))
"""
multiclass_positive_predictive_value(m, average, weights...) = subtracted_from_one(
    average,
    multiclass_false_discovery_rate,
    m,
    weights...,
)

"""
$(docstring(
    "Functions.multiclass_fscore",
    sig="(m, β, average[, weights])",
    the=true,
))*"\n Note that the `MicroAvg` score is insenstive to `β`. "
"""
multiclass_fscore(m, beta, average::MicroAvg) =
    multiclass_true_positive_rate(m, MicroAvg())
@inline function multiclass_fscore(m, beta, average::NoAvg)
    β2 = beta^2
    TP = multiclass_true_positive(m)
    FN = multiclass_false_negative(m)
    FP = multiclass_false_positive(m)
    (1 + β2) * TP ./ ((1 + β2) * TP + β2 * FN + FP)
end
multiclass_fscore(m, beta, average::NoAvg, weights) =
    weights .* multiclass_fscore(m, beta, NoAvg())
multiclass_fscore(m, beta, average::MacroAvg, weights=nothing) =
    API.aggregate(multiclass_fscore(m, beta, NoAvg()); weights, skipnan=true)

end # module

using .Functions

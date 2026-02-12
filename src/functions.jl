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


# # CONFUSION MATRIX AT THRESHOLDS


"""
    _idx_unique_sorted(v)

*Private method.*

Return the index of the first appearance of each element within `v`, under the untested
assumption that `v` is sorted in decreasing order.

```julia-repl
julia> [5, 5, 4, 3, 3, 3, 2, 1] |> _idx_unique_sorted
5-element Vector{Int64}:
 1
 3
 4
 7
 8
```

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

const DOC_YHAT_Y =
"""

Here `ŷ` is a vector of predicted numerical probabilities of the specified
`positive_class`, which is one of two possible values occurring in the provided vector
`y` of ground truth observations.

The returned probability `thresholds` are the distinct values taken on by `ŷ`, listed in
descending order. In particular, `0` and `1` are only included if they are present in `ŷ`.

"""

DOC_THRESHOLDS(; counts="counts") =
"""

If `thresholds` has length `k`, the interval [0, 1] is partitioned into `k+1` bins.
The $counts are constant within each bin:

- `[0.0, thresholds[k])`
- `[thresholds[k], thresholds[k - 1])`
- ...
- `[thresholds[1], 1]`

"""

const DOC_CONFUSION_CHECK = "Assumes there are no more than two classes but does "*
    "not check this. Does not check that "*
    "`positive_class` is one of the observed classes. "

const DOC_CONFUSION_AT_THRESHOLDS(;middle=DOC_YHAT_Y, footer=DOC_CONFUSION_CHECK) =
"""

For a binary classification problem, return probability thresholds and corresponding
confusion matrix entries, suitable for generating ROC curves and precision-recall curves
(and variations on these). Primarily intended as a backend for implementations of those
two cases.

$middle

$(DOC_THRESHOLDS())

Consequently, `TN`, `FP`, `FN` and `TP`, will each have length `k + 1` in that case.

The `j`th raw confusion matrix will be `reshape([TN[j], FP[j], FN[j], TP[j]], 2, 2)`,
according to conventions used elsewhere in StatisticalMeasures.jl, which explains the
chosen order for the return value.

$footer

"""

"""
    Functions.confusion_counts_at_thresholds(ŷ, y, positive_class) ->
        (TN, FP, FN, TP), thresholds

$(DOC_CONFUSION_AT_THRESHOLDS())

"""
function confusion_counts_at_thresholds(scores, y, positive_class)
    n = length(y)

    ranking = sortperm(scores, rev=true)

    scores_sort = scores[ranking]
# Sort samples by score in descending order
# This lets us easily count predictions by threshold: for any threshold t,
# all samples with score ≥ t come before those with score < t
ranking = sortperm(scores, rev=true)
sorted_scores = scores[ranking]
sorted_labels  = (y[ranking] .== positive_class)  

     # Find where unique thresholds begin
     # Since scores are sorted descending, each unique score value marks a threshold
     # Example: scores [0.5, 0.5, 0.2, 0.2, 0.1] → thresholds start at indices [1, 3, 5]
    threshold_indices = _idx_unique_sorted(sorted_scores)
    thresholds = sorted_scores[threshold_indices]

    # detailed computations with example:
    # sorted_labels = [  1   0   0   1   0   0   1]
    # s          = [0.5 0.5 0.2 0.2 0.1 0.1 0.1] thresh are 0.5 0.2, 0.1 // idx [1, 3, 5]
    # ŷ          = [  0   0   0   0   0   0   0] (0.5 - 1.0] # no pos pred
    # ŷ          = [  1   1   0   0   0   0   0] (0.2 - 0.5] # 2 pos pred
    # ŷ          = [  1   1   1   1   0   0   0] (0.1 - 0.2] # 4 pos pred
    # ŷ          = [  1   1   1   1   1   1   1] [0.0 - 0.1] # all pos pre
    # Count total positives and negatives in the dataset
    cum_positives = cumsum(sorted_labels)   # running count of true positives  # [1, 1, 1, 2, 2, 2, 3]
    P = cum_positives[end]   # total number of observed positives (3)
    N = n - P     # total number of observed negatives (4)
    # For each threshold (except the highest), count predictions
    # At a given threshold starting at index i, all samples 1..(i-1) are predicted positive
    # Example: threshold at index 3 → samples 1-2 predicted positive (2 samples)
    n_ŷ_pos = threshold_indices[2:end]  .- 1    # [2, 4] implicit [0, 2, 4, 7]
    
    # Compute true positives and false positives 
    tp = cum_positives[n_ŷ_pos]                 # [1, 2] implicit [0, 1, 2, 3]
    fp = n_ŷ_pos .- tp               # [1, 2] implicit [0, 1, 2, 4]

    # add end points
    # - First endpoint: threshold > max score → no positive predictions
    # - Last endpoint: threshold ≤ min score → all samples predicted positive
    tp = [0, tp..., P] # [0, 1, 2, 3]
    fp = [0, fp..., N] # [0, 1, 2, 4]
    
    # Derive the remaining confusion matrix entries
    fn = P .- tp       # [3, 2, 1, 0]
    tn = N .- fp       # [4, 3, 2, 0]

    return (tn, fp, fn, tp), thresholds
end


# # ROC CURVE

const DOC_ROC(;middle=DOC_YHAT_Y, footer=DOC_CONFUSION_CHECK) =
"""

Return data for plotting the receiver operator characteristic (ROC curve) for a binary
classification problem.

$middle

$(DOC_THRESHOLDS(counts="`true_positive_rate` and `false_positive_rate`"))

Accordingly, `true_positive_rates` and `false_positive_rates` have length `k+1` in that
case.

To plot the curve using your favorite plotting library, do something like
`plot(false_positive_rates, true_positive_rates)`.

$footer
"""

"""
    Functions.roc_curve(ŷ, y, positive_class) ->
        false_positive_rates, true_positive_rates, thresholds

$(DOC_ROC())

For a method with checks, see [`StatisticalMeasures.roc_curve`](@ref). See also
[`Functions.confusion_counts_at_thresholds`](@ref).

"""
function roc_curve(scores, y, positive_class)
    (tn, fp, fn, tp), thresholds =
        confusion_counts_at_thresholds(scores, y, positive_class)

    N = tn[1] # num observed negatives
    P = fn[1] # num observed positives

    tpr = tp ./ P
    fpr = fp ./ N

    return fpr, tpr, thresholds
end


# # PRECISION RECALL CURVE

tamed_divide(a, b) = b == 0 ? 0 : a/b

const DOC_ROC_CHECK = DOC_CONFUSION_CHECK*
    "That failing to be the case, each returned recall will be `Inf` or `NaN`. "

const DOC_PRECISION_RECALL(;middle=DOC_YHAT_Y, footer=DOC_ROC_CHECK) =
"""

Return data for plotting the precision-recall curve (PR curve) for a binary classification
problem. The first point on the corresponding curve is always `(recall, precision) = (0,
1)`, while the last point is always `(recall, precision) = (1, p)` where `p` is the
proportion of positives in the observed sample `y`.

$middle

$(DOC_THRESHOLDS(counts="precison and recall"))

Accordingly, `precisions` and `recalls` have length `k+1` in that case.

To plot the curve using your favorite plotting library, do something like
`plot(recalls, precisions)`.

$footer
"""

"""
    Functions.precision_recall_curve(ŷ, y, positive_class) ->
        precisions, recalls, thresholds

$(DOC_PRECISION_RECALL())

See also [`StatisticalMeasures.precision_recall_curve`](@ref), which includes some
checks, and [`Functions.confusion_counts_at_thresholds`](@ref).

"""
function precision_recall_curve(scores, y, positive_class)
    (tn, fp, fn, tp), thresholds =
        confusion_counts_at_thresholds(scores, y, positive_class)

    k = length(tp)
    precisions = Vector{Float64}(undef, k)
    @. precisions = tamed_divide(tp, tp + fp)
    # force precision = 1 at threshold -> 1:
    precisions[1] = 1

    recalls = Vector{Float64}(undef, k)
    P = fn[1] # num observed positives
    @. recalls = tp / P
    return recalls, precisions, thresholds
end


# # AVERAGE PRECISION

const DOC_AVERAGE_PRECISION =
"""

Average precision is the area under the empirical precision-recall curve, understood as a
step function. This is to be contrasted with measures going under the name "area
under the precision-recall curve", in which the step function is usually replaced by a
piece-wise linear approximation. Generally, differences between the two are only obvious
when the number of observations is small, but it is faster to compute average precision.

Reference: Wikipedia entry, [Average
precision](https://en.wikipedia.org/w/index.php?title=Information_retrieval&oldid=793358396#Average_precision)

# Definition

Adopting each distinct predicted probability ``p_1, p_2, \\ldots, p_k`` for the positive
class as a soft probability threshold for predicting an actual class, and assuming these
thresholds are arranged in decreasing order, we obtain corresponding recalls ``R_1, R_2,
\\ldots, R_k`` (monotonically increasing) and precisions ``P_1, P_2, \\ldots,
P_k``. Adding an extra recall, ``R_{k+1} = 1``, the average precision implemented here is
defined as

``\\sum_{j=1}^k P_j (R_{j+1} - R_j)``

In some other implementations, such as
[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score),
``P_j`` is replaced by ``P_{j+1}``. However, this requires the definition of a precision
for unit recall, in the case the predicted positive class probabilities exclude `1.0`, and
this is avoided here.

"""

"""
    function average_precision(ŷ, y, positive_class)

Return the average precision corresponding to a vector `ŷ` of predicted numerical
probabilities of the specified `positive_class`, which is one of two possible values
occurring in the accompanying vector `y` of ground truth observations.

$DOC_AVERAGE_PRECISION

$DOC_CONFUSION_CHECK Method requires at least one observation, but this is not checked.

"""
function average_precision(ŷ, y, positive_class)

    recalls, precisions, _ = precision_recall_curve(ŷ, y, positive_class)
    area = 0.0

    # `recalls` will have length at least two:
    length(recalls) > 2 || return 1.0

    r = recalls[1]

    # We ignore the last precision, as this does not correspond to any predicted
    # probability, but is rather an artifact to ensure precision-recall curves always have
    # a recall=1 point. See the definition in the docstring.
    for i in 1:length(precisions[1:(end -1)])
        r_next = recalls[i + 1]
        Δr = r_next - r
        r = r_next
        area += precisions[i]*Δr
    end

    return area
end


# # AUC

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

"""
    Functions.cbi(
        probability_of_positive, ground_truth_observations, positive_class,
        nbins, binwidth, ma=maximum(scores), mi=minimum(scores), cor=corspearman
    )
    Return the Continuous Boyce Index (CBI) for a vector of probabilities and ground truth observations.

"""
function cbi(
    scores, y, positive_class;
    verbosity, nbins, binwidth,
    max=maximum(scores), min=minimum(scores), cor=StatsBase.corspearman
)
    binstarts = range(min, stop=max-binwidth, length=nbins)
    binends = binstarts .+ binwidth

    sorted_indices = sortperm(scores)
    sorted_scores = view(scores, sorted_indices)
    sorted_y = view(y, sorted_indices)

    n_positive = zeros(Int, nbins)
    n_total = zeros(Int, nbins)
    empty_bins = falses(nbins)
    any_empty = false

    @inbounds for i in 1:nbins
        bin_index_first = searchsortedfirst(sorted_scores, binstarts[i])
        bin_index_last = searchsortedlast(sorted_scores, binends[i])
        if bin_index_first > bin_index_last
            empty_bins[i] = true
            any_empty = true
        end
        @inbounds for j in bin_index_first:bin_index_last
            if sorted_y[j] == positive_class
                n_positive[i] += 1
            end
        end
        n_total[i] = bin_index_last - bin_index_first + 1
    end
    if any_empty
        verbosity > 1 && @info "removing $(sum(empty_bins)) bins without any observations"
        deleteat!(n_positive, empty_bins)
        deleteat!(n_total, empty_bins)
        binstarts = binstarts[.!empty_bins]
    end

    # calculate "PE-ratios" - a bunch of things cancel out but that does not matter for
    # any correlation calculation
    PE_ratios = n_positive ./ n_total
    return cor(PE_ratios, binstarts)
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
))*"\n Note that the `MicroAvg` score is insensitive to `β`. "
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

"""
    ConfusionMatrices

A module providing confusion matrix basics.

"""
module ConfusionMatrices

using CategoricalArrays
using OrderedCollections
import ..Functions

const CM  = "ConfusionMatrices"
const CatArrOrSub{T, N} =
    Union{CategoricalArray{T, N}, SubArray{T, N, <:CategoricalArray}}

function WARN_UNORDERED(levels)
    ret = "Levels not explicitly ordered. "*
        "Using the order $levels. "
    if length(levels) == 2
        ret *= "The \"positive\" level is $(levels[2]). "
    end
    ret
end

const ERR_INDEX_ACCESS_DENIED = ErrorException(
    "Direct access by index of unordered confusion matrices dissallowed. "*
        "Access by level, as in `some_confusion_matrix(\"male\", \"female\")` or first "*
        "extract regular matrix using `$CM.matrix(some_confusion_matrix)`. "
)

const ERR_ORPHANED_OBSERVATIONS = ArgumentError(
    "Classes observed in inputs but not accounted for in specified levels. "
)

const DOC_REF =
    "[*Confusion matrix* wikipedia "*
    "article](https://en.wikipedia.org/wiki/Confusion_matrix)"

const DOC_ORDER_REQUIREMENTS =
"""
Elements of a confusion matrix can always be accessed by level - see the example below. To
flag the confusion matrix as ordered, and hence index-accessible, do one of the following:

- Supply ordered `CategoricalArray` inputs `ŷ` and `y`

- Explicitly specify `levels` or one of `rev`, `perm`

Note that `==` for two confusion matrices is stricter when both are ordered.
"""

const DOC_OPTIMISED =
"""
Method is optimized for `CategoricalArray` inputs with `levels` inferred. In that case
`levels` will be the complete internal class pool, and not just the observed levels.
"""

const DOC_ROWCOL =
    "Predicted classes are constant on rows, ground truth "*
    "classes are constant on columns. "

const DOC_EG =
"""
import StatisticalMeasures.ConfusionMatrices as CM

y = ["a", "b", "a", "a", "b", "a", "a", "b", "b", "a"]
ŷ = ["b", "a", "a", "b", "a", "b", "b", "b", "a", "a"]

julia> cm = CM.confmat(ŷ, y)
              ┌───────────────────────────┐
              │       Ground Truth        │
┌─────────────┼─────────────┬─────────────┤
│  Predicted  │      a      │      b      │
├─────────────┼─────────────┼─────────────┤
│      a      │      2      │      3      │
├─────────────┼─────────────┼─────────────┤
│      b      │      4      │      1      │
└─────────────┴─────────────┴─────────────┘
"""

function DOC_OPTIONS(; binary=false, return_type=false, average=false, beta=false)
    ret =
        """

        # Keyword options

        """
    ret *= !beta ? "" :
        """

        - `beta=1.0`: parameter in the range ``[0,∞]``, emphasizing recall over precision
          for `beta > 1`, except in the case `average=MicroAvg()`, when it has no
          effect.

        """
    ret *= !average ? "" :
        """

        - `average=MacroAvg()`: one of: `NoAvg()`, `MacroAvg()`, `MicroAvg()` (names owned
          and exported by StatisticalMeasuresBase.jl.) See J. Opitz and S. Burst
          [(2019)](https://arxiv.org/abs/1911.03347). "Macro F1 and Macro F1", *arXiv*.

        """
    ret *= !return_type ? "" :
        """

        - `return_type=LittleDict`: type of returned measurement for `average=NoAvg()`
          case; if `LittleDict`, then keyed on levels of the target; can also be `Vector`

        """
    ret *=
        """

        - `levels::Union{Vector,Nothing}=nothing`: if `nothing`, levels are inferred from
           `ŷ` and `y` and, by default, ordered according to the element type of `y`.

        - `rev=false`: in the case of binary data, whether to reverse the `levels` (as
          inferred or specified); a `nothing` value is the same as `false`.

        """
    ret *= binary ?  "" :
        """

        - `perm=nothing`: in the general case, a permutation representing a re-ordering of
          `levels` (as inferred or specified); e.g., `perm = [1,3,2]` for data with three
          classes.

        """
    ret *=
        """

        - `checks=true`: when true, specified `levels` are checked to see they include all
          observed levels; set to `false` for speed.

        """
end


# TYPE AND CONSTRUCTORS

"""
    $CM.ConfusionMatrix{N,O,L}

Wrapper type for confusion matrices.

# Type parameters

- `N ≥ 2`: number of levels (classes)

- `O`: `true` if levels are explicitly understood to be ordered

- `L`: type of labels

$DOC_ROWCOL

See the $DOC_REF for more information.

# Public interface

Instances can be constructed directly using the `ConfusionMatrix` constructor (two methods
documented below) or, more typically, using [`$CM.confmat`](@ref). Other methods are:
[`$CM.matrix`](@ref) (to extract raw matrix), [`levels`](@ref), and [`isordered`](@ref).

Two instances are considered `==` if:

- The associated levels agree, *as sets*

- If both instances are ordered, then the levels also agree as vectors

- Access-by-level behaviour is the same (see below)

Instances need not have the same underlying matrix to be `==`.

Access elements via level as shown in this example:

```julia
$DOC_EG

julia> cm("a", "b")
3
```
Access by index is also possible, if the confusion matrix is ordered. Otherwise, you can
first extract the underlying matrix with [`$CM.matrix`](@ref). For options creating
ordered confusion matrices, see [`$CM.confmat`](@ref).

"""
struct ConfusionMatrix{N,O,L}
    mat::Matrix{Int}
    index_given_level::LittleDict{L, Int, NTuple{N,L}, NTuple{N,Int}}
end

"""
    $CM.ConfusionMatrix(m, index_given_level; ordered=false, checks=true)

Return an instance of [`$CM.ConfusionMatrix`](@ref) associating an ``n`` x ``n`` matrix
`m` with a dictionary `index_given_level`, mapping levels to integer indexes.

*Note.* Most users will construct confusion matrices using [`$CM.confmat`](@ref).

Specify `checks=false` to skip argument checks.

See also [`$CM.confmat`](@ref).

"""
function ConfusionMatrix(
    m,
    dic::OrderedCollections.FrozenLittleDict{L, I};
    checks=true,
    ordered=false,
    ) where {L,I<:Integer}
    s = size(m)
    N = s[1]
    if checks
        N == s[2] || throw(ArgumentError("Expected a square matrix."))
        N > 1 || throw(ArgumentError("Expected a matrix of size ≥ 2x2."))
        length(unique(keys(dic))) == N || throw(ArgumentError(
        "Expected dictionary with $N unique keys (levels) as "*
            "provided matrix has $N rows). "
        ))
        Set(values(dic)) == Set(1:N) || throw(ArgumentError(
        "Given matrix is $N x $N, expected dictionary values "*
            "to be integers from 1 to $N. "
        ))
    end
    
    ConfusionMatrix{N,ordered,L}(m, dic)
end
function ConfusionMatrix(
    m,
    dic::AbstractDict;
    checks=true,
    ordered=false,
    ) 
    ConfusionMatrix(m, freeze(dic); checks, ordered)
end

"""
    $CM.ConfusionMatrix(m, levels::AbstractVector; ordered=false, checks=true)

Return an instance of [`$CM.ConfusionMatrix`](@ref) associating an ``n`` x ``n`` matrix
 `m` with ``n`` `levels` (classes). Indices are associated with `levels` according to how
 the elements of `levels` are ordered.

*Note.* Most users will construct confusion matrices using [`$CM.confmat`](@ref).

Specify `checks=false` to skip argument checks.

See also [`$CM.confmat`](@ref).

"""
function ConfusionMatrix(m, levels::AbstractVector{L}; ordered=false, checks=true) where L
    s = size(m)
    N = s[1]
    if checks
        N == s[2] || throw(ArgumentError("Expected a square matrix."))
        N > 1 || throw(ArgumentError("Expected a matrix of size ≥ 2x2."))
        length(levels) == N || throw(ArgumentError(
            "Matrix dimension does not match number of levels provided. "
        ))
    end
    index_given_level =
        LittleDict{L, Int, Vector{L}, Vector{Int}}(levels, eachindex(levels))
    ConfusionMatrix(m, index_given_level; ordered, checks=false)
end


# BASIC FUNCTIONALITY

#  ## Equality:

import Base.(==)
function ==(
    cm1::ConfusionMatrix{N1,O1,L1},
    cm2::ConfusionMatrix{N2,O2,L2},
    ) where {N1,O1,L1,N2,O2,L2}
    N1 == N2 || return false

    levels1 = levels(cm1)
    levels2 = levels(cm2)

    # in case *both* conf matrices are ordered:
    if isordered(cm1) && isordered(cm2)
        levels1 == levels2 || return false
        matrix(cm1) == matrix(cm2) || return false
        return true
    end

    # all other cases:
    Set(levels1) == Set(levels2) || return false
    for j in 1:N1
        for i in 1:N1
            cm1(levels1[i], levels1[j]) == cm2(levels1[i], levels1[j]) || return false
        end
    end
    true
end


# ## Arithmetic

# no `zero()` is possible as this would need labels as type parameters, and the labels
# will generally not have `Symbol` type

import Base.(+)
import Base.(*)

+(
    cm1::ConfusionMatrix{N,O,L},
    cm2::ConfusionMatrix{N,O,L},
) where {N,O,L} = ConfusionMatrix{N,O,L}(
    cm1.mat + cm2.mat,
    cm1.index_given_level,
)

*(cm::ConfusionMatrix{N,O,L}, x::Int) where {N,O,L} =
    ConfusionMatrix{N,O,L}(cm.mat*x, cm.index_given_level)

*(x::Int, cm::ConfusionMatrix{N,O,L}) where {N,O,L} = cm*x


# ## Accessor functions

"""
    levels(m::$CM.ConfusionMatrix)

Return the levels associated with the confusion matrix `m`, in the order consistent with
the regular matrix returned by `$CM.matrix(cm)`.

"""
CategoricalArrays.levels(cm::ConfusionMatrix) =
    sort!(collect(keys(cm.index_given_level)), by=k->cm.index_given_level[k])

"""
    $CM.matrix(m::ConfusionMatrix; warn=true)

Return the regular `Matrix` associated with confusion matrix `m`.

"""
matrix(cm::ConfusionMatrix{N,true}; kwargs...) where N = cm.mat
@inline function matrix(cm::ConfusionMatrix{N,false}; warn=true) where N
    warn && @warn WARN_UNORDERED(levels(cm))
    cm.mat
end

"""
    $CM.isordered(m::ConfusionMatrix)

Return `true` if and only if the levels associated with `m` have been explicitly ordered.

"""
CategoricalArrays.isordered(cm::ConfusionMatrix{N,O,L}) where {N,O,L} = O

Base.getindex(cm::ConfusionMatrix{N,true}, I...) where N =
    Base.getindex(cm.mat, I...)

Base.getindex(cm::ConfusionMatrix{N,false}, I...) where N =
    throw(ERR_INDEX_ACCESS_DENIED)

@inline function (cm::ConfusionMatrix)(lev1, lev2)
    d = cm.index_given_level
    @inbounds return cm.mat[d[lev1], d[lev2]]
end


# COMPUTING A CONFUSION MATRIX FROM OBSERVATIONS AND PREDICTIONS

# ## Some helpers

function combined_levels(ŷ, y)
    unsorted = Set(skipmissing(y))
    union!(unsorted, skipmissing(ŷ))
    sort(collect(unsorted))
end

"""
    permutation(perm, rev, levels)

*Private method.*

Try to check that `perm` and `rev` make sense and are consistent with `levels` and return
the actual permutation to be used (which could be `nothing`).

"""
function permutation(perm, rev, levels::Nothing)
     !isnothing(perm) && !isnothing(rev) && throw(ArgumentError(
         "You cannot specify both `rev` and `perm`."
     ))
    perm
end
permutation(perm, rev, levels) = _permutation(perm, rev, length(levels))
_permutation(perm::Nothing, rev::Nothing, nlevels) = nothing
function _permutation(perm, rev::Nothing, nlevels)
    length(perm) == nlevels || throw(ArgumentError(
        "The length of `perm` must match the number of "*
            "number of levels."
    ))
    Set(perm) == Set(collect(1:nlevels)) || throw(ArgumentError(
        "`perm` must specify a valid permutation of "*
            "`[1, 2, ..., c]`, where `c` is "*
            "number of levels."
    ))
    perm
end
function _permutation(perm::Nothing, rev, nlevels)
    nlevels == 2 || throw(ArgumentError(
        "Keyword `rev` can only be used in binary case."
    ))
    [2, 1]
end
_permutation(perm, rev, nlevels) = throw(ArgumentError(
    "You cannot specify both `rev` and `perm`."
))

apply(perm, levels) = levels[perm]
apply(::Nothing, levels) = levels
get(f, x) = f(x)
get(dic::AbstractDict, x) = dic[x]

# `classes(...)` is same as `levels(...)` except it returns `CategoricalValue`s:
classes(p::CategoricalPool) = [p[i] for i in 1:length(p)]
classes(x::CategoricalValue) = classes(CategoricalArrays.pool(x))
classes(v::CategoricalArray) = classes(CategoricalArrays.pool(v))
classes(v::SubArray{<:Any, <:Any, <:CategoricalArray}) = classes(parent(v))

"""
    $CM.confmat(ŷ, y, levels=nothing, rev=false, perm=nothing, checks=true)

Return the confusion matrix corresponding to predictions `ŷ` and ground truth observations
`y`. Whenever `missing` occurs the corresponding prediction-ground-truth pair is skipped
in the counting.

$DOC_ORDER_REQUIREMENTS

$DOC_OPTIMISED

```julia
$DOC_EG

julia> cm("a", "b")
3

julia> CM.matrix(cm)
┌ Warning: Confusion matrix levels not explicitly ordered. Using the order, ["a", "b"].
└ @ StatisticalMeasures.ConfusionMatrices ~/MLJ/StatisticalMeasures/src/confusion_matrices.jl:120
2×2 Matrix{Int64}:
 2  3
 4  1

ordered_cm = CM.confmat(ŷ, y, levels=["b", "a"])

julia> ordered_cm("a", "b")
3

julia> CM.matrix(ordered_cm)
2×2 Matrix{Int64}:
 1  4
 3  2

julia> ordered_cm[2, 1]
3

```

$(DOC_OPTIONS())

See also [`$CM.ConfusionMatrix`](@ref), and the [*Confusion matrix* wikipedia
article](https://en.wikipedia.org/wiki/Confusion_matrix).

"""
function confmat(
    ŷ,
    y;
    levels::Union{Nothing,Vector}=nothing,
    perm::Union{Nothing,Vector{<:Integer}}=nothing,
    rev::Union{Nothing,Bool}=nothing,
    checks=true,
    )
    if checks && !isnothing(levels)
        issubset(combined_levels(ŷ, y), levels) ||
            throw(ERR_ORPHANED_OBSERVATIONS)
    end
    confmat(ŷ, y, levels, perm, rev)
end


# ## Second stage methods to handle the different argument types:

# Note: `indexer` below is like `index_given_level` but could be a function instead of a
# dictionary.

# fallback:
function confmat(ŷ, y, _levels, _perm, rev)
    levels = isnothing(_levels) ? combined_levels(ŷ, y) : _levels
    ordered = !all(isnothing.((_levels, _perm, rev)))

    # get the actual permutation to be used, or `nothing` if not permuting
    perm = permutation(_perm,  rev, levels)

    levels = apply(perm, levels)
    indexer = LittleDict(levels[i] => i for i in eachindex(levels)) |> freeze
    _confmat(ŷ, y, indexer, levels, ordered)
end

function confmat(
    ŷ::CatArrOrSub,
    y::CatArrOrSub,
    _levels::Nothing,
    _perm::Nothing,
    rev::Nothing
    )

    levels = classes(y)
    ordered =  isordered(y) && isordered(ŷ) || !all(isnothing.((_levels, _perm, rev)))

    indexer = levelcode

    _confmat(ŷ, y, indexer, levels, ordered)
end

function confmat(
    ŷ::CatArrOrSub,
    y::CatArrOrSub,
    _levels::Nothing,
    _perm,
    rev,
    )

    levels = classes(y)
    ordered =  isordered(y) && isordered(ŷ) || !all(isnothing.((_levels, _perm, rev)))

    # get the actual permutation to be used, or `nothing` if not permuting
    perm = permutation(_perm,  rev, levels)
    levels = apply(perm, levels)

    iperm = invperm(perm)
    indexer(x) = iperm[levelcode(x)]

    _confmat(ŷ, y, indexer, levels, ordered)
end


# ## Final method to do the computation
function _confmat(ŷ, y, indexer::F, levels, ordered) where F
    nc = length(levels)
    cmat = zeros(Int, nc, nc)
    @inbounds for i in eachindex(y)
        (ismissing(y[i]) || ismissing(ŷ[i])) && continue
        cmat[get(indexer, ŷ[i]), get(indexer, y[i])] += 1
    end
    return ConfusionMatrix(cmat, levels; ordered, checks=false)
end

function _confmat(ŷ, y, indexer::AbstractDict{L,I}, levels, ordered) where {L,I<:Integer}
    nc = length(levels)
    cmat = zeros(Int, nc, nc)
    @inbounds for i in eachindex(y)
        (ismissing(y[i]) || ismissing(ŷ[i])) && continue
        cmat[get(indexer, ŷ[i]), get(indexer, y[i])] += 1
    end
    return ConfusionMatrix(cmat, indexer; ordered, checks=false)
end


# DISPLAY

splitw(w::Int) = (sp1 = div(w, 2); sp2 = w - sp1; (sp1, sp2))

function Base.show(stream::IO, m::MIME"text/plain", cm::ConfusionMatrix{N}
                   ) where N
    labels = string.(levels(cm))
    width    = displaysize(stream)[2]
    mincw    = ceil(Int, 12/N)
    cw       = max(length(string(maximum(cm.mat))),maximum(length.(labels)),mincw)
    textlim  = 9
    firstcw  = max(length(string(maximum(cm.mat))),maximum(length.(labels)),textlim)    
    totalwidth = firstcw + cw * N + N + 2
    width < totalwidth && (show(stream, m, cm.mat); return)

    iob     = IOBuffer()
    wline   = s -> write(iob, s * "\n")
    splitcw = s -> (w = cw - length(s); splitw(w))
    splitfirstcw = s -> (w = firstcw - length(s); splitw(w))
    cropw   = s -> length(s) > textlim ? s[1:prevind(s, textlim)] * "…" : s

    # 1.a top box
    " "^(firstcw+1) * "┌" * "─"^((cw + 1) * N - 1) * "┐" |> wline
    gt = "Ground Truth"
    w  = (cw + 1) * N - 1 - length(gt)
    sp1, sp2 = splitw(w)
    " "^(firstcw+1) * "│" * " "^sp1 * gt * " "^sp2 * "│" |> wline
    # 1.b separator
    "┌" * "─"^firstcw * "┼" * ("─"^cw * "┬")^(N-1) * "─"^cw * "┤" |> wline
    # 2.a description line
    pr = "Predicted"
    sp1, sp2 = splitfirstcw(pr)
    partial = "│" * " "^sp1 * pr * " "^sp2 * "│"
    for c in 1:N
        # max = 10
        s = labels[c] |> cropw
        sp1, sp2 = splitcw(s)
        partial *= " "^sp1 * s * " "^sp2 * "│"
    end
    partial |> wline
    # 2.b separating line
    "├" * "─"^firstcw * "┼" * ("─"^cw * "┼")^(N-1) * ("─"^cw * "┤") |> wline
    # 2.c line by line
    for c in 1:N
        # line
        s  = labels[c] |> cropw
        sp1, sp2 = splitfirstcw(s)
        partial = "│" * " "^sp1 * s * " "^sp2 * "│"
        for r in 1:N
            e = string(cm.mat[c, r])
            sp1, sp2 = splitcw(e)
            partial *= " "^sp1 * e * " "^sp2 * "│"
        end
        partial |> wline
        # separator
        if c < N
            "├" * "─"^firstcw * "┼" * ("─"^cw * "┼")^(N-1) * ("─"^cw * "┤") |> wline
        end
    end
    # 2.d final line
    "└" * "─"^firstcw * "┴" * ("─"^cw * "┴")^(N-1) * ("─"^cw * "┘") |> wline
    write(stream, take!(iob))
end

function Base.show(stream::IO, cm::ConfusionMatrix{N}) where N
    mat = matrix(cm, warn=false)
    print(stream, "ConfusionMatrix{$N}($(repr(mat)))")
end

# ## STATISTICAL FUNCTIONS ON CONFUSION MATRICES

# accuracy, kappa, matthew's correlation:
for f in [
    :accuracy,
    :kappa,
    :matthews_correlation,
    ]
    quote
        $f(cm::ConfusionMatrix, args...; kwargs...) =
            Functions.$f(matrix(cm, warn=false), args...; kwargs...)
    end |> eval
end

# order sensitive binary functions:
for f in [
    :true_positive,
    :true_negative,
    :false_positive,
    :false_negative,
    :true_positive_rate,
    :true_negative_rate,
    :false_positive_rate,
    :false_negative_rate,
    :false_discovery_rate,
    :negative_predictive_value,
    :positive_predictive_value,
    :fscore,
    ]
    quote
        $f(cm::ConfusionMatrix, args...; kwargs...) =
            Functions.$f(matrix(cm), args...; kwargs...)
    end |> eval
end

# multiclass based on one-against many binary functions:
for f in [
    :multiclass_true_positive,
    :multiclass_true_negative,
    :multiclass_false_positive,
    :multiclass_false_negative,
    :multiclass_true_positive_rate,
    :multiclass_true_negative_rate,
    :multiclass_false_positive_rate,
    :multiclass_false_negative_rate,
    :multiclass_false_discovery_rate,
    :multiclass_negative_predictive_value,
    :multiclass_positive_predictive_value,
    :multiclass_fscore,
    ]
    quote
        $f(cm::ConfusionMatrix, args...; kwargs...) =
            Functions.$f(matrix(cm, warn=false), args...; kwargs...)
    end |> eval
end



end # module

using .ConfusionMatrices

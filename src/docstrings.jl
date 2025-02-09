const PROPER_SCORING_RULES = "Gneiting and Raftery [(2007)]"*
    "(https://doi.org/10.1198/016214506000001437), \"Strictly"*
    "Proper Scoring Rules, Prediction, and Estimation\""
const DOC_FINITE =
    "`Union{Finite,Missing}` (multiclass classification)"
const DOC_FINITE_BINARY =
    "`Union{Finite{2},Missing}` (binary classification)"
const DOC_ORDERED_FACTOR =
    "`Union{OrderedFactor,Missing}` (classification of ordered target)"
const DOC_ORDERED_FACTOR_BINARY =
    "`Union{OrderedFactor{2},Missing}` "*
    "(binary classification where definition of \"positive\" class matters)"
const DOC_CONTINUOUS = "`Union{Continuous,Missing}` (regression)"
const DOC_COUNT = "`Union{Count,Missing}`"
const DOC_MULTI = "`Union{Missing,T}` where `T` is `Continuous` "*
    "or `Count` (for respectively continuous or discrete Distribution.jl objects in "*
    "`ŷ`) or  `OrderedFactor` or `Multiclass` "*
    "(for `UnivariateFinite` distributions in `ŷ`)"
const DOC_INFINITE = "`Union{Infinite,Missing}`"
const INVARIANT_LABEL =
    "This metric is invariant to class reordering."
const VARIANT_LABEL =
    "This metric is *not* invariant to class re-ordering"
const DOC_WEIGHTS = "Any iterator with a `length` generating `Real` elements can be used "*
    "for `weights`. "
const DOC_CLASS_WEIGHTS =
    "The keys of `class_weights` should include all conceivable values for "*
    "observations in `y`, and values should be `Real`. "
DOC_MULTITARGET(C) =
    "Specifically, compute the multi-target version of [`$C`](@ref). "*
    "Some kinds of tabular input are supported.\n\n"*
    "In array arguments the last dimension is understood to be the observation "*
    "dimension. "*
    "The `atomic_weights` are weights for each "*
    "component of the "*
    "multi-target. Unless equal to `nothing` (uniform weights) "*
    "the length of `atomic_weights` will generally match the number of columns of "*
    "`y`, if `y` is a table, or the number of "*
    "rows of `y`, if `y` is a matrix. "
const DOC_SUPPORTS_TABLES =
    "Alternatively, `y` and `ŷ` can be some types of table, provided elements have "*
    "the approprate scitype. "
_decorate(s::AbstractString) = "`$s`"
_decorate(v) = join(_decorate.(v), ", ")

"""
    docstring(C; body="", aliases="", footer="", scitype="")

*Private method.*

Here `C` should be a constructor such that all measures constructed using `C` have the
same trait values as `C()`.

"""
function docstring(signature; body="", footer="", scitype="")
    ex = Meta.parse(signature)
    if @capture(ex, Cex_())
        C = eval(Cex)
        n_params = 0
    elseif @capture(ex, Cex_(; args__))
        C = eval(Cex)
        n_params = length(args)
    else
        error(
        "First `docstring` argument must have the form `Constructor()` or "*
            "`Constructor(; p1=p1_default, p2=p2_default, ...)`. "
        )
    end
    m = C()
    _aliases = _decorate(measures()[C].aliases)
    human_name = API.human_name(m)
    kind_of_proxy = API.kind_of_proxy(m)
    if isempty(scitype)
        scitype = "`$(API.observation_scitype(m))`"
    end

    ret =
        """
            $signature

        """
    ret *= "Return a callable measure for computing the $human_name"
    isempty(_aliases) ||
        (ret  *= ". Aliases: "*
         "$_aliases")
    ret *= ".\n\n"
    m_str = n_params == 0 ? "$C()" : "m"
    ret *= "    $m_str(ŷ, y)\n"
    API.supports_weights(m) &&
        (ret *= "    $m_str(ŷ, y, weights)\n")
    API.supports_class_weights(m) &&
        (ret *= "    $m_str(ŷ, y, class_weights::AbstractDict)\n")
    API.supports_weights(m) && API.supports_class_weights(m) &&
        (ret *= "    $m_str(ŷ, y, weights, class_weights::AbstractDict)\n")

    ret *= "\n"
    if n_params > 0
        ret *= "Evaluate some measure `m` returned by the `$C` constructor "*
            "(e.g., `m = $C()`) on "*
            "predictions `ŷ`, "*
            "given ground truth observations `y`. "
    else
        ret *= "Evaluate `$m_str` on "*
            "predictions `ŷ`, "*
            "given ground truth observations `y`. "
    end
    isempty(body) || (ret *= "$body\n\n")
    API.supports_weights(m) &&
        (ret *= DOC_WEIGHTS)
    API.supports_class_weights(m) &&
        (ret *= DOC_CLASS_WEIGHTS)
    API.can_report_unaggregated(m) &&
        (ret *= "\n\nMeasurements are aggregated. "*
        "To obtain a separate measurement for each observation, "*
        "use the syntax `measurements($m_str, ŷ, y)`. ")
    ret *= "Generally, an observation `obs` in `MLUtils.eachobs(y)` is expected to satisfy "*
        "`ScientificTypes.scitype(obs)<:`$scitype. "
    # if kind_of_proxy == LearnAPI.Point()
    #     ret *= "The same is true for `ŷ`. "
    # else
    #     ret *= "Each observation in `ŷ` should be this kind of proxy for actual "*
    #         "observations: `$kind_of_proxy`. "
    # end
    if API.can_consume_tables(m)
        ret *= DOC_SUPPORTS_TABLES
    end
    isempty(footer) ||(ret *= "\n\n$footer")
    ret *= "\n\nFor a complete dictionary of available measures, "*
        "keyed on constructor, run "*
        "[`measures()`](@ref). "
    ret *= "\n\n# Traits\n"
    metadata = measures()[C]
    for trait in keys(metadata)
        trait == :aliases && continue
        value  = getproperty(metadata, trait)
        ret *= "    $trait = $value\n"
    end
    return ret
end

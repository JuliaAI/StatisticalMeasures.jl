const TRAITS_GIVEN_CONSTRUCTOR = LittleDict()

const ERR_BAD_CONSTRUCTOR = ArgumentError(
    "Constructor must have a zero argument method. "
)

const ERR_BAD_KWARG(trait) = ArgumentError(
    "`$trait` is not a valid measure trait name. You must choose one of these: "*
        "$(API.OVERLOADABLE_TRAITS_LIST). "
)

"""
    measures(; trait_options...)

*Experimental* and subject to breaking behavior between patch releases.

Return a dictionary, `dict`, keyed on measure constructors provided by
StatisticalMeasures.jl. The value of `dict[constructor]` provides information about traits
(measure "metadata") shared by all measures constructed using the syntax
`constructor(args...)`.

# Trait options

One can filter on the basis of measure trait values, as shown in this example:

```
using StatisticalMeasures
import ScientificTypesBase.Multiclass

julia> measures(
    observation_scitype = Union{Missing,Multiclass},
    supports_class_weights = true,
)
```

---

    measures(y; trait_filters...)
    measures(yhat, y; trait_filters...)

*Experimental* and subject to breaking behavior between patch releases.

Assuming, ScientificTypes.jl has been imported, find measures that can be applied to data
with the specified data arguments `(y,)` or `(yhat, y)`. It is assumed that the arguments
contain multiple observations (have types implementing `MLUtils.getobs`).

Returns a dictionary keyed on the constructors of such measures. Additional
`trait_filters` are the same as for the zero argument `measures` method.

```julia
using StatisticalArrays
using ScientificTypes

julia> measures(rand(3), rand(3), supports_weights=false)
LittleDict{Any, Any, Vector{Any}, Vector{Any}} with 1 entry:
  RSquared => (aliases = ("rsq", "rsquared"), consumes_multiple_observations = true, can_re…
```

*Warning.* Matching is based only on the *first* observation of the arguments provided,
and must be interpreted carefully if, for example, `y` or `yhat` are vectors with `Union`
or other abstract element types.

"""
measures(; trait_options...) = filter(TRAITS_GIVEN_CONSTRUCTOR) do (_, metadata)
    trait_value_pairs = collect(trait_options)
    traits = first.(trait_value_pairs)
    for trait in traits
        trait in API.OVERLOADABLE_TRAITS || throw(ERR_BAD_KWARG(trait))
    end
    all(trait_value_pairs) do pair
        trait = first(pair)
        value = last(pair)
        getproperty(metadata, trait) == value
    end
end

"""
    measures(needle::Union{AbstractString,Regex}; trait_options...)

*Experimental* and subject to breaking behavior between patch releases.

Find measures that contain `needle` in their document string.  Returns a dictionary keyed
on the constructors of such measures.

```
julia> measures("Matthew")
LittleDict{Any, Any, Vector{Any}, Vector{Any}} with 1 entry:
  MatthewsCorrelation => (aliases = ("matthews_correlation", "mcc"), consumes_multiple_obse…
```
"""
function measures(needle::Union{AbstractString,Regex}; trait_options...)
    filter(measures(; trait_options...)) do (constructor, _)
        doc = Base.Docs.doc(constructor) |> string
        occursin(needle, doc)
    end
end

"""
    StatisticalMeasures.register(constructor, aliases=String[])

**Private method.**

Add the specified measure `constructor` to the keys of the dictionary returned by
`$API.measures()`.

At registration, the dictionary value assigned to the key `constructor` is a named tuple
keyed on trait names, with values the corresponding traits of the measure
`constructor()`. Registration implies a contract that these trait values are the same for
all measures of the form `constructor(args...)`. An additional key of the named tuple is
`:aliases`, which has the specified `aliases` as value. The aliases must be `String`s.

"""
function register(constructor, aliases...)
    measure = try
        constructor()
    catch ex
        @error ERR_BAD_CONSTRUCTOR
        throw(ex)
    end
    traits = NamedTuple{tuple(API.METADATA_TRAITS...)}(tuple(
        (trait(measure) for trait in  API.METADATA_TRAIT_FUNCTIONS)...
            ))
    extended_traits = (; aliases, traits...)
    TRAITS_GIVEN_CONSTRUCTOR[constructor] = extended_traits
end

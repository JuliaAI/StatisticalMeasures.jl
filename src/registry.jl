const TRAITS_GIVEN_CONSTRUCTOR = LittleDict()

const ERR_BAD_CONSTRUCTOR = ArgumentError(
    "Constructor must have a zero argument method. "
)

const ERR_BAD_KWARG(trait) = ArgumentError(
    "`$trait` is not a valid measure trait name. You must choose one of these: "*
        "$(API.OVERLOADABLE_TRAITS_LIST). "
)

"""
    measures(; filter_options...)

*Experimental* and subject to breaking behavior between patch releases.

Return a dictionary, `dict`, keyed on measure constructors provided by
StatisticalMeasures.jl. The value of `dict[constructor]` provides information about traits
(measure "metadata") shared by all measures constructed using the syntax
`constructor(args...)`.

# Filter options

One can filter on the basis of measure trait values, as shown in this example:

```
using StatisticalMeasures
using ScientificTypes

julia> measures(
    observation_scitype = Union{Missing,Multiclass},
    supports_class_weights = true,
)
```

For more general searches, use a `filter(measures()) do (_, metadata) ... end`
construction.

"""
measures(; kwargs...) = filter(TRAITS_GIVEN_CONSTRUCTOR) do (_, metadata)
    trait_value_pairs = collect(kwargs)
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
    measures(needle::Union{AbstractString,Regex})

*Experimental* and subject to breaking behavior between patch releases.

Return a dictionary keyed on measure constructors that contain `needle` in their document
strings.

```
julia> measures("root")
LittleDict{Any, Any, Vector{Any}, Vector{Any}} with 8 entries:
  RootMeanSquaredError          => (aliases = ("rms", "rmse", "root_mean_squared_error"), c…
  MultitargetRootMeanSquaredEr… => (aliases = ("multitarget_rms", "multitarget_rmse", "mult…
  RootMeanSquaredLogError       => (aliases = ("rmsl", "rmsle", "root_mean_squared_log_erro…
  MultitargetRootMeanSquaredLo… => (aliases = ("multitarget_rmsl", "multitarget_rmsle", "mu…
  RootMeanSquaredLogProportion… => (aliases = ("rmslp1",), consumes_multiple_observations =…
  MultitargetRootMeanSquaredLo… => (aliases = ("multitarget_rmslp1",), consumes_multiple_ob…
  RootMeanSquaredProportionalE… => (aliases = ("rmsp",), consumes_multiple_observations = t…
  MultitargetRootMeanSquaredPr… => (aliases = ("multitarget_rmsp",), consumes_multiple_obse…
```

"""
function measures(needle::Union{AbstractString,Regex}; kwargs...)
    filter(measures(; kwargs...)) do (constructor, _)
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

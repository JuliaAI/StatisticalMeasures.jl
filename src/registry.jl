const TRAITS_GIVEN_CONSTRUCTOR = LittleDict()

const ERR_BAD_CONSTRUCTOR = ArgumentError(
    "Constructor must have a zero argument method. "
)

"""
    StatisticalMeasusures.measures()

*Experimental* and subject to breaking behavior between patch releases.

Return a dictionary, `dict`, keyed on measure constructors provided by
StatisticalMeasures.jl. The value of `dict[constructor]` provides information about
traits shared by all measures constructed using the syntax `constructor(args...)`.

"""
measures() = TRAITS_GIVEN_CONSTRUCTOR

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

const PATH_TO_DOCS_SRC = joinpath(@__DIR__, "src")


function table(constructors)
    traits_given_constructor = measures()
    sort!(constructors, by = string)
    sort!(
        constructors,
        by = c-> string(traits_given_constructor[c].observation_scitype),
        rev=true,
    )
    table = "| constructor / instance aliases | observation scitype              |\n"
    table *= "|:-------------------------------|:---------------------------------|\n"
    for c in constructors
        traits = traits_given_constructor[c]
        aliases = join(map(a->"`$a`", traits.aliases), ", ")
        scitype = traits.observation_scitype
        proxy = split(string(traits.kind_of_proxy), '.') |> last
        table *= "| [`$c`](@ref) | `$scitype` |\n"
#        table *= "| $aliases     |            |\n"
    end
    table
end

function alias_table()
    traits_given_constructor = measures()
    alias_constructor_pairs = []
    for constructor in keys(traits_given_constructor)
        for alias in traits_given_constructor[constructor].aliases
            push!(alias_constructor_pairs, ("[`$constructor`](@ref)", "`$alias`"))
        end
    end
    table = "| alias | constructed with |\n"
    table *= "|:-----|:-----------------|\n"
    for pair in sort!(alias_constructor_pairs, by=last)
        table *= "| $(last(pair)) | $(first(pair)) |\n"
    end
    table
end

function write_measures_page(path=PATH_TO_DOCS_SRC)
    pagename = "_auto_generated_list_of_measures.md"
    pagepath = joinpath(path, pagename)
    traits_given_constructor = measures()
    all_constructors = keys(traits_given_constructor) |> collect
    open(pagepath, "w") do stream
        page =
"""
# The Measures

### Quick links

- [List of aliases](@ref aliases)
- [Classification measures (non-probabilistic)](@ref)
- [Regression measures (non-probabilistic)](@ref)
- [Probabilistic measures](@ref)

## Scientific type of observations

Measures can be classified according to the [scientific
type](https://juliaai.github.io/ScientificTypes.jl/dev/) of the target observations they
consume (given by the value of the trait,
[`$StatisticalMeasuresBase.observation_scitype(measure)`](@ref)):

| observation scitype | meaning                                                |
|:--------------------|:-------------------------------------------------------|
| `Finite`            | general classification                                 |
| `Finite{2}=Binary`  | binary classification                                  |
| `OrderedFactor`     | classification (class order matters)                   |
| `OrderedFactor{2}`  | binary classification (order matters)                  |
| `Continuous`        | regression                                             |
| `Infinite`          | regression, including integer targets for `Count` data |
| `AbstractArray{T}`  | multitarget version of `T`, some tabular data okay     |

Measures are not strict about data conforming to the declared observation scitype. For
example, where `OrderedFactor{2}` is expected, `Finite{2}` will work, and in fact most
eltypes will work, so long as there are only two classes. However, you may get warnings
that mitigate possible misinterpretations of results (e.g., about which class is the
"positive" one). Some warnings can be suppressed by explicitly specifying measure
parameters, such as `levels`.

To be 100% safe and avoid warnings, use data with the recommended observation scitype.

## On multi-target measures and tabular data

All multi-target measures below (the ones with `AbstractArray` observation scitypes) also
handle some forms of tabular input, including `DataFrame`s and Julia's native "row table"
and "column table" formats. This is not reflected by the declared observation
scitype. Instead, you can inspect the trait
[`StatisticalMeasuresBase.can_consume_tables`](@ref) or consult the measure document
string.

"""
        page *= "\n## Classification measures (non-probabilistic)\n\n"
        constructors = filter(all_constructors) do c
            traits = traits_given_constructor[c]
            traits.kind_of_proxy == LearnAPI.LiteralTarget() &&
                traits.observation_scitype <: Union{
                    Union{Missing,Finite},
                    AbstractArray{<:Union{Missing,Finite}},
                }
        end
        page *= table(constructors)
        page *= "\n## Regression measures (non-probabilistic)\n\n"
        constructors = filter(all_constructors) do c
            traits = traits_given_constructor[c]
            traits.kind_of_proxy == LearnAPI.LiteralTarget() &&
                traits.observation_scitype <: Union{
                    Union{Missing,Infinite},
                    AbstractArray{<:Union{Missing,Infinite}},
                }
        end
        page *= table(constructors)
        page *= "\n## Probabilistic measures\n\n"*
            "These are measures where each prediction is a "*
            "probability mass or density function, over "*
            "the space of possible ground truth observations. Specifically, "*
            "[`StatisticalMeasuresBase.kind_of_proxy(measure)`](@ref) "*
            "`== LearnAPI.Distribution()`.\n\n"
        constructors = filter(all_constructors) do c
            traits_given_constructor[c].kind_of_proxy == LearnAPI.Distribution()
        end
        page *= table(constructors)
        page *= "## [List of aliases](@id aliases)\n\n"
        page *= "Some of the measures constructed using specific parameter values have "*
            "pre-defined names associated with them that are exported by "*
            "StatisticalMeasures.jl "*
            "These are called *aliases*.\n\n"
        page  *= alias_table()
        page *= "\n## Reference\n\n"
        page *= "```@docs\n"
        for c in all_constructors
            page *= "$c\n"
        end
        page *= "```"

        write(stream, page)
        nothing
    end
end

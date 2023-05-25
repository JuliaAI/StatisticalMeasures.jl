using Documenter
using StatisticalMeasures
using StatisticalMeasures.StatisticalMeasuresBase
using StatisticalMeasures.LearnAPI
using ScientificTypesBase
using ScientificTypes

const REPO="github.com/JuliaAI/StatisticalMeasures.jl"

include("make_tools.jl")
write_measures_page()

makedocs(;
         modules=[
             StatisticalMeasures,
             StatisticalMeasuresBase,
             ConfusionMatrices,
             Functions,
         ],
    format=Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages=[
        "Overview" => "index.md",
        "Examples of usage" => "examples_of_usage.md",
        "The Measures" => "auto_generated_list_of_measures.md",
        "Confusion Matrices" => "confusion_matrices.md",
        "Receiver Operator Characteristics" => "roc.md",
        "Tools" => "tools.md",
        "Reference" => "reference.md",
    ],
    repo="https://$REPO/blob/{commit}{path}#L{line}",
    sitename="StatisticalMeasures.jl"
)

deploydocs(
    ; repo=REPO,
    devbranch="dev",
    push_preview=false,
)

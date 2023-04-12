# Tools

| method                                        | description                                                                          |
|:----------------------------------------------|:-------------------------------------------------------------------------------------|
| [`measurements`](@ref)`(measure, ...)`        | for obtaining per-observation measurements, instead of aggregated ones               |
| [`measures()`](@ref)                          | dictionary of traits keyed on measure constructors                                   |
| [`unfussy(measure)`](@ref)                    | new measure without argument checks¹                                                 |
| [`multimeasure`](@ref)`(measure; options...)` | wrapper to broadcast measures over multiple observations                             |
| [`robust_measure(measure)`](@ref)             | wrapper to silently treat unsupported weights as uniform                             |
| [`Measure(measure)`](@ref)                    | wrapper for 3rd party measures with different calling syntax (e.g. LossFunctions.jl) |


¹For measures provided by StatisticalMeasures; behaviour for general measures may differ.

For more on defining your own measures, see the [StatisticalMeasuresBase.jl
documentation](https://juliaai.github.io/StatisticalMeasuresBase.jl/dev/).

  
```@docs
measurements
measures
unfussy
multimeasure
robust_measure
Measure
```

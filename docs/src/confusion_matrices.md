# Confusion Matrices

Users typically construct confusion matrices using the `confmat` measure, or other
variants that can be constructed using the [`ConfusionMatrix`](@ref) constructor. See
[Examples of Usage](@ref) for examples. (A pure function version of `confmat`, with options, is
[`ConfusionMatrices.confmat`](@ref).)

The `ConfusionMatrices` submodule of StatisticalMeasures.jl provides some methods for
extracting data from these matrices, detailed below.

| method                                    | description                                                      |
|:------------------------------------------|:-----------------------------------------------------------------|
| `cm[i, j]`                                | count for a  `i`th class prediction and `j`th class ground truth |
| `cm(p, g)`                                | count for a  class `p` prediction and class `g` ground truth     |
| [`ConfusionMatrices.matrix(cm)`](@ref)    | return the raw matrix associated with confusion matrix `cm`      |
| [`ConfusionMatrices.isordered(cm)`](@ref) | `true` if levels of `cm` have been explicitly ordered            |
| [`ConfusionMatrices.levels(cm)`](@ref)    | return the target levels (classes) associated with `cm`          |

## Reference

```@docs
ConfusionMatrices.matrix
ConfusionMatrices.isordered
ConfusionMatrices.levels
ConfusionMatrices.ConfusionMatrix
ConfusionMatrices.confmat
```

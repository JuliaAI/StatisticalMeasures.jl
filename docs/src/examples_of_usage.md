# Examples of Usage

### [Calling syntax](@id calling)

A measure `m` is called with this syntax:

```julia
m(ŷ, y)
m(ŷ, y, weights)
m(ŷ, y, class_weights::AbstractDict)
m(ŷ, y, weights, class_weights)
```
where `y` is ground truth and `ŷ` predictions. This package provides measure *constructors*, such as `BalancedAccuracy`:

```@example 19
using StatisticalMeasures
using StatisticalMeasures

m = BalancedAccuracy(adjusted=true)
m(["O", "X", "O", "X"], ["X", "X", "X", "O"], [1, 2, 1, 2])
```

[Aliases](@ref aliases) are provided for commonly applied instances:

```@example 19
bacc == BalancedAccuracy() == BalancedAccuracy(adjusted=false)
```

### Contents

- [Binary classification](@ref)
- [Multi-class classification](@ref)
- [Probabilistic classification](@ref)
- [Non-probabilistic regression](@ref)
- [Probabilistic regression](@ref)
- [Custom multi-target measures](@ref)
- [Using losses from LossFunctions.jl](@ref)


## Binary classification

```@example 20
using StatisticalMeasures
using CategoricalArrays

# ground truth:
y = categorical(
        ["X", "X", "X", "O", "X", "X", "O", "O", "X"],
        ordered=true,
)

# prediction:
ŷ = categorical(
   ["O", "X", "O", "X", "O", "O", "O", "X", "X"],
   levels=levels(y),
   ordered=true,
)

accuracy(ŷ, y)
```

```@example 20
weights = [1, 2, 1, 2, 1, 2, 1, 2, 1]
accuracy(ŷ, y, weights)
```

```@example 20
class_weights = Dict("X" => 10, "O" => 1)
accuracy(ŷ, y, class_weights)
```

```@example 20
accuracy(ŷ, y, weights, class_weights)
```

To get individual per-observation weights, use [`measurements`](@ref):

```@example 20
measurements(accuracy, ŷ, y, weights, class_weights)
```

```@example 20
kappa(ŷ, y)
```

```@example 20
mat = confmat(ŷ, y)
```

Some measures can be applied directly to confusion matrices:

```@example 20
kappa(mat)
```

## Multi-class classification

```@example 23
using StatisticalMeasures
using CategoricalArrays
import Random
Random.seed!()

y = rand("ABC", 1000) |> categorical
ŷ = rand("ABC", 1000) |> categorical
class_weights = Dict('A' => 1, 'B' =>2, 'C' => 10)
MulticlassFScore(beta=0.5, average=MacroAvg())(ŷ, y,  class_weights)
```

```@example 23
MulticlassFScore(beta=0.5, average=NoAvg())(ŷ, y,  class_weights)
```

Unseen classes are tracked, when using `CategoricalArrays`, as here:

```@example 23
# find 'C'-free indices
mask = y .!= 'C' .&& ŷ .!= 'C';
# and remove:
y = y[mask]
ŷ = ŷ[mask]
'C' in y ∪ ŷ
```

```@example 23
confmat(ŷ, y)
```

## Probabilistic classification

To mitigate ambiguity around representations of predicted probabilities, a probabilistic
prediction of categorical data is expected to be represented by a `UnivariateFinite`
distribution, from the package
[CategoricalDistributions.jl](https://github.com/JuliaAI/CategoricalDistributions.jl). This
is the form delivered, for example, by
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) classification models.

```@example 24
using StatisticalMeasures
using CategoricalArrays
using CategoricalDistributions

y = categorical(["X", "O", "X", "X", "O", "X", "X", "O", "O", "X"], ordered=true)
X_probs = [0.3, 0.2, 0.4, 0.9, 0.1, 0.4, 0.5, 0.2, 0.8, 0.7]
ŷ = UnivariateFinite(["O", "X"], X_probs, augment=true, pool=y)
ŷ[1]
```

```@example 24
auc(ŷ, y)
```

```@example 24
measurements(log_loss, ŷ, y)
```

```@example 24
measurements(brier_score, ŷ, y)
```

We note in passing that `mode` and `pdf` methods can be applied to `UnivariateFinite`
distributions. So, for example, we can do:

```@example 24
confmat(mode.(ŷ), y)
```

## Non-probabilistic regression

```@example 25
using StatisticalMeasures

y = [0.1, -0.2, missing, 0.7]
ŷ = [-0.2, 0.1, 0.4, 0.7]
rsquared(ŷ, y)
```
```@example 25
weights = [1, 3, 2, 5]
rms(ŷ, y, weights)
```
```@example 25
measurements(LPLoss(p=2.5), ŷ, y, weights)
```

Here's an example of a multi-target regression measure, for data with 3 observations of a
2-component target:

```@example 25
# last index is observation index:
y = [1 2 3; 2 4 6]
ŷ = [2 3 4; 4 6 8]
weights = [8, 7, 6]
ŷ - y
```

```@example 25
MultitargetLPLoss(p=2.5)(ŷ, y, weights)
```

```@example 25
# one "atomic weight" per component of target:
MultitargetLPLoss(p=2.5, atomic_weights = [1, 10])(ŷ, y, weights)
```

Some tabular formats (e.g., `DataFrame`) are also supported:

```@example 25
using Tables
t = y' |> Tables.table |> Tables.rowtable
t̂ = ŷ' |> Tables.table |> Tables.rowtable
MultitargetLPLoss(p=2.5)(ŷ, y, weights)
```

## Probabilistic regression

```@example 26
using StatisticalMeasures
import Distributions:Poisson, Normal
import Random.seed!
seed!()

y = rand(20)
ŷ = [Normal(rand(), 0.5) for i in 1:20]
ŷ[1]
```

```@example 26
log_loss(ŷ, y)
```

```@example 26
weights = rand(20)
log_loss(ŷ, y, weights)
```

```@example 26
weights = rand(20)
measurements(log_loss, ŷ, y, weights)
```

An example with `Count` (integer) data:

```@example 26
y = rand(1:10, 20)
ŷ = [Poisson(10*rand()) for i in 1:20]
ŷ[1]
```

```@example 26
brier_loss(ŷ, y)
```

## Custom multi-target measures

Here's an example of constructing a multi-target regression measure, for data with 3
observations of a 2-component target:

```@example 27
using StatisticalMeasures

# last index is observation index:
y = ["X" "O" "O"; "O" "X" "X"]
ŷ = ["O" "X" "O"; "O" "O" "O"]
```

```@example 27
# if prescribed, we need one "atomic weight" per component of target:
multitarget_accuracy= multimeasure(accuracy, atomic_weights=[1, 2])
multitarget_accuracy(ŷ, y)
```

```@example 27
measurements(multitarget_accuracy, ŷ, y)
```

```@example 27
# one weight per observation:
weights = [1, 2, 10]
measurements(multitarget_accuracy, ŷ, y, weights)
```

See [`multimeasure`](@ref) for options. Refer to the
[StatisticalMeausureBase.jl](https://github.com/JuliaAI/StatisticalMeasuresBase.jl/actions)
documentation for advanced measure customization.


## Using losses from LossFunctions.jl

The margin losses in LossFunctions.jl can be regarded as binary probabilistic measures,
but they cannot be directly called on `CategoricalValue`s and `UnivariateFinite`
distributions, as we do for similar measures provided by `StatisticalMeasures` (see
[Probabilistic classification](@ref) above). If we want this latter behavior, then we need
to wrap these losses using `Measure`:

```@example 28
using StatisticalMeasures
import LossFunctions as LF

loss = Measure(LF.L1HingeLoss())
```

This loss can only called on scalars (true for LossFunctions.jl losses since v0.10):

```@example 28
using CategoricalArrays
using CategoricalDistributions

y = categorical(["X", "O", "X", "X"], ordered=true)
X_probs = [0.3, 0.2, 0.4, 0.9]
ŷ = UnivariateFinite(["O", "X"], X_probs, augment=true, pool=y)
loss(ŷ[1], y[1])
```

This is remedied with the `multimeasure` wrapper:

```@example 28
import StatisticalMeasuresBase.Sum

loss_on_vectors = multimeasure(loss, mode=Sum())
loss_on_vectors(ŷ, y)
```

```@example 28
class_weights = Dict("X"=>1, "O"=>10)
loss_on_vectors(ŷ, y, class_weights)
```

```@example 28
measurements(loss_on_vectors, ŷ, y)
```

Wrap again, as shown in the preceding section, to get a multi-target version.

For distance-based loss functions, wrapping in `Measure` is not strictly
necessary, but does no harm.

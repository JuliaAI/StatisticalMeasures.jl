# Receiver Operator Characteristics

## Example

```@example 70
using StatisticalMeasures
using CategoricalArrays
using CategoricalDistributions

# ground truth:
y = categorical(["X", "O", "X", "X", "O", "X", "X", "O", "O", "X"], ordered=true)

# probabilistic predictions:
X_probs = [0.3, 0.2, 0.4, 0.9, 0.1, 0.4, 0.5, 0.2, 0.8, 0.7]
ŷ = UnivariateFinite(["O", "X"], X_probs, augment=true, pool=y)
ŷ[1]
```

```julia
using Plots
curve = roc_curve(ŷ, y)
plt = plot(curve, legend=false)
plot!(plt, xlab="false positive rate", ylab="true positive rate")
plot!([0, 1], [0, 1], linewidth=2, linestyle=:dash, color=:black)
```

![](assets/roc_curve.png)


```@example 70
auc(ŷ, y) # maximum possible is 1.0
```

## Reference

```@docs
roc_curve
```

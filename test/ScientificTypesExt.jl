using ScientificTypes
using ScientificTypes.Tables

n = 10
p = 3
y    = rand(p, n)
yhat = rand(p, n)
t = y'       |> Tables.table |> Tables.columntable
that = yhat' |> Tables.table |> Tables.columntable

ms = measures(t)
@test all(ms) do (_, metadata)
    AbstractVector{Continuous} <: metadata.observation_scitype
end

ms = measures(that, t)
@test all(ms) do (_, metadata)
    AbstractVector{Continuous} <: metadata.observation_scitype
end

y = categorical(rand("ab", n))
yhat = UnivariateFinite(levels(y), rand(n), augment=true, pool=y)
ms2 = measures((yhat, y))
@test all(ms2) do (_, metadata)
    Multiclass{2} <: metadata.observation_scitype &&
        metadata.kind_of_proxy == LearnAPI.Distribution()
end

@test isempty(intersect(keys(ms), keys(ms2)))

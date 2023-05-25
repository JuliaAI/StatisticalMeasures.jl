module ScientificTypesExt
using ScientificTypes
import ScientificTypes.Tables
using StatisticalMeasures
import StatisticalMeasuresBase.MLUtils
import Distributions
import LearnAPI

# # HELPERS

guess_observation_scitype(y) = guess_observation_scitype(y, Val(Tables.istable(y)))
guess_observation_scitype(y, ::Val{false}) = MLUtils.getobs(y, 1) |> scitype
guess_observation_scitype(table, ::Val{true}) =
    MLUtils.getobs(table, 1) |> collect |> scitype


# # MEASURE SEARCH BASED ON ARGUMENTS

StatisticalMeasures.measures(y; kwargs...) = filter(measures(; kwargs...)) do (_, metadata)
    guess_observation_scitype(y) <: metadata.observation_scitype
end

function StatisticalMeasures.measures(yhat, y; trait_filters...)
    y_scitype = guess_observation_scitype(y)
    yhat_scitype = guess_observation_scitype(yhat)
    filter(measures(; trait_filters...)) do (_, metadata)
        requirement1 = y_scitype <: metadata.observation_scitype
        proxy = metadata.kind_of_proxy
        requirement2 = if proxy == LearnAPI.LiteralTarget()
            yhat_scitype <: metadata.observation_scitype
        elseif proxy == LearnAPI.Distribution()
            yhat_scitype <: Density{<:y_scitype}
        else
            false
        end
        requirement3 = !Tables.istable(y) || metadata.can_consume_tables
        requirement1 && requirement2 && requirement3
    end
end

end # module

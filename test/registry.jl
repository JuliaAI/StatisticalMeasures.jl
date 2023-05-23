API.register(LPLossOnScalars)
API.register(LPLossOnVectors, "l2")
metadata = API.measures()[LPLossOnScalars]
measure = LPLossOnScalars()

@testset "register" begin
    @test Set(keys(API.measures())) == Set([LPLossOnScalars, LPLossOnVectors])
    for trait in API.METADATA_TRAITS
        trait_ex = QuoteNode(trait)
        quote
            @test API.$trait(measure) == getproperty(metadata, $trait_ex)
        end |> eval
    end
    @test measures()[LPLossOnVectors].aliases == ("l2", )
end

@testset "search for needle in docstring" begin
    ms = measures("Matthew")
    @test [keys(ms)...] == [MatthewsCorrelation,]
    @test measures()[MatthewsCorrelation] == ms[MatthewsCorrelation]
end

@testset "search using trait values" begin
    ms = measures(
        observation_scitype = Union{Missing,Multiclass},
        supports_class_weights = true,
    )
    # test filter only catches true matches:
    @test all(ms) do (_, metadata)
        metadata.observation_scitype == Union{Missing,Multiclass} &&
            metadata.supports_class_weights
    end
    # find on basis of a mutually exclusive condition:
    ms! = measures(
        observation_scitype = Union{Missing,Multiclass},
        supports_class_weights = false,
    )
    # check no  matches in common:
    @test isempty(intersect(keys(ms), keys(ms!)))
end

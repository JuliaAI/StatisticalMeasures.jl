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

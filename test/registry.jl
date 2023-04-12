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
    @test API.measures()[LPLossOnVectors].aliases == ("l2", )
end

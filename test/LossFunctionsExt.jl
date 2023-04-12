import LossFunctions as LF

@testset "a DistanceLoss" begin
    rng  = srng(123)
    N = 10
    y = rand(rng, N)
    yhat = rand(rng, N)
    weights = rand(rng, N)
    for measure in [
        LF.LPDistLoss(2) |> Measure |> multimeasure,
        LF.LPDistLoss(2) |> Measure |> supports_missings_measure |> multimeasure,
        ]
        @test measure(yhat, y) ≈ l2(yhat, y)
        @test measurements(measure, yhat, y) ≈
            measurements(l2, yhat, y)
        @test measure(yhat, y, weights) ≈
            l2(yhat, y, weights)
        @test measurements(measure, yhat, y, weights) ≈
            measurements(l2, yhat, y, weights)
    end
    m = Measure(LF.LPDistLoss(2))
    @test m(yhat[1], y[1]) ≈ (yhat[1] - y[1])^2
    @test API.orientation(m) == API.Loss()
    @test API.observation_scitype(multimeasure(m)) == Infinite
end

@testset "a MarginLoss" begin
    rng  = srng(123)
    N = 100
    y = categorical(rand(rng, "-+", N), levels=['-', '+'], ordered=true)
    yhat = UnivariateFinite(['-', '+'], rand(N), augment=true, pool=y)
    zhat = [yhat...] # non-performant representation
    weights = rand(rng, N)
    class_weights = Dict('-' => rand(rng), '+' => rand(rng))
    for measure in [
        LF.ZeroOneLoss() |> Measure |> multimeasure,
        LF.ZeroOneLoss() |> Measure |> supports_missings_measure |> multimeasure,
        ]
        for xhat in [yhat, zhat]
            # `mcr` is `misclassification_rate`:
            @test measure(xhat, y) ≈ mcr(mode.(xhat), y)
            @test measurements(measure, xhat, y) ≈
                measurements(mcr, mode.(xhat), y)
            @test measure(xhat, y, weights, class_weights) ≈
                mcr(mode.(xhat), y, weights, class_weights)
            @test measurements(measure, xhat, y, weights, class_weights) ≈
                measurements(mcr, mode.(xhat), y, weights, class_weights)
        end
    end
    m = Measure(LF.ZeroOneLoss())
    @test m(yhat[1], y[1]) == float( mode(yhat[1]) != y[1])
    @test API.orientation(m) == API.Loss()
    @test API.observation_scitype(multimeasure(m)) == Finite{2}
end

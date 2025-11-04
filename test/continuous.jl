rng = srng(666899)

@testset "continuous measures" begin

    y    = [1, 42,      2, 3, missing, 4]
    yhat = [4, missing, 3, 2, 42,      1]
    w =    [1, 42,      2, 4, 42,      3]

    # include missing value and testing of weights for some of the measures:
    @test isapprox(l1(yhat, y), (3 + 1 + 1 + 3)/length(y))
    @test isapprox(LPLoss()(yhat, y), (3^2 + 1^2 + 1^2 + 3^2)/length(y))
    @test isapprox(LPLoss(p=2.5)(yhat, y), (3^2.5 + 1^2.5 + 1^2.5 + 3^2.5)/length(y))
    @test isapprox(LPSumLoss()(yhat, y), (3^2 + 1^2 + 1^2 + 3^2))
    @test isapprox(LPSumLoss(p=2.5)(yhat, y), (3^2.5 + 1^2.5 + 1^2.5 + 3^2.5))
    @test isapprox(l1(yhat, y, w), (1*3 + 2*1 + 4*1 + 3*3)/length(w))
    @test isapprox(rms(yhat, y), sqrt((3^2 + 1^2 + 1^2 + 3^2)/length(y)))
    @test isapprox(rms(yhat, y, w), sqrt((1*3^2 + 2*1^2 + 4*1^2 + 3*3^2)/length(w)))
    @test isapprox(sum(skipmissing(measurements(l2, yhat, y)))/length(y), l2(yhat, y))

    # the rest:
    y    = [1, 2, 3, 4]
    yhat = [2, 3, 4, 5]
    @test isapprox(rmsl(yhat, y),
                   sqrt((log(1/2)^2 + log(2/3)^2 + log(3/4)^2 + log(4/5)^2)/4))
    @test isapprox(rmslp1(yhat, y),
                   sqrt((log(2/3)^2 + log(3/4)^2 + log(4/5)^2 + log(5/6)^2)/4))
    @test isapprox(rmsp(yhat, y), sqrt((1 + 1/4 + 1/9 + 1/16)/4))
    @test isapprox(mape(yhat, y), (1/1 + 1/2 + 1/3 + 1/4)/4)

    y    = rand(rng, 4)
    yhat = rand(rng, 4)
    @test isapprox(log_cosh(yhat, y), mean(log.(cosh.(yhat - y))))
    @test rsq(yhat, y) == 1 - sum((yhat - y).^2)/sum((y .- mean(y)).^2)
    let
        num = sum((yhat - y).^2)
        den = sum((abs.(yhat .- mean(y)) .+ abs.(y .- mean(y))).^2)
        @test isapprox(willmott_d(yhat, y), den == 0 ? (num == 0 ? 1.0 : 0.0) : 1 - num/den)
        # additional tests for willmott_d
        @test willmott_d(yhat, yhat) == 1
        @test willmott_d(y, y) == 1
        @test willmott_d(yhat .+ 1, zeros(length(yhat))) == 0  # yhat .+ 1 ensures it's not all zeros
        @test willmott_d(zeros(4), zeros(4)) == 1
    end

    # a multi-target test where there is a parameter:
    y   = rand(rng, 2, 10)
    yhat = rand(rng, 2, 10)
    w = rand(rng, 2)
    μ = MultitargetRootMeanSquaredLogProportionalError(
        offset=2.5,
        atomic_weights=w,
    )(yhat, y)
    ms = [RootMeanSquaredLogProportionalError(2.5)(ηhat, η, w)
          for (ηhat, η) in MLUtils.eachobs((yhat, y))]
    @test aggregate(ms,  mode=API.RootMean()) ≈  μ

    # multi-target tests, ignoring parameters:
    for Single in [
        :LPLoss,
        :LPSumLoss,
        :RootMeanSquaredError,
        :RootMeanSquaredLogError,
        :RootMeanSquaredLogProportionalError,
        :RootMeanSquaredProportionalError,
        :MeanAbsoluteProportionalError,
        :LogCoshLoss,
        ]
        Multi = "Multitarget$(Single)" |> Symbol
        t = quote
            μ = $Multi(atomic_weights=$w)($yhat, $y)
            ms = [$Single()(ηhat, η, $w)
                  for (ηhat, η) in MLUtils.eachobs(($yhat, $y))]
            aggregate(ms,  mode=API.external_aggregation_mode($Single())) ≈  μ
        end |> eval
    end
end

true

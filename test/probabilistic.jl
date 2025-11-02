@testset "AreaUnderCurve" begin
    # this is random binary and random scores generated with numpy
    # then using roc_auc_score from sklearn to get the AUC
    # we check that we recover a comparable AUC and that it's invariant
    # to ordering.
    c = ["neg", "pos"]
    y = categorical(c[[0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                     1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
                     1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0,
                     1, 0] .+ 1])
    probs = [
        0.90237535, 0.41276349, 0.94511611, 0.08390761, 0.55847392,
        0.26043136, 0.78565351, 0.20133953, 0.7404382 , 0.15307601,
        0.59596716, 0.8169512 , 0.88200483, 0.23321489, 0.94050483,
        0.27593662, 0.60702176, 0.36427036, 0.35481784, 0.06416543,
        0.45576954, 0.12354048, 0.79830435, 0.15799818, 0.20981099,
        0.43451663, 0.24020098, 0.11401055, 0.25785748, 0.86490263,
        0.75715379, 0.06550534, 0.12628999, 0.18878245, 0.1283757 ,
        0.76542903, 0.8780248 , 0.86891113, 0.24835709, 0.06528076,
        0.72061354, 0.89451634, 0.95634394, 0.07555979, 0.16345437,
        0.43498831, 0.37774708, 0.31608861, 0.41369339, 0.95691113]

    ŷ = UnivariateFinite(y[1:2], probs, augment=true)
    @test isapprox(auc(ŷ, y), 0.455716, rtol=1e-4)
    ŷ_unwrapped = [ŷ...]
    @test isapprox(auc(ŷ_unwrapped, y), 0.455716, rtol=1e-4)

    # reversing the roles of positive and negative should return very
    # similar score
    y2 = deepcopy(y);
    levels!(y2, reverse(levels(y2)));
    @test y == y2
    ŷ2 = UnivariateFinite(y2[1:2], probs, augment=true)
    # check positive class has changed:
    @test levels(y) != levels(y2)
    @test CategoricalArrays.levels(ŷ) !=
        CategoricalArrays.levels(ŷ2)
    # check probability assignments are not changed:
    @test pdf.(ŷ, "pos") == pdf.(ŷ2, "pos")
    # test auc is the same:
    @test isapprox(auc(ŷ2, y2), auc(ŷ, y), rtol=1e-4)
end

@testset "NonMissingCatArrOrSub" begin
    y = categorical(['a', 'b', 'a'])
    @test y isa StatisticalMeasures.NonMissingCatArrOrSub
    @test view(y, 1:2) isa StatisticalMeasures.NonMissingCatArrOrSub
    @test !(unwrap.(y) isa StatisticalMeasures.NonMissingCatArrOrSub)
    y = categorical([missing, 'a', 'b', 'a'])
    @test !(y isa StatisticalMeasures.NonMissingCatArrOrSub)
    @test !(view(y, 1:2) isa StatisticalMeasures.NonMissingCatArrOrSub)
end

@testset "Log, Brier, Spherical - finite case" begin
    y = categorical(collect("abb"))
    L = [y[1], y[2]]
    d1 = UnivariateFinite(L, [0.1, 0.9]) # a
    d2 = UnivariateFinite(L, Float32[0.4, 0.6]) # b
    d3 = UnivariateFinite(L, [0.2, 0.8]) # b
    yhat = [d1, d2, d3]
    ym = vcat(y, [missing,])
    yhatm = vcat(yhat, [d3, ])

    # versions of yhat that is a UnivariateFiniteArray:
    yhat_u = UnivariateFinite(L, [0.1 0.9; 0.4 0.6; 0.2 0.8])
    @test Float32.(pdf.(yhat, 'a')) ≈ pdf.(yhat_u, 'a')

    @test log_loss(yhat, y) ≈
        Float32(-(log(0.1) + log(0.6) + log(0.8))/3)  ≈ log_loss(yhat_u, y)
    @test log_loss(yhatm, ym) ≈
        Float32(-(log(0.1) + log(0.6) + log(0.8))/4)
    w = rand(srng(123), 3)
    @test Float32(log_loss(yhat, y, w)) ≈
        -Float32((w[1]*log(0.1) + w[2]*log(0.6) + w[3]*log(0.8))/length(w)) ≈
        Float32(log_loss(yhat_u, y, w))
    w_class = Dict('a' => 4.3, 'b' => 2.9)
    @test Float32(log_loss(yhat, y, w_class)) ≈
        -Float32((4.3*log(0.1) + 2.9*log(0.6) + 2.9*log(0.8))/3) ≈
        Float32(log_loss(yhat_u, y, w_class))
    @test Float32(log_score(yhat, y)) ≈ -Float32(log_loss(yhat, y)) ≈
        -Float32(log_loss(yhat_u, y))
    @test log_score(yhat, y, w, w_class) ≈ - log_loss(yhat, y, w, w_class)
    @test measurements(log_loss, yhat_u, y) ≈ [-log(0.1), -log(0.6), -log(0.8)]
    @test measurements(log_loss, yhat_u, y, w) ≈
        w .* [-log(0.1), -log(0.6), -log(0.8)]
    @test measurements(log_loss, yhat_u, y, w_class) ≈
        [-4.3*log(0.1), -2.9*log(0.6), -2.9*log(0.8)]

    yhat = yhat_u

    # sklearn test
    # >>> from sklearn.metrics import log_loss
    # >>> log_loss(["spam", "ham", "ham", "spam","ham","ham"],
    #    [[.1, .9], [.9, .1], [.8, .2], [.35, .65], [0.2, 0.8], [0.3,0.7]])
    # 0.6130097025803921
    y2 = categorical(["spam", "ham", "ham", "spam", "ham", "ham"])
    L2 = CategoricalArrays.levels(y2[1])
    probs = vcat([.1 .9], [.9 .1], [.8 .2], [.35 .65], [0.2 0.8], [0.3 0.7])
    yhat2 = UnivariateFinite(L2, probs)
    y2m = vcat(y2, [missing,])
    yhat2m = UnivariateFinite(L2, vcat(probs, [0.1 0.9]))
    @test log_loss(yhat2, y2) ≈ 0.6130097025803921

    ## Brier
    scores = measurements(BrierScore(), yhat, y)
    @test Float32.(scores) ≈ [-1.62, -0.32, -0.08]
    score = BrierScore()(yhatm, ym)
    @test isapprox(score, sum([-1.62, -0.32, -0.08])/4, atol=1e-6)
    @test BrierLoss()(yhat, y) == -BrierScore()(yhat, y)
    # sklearn test
    # >>> from sklearn.metrics import brier_score_loss
    # >>> brier_score_loss([1, 0, 0, 1, 0, 0], [.9, .1, .2, .65, 0.8, 0.7])
    # 0.21875 NOTE: opposite orientation
    @test -BrierScore()(yhat2, y2) / 2 ≈ 0.21875
    probs2 = [[.1, .9], [Float32(0.9), Float32(1) - Float32(0.9)], [.8, .2],
              [.35, .65], [0.2, 0.8], [0.3, 0.7]]
    yhat3 = [UnivariateFinite(L2, prob) for prob in probs2]
    @test -BrierScore()(yhat3, y2) / 2 ≈ 0.21875
    @test BrierLoss()(yhat3, y2) / 2 ≈ -mean(BrierScore()(yhat3, y2) / 2)

    # Spherical
    s = SphericalScore() # SphericalScore(2)
    norms = [norm(probs[i,:]) for i in 1:size(probs, 1)]
    scores =  (pdf.(yhat2, y2) ./ norms)
    @test scores ≈  measurements(s, yhat2, y2)
    @test mean(scores) ≈ s(yhat2, y2)
    w = rand(srng(13), length(y2))
    @test w .* scores ≈ measurements(s, yhat2, y2, w)
    # non-performant version:
    yhat4 = [yhat2...]
    @test mean(scores) ≈  s(yhat4, y2)
end

@testset "LogScore, BrierScore, SphericalScore - infinite case" begin
    uniform = Distributions.Uniform(2, 5)
    betaprime = Distributions.BetaPrime()
    discrete_uniform = Distributions.DiscreteUniform(2, 5)
    w = [2, 3]

    # brier
    yhat = [uniform, uniform]
    @test isapprox(measurements(brier_score, yhat, [1.0, 1.0]) |> last, -1/3)
    @test isapprox(measurements(brier_score, yhat, [missing, 4.0]) |> last,  1/3)
    @test isapprox(measurements(brier_score, yhat, [1.0, 1.0], w) |> last, -1)
    yhat = [discrete_uniform, discrete_uniform]
    @test isapprox(
        measurements(brier_score, yhat, [missing, 1.0]) |> last,
        -1/4,
        )
    @test isapprox(brier_score(yhat, [4.0, 4.0]), 1/4)

    # spherical
    yhat = [uniform, uniform]
    @test isapprox(measurements(spherical_score, yhat, [1.0, 1.0]), [0, 0])
    @test isapprox(
        measurements(spherical_score, yhat, [NaN, 4.0]),
        [0, 1/sqrt(3),],
    )
    @test isapprox(
        measurements(spherical_score, yhat, [4.0, 4.0], w),
        [2/sqrt(3), sqrt(3),],
    )
    yhat = [discrete_uniform, discrete_uniform]
    @test isapprox(measurements(spherical_score, yhat, [NaN, 1.0]), [0, 0])
    @test isapprox(measurements(spherical_score, yhat, [4.0, 4.0]), [1/2, 1/2])

    # log
    yhat = [uniform, uniform]
    @test log_score(yhat, [4.0, 4.0]) ≈ mean([-log(3), -log(3)])
    @test log_score(yhat, [4.0, 4.0], w) ≈ sum([-2*log(27)/3, -log(27)])/2
    yhat = [discrete_uniform, discrete_uniform]
    # issue https://github.com/JuliaStats/Distributions.jl/issues/1392
    @test_broken  isapprox(log_score(yhat, [missing, 4.0]), [-log(4),])

    # errors
    @test_throws(StatisticalMeasures.ERR_L2_NORM,
                 brier_score([betaprime, betaprime], [1.0, 1.0]))
    s = SphericalScore(alpha=1)
    @test_throws StatisticalMeasures.ERR_UNSUPPORTED_ALPHA s(yhat, [1.0, 1.0])
end

@testset "l2_check" begin
    d = Distributions.Normal()
    yhat = Union{Distributions.Sampleable,Missing}[d, d, missing]
    y = ones(3)
    @test isnothing(StatisticalMeasures.l2_check("dummy", yhat, y))
    @test isnothing(StatisticalMeasures.l2_check("dummy", UnivariateFinite[], Float64[]))
end

true

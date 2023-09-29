@testset "ROC" begin
    perm = [4, 7, 2, 1, 3, 8, 5, 6]
    y = [0   0   0   1   0   1   1   0][perm] |> vec |> categorical
    s = [0.0 0.1 0.1 0.1 0.2 0.2 0.5 0.5][perm] |> vec
    ŷ = UnivariateFinite([0, 1], s, augment=true, pool=y)
    @test_throws(
        StatisticalMeasures.ERR_NEED_CATEGORICAL,
        roc_curve(ŷ, CategoricalArrays.unwrap.(y)),
    )
    @test_throws(
        StatisticalMeasures.ERR_NEED_CATEGORICAL,
        roc_curve(s, y),
    )
    @test_throws(
        StatisticalMeasures.ERR_ROC2,
        roc_curve(ŷ, categorical([0,1,2, fill(0, 7)...])),
    )
    @test_throws(
        StatisticalMeasures.ERR_ROC1,
        roc_curve(UnivariateFinite([0, 1, 2], rand(0:2,10,3), pool=missing), y)
    )
    @test_throws(
        API.ERR_POOL,
    roc_curve(ŷ, categorical([1, 2, 2, 2, 2, 2, 1, 2]))
    )

    fprs, tprs, ts = @test_logs(
        (:warn, ConfusionMatrices.WARN_UNORDERED([0, 1])),
        roc_curve(ŷ, y),
     )

    sk_fprs = [0. , 0.2, 0.4, 0.8, 1. ]
    sk_tprs = [0. , 0.33333333, 0.66666667, 1., 1.]

    @test fprs ≈ sk_fprs
    @test tprs ≈ sk_tprs

    y = categorical([  0   0   0   1   0   1   1   0] |> vec, ordered=true)
    s = [0.0 0.1 0.1 0.1 0.2 0.2 0.5 0.5] |> vec
    ŷ = UnivariateFinite([0, 1], s, augment=true, pool=y)

    fprs2, tprs2, ts2 = @test_logs roc_curve(ŷ, y)
    @test fprs2 == fprs
    @test tprs2 == tprs
    @test ts2 == ts
end

true

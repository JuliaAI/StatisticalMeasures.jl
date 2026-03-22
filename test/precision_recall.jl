@testset "precision_recall_curve" begin
    perm = [4, 7, 2, 1, 3, 8, 5, 6]
    y = [0   0   0   1   0   1   1   0][perm] |> vec |> categorical
    s = [0.0 0.1 0.1 0.1 0.2 0.2 0.5 0.5][perm] |> vec
    ŷ = UnivariateFinite([0, 1], s, augment=true, pool=y)
    @test_throws(StatisticalMeasures.ERR_NEED_CATEGORICAL_PR,
        precision_recall_curve(ŷ, CategoricalArrays.unwrap.(y)),
    )
    @test_throws(
        StatisticalMeasures.ERR_NEED_CATEGORICAL_PR,
        precision_recall_curve(s, y),
    )
    @test_throws(
        StatisticalMeasures.ERR_PR2,
        precision_recall_curve(ŷ, categorical([0,1,2, fill(0, 7)...])),
    )
    @test_throws(
        StatisticalMeasures.ERR_PR1,
        precision_recall_curve(UnivariateFinite([0, 1, 2], rand(0:2,10,3), pool=missing), y)
    )
    @test_throws(
        API.ERR_POOL,
        precision_recall_curve(ŷ, categorical([1, 2, 2, 2, 2, 2, 1, 2]))
    )

    recalls, precisions, ts = @test_logs(
        (:warn, StatisticalMeasures.warning_unordered([0, 1])),
        precision_recall_curve(ŷ, y),
     )

    core_function_recalls, core_function_precisions =
        Functions.precision_recall_curve(s, y, 1)

    @test precisions == core_function_precisions
    @test recalls == core_function_recalls

    y = categorical([  0   0   0   1   0   1   1   0] |> vec, ordered=true)
    s = [0.0 0.1 0.1 0.1 0.2 0.2 0.5 0.5] |> vec
    ŷ = UnivariateFinite([0, 1], s, augment=true, pool=y)

    recalls2, precisions2, ts2 = @test_logs precision_recall_curve(ŷ, y)
    @test precisions2 == precisions
    @test recalls2 == recalls
    @test ts2 == ts
end

true

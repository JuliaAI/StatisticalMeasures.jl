rng = srng(51803)

@testset "confusion matrix, as measure" begin
    # CM.confmat is tested in /test/confusion_matrices.jl. We need only
    # interface tests here:
    @test API.orientation(confmat) == API.Unoriented()
    z = rand(rng, "ab", 100);
    zhat = rand(rng, "ab", 100);
    for (y, yhat) in [
        (z, zhat),
        (categorical(z, ordered=true), categorical(zhat, ordered=true)),
        ]
        @test confmat(yhat, y) == CM.confmat(yhat, y)
        @test ConfusionMatrix(levels=['b', 'a'])(yhat, y) ==
            CM.confmat(yhat, y, levels=['b', 'a'])
        @test ConfusionMatrix(rev=true)(yhat, y) ==
            CM.confmat(yhat, y, rev=true)
        @test ConfusionMatrix(perm=[2, 1])(yhat, y) ==
            CM.confmat(yhat, y, perm=[2, 1])
        @test ConfusionMatrix(perm=[2, 1], levels=['b', 'a'])(yhat, y) ==
            CM.confmat(yhat, y, perm=[2, 1], levels=['b', 'a'])
        @test ConfusionMatrix(rev=true, levels=['b', 'a'])(yhat, y) ==
            CM.confmat(yhat, y, rev=true, levels=['b', 'a'])
        @test_throws ArgumentError ConfusionMatrix(rev=true, perm=[2, 1])
        @test_throws ArgumentError ConfusionMatrix(rev=true, levels=['a', 'b', 'c'])
        @test_throws ArgumentError ConfusionMatrix(perm=[1,2,3], levels=['a', 'b'])
    end
end

@testset "misclassification_rate" begin
    y    = categorical(collect("asdfasdfaaassdd"))
    yhat = categorical(collect("asdfaadfaasssdf"))
    w = 1:15
    ym = vcat(y, [missing,])
    yhatm = vcat(yhat, [missing,])
    wm = 1:16
    @test misclassification_rate(yhat, y) ≈ sum(y .!= yhat)/length(y)
    @test misclassification_rate(yhatm, ym) ≈ sum(y .!= yhat)/length(ym)
    @test misclassification_rate(yhat, y, w) ≈ (6*1 + 11*1 + 15*1) / length(w)
    @test misclassification_rate(yhatm, ym, wm) ≈ (6*1 + 11*1 + 15*1) / length(wm)
end

@testset "mcr, acc, bacc, mcc" begin
    y = categorical(['m', 'f', 'n', 'f', 'm', 'n', 'n', 'm', 'f'])
    ŷ = categorical(['f', 'f', 'm', 'f', 'n', 'm', 'n', 'm', 'f'])
    @test accuracy(ŷ, y) ≈ 1-mcr(ŷ,y) ≈
        accuracy(CM.confmat(ŷ, y))  ≈
        1-mcr(CM.confmat(ŷ, y))
    w = rand(rng,length(y))
    @test accuracy(ŷ, y, w) ≈ mean(w) - mcr(ŷ,y,w) ≈ sum(w[y .== ŷ])/length(w)

    ## balanced accuracy
    y = categorical([
        3, 4, 1, 1, 1, 4, 1, 3, 3, 1, 2, 3, 1, 3, 3, 3, 2, 4, 3, 2, 1, 3,
        3, 1, 1, 1, 2, 4, 1, 4, 4, 4, 1, 1, 4, 4, 3, 1, 2, 2, 3, 4, 2, 1,
        2, 2, 3, 2, 2, 3, 1, 2, 3, 4, 1, 2, 4, 2, 1, 4, 3, 2, 3, 3, 3, 1,
        3, 1, 4, 3, 1, 2, 3, 1, 2, 2, 4, 4, 1, 3, 2, 1, 4, 3, 3, 1, 3, 1,
        2, 2, 2, 2, 2, 3, 2, 1, 1, 4, 2, 2])
    ŷ = categorical([
        2, 3, 2, 1, 2, 2, 3, 3, 2, 4, 2, 3, 2, 4, 3, 4, 4, 2, 1, 3, 3, 3,
        3, 3, 2, 4, 4, 3, 4, 4, 1, 2, 3, 2, 4, 1, 2, 3, 1, 4, 2, 2, 1, 2,
        3, 2, 2, 4, 3, 2, 2, 2, 1, 2, 2, 1, 3, 1, 4, 1, 2, 1, 2, 4, 3, 2,
        4, 3, 2, 4, 4, 2, 4, 3, 2, 3, 1, 2, 1, 2, 1, 2, 3, 1, 1, 3, 4, 2,
        4, 4, 2, 1, 3, 2, 2, 4, 1, 1, 4, 1])
    w = [
        0.5, 1.4, 0.6, 1. , 0.1, 0.5, 1.2, 0.2, 1.8, 0.3, 0.6, 2.2, 0.1,
        1.4, 0.2, 0.4, 0.6, 2.1, 0.7, 0.2, 0.9, 0.4, 0.7, 0.3, 0.1, 1.7,
        0.2, 0.7, 1.2, 1. , 0.9, 0.4, 0.5, 0.5, 0.5, 1. , 0.3, 0.1, 0.2,
        0. , 2.2, 0.8, 0.9, 0.8, 1.3, 0.2, 0.4, 0.7, 1. , 0.7, 1.7, 0.7,
        1.1, 1.8, 0.1, 1.2, 1.8, 1. , 0.1, 0.5, 0.6, 0.7, 0.6, 1.2, 0.6,
        1.2, 0.5, 0.5, 0.8, 0.2, 0.6, 1. , 0.3, 1. , 0.2, 1.1, 1.1, 1.1,
        0.6, 1.4, 1.2, 0.3, 1.1, 0.2, 0.5, 1.6, 0.3, 1. , 0.3, 0.9, 0.9,
        0. , 0.6, 0.6, 0.4, 0.5, 0.4, 0.2, 0.9, 0.4]
    sk_bacc = 0.17493386243386244 # note: sk-learn reverses ŷ and y
    @test bacc(ŷ, y) ≈ sk_bacc
    sk_adjusted_bacc =  -0.10008818342151675
    @test BalancedAccuracy(adjusted=true)(ŷ, y) ≈ BalancedAccuracy(adjusted=true)(CM.confmat(ŷ, y)) ≈ sk_adjusted_bacc
    sk_bacc_w = 0.1581913163016446
    @test bacc(ŷ, y, w) ≈ sk_bacc_w
    sk_adjusted_bacc_w = -0.1224115782644738
    @test BalancedAccuracy(adjusted=true)(ŷ, y, w) ≈ sk_adjusted_bacc_w

    ## matthews correlation
    sk_mcc = -0.09759509982785947
    @test mcc(ŷ, y) == matthews_correlation(ŷ, y) ≈ sk_mcc

end

@testset "kappa" begin
    # core functionality tested in /test/functions.jl

    # Binary case
    y_b = categorical([2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2,
                       2, 2, 2, 2, 1, 1, 1, 2, 2])
    ŷ_b = categorical([1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2,
                       1, 2, 1, 2, 2, 2, 2, 2, 2])
    @test kappa(ŷ_b, y_b) ≈ CM.kappa(CM.confmat(ŷ_b, y_b))

    # Multiclass case
    y_m = categorical([5, 5, 3, 5, 4, 4, 2, 2, 3, 2, 5, 2, 4, 3, 2, 1, 1, 5, 1, 4, 2,
                       5, 4, 5, 2, 3, 3, 4, 2, 4])
    ŷ_m = categorical([1, 1, 1, 5, 4, 2, 1, 3, 4, 4, 2, 5, 4, 4, 1, 5, 5, 2, 3, 3, 1,
                       3, 2, 5, 5, 2, 3, 2, 5, 3])
    @test kappa(ŷ_m, y_m) ≈ CM.kappa(CM.confmat(ŷ_m, y_m))
end

@testset "fscore" begin
    # core functionality tested in /test/functions.jl

    y = [missing, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2,
         2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1,
         2, 2, 2, 1]
    ŷ = [1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2,
         1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2,
         1, 2, 2, missing]

    @test_logs (:warn, CM.WARN_UNORDERED([1, 2])) f1score(ŷ, y)
    f05 = @test_logs FScore(0.5, levels=[1, 2])(ŷ, y)
    sk_f05 = 0.625
    @test f05 ≈ sk_f05 # m.fbeta_score(y, yhat, 0.5, pos_label=2)
    cm = CM.confmat(ŷ, y, rev=true)
    @test FScore(β=0.5, rev=true)(ŷ, y) ≈ CM.fscore(cm, 0.5)
    @test FScore(beta=0.5, levels=[2, 1])(ŷ, y) ≈ CM.fscore(cm, 0.5)

    @test_throws StatisticalMeasures.ERR_NONBINARY_LEVELS FScore(levels=[1, 2, 3])
end

@testset "truepositive and cousins" begin
    y = coerce([missing, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2,
                2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1,
                2, 2, 2, 1], Union{Missing,OrderedFactor})
    ŷ = coerce([1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2,
                1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2,
                1, 2, 2, missing], Union{Missing,OrderedFactor})

    # "synchronized" removal of missings:
    (z, ẑ) = StatisticalMeasures.StatisticalMeasuresBase.skipinvalid(y, ŷ)

    m = TruePositive()
    @test m(ŷ, y) == truepositive(ŷ, y) == sum((ẑ .== 2) .& (z .== 2))
    m = TruePositive(rev=true)
    @test m(ŷ, y) == truenegative(ŷ, y)
    m = TrueNegative()
    @test m(ŷ, y) == truenegative(ŷ, y) == sum((ẑ .== 1) .& (z .== 1))
    m = FalsePositive()
    @test m(ŷ, y) == falsepositive(ŷ, y) == sum((ẑ .== 2) .& (z .== 1))
    m = FalseNegative()
    @test m(ŷ, y) == falsenegative(ŷ, y) == sum((ẑ .== 1) .& (z .== 2))

    TP = true_positive(ẑ, z)
    TN = true_negative(ẑ, z)
    FP = false_positive(ẑ, z)
    FN = false_negative(ẑ, z)
    m = TruePositiveRate()
    @test m(ŷ, y) == tpr(ŷ, y) == truepositive_rate(ŷ, y) == TP/(TP +FN)
    m = TrueNegativeRate()
    @test m(ŷ, y) == tnr(ŷ, y) == truenegative_rate(ŷ, y) == TN/(TN +FP)
    m = FalsePositiveRate()
    @test m(ŷ, y) == fpr(ŷ, y) == falsepositive_rate(ŷ, y) == FP/(FP + TN)
    m = FalseNegativeRate()
    @test m(ŷ, y) == fnr(ŷ, y) == falsenegative_rate(ŷ, y) == FN/(FN + TP)
    m = FalseDiscoveryRate()
    @test m(ŷ, y) == fdr(ŷ, y) == falsediscovery_rate(ŷ, y) == FP/(FP + TP)
    m = PositivePredictiveValue()
    @test m(ŷ, y) == StatisticalMeasures.precision(ŷ, y) == TP/(TP + FP)
    m = NegativePredictiveValue()
    @test m(ŷ, y) == negative_predictive_value(ŷ, y) == TN/(TN + FN)

    # check synonyms
    m = TruePositiveRate()
    @test m(ŷ, y) == tpr(ŷ, y)
    m = TrueNegativeRate()
    @test m(ŷ, y) == tnr(ŷ, y)
    m = FalsePositiveRate()
    @test m(ŷ, y) == fpr(ŷ, y) == fallout(ŷ, y)
    m = FalseNegativeRate()
    @test m(ŷ, y) == fnr(ŷ, y) == miss_rate(ŷ, y)
    m = FalseDiscoveryRate()
    @test m(ŷ, y) == fdr(ŷ, y)
    m = PositivePredictiveValue()
    @test m(ŷ, y) == ppv(ŷ, y)
    m = TruePositiveRate()
    @test m(ŷ, y) == tpr(ŷ, y) == recall(ŷ, y) ==
        sensitivity(ŷ, y) == hit_rate(ŷ, y)
    m = TrueNegativeRate()
    @test m(ŷ, y) == tnr(ŷ, y) == specificity(ŷ, y) == selectivity(ŷ, y)

    # integration test
    m = BalancedAccuracy()
    @test m(ẑ, z) == bacc(ẑ, z) == (tpr(ẑ, z) + tnr(ẑ, z))/2

    ### External comparisons
    sk_prec = 0.6111111111111112 # m.precision_score(y, yhat, pos_label=2)
    @test StatisticalMeasures.precision(ŷ, y) ≈ sk_prec
    sk_rec = 0.6875
    @test recall(ŷ, y) == sk_rec # m.recall_score(y, yhat, pos_label=2)

    # reversion mechanism
    sk_prec_rev = 0.5454545454545454
    prec_rev = PositivePredictiveValue(rev=true)
    @test prec_rev(ŷ, y) ≈ sk_prec_rev
    sk_rec_rev = 0.46153846153846156
    rec_rev = TruePositiveRate(rev=true)
    @test rec_rev(ŷ, y) ≈ sk_rec_rev
end

@testset "selected traits for binary measures" begin
    for m in (accuracy, recall, StatisticalMeasures.precision, f1score, specificity)
        @test API.consumes_multiple_observations(m)
        m == accuracy || @test !API.can_report_unaggregated(m)
        @test API.kind_of_proxy(m) == StatisticalMeasures.LearnAPI.LiteralTarget()
        @test API.observation_scitype(m) <: Union{Missing,Finite}
        @test !API.can_consume_tables(m)
        m == accuracy && @test API.supports_weights(m)
        m == accuracy && @test API.supports_class_weights(m)
        m == accuracy || @test !API.supports_weights(m)
        m == accuracy || @test !API.supports_class_weights(m)
        @test API.orientation(m) == API.Score()
        @test API.external_aggregation_mode(m) == API.Mean()
        m == accuracy    && (@test API.human_name(m) == "accuracy")
        m == recall      && (@test API.human_name(m) == "true positive rate")
        m == StatisticalMeasures.precision &&
            (@test API.human_name(m) == "positive predictive value")
        m == f1score     && (@test API.human_name(m) == "``F_β`` score")
        m == specificity && (@test API.human_name(m) == "true negative rate")
    end
    # e = info(auc)
    # @test e.name == "AreaUnderCurve"
    # @test e.target_scitype ==
    #     Union{AbstractArray{<:Union{Missing,Multiclass{2}}},
    #           AbstractArray{<:Union{Missing,OrderedFactor{2}}}}
    # @test e.prediction_type == :probabilistic
    # @test e.reports_each_observation == false
    # @test e.is_feature_dependent == false
    # @test e.supports_weights == false
end

@testset "mutliclass measures based on confusion marrix" begin
    y = coerce([1, 2, 0, 2, 1, 0, 0, 1, 2, 2, 2, 1, 2,
                            2, 1, 0, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1,
                            2, 2, 2], OrderedFactor)
    ŷ = coerce([2, 0, 2, 2, 2, 0, 1, 2, 1, 2, 0, 1, 2,
                            1, 1, 1, 2, 0, 1, 2, 1, 2, 2, 2, 1, 2,
                            1, 2, 2], OrderedFactor)
    class_w = Dict(0=>0,2=>2,1=>1)
    cm = ConfusionMatrices.confmat(ŷ, y)

    #               ┌─────────────────────────────────────────┐
    #               │              Ground Truth               │
    # ┌─────────────┼─────────────┬─────────────┬─────────────┤
    # │  Predicted  │      0      │      1      │      2      │
    # ├─────────────┼─────────────┼─────────────┼─────────────┤
    # │      0      │      1      │      1      │      2      │
    # ├─────────────┼─────────────┼─────────────┼─────────────┤
    # │      1      │      2      │      4      │      4      │
    # ├─────────────┼─────────────┼─────────────┼─────────────┤
    # │      2      │      1      │      6      │      8      │
    # └─────────────┴─────────────┴─────────────┴─────────────┘

    cm_tp   = [1; 4; 8]
    cm_tn   = [22; 12; 8]
    cm_fp   = [1+2; 2+4; 1+6]
    cm_fn   = [2+1; 1+6; 2+4]
    cm_prec = cm_tp ./ ( cm_tp + cm_fp  )
    cm_rec  = cm_tp ./ ( cm_tp + cm_fn  )

    # Check if is positive
    m = MulticlassTruePositive(;return_type=Vector)
    @test  [0; 0; 0] <= m(ŷ, y) == cm_tp
    m = MulticlassTrueNegative(;return_type=Vector)
    @test  [0; 0; 0] <= m(ŷ, y) == cm_tn
    m = MulticlassFalsePositive(;return_type=Vector)
    @test  [0; 0; 0] <= m(ŷ, y) == cm_fp
    m = MulticlassFalseNegative(;return_type=Vector)
    @test  [0; 0; 0] <= m(ŷ, y) == cm_fn

    # Check if is in [0,1]
    m = MulticlassTruePositiveRate(average=NoAvg();return_type=Vector)
    @test  [0; 0; 0] <= m(ŷ, y) == cm_tp ./ (cm_fn.+cm_tp) <= [1; 1; 1]
    m = MulticlassTrueNegativeRate(average=NoAvg();return_type=Vector)
    @test  [0; 0; 0] <= m(ŷ, y) == cm_tn ./ (cm_tn.+cm_fp) <= [1; 1; 1]
    m = MulticlassFalsePositiveRate(average=NoAvg();return_type=Vector)
    @test  [0; 0; 0] <= m(ŷ, y) == 1 .- cm_tn ./ (cm_tn.+cm_fp) <= [1; 1; 1]
    m = MulticlassFalseNegativeRate(average=NoAvg();return_type=Vector)
    @test  [0; 0; 0] <= m(ŷ, y) == 1 .- cm_tp ./ (cm_fn.+cm_tp) <= [1; 1; 1]

    #`NoAvg()` and `LittleDict`
    @test collect(values(MulticlassPositivePredictiveValue(average=NoAvg())(cm))) ≈
        collect(values(MulticlassPositivePredictiveValue(average=NoAvg())(ŷ, y))) ≈
        cm_prec
    @test MulticlassPositivePredictiveValue(average=MacroAvg())(cm) ≈
        MulticlassPositivePredictiveValue(average=MacroAvg())(ŷ, y) ≈ mean(cm_prec)
    @test collect(keys(MulticlassPositivePredictiveValue(average=NoAvg())(cm)))  ==
        collect(keys(MulticlassPositivePredictiveValue(average=NoAvg())(ŷ, y))) ==
        [0; 1; 2]
    @test collect(values(MulticlassTruePositiveRate(average=NoAvg())(cm))) ≈
        collect(values(MulticlassTruePositiveRate(average=NoAvg())(ŷ, y))) ≈
        cm_rec
    @test collect(values(MulticlassFScore(average=NoAvg())(cm))) ≈
        collect(values(MulticlassFScore(average=NoAvg())(ŷ, y))) ≈
        2 ./ ( 1 ./ cm_prec + 1 ./ cm_rec )

    #`NoAvg()` and `LittleDict` with class weights
    @test collect(values(MulticlassPositivePredictiveValue(average=NoAvg())(cm, class_w))) ≈
        collect(values(MulticlassPositivePredictiveValue(average=NoAvg())(ŷ, y, class_w))) ≈
        cm_prec .* [0; 1; 2]
    @test collect(values(MulticlassTruePositiveRate(average=NoAvg())(cm, class_w))) ≈
        collect(values(MulticlassTruePositiveRate(average=NoAvg())(ŷ, y, class_w))) ≈
        cm_rec .* [0; 1; 2]
    @test collect(values(MulticlassFScore(average=NoAvg())(cm, class_w))) ≈
        collect(values(MulticlassFScore(average=NoAvg())(ŷ, y, class_w))) ≈
        2 ./ ( 1 ./ cm_prec + 1 ./ cm_rec ) .* [0; 1; 2]

    #`MacroAvg()` and `LittleDict`
    macro_prec = MulticlassPositivePredictiveValue(average=MacroAvg())
    macro_rec  = MulticlassTruePositiveRate(average=MacroAvg())

    @test macro_prec(cm)    ≈ macro_prec(ŷ, y)    ≈ mean(cm_prec)
    @test macro_rec(cm)     ≈ macro_rec(ŷ, y)     ≈ mean(cm_rec)
    @test macro_f1score(cm) ≈ macro_f1score(ŷ, y) ≈
        mean(2 ./ ( 1 ./ cm_prec + 1 ./ cm_rec ))

    #`MicroAvg()` and `LittleDict`
    micro_prec = MulticlassPositivePredictiveValue(average=MicroAvg())
    micro_rec  = MulticlassTruePositiveRate(average=MicroAvg())

    @test micro_prec(cm)    == micro_prec(ŷ, y)    == sum(cm_tp) ./ sum(cm_fp.+cm_tp)
    @test micro_rec(cm)     == micro_rec(ŷ, y)     == sum(cm_tp) ./ sum(cm_fn.+cm_tp)
    @test micro_f1score(cm) == micro_f1score(ŷ, y) ==
        2 ./ ( 1 ./ ( sum(cm_tp) ./ sum(cm_fp.+cm_tp) ) + 1 ./
        ( sum(cm_tp) ./ sum(cm_fn.+cm_tp) ) )

    #`NoAvg()` and `Vector` with class weights
    vec_precision = MulticlassPositivePredictiveValue(return_type=Vector)
    vec_recall    = MulticlassTruePositiveRate(return_type=Vector)
    vec_f1score   = MulticlassFScore(return_type=Vector)

    @test vec_precision(cm, class_w) ≈ vec_precision(ŷ, y, class_w) ≈
        mean(cm_prec .* [0; 1; 2])
    @test vec_recall(cm, class_w)    ≈ vec_recall(ŷ, y, class_w)    ≈
        mean(cm_rec .* [0; 1; 2])
    @test vec_f1score(cm, class_w)   ≈ vec_f1score(ŷ, y, class_w)   ≈
        mean(2 ./ ( 1 ./ cm_prec + 1 ./ cm_rec ) .* [0; 1; 2])

    #`MacroAvg()` and `Vector`
    v_ma_prec = MulticlassPositivePredictiveValue(average=MacroAvg(),
                                    return_type=Vector)
    v_ma_rec  = MulticlassTruePositiveRate(average=MacroAvg(), return_type=Vector)
    v_ma_f1   = MulticlassFScore(average=MacroAvg(), return_type=Vector)

    @test v_ma_prec(cm) ≈ v_ma_prec(ŷ, y) ≈ mean(cm_prec)
    @test v_ma_rec(cm)  ≈ v_ma_rec(ŷ, y)  ≈ mean(cm_rec)
    @test v_ma_f1(cm)   ≈ v_ma_f1(ŷ, y)   ≈ mean(2 ./ ( 1 ./ cm_prec + 1 ./ cm_rec ))

    #`MacroAvg()` and `Vector` with class weights
    @test v_ma_prec(cm, class_w) ≈ v_ma_prec(ŷ, y, class_w) ≈
        mean(cm_prec .* [0, 1, 2])
    @test v_ma_rec(cm, class_w)  ≈ v_ma_rec(ŷ, y, class_w)  ≈
        mean(cm_rec .* [0, 1, 2])
    @test v_ma_f1(cm, class_w)   ≈ v_ma_f1(ŷ, y, class_w)   ≈
        mean(2 ./ ( 1 ./ cm_prec + 1 ./ cm_rec ) .* [0, 1, 2])

    #`MicroAvg()` and `Vector`
    v_mi_prec = MulticlassPositivePredictiveValue(average=MicroAvg(), return_type=Vector)
    v_mi_rec  = MulticlassTruePositiveRate(average=MicroAvg(), return_type=Vector)
    v_mi_f1   = MulticlassFScore(average=MicroAvg(), return_type=Vector)

    @test v_mi_prec(cm) == v_mi_prec(ŷ, y) == sum(cm_tp) ./ sum(cm_fp.+cm_tp)
    @test v_mi_rec(cm)  == v_mi_rec(ŷ, y)  == sum(cm_tp) ./ sum(cm_fn.+cm_tp)
    @test v_mi_f1(cm)   == v_mi_f1(ŷ, y)   ==
        2 ./ ( 1 ./ ( sum(cm_tp) ./ sum(cm_fp.+cm_tp) ) + 1 ./
        ( sum(cm_tp) ./ sum(cm_fn.+cm_tp) ) )
end

@testset "issue MLJBase.jl #630" begin
    # multiclass fscore corner case of absent class

    y = coerce([1, 2, 2, 2, 3], OrderedFactor)[1:4]
    # [1, 2, 2, 2] # but 3 is in the pool
    yhat = reverse(y)
    # [2, 2, 2, 1]

    # In this case, assigning "3" as "positive" gives all true negative,
    # and so NaN for that class's contribution to the average F1Score,
    # which should accordingly be skipped.

    # positive class | TP | FP | FN | score for that class
    # --------------|----|----|----|---------------------
    #  1            | 0  | 1  | 2  | 0
    #  2            | 2  | 1  | 1  | 2/3
    #  3            | 0  | 0  | 0  | NaN

    # When taking averages in StatisticalMeasures.jl (see the `aggregate` docstring) the
    # convention, when skipping invalid entries, is to replace them with zero (ie, skip
    # them but they still get counted in the normalization factor. So the mean score with
    # NaN skipping is (2/3)/3 = 2/9:
    @test MulticlassFScore()(yhat, y) ≈ 2/9
end

@testset "Metadata multiclass" begin
    for m in (MulticlassTruePositiveRate(), MulticlassPositivePredictiveValue(),
              MulticlassFScore(), MulticlassTrueNegativeRate())
        m == MulticlassTruePositiveRate() &&
            (@test API.human_name(m) == "multi-class true positive rate")
        m  == MulticlassPositivePredictiveValue()   &&
            (@test API.human_name(m) == "multi-class positive predictive value")
        m == MulticlassFScore &&
            (@test API.human_name(m) == "multi-class F_β score")
        m == MulticlassTrueNegativeRate &&
            (@test API.human_name(m) == "multi-class true negative rate")
        @test API.observation_scitype(m) <: Union{Missing,Finite}
        @test API.kind_of_proxy(m) == StatisticalMeasures.LearnAPI.LiteralTarget()
        @test API.orientation(m) == API.Score()
        @test API.can_report_unaggregated(m) == false
        @test !API.supports_weights(m)
        @test API.supports_class_weights(m)
        @test !API.can_consume_tables(m)
        @test API.external_aggregation_mode(m) == API.Mean()
    end
end

@testset "More multiclass metrics" begin
    y = coerce(categorical([missing, 1, 2, 0, 2, 1, 0, 0, 1, 2, 2, 2, 1, 2,
                            2, 1, 0, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1,
                            2, 2, 2, 0]), Union{Missing,Multiclass})
    ŷ = coerce(categorical([0, 2, 0, 2, 2, 2, 0, 1, 2, 1, 2, 0, 1, 2,
                            1, 1, 1, 2, 0, 1, 2, 1, 2, 2, 2, 1, 2,
                            1, 2, 2, missing]), Union{Missing,Multiclass})
    w = Dict(0=>1, 1=>2, 2=>3) #class_w
    cm = ConfusionMatrices.confmat(ŷ, y)
    mat = ConfusionMatrices.matrix(cm, warn=false)
    # check all constructors
    m = MulticlassTruePositive()
    @test m(ŷ, y) == multiclass_truepositive(ŷ, y)
    @test Functions.multiclass_true_positive(mat) == collect(values(m(ŷ, y)))
    m = MulticlassTrueNegative()
    @test m(ŷ, y) == multiclass_truenegative(ŷ, y)
    m = MulticlassFalsePositive()
    @test m(ŷ, y) == multiclass_falsepositive(ŷ, y)
    m = MulticlassFalseNegative()
    @test m(ŷ, y) == multiclass_falsenegative(ŷ, y)
    m = MulticlassTruePositiveRate()
    @test m(ŷ, y) == multiclass_tpr(ŷ, y) ==
        multiclass_truepositive_rate(ŷ, y)
    @test m(ŷ, y, w) == multiclass_tpr(ŷ, y, w) ==
        multiclass_truepositive_rate(ŷ, y, w)
    m = MulticlassTrueNegativeRate()
    @test m(ŷ, y) == multiclass_tnr(ŷ, y) ==
        multiclass_truenegative_rate(ŷ, y)
    @test m(ŷ, y, w) == multiclass_tnr(ŷ, y, w) ==
        multiclass_truenegative_rate(ŷ, y, w)
    m = MulticlassFalsePositiveRate()
    @test m(ŷ, y) == multiclass_fpr(ŷ, y) ==
        multiclass_falsepositive_rate(ŷ, y)
    @test m(ŷ, y, w) == multiclass_fpr(ŷ, y, w) ==
        multiclass_falsepositive_rate(ŷ, y, w)
    m = MulticlassFalseNegativeRate()
    @test m(ŷ, y) == multiclass_fnr(ŷ, y) ==
        multiclass_falsenegative_rate(ŷ, y)
    @test m(ŷ, y, w) == multiclass_fnr(ŷ, y, w) ==
        multiclass_falsenegative_rate(ŷ, y, w)
    m = MulticlassFalseDiscoveryRate()
    @test m(ŷ, y) == multiclass_fdr(ŷ, y) ==
        multiclass_falsediscovery_rate(ŷ, y)
    @test m(ŷ, y, w) == multiclass_fdr(ŷ, y, w) ==
        multiclass_falsediscovery_rate(ŷ, y, w)
    m = MulticlassPositivePredictiveValue()
    @test m(ŷ, y) == multiclass_precision(ŷ, y)
    @test m(ŷ, y, w) == multiclass_precision(ŷ, y, w)
    m = MulticlassNegativePredictiveValue()
    @test m(ŷ, y) == multiclass_npv(ŷ, y)
    @test m(ŷ, y, w) == multiclass_npv(ŷ, y, w)
    m = MulticlassFScore()
    @test m(ŷ, y) == macro_f1score(ŷ, y)
    @test m(ŷ, y, w) == macro_f1score(ŷ, y, w)
    m = MulticlassTruePositiveRate()
    @test m(ŷ, y) == multiclass_tpr(ŷ, y)
    @test m(ŷ, y, w) == multiclass_tpr(ŷ, y, w)
    @test m(ŷ, y) == multiclass_sensitivity(ŷ, y) ==
        multiclass_hit_rate(ŷ, y)
    @test m(ŷ, y, w) == multiclass_sensitivity(ŷ, y, w) ==
        multiclass_hit_rate(ŷ, y, w)
    m = MulticlassTrueNegativeRate()
    @test m(ŷ, y) == multiclass_tnr(ŷ, y) == multiclass_specificity(ŷ, y) ==
        multiclass_selectivity(ŷ, y)
    @test m(ŷ, y, w) == multiclass_tnr(ŷ, y, w) ==
        multiclass_specificity(ŷ, y, w) == multiclass_selectivity(ŷ, y, w)
end


@testset "Additional multiclass tests" begin
    table = reshape(collect("aabbbccccddbabccbacccd"), 11, 2)
    table = coerce(table, Multiclass);
    yhat = table[:,1] # ['a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'd', 'd']
    y    = table[:,2] # ['b', 'a', 'b', 'c', 'c', 'b', 'a', 'c', 'c', 'c', 'd']
    class_w = Dict('a'=>7, 'b'=>5, 'c'=>2, 'd'=> 0)

    # class | TP | FP | TP + FP | precision | FN | TP + FN | recall
    # ------|----|----|------------------------------------|-------
    # a     | 1  | 1  | 2       | 1/2       | 1  | 2       | 1/2
    # b     | 1  | 2  | 3       | 1/3       | 2  | 3       | 1/3
    # c     | 2  | 2  | 4       | 1/2       | 3  | 5       | 2/5
    # d     | 1  | 1  | 2       | 1/2       | 0  | 1       | 1

    # helper:
    inverse(x) = 1/x
    harmonic_mean(x, y; beta=1.0) =
        (1 + inverse(beta^2))*inverse(mean(inverse(beta^2*x)+ inverse(y)))

    # precision:
    p_macro = mean([1/2, 1/3, 1/2, 1/2])
    @test MulticlassPositivePredictiveValue()(yhat, y) ≈ p_macro
    p_macro_w = sum([7/2, 5/3, 2/2, 0/2])/4
    @test MulticlassPositivePredictiveValue()(yhat, y, class_w) ≈ p_macro_w
    p_micro = (1 + 1 + 2 + 1)/(2 + 3 + 4 + 2)
    @test p_micro ≈
        @test_logs((:warn, StatisticalMeasures.WARN_MICRO_IGNORING_WEIGHTS),
                     MulticlassPositivePredictiveValue(average=MicroAvg())(yhat, y, class_w))
    @test MulticlassPositivePredictiveValue(average=MicroAvg())(yhat, y) ≈ p_micro

    # recall:
    r_macro = mean([1/2, 1/3, 2/5, 1])
    @test MulticlassTruePositiveRate(average=MacroAvg())(yhat, y) ≈ r_macro
    r_macro_w = sum([7/2, 5/3, 4/5, 0/1])/4
    @test MulticlassTruePositiveRate(average=MacroAvg())(yhat, y, class_w) ≈ r_macro_w
    r_micro = (1 + 1 + 2 + 1)/(2 + 3 + 5 + 1)
    @test r_micro ≈
        @test_logs((:warn, StatisticalMeasures.WARN_MICRO_IGNORING_WEIGHTS),
                     MulticlassTruePositiveRate(average=MicroAvg())(yhat, y, class_w))
    @test MulticlassPositivePredictiveValue(average=MicroAvg())(yhat, y) ≈ r_micro

    # fscore:
    harm_means = [harmonic_mean(1/2, 1/2),
                     harmonic_mean(1/3, 1/3),
                     harmonic_mean(1/2, 2/5),
                     harmonic_mean(1/2, 1)]
    f1_macro = mean(harm_means)
    @test MulticlassFScore(average=MacroAvg())(yhat, y) ≈ f1_macro
    @test MulticlassFScore(average=NoAvg(),
                           return_type=Vector)(yhat, y, class_w) ≈
        [7, 5, 2, 0] .* harm_means
    f1_macro_w = sum([7, 5, 2, 0] .* harm_means)/4
    @test MulticlassFScore(average=MacroAvg())(yhat, y, class_w) ≈ f1_macro_w
    f1_micro = harmonic_mean(p_micro, r_micro)
    @test f1_micro ≈
        @test_logs((:warn, StatisticalMeasures.WARN_MICRO_IGNORING_WEIGHTS),
                     MulticlassFScore(average=MicroAvg())(yhat, y, class_w))
    @test MulticlassFScore(average=MicroAvg())(yhat, y) ≈ f1_micro

    # fscore, β=1/3:
    harm_means = [harmonic_mean(1/2, 1/2, beta=1/3),
                     harmonic_mean(1/3, 1/3, beta=1/3),
                     harmonic_mean(1/2, 2/5, beta=1/3),
                     harmonic_mean(1/2, 1, beta=1/3)]
    f1_macro = mean(harm_means)
    @test MulticlassFScore(β=1/3, average=MacroAvg())(yhat, y) ≈ f1_macro
    @test MulticlassFScore(β=1/3,
                           average=NoAvg(),
                           return_type=Vector)(yhat, y, class_w) ≈
        [7, 5, 2, 0] .* harm_means
    f1_macro_w = sum([7, 5, 2, 0] .* harm_means)/4
    @test MulticlassFScore(β=1/3,
                           average=MacroAvg())(yhat, y, class_w) ≈ f1_macro_w
    # micro fscore is insensitive to β:
    f1_micro = harmonic_mean(p_micro, r_micro, beta=42.8)
    @test f1_micro ≈
        @test_logs((:warn, StatisticalMeasures.WARN_MICRO_IGNORING_WEIGHTS),
                   MulticlassFScore(β=1/3,
                                    average=MicroAvg())(yhat, y, class_w))
    @test MulticlassFScore(β=1/3, average=MicroAvg())(yhat, y) ≈ f1_micro
end

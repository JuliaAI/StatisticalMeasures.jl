rng = srng(34234)

@testset "ROC" begin
    y = [  0   0   0   1   0   1   1   0] |> vec
    s = [0.0 0.1 0.1 0.1 0.2 0.2 0.5 0.5] |> vec

    fprs, tprs, ts = Functions.roc_curve(s, y, 1)

    # sk-learn:
    sk_fprs = [0. , 0.2, 0.4, 0.8, 1. ]
    sk_tprs = [0. , 0.33333333, 0.66666667, 1., 1.]

    @test fprs ≈ sk_fprs
    @test tprs ≈ sk_tprs
end

@testset "AUC" begin
    # this is random binary and random scores generated with numpy
    # then using roc_auc_score from sklearn to get the AUC
    # we check that we recover a comparable AUC and that it's invariant
    # to ordering.
    y = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
         1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
         1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0,
         1, 0]
    scores = [
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

    @test isapprox(Functions.auc(scores, y, 1), 0.455716, rtol=1e-4)

    # reversing the roles of positive and negative should return very
    # similar score:
    @test isapprox(
        Functions.auc(1 .- scores, y, 0),
        Functions.auc(scores, y, 1),
        rtol=1e-4,
    )

    # The auc algorithm should be able to handle the case of repeated scores
    # (positive-class probabilities).  We check this by comparing our auc with that gotten
    # from roc_auc_score from sklearn.
    y = [1, 1, 0, 0, 1, 1, 0]
    scores = [0.8,0.7,0.5,0.5,0.5,0.5,0.3]
    @test isapprox(Functions.auc(scores, y, 1), 0.8333333333333334, rtol=1e-16)
end

@testset "accuracy" begin
    y = rand(rng, "abc", 100)
    ŷ =  rand(rng, "abc", 100)
    cm = CM.confmat(y, ŷ)
    m = CM.matrix(cm, warn=false)
    @test Functions.accuracy(m) ≈ mean(y .== ŷ)
end

@testset "kappa" begin
    # Binary case
    y_b = [2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2,
           2, 2, 2, 2, 1, 1, 1, 2, 2]
    ŷ_b = [1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2,
           1, 2, 1, 2, 2, 2, 2, 2, 2]
    m_b =  CM.matrix(CM.confmat(y_b, ŷ_b), warn=false)
    m_b2 = CM.matrix(CM.confmat(y_b, ŷ_b, rev=true), warn=false)
    p0_b = (4+10)/30
    pe_b = (13*11 + 17*19)/(30*30)

    # Multiclass case
    y_m = [5, 5, 3, 5, 4, 4, 2, 2, 3, 2, 5, 2, 4, 3, 2, 1, 1, 5, 1, 4, 2,
           5, 4, 5, 2, 3, 3, 4, 2, 4]
    ŷ_m = [1, 1, 1, 5, 4, 2, 1, 3, 4, 4, 2, 5, 4, 4, 1, 5, 5, 2, 3, 3, 1,
           3, 2, 5, 5, 2, 3, 2, 5, 3]
    m_m = CM.matrix(CM.confmat(ŷ_m, y_m), warn=false)
    m_m2 = CM.matrix(CM.confmat(ŷ_m, y_m, perm=[4,5,1,3,2]), warn=false)
    p0_m = 5/30
    pe_m = (3*6 + 8*6 + 5*6 + 7*5 + 7*7)/(30*30)

    # perfect correlation:
    m = CM.matrix(CM.confmat(y_m, y_m), warn=false)

    # Tests
    @test Functions.kappa(m_m) ≈ (p0_m - pe_m)/(1 - pe_m)
    @test Functions.kappa(m_b) ≈ (p0_b - pe_b)/(1 - pe_b)
    @test Functions.kappa(m_m2) ≈ Functions.kappa(m_m)
    @test Functions.kappa(m_b2) ≈ Functions.kappa(m_b)
    @test Functions.kappa(m) ≈ 1.0
end

@testset "matthews_correlation" begin
    y = [
        3, 4, 1, 1, 1, 4, 1, 3, 3, 1, 2, 3, 1, 3, 3, 3, 2, 4, 3, 2, 1, 3,
        3, 1, 1, 1, 2, 4, 1, 4, 4, 4, 1, 1, 4, 4, 3, 1, 2, 2, 3, 4, 2, 1,
        2, 2, 3, 2, 2, 3, 1, 2, 3, 4, 1, 2, 4, 2, 1, 4, 3, 2, 3, 3, 3, 1,
        3, 1, 4, 3, 1, 2, 3, 1, 2, 2, 4, 4, 1, 3, 2, 1, 4, 3, 3, 1, 3, 1,
        2, 2, 2, 2, 2, 3, 2, 1, 1, 4, 2, 2]
    ŷ = [
        2, 3, 2, 1, 2, 2, 3, 3, 2, 4, 2, 3, 2, 4, 3, 4, 4, 2, 1, 3, 3, 3,
        3, 3, 2, 4, 4, 3, 4, 4, 1, 2, 3, 2, 4, 1, 2, 3, 1, 4, 2, 2, 1, 2,
        3, 2, 2, 4, 3, 2, 2, 2, 1, 2, 2, 1, 3, 1, 4, 1, 2, 1, 2, 4, 3, 2,
        4, 3, 2, 4, 4, 2, 4, 3, 2, 3, 1, 2, 1, 2, 1, 2, 3, 1, 1, 3, 4, 2,
        4, 4, 2, 1, 3, 2, 2, 4, 1, 1, 4, 1]

    m  = CM.matrix(CM.confmat(ŷ, y), warn=false)
    sk_mcc = -0.09759509982785947
    @test Functions.matthews_correlation(m) ≈ sk_mcc

    # invariance to relabelling:
    m2 = CM.matrix(CM.confmat(ŷ, y, perm=[3, 1, 2, 4]), warn=false)
    @test Functions.matthews_correlation(m2) ≈ sk_mcc

    # Issue MLJBase.jl issue #381
    m = [29488 13017; 12790 29753]
    @test Functions.matthews_correlation(m) ≈ 0.39312321239417797
end

@testset "truepositive and cousings" begin
    y = ["-", "+", "-", "-", "+", "-", "-", "+", "+", "-"]
    ŷ = ["+", "-", "-", "+", "-", "+", "+", "+", "-", "-"]
    m  = CM.matrix(CM.confmat(ŷ, y, levels=["-", "+"]))

    #               ┌───────────────────────────┐
    #               │       Ground Truth        │
    # ┌─────────────┼─────────────┬─────────────┤
    # │  Predicted  │      -      │      +      │
    # ├─────────────┼─────────────┼─────────────┤
    # │      -      │      2      │      3      │
    # ├─────────────┼─────────────┼─────────────┤
    # │      +      │      4      │      1      │
    # └─────────────┴─────────────┴─────────────┘

    @test Functions.true_positive(m) == 1   # TP
    @test Functions.true_negative(m) == 2   # TN
    @test Functions.false_positive(m) == 4  # FP
    @test Functions.false_negative(m) == 3  # FN
    @test Functions.true_positive_rate(m) ≈ 1/(1 + 3)
    @test Functions.true_negative_rate(m) ≈ 2/(2 + 4)
    @test Functions.false_positive_rate(m) ≈ 4/(2 + 4)
    @test Functions.false_negative_rate(m) ≈ 3/(1 + 3)
    @test Functions.false_discovery_rate(m) ≈ 4/(4+ 1)
    @test Functions.negative_predictive_value(m) ≈ 2/(2 + 3)
    @test Functions.positive_predictive_value(m) ≈ 1/(4 + 1)
end

@testset "fscore" begin
    y2 = rand(rng, "ab", 1000)
    ŷ2 = rand(rng, "ab", 1000)
    m  = CM.matrix(CM.confmat(ŷ2, y2), warn=false)
    recall = Functions.true_positive_rate(m)
    precision = Functions.positive_predictive_value(m)
    @test Functions.fscore(m) ≈ 2.0 / (1.0 / recall + 1.0 / precision)

    # comparison with sklearn:
    y =[1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2,
        2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1,
        2, 2, 2]
    ŷ = [1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2,
         1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2,
         1, 2, 2]
    m  = CM.matrix(CM.confmat(ŷ, y), warn=false)
    sk_f05 = 0.625
    @test Functions.fscore(m, 0.5) ≈ sk_f05 # m.fbeta_score(y, yhat, 0.5, pos_label=2)
end

@testset "multiclass_truepositive and cousins" begin

    # synthesize 2 x 2 matrix of positive integers `m` and it's [2,1]-permutation:
    m = rand(rng, 1:1_000_000, 2, 2)
    mrev = Matrix{Int}(undef, 2, 2)
    perm = [2, 1]
    for j in 1:2, i in 1:2
        mrev[i, j] = m[perm[i], perm[j]]
    end

    weights= rand(rng, 2)
    w1, w2 = weights

    # atomic functions:
    for f in [:true_positive,  :true_negative, :false_positive, :false_negative]
        multi_f = "multiclass_$f" |> Symbol
        quote
            @test Functions.$multi_f($m) == [Functions.$f($mrev), Functions.$f($m)]
        end |> eval
    end

    # derived functions - no averaging:
    for f in [
        :true_positive_rate,
        :true_negative_rate,
        :false_positive_rate,
        :false_negative_rate,
        :false_discovery_rate,
        :negative_predictive_value,
        :positive_predictive_value,
        ]
        multi_f = "multiclass_$f" |> Symbol
        quote
            @test Functions.$multi_f($m, Functions.NoAvg()) ≈
                [Functions.$f($mrev), Functions.$f($m)]
            @test Functions.$multi_f($m, Functions.NoAvg(), $weights) ≈
                [$w1*Functions.$f($mrev), $w2*Functions.$f($m)]
        end |> eval
    end

    # derived functions - macro averaging:
    for f in [
        :true_positive_rate,
        :true_negative_rate,
        :false_positive_rate,
        :false_negative_rate,
        :false_discovery_rate,
        :negative_predictive_value,
        :positive_predictive_value,
        ]
        multi_f = "multiclass_$f" |> Symbol
        quote
            @test Functions.$multi_f($m, Functions.MacroAvg()) ≈
                [Functions.$f($mrev), Functions.$f($m)] |> mean
            @test Functions.$multi_f($m, Functions.MacroAvg(), $weights) ≈
                sum([$w1*Functions.$f($mrev), $w2*Functions.$f($m)])/length($weights)
        end |> eval
    end

    # micro averaging:
    tp = Functions.true_positive
    fn = Functions.false_negative
    multirecall = Functions.multiclass_true_positive_rate
    TP = tp(mrev) + tp(m)
    FN = fn(mrev) + fn(m)
    @test multirecall(m, Functions.MicroAvg()) ≈ TP/(TP + FN)

end

@testset "multiclass fscore" begin
    table = reshape(collect("aabbbccccddbabccbacccd"), 11, 2)
    yhat = table[:,1] # ['a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'd', 'd']
    y    = table[:,2] # ['b', 'a', 'b', 'c', 'c', 'b', 'a', 'c', 'c', 'c', 'd']
    m = ConfusionMatrices.confmat(yhat, y, levels=['a', 'b', 'c', 'd']) |>
        ConfusionMatrices.matrix
    weights = [7, 5, 2, 0]

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

    harm_means = [harmonic_mean(1/2, 1/2),
                  harmonic_mean(1/3, 1/3),
                  harmonic_mean(1/2, 2/5),
                  harmonic_mean(1/2, 1)]
    f1_macro = mean(harm_means)
    @test Functions.multiclass_fscore(m, 1.0, Functions.MacroAvg()) ≈ f1_macro
    @test Functions.multiclass_fscore(m, 1.0, Functions.NoAvg(), weights) ≈
        [7, 5, 2, 0] .* harm_means
    f1_macro_w = sum(weights .* harm_means)/length(weights)
    @test Functions.multiclass_fscore(m, 1.0, Functions.MacroAvg(), weights) ≈ f1_macro_w
    recall_micro = (1 + 1 + 2 + 1)/(2 + 3 + 4 + 2)
    @test Functions.multiclass_fscore(m, rand(), Functions.MicroAvg()) ≈
        recall_micro

    # fscore, β=1/3:
    harm_means = [harmonic_mean(1/2, 1/2, beta=1/3),
                     harmonic_mean(1/3, 1/3, beta=1/3),
                     harmonic_mean(1/2, 2/5, beta=1/3),
                     harmonic_mean(1/2, 1, beta=1/3)]
    f1_macro = mean(harm_means)
    @test Functions.multiclass_fscore(m, 1/3, Functions.MacroAvg()) ≈ f1_macro
    @test Functions.multiclass_fscore(m, 1/3, Functions.NoAvg(), weights) ≈
        [7, 5, 2, 0] .* harm_means
    f1_macro_w = sum(weights .* harm_means)/length(weights)
    @test Functions.multiclass_fscore(m, 1/3, Functions.MacroAvg(), weights) ≈ f1_macro_w
end

true

const CM = StatisticalMeasures.ConfusionMatrices

@testset "constructors, `matrix`, equality, element access, arithmetic" begin
    m = [1 2; 3 4]
    index_given_level = Dict("A" => 1, "B" => 2)
    levels = ["A", "B"]
    @test_throws ArgumentError CM.ConfusionMatrix(m, ["A",])
    @test_throws ArgumentError CM.ConfusionMatrix(Matrix{Int}(undef, 0, 0), levels)
    @test_throws ArgumentError CM.ConfusionMatrix(m, Dict("A" => 1, "B" => 2, "C" => 1))
    @test_throws ArgumentError CM.ConfusionMatrix(m, Dict("A" => 2, "B" => 3))
    cm = CM.ConfusionMatrix(m, index_given_level)
    @test !isordered(cm)
    @test cm == CM.ConfusionMatrix(m, levels)
    @test cm !== CM.ConfusionMatrix(m, reverse(levels))
    n = [4 3; 2 1]
    rev_index_given_level = Dict("B" => 1, "A" => 2)
    @test cm == CM.ConfusionMatrix(n, rev_index_given_level)
    mat = @test_logs(
        (:warn, CM.WARN_UNORDERED(levels)),
        CM.matrix(cm),
    )
    @test mat == m
    @test_throws CM.ERR_INDEX_ACCESS_DENIED cm[1,1]
    for c in levels, d in levels
        @test cm(c, d) == m[index_given_level[c], index_given_level[d]]
    end

    ordered_cm = CM.ConfusionMatrix(m, rev_index_given_level, ordered=true)
    @test CM.ConfusionMatrix(n, index_given_level, ordered=true) != ordered_cm
    @test @test_logs CM.matrix(ordered_cm) == m
    for i in 1:2, j in 1:2
        @test ordered_cm[i, j] == m[i, j]
    end
    for c in levels, d in levels
        @test cm(c, d) == m[index_given_level[c], index_given_level[d]]
    end

    # arithmetic:
    @test CM.matrix(3*cm + cm*4, warn=false) == 7*CM.matrix(cm, warn=false)
    @test 3cm + cm*4 == 7cm == cm*7
    @test sum([cm, cm, cm]) == 3cm
end

@testset "confmat" begin
    yraw = ['m',     'm', 'f', 'n', missing, 'f', 'm', 'n', 'n', 'm', 'f']
    ŷraw = [missing, 'f', 'f', 'm', 'f',     'f', 'n', 'm', 'n', 'm', 'f']
    y = categorical(yraw)
    ŷ = categorical(ŷraw)
    l = levels(y) # f, m, n
    cm = CM.confmat(ŷ, y)
    ẑ, z = StatisticalMeasures.StatisticalMeasuresBase.skipinvalid(ŷ, y)
    e(c1, c2) = sum((ẑ .== c1) .& (z .== c2))
    for c1 in l, c2 in l
        @test cm(c1, c2) == e(c1, c2)
    end

    cm2 = CM.confmat(ŷraw, yraw)
    @test cm2 == cm

    perm = [3, 1, 2]
    l2 = l[perm]
    cm2 = CM.confmat(ŷ, y; perm=perm)
    @test cm2 == cm
    @test levels(cm2) == l[perm]

    cm2 = CM.confmat(ŷ, y; perm=perm)
    @test cm2 == cm
    @test levels(cm2) == l[perm]

    # more binary tests:
    rng = srng(13)
    yraw = rand(rng, "ab", 50)
    ŷraw = rand(rng, "ab", 50)
    eee(c1, c2) = sum((ŷraw .== c1) .& (yraw .== c2))
    for (ŷ, y) in  [(ŷraw, yraw), categorical.((ŷraw, yraw))]
        @test_throws CM.ERR_ORPHANED_OBSERVATIONS CM.confmat(ŷ, y, levels=['a',])
        l = ['b', 'a']
        cm = CM.confmat(ŷ, y, levels=l)
        @test isordered(cm)
        @test levels(cm) == l
        for c1 in l, c2 in l
            @test cm(c1, c2) == eee(c1, c2)
        end

        cm2 = CM.confmat(ŷ, y, levels=l, rev=true)
        @test isordered(cm2)
        @test cm2 != cm # because they are both ordered but order is different
        @test levels(cm2) == reverse(l)
        for c1 in l, c2 in l
            @test cm2(c1, c2) == cm(c1, c2)
        end
    end

    # more multiclass tests:
    yraw = rand(rng, "abc", 100)
    ŷraw = rand(rng, "abc", 100)
    for (ŷ, y) in  [(ŷraw, yraw), categorical.((ŷraw, yraw))]
        l = ['b', 'a', 'c']
        cm = CM.confmat(ŷ, y, levels=l)
        @test isordered(cm)
        @test levels(cm) == l
        for c1 in l, c2 in l
            @test cm(c1, c2) == eee(c1, c2)
        end

        perm=[3, 1, 2]
        cm2 = CM.confmat(ŷ, y; levels=l, perm)
        @test isordered(cm2)
        @test cm2 != cm # because they are both ordered but order is different
        @test levels(cm2) == l[perm]
        for c1 in l, c2 in l
            @test cm2(c1, c2) == cm(c1, c2)
        end
    end

    y = categorical([1,2,3,1,2,3,1,2,3])
    ŷ = categorical([1,2,3,1,2,3,1,2,3])
    @test_throws ArgumentError CM.confmat(ŷ, y, rev=true)

    # silly test for display
    ŷ = coerce(y, OrderedFactor)
    y = coerce(y, OrderedFactor)
    iob = IOBuffer()
    Base.show(iob, MIME("text/plain"), CM.confmat(ŷ, y))
    siob = String(take!(iob))
    @test strip(siob) == strip("""
                  ┌──────────────┐
                  │ Ground Truth │
        ┌─────────┼────┬────┬────┤
        │Predicted│ 1  │ 2  │ 3  │
        ├─────────┼────┼────┼────┤
        │    1    │ 3  │ 0  │ 0  │
        ├─────────┼────┼────┼────┤
        │    2    │ 0  │ 3  │ 0  │
        ├─────────┼────┼────┼────┤
        │    3    │ 0  │ 0  │ 3  │
        └─────────┴────┴────┴────┘""")

    # yet more binary tests:
    y = categorical([1, 2, 1, 2, 1, 1, 2], ordered=true)
    ŷ = categorical([1, 2, 2, 2, 2, 1, 2], ordered=true)
    cm = CM.confmat(ŷ, y)
    TN = sum(ŷ .== y .== 1) # pred and true = - (1)
    TP = sum(ŷ .== y .== 2) # pred and true = + (2)
    FP = sum(ŷ .!= y .== 1) # pred + (2) and true - (1)
    FN = sum(ŷ .!= y .== 2) # pred - (1) and true + (2)
    @test cm[1,1] == TN
    @test cm[2,2] == TP
    @test cm[1,2] == FN
    @test cm[2,1] == FP

    ym = categorical([1, missing, 2, 1, 2, 1, 1, 1, 2], ordered=true)
    ŷm = categorical([1, 2,       2, 2, 2, missing, 2, 1, 2], ordered=true)
    cm = CM.confmat(ŷ, y)
    TN = sum(skipmissing(ŷ .== y .== 1)) # pred and true = - (1)
    TP = sum(skipmissing(ŷ .== y .== 2)) # pred and true = + (2)
    FP = sum(skipmissing(ŷ .!= y .== 1)) # pred + (2) and true - (1)
    FN = sum(skipmissing(ŷ .!= y .== 2)) # pred - (1) and true + (2)
    @test cm[1,1] == TN
    @test cm[2,2] == TP
    @test cm[1,2] == FN
    @test cm[2,1] == FP

    cm2 = CM.confmat(ŷ, y; rev=true)
    @test cm2[1,1] == cm[2,2]
    @test cm2[1,2] == cm[2,1]
    @test cm2[2,2] == cm[1,1]
    @test cm2[2,1] == cm[1,2]
end

true

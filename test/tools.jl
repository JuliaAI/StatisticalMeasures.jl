using CategoricalDistributions
import StatisticalMeasuresBase as API

@testset "check_pools" begin
    y = categorical(collect("aba"), ordered=true)
    yhat = UnivariateFinite(levels(y), rand(3), augment=true)
    y = [y..., missing]
    yhat = [yhat..., missing]
    @test isnothing(API.check_pools(yhat, y))
end

true

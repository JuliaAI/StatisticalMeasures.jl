@setup_workload begin
    y = rand(2, 3)
    wts = rand(3)
    class_wts = Dict('a'=> 2, 'b'=> 3)
    y2 = CategoricalArrays.categorical(collect("aba"), ordered=true)
    yhat2 = UnivariateFinite(CategoricalArrays.levels(y2), rand(3), augment=true)
    @compile_workload begin
        MultitargetLPLoss()(y, y)
        MultitargetLPLoss()(y, y, wts)
        accuracy(y2, y2)
        log_loss(yhat2, y2)
        confmat(y2, y2)
        roc_curve(yhat2, y2)
    end
end

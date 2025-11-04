# This section is for measures that take `Infinite` input (vectors, arrays or tables)
# and where the prediction has the same form as the ground truth, i.e.,`kind_of_proxy =
# LearnAPI.Point` (in particular, is not a probabilistic prediction).

# If a new measure can be derived from a single scalar function with zero or one
# parameters, then follow the examples that use the `@combination` macro. Otherwise you
# will need to implement calling behaviour directly. See, e.g, `RSquared`.

# -------------------------------------------------------------------
# LPLoss, MultitargetLPLoss

# defines LPLoss and MultitargetLPLoss constructors:
@combination(
    LPLoss(; p=2) = multimeasure(Functions.pth_power_of_absolute_difference, mode=Mean()),
    kind_of_proxy = LearnAPI.Point(),
    observation_scitype = Infinite,
    orientation = Loss(),
    human_name = "``L^p`` loss",
)

# ## LPLoss

# so that constructor appears in dictionary returned by `measures()`:
StatisticalMeasures.register(
    LPLoss,
    "l1",
    "l2",
    "mae",
    "mav",
    "mean_absolute_error",
    "mean_absolute_value"
)

const LPLossDoc = docstring(
    "LPLoss(; p=2)",
    scitype=DOC_INFINITE,
    body=
"""
Specifically, return the mean of ``|ŷ_i - y_i|^p`` over all pairs of observations ``(ŷ_i,
y_i)`` in `(ŷ, y)`, or more generally, the mean of weighted versions of those values. For
the weighted *sum* use [`LPSumLoss`](@ref) instead.
""",
)

"$LPLossDoc"
LPLoss
"$LPLossDoc"
const l1 = LPLoss(1)
"$LPLossDoc"
const l2 = LPLoss(2)
"$LPLossDoc"
const mae = l1
"$LPLossDoc"
const mav = l1
"$LPLossDoc"
const mean_absolute_error = l1
"$LPLossDoc"
const mean_absolute_value = l1
"$LPLossDoc"

# ## MultitargetLPLoss

register(
    MultitargetLPLoss,
    "multitarget_l1",
    "multitarget_l2",
    "multitarget_mae",
    "multitarget_mav",
    "multitarget_mean_absolute_error",
    "multitarget_mean_absolute_value"
)

const MultitargetLPLossDoc = docstring(
    "MultitargetLPLoss(; p=2, atomic_weights=nothing)",
    body=DOC_MULTITARGET(LPLoss),
    scitype="`AbstractArray{<:Union{Missing,Infinite}}`",
)

"$MultitargetLPLossDoc"
MultitargetLPLoss
"$MultitargetLPLossDoc"
const multitarget_l1 = MultitargetLPLoss(1)
"$MultitargetLPLossDoc"
const multitarget_l2 = MultitargetLPLoss(2)
"$MultitargetLPLossDoc"
const multitarget_mae = multitarget_l1
"$MultitargetLPLossDoc"
const multitarget_mav = multitarget_l1
"$MultitargetLPLossDoc"
const multitarget_mean_absolute_error = multitarget_l1
"$MultitargetLPLossDoc"
const multitarget_mean_absolute_value = multitarget_l1
"$MultitargetLPLossDoc"

# -------------------------------------------------------------------
# LPSumLoss, MultitargetLPSumLoss

# defines LPSumLoss and MultitargetLPSumLoss constructors:
@combination(
    LPSumLoss(; p=2) =
        multimeasure(Functions.pth_power_of_absolute_difference, mode=Sum()),
    kind_of_proxy = LearnAPI.Point(),
    observation_scitype = Infinite,
    orientation = Loss(),
    human_name = "``L^p`` sum loss",
)

# ## LPSumLoss

StatisticalMeasures.register(
    LPSumLoss,
    "l1_sum",
    "l2_sum",
)

const LPSumLossDoc = docstring(
    "LPSumLoss(; p=2)",
    scitype=DOC_INFINITE,
    body=
"""
Specifically, compute the (weighted) sum of ``|ŷ_i - yᵢ|^p`` over all pairs of observations
``(ŷ_i, yᵢ)`` in `(ŷ, y)`. For the weighted *mean* use [`LPLoss`](@ref) instead.
""",
)

"$LPSumLossDoc"
LPSumLoss
"$LPSumLossDoc"
const l1_sum = LPSumLoss(1)
"$LPSumLossDoc"
const l2_sum = LPSumLoss(2)
"$LPSumLossDoc"

# ## MultitargetLPSumLoss

register(
    MultitargetLPSumLoss,
    "multitarget_l1_sum",
    "multitarget_l2_sum",
)

const MultitargetLPSumLossDoc = docstring(
    "MultitargetLPSumLoss(; p=2, atomic_weights=nothing)",
    body=DOC_MULTITARGET(LPSumLoss),
    scitype="`AbstractArray{<:Union{Missing,Infinite}}`",
)

"$MultitargetLPSumLossDoc"
MultitargetLPSumLoss
"$MultitargetLPSumLossDoc"
const multitarget_l1_sum = MultitargetLPSumLoss(1)
"$MultitargetLPSumLossDoc"
const multitarget_l2_sum = MultitargetLPSumLoss(2)

# ----------------------------------------------------------------
# RootMeanSquaredError, MultitargetMeanSquaredError

# define constructors:
@combination(
    RootMeanSquaredError() = multimeasure(Functions.absolute_difference, mode=RootMean()),
    kind_of_proxy = LearnAPI.Point(),
    observation_scitype = Infinite,
    orientation = Loss(),
)

# ## RootMeanSquaredError

register(RootMeanSquaredError, "rms", "rmse", "root_mean_squared_error")

const RootMeanSquaredErrorDoc = docstring(
    "RootMeanSquaredError()",
    scitype=DOC_INFINITE,
    body=
"""

Specifically, compute the mean of ``|y_i-ŷ_i|^2`` over all pairs of observations ``(ŷ_i,
y_i)`` in `(ŷ, y)`, and return the square root of the result. More generally,
pre-multiply the squared deviations by the specified weights.

""",
)

"$RootMeanSquaredErrorDoc"
RootMeanSquaredError
"$RootMeanSquaredErrorDoc"
const rms = RootMeanSquaredError()
"$RootMeanSquaredErrorDoc"
const rmse = rms
"$RootMeanSquaredErrorDoc"
const root_mean_squared_error = rms

# ## MultitargetRootMeanSquaredError

register(
    MultitargetRootMeanSquaredError,
    "multitarget_rms",
    "multitarget_rmse",
    "multitarget_root_mean_squared_error",
)

const MultitargetRootMeanSquaredErrorDoc = docstring(
    "MultitargetRootMeanSquaredError(; atomic_weights=nothing)",
    body=DOC_MULTITARGET(RootMeanSquaredError),
    scitype="`AbstractArray{<:Union{Missing,Infinite}}`",
)

"$MultitargetRootMeanSquaredErrorDoc"
MultitargetRootMeanSquaredError
"$MultitargetRootMeanSquaredErrorDoc"
const multitarget_rms = MultitargetRootMeanSquaredError()
"$MultitargetRootMeanSquaredErrorDoc"
const multitarget_rmse = multitarget_rms
"$MultitargetRootMeanSquaredErrorDoc"
const multitarget_root_mean_squared_error = multitarget_rms

# ----------------------------------------------------------------
# RootMeanSquaredLogError, MultitargetRootMeanSquaredLogError

# defines constructors:
@combination(
RootMeanSquaredLogError() =
    multimeasure(Functions.absolute_difference_of_logs, mode=RootMean()),
    kind_of_proxy = LearnAPI.Point(),
    observation_scitype = Infinite,
    orientation = Loss(),
)

# ## RootMeanSquaredLogError

register(RootMeanSquaredLogError, "rmsl", "rmsle", "root_mean_squared_log_error")

const RootMeanSquaredLogErrorDoc = docstring(
    "RootMeanSquaredLogError()",
    scitype=DOC_INFINITE,
    body=
"""
Specifically, return the mean of ``(\\log(y)_i - \\log(ŷ_i))^2`` over all pairs of
observations ``(ŷ_i, y_i)`` in `(ŷ, y)`, and return the square root of the result. More
generally, pre-multiply the values averaged by the specified weights. To include an offset,
use [`RootMeanSquaredLogProportionalError`](@ref) instead.

""",
)

"rmsl", "rmsle", "root_mean_squared_log_error"

"$RootMeanSquaredLogErrorDoc"
RootMeanSquaredLogError
"$RootMeanSquaredLogErrorDoc"
const rmsl = RootMeanSquaredLogError()
"$RootMeanSquaredLogErrorDoc"
const rmsle = rmsl
"$RootMeanSquaredLogErrorDoc"
const root_mean_squared_log_error = rmsl

# ## MultitargetRootMeanSquaredLogError

register(
    MultitargetRootMeanSquaredLogError,
    "multitarget_rmsl",
    "multitarget_rmsle",
    "multitarget_root_mean_squared_log_error",
)

const MultitargetRootMeanSquaredLogErrorDoc = docstring(
    "MultitargetRootMeanSquaredLogError(; atomic_weights=nothing)",
    body=DOC_MULTITARGET(RootMeanSquaredLogError),
    scitype="`AbstractArray{<:Union{Missing,Infinite}}`",
)

"$MultitargetRootMeanSquaredLogErrorDoc"
MultitargetRootMeanSquaredLogError
"$MultitargetRootMeanSquaredLogErrorDoc"
const multitarget_rmsl = MultitargetRootMeanSquaredLogError()
"$MultitargetRootMeanSquaredLogErrorDoc"
const multitarget_rmsle = multitarget_rmsl
"$MultitargetRootMeanSquaredLogErrorDoc"
const multitarget_root_mean_squared_log_error = multitarget_rmsl

# ---------------------------------------------------------------------------
#  RootMeanSquaredLogProportionalError,
#  MultitargetRootMeanSquaredLogProportionalError

# define constructors:
@combination(
    RootMeanSquaredLogProportionalError(; offset=1) =
        multimeasure(Functions.absolute_difference_of_logs, mode=RootMean()),
    kind_of_proxy = LearnAPI.Point(),
    observation_scitype = Infinite,
    orientation = Loss(),
)

# ## RootMeanSquaredLogProportionalError,

StatisticalMeasures.register(RootMeanSquaredLogProportionalError, "rmslp1")

const RootMeanSquaredLogProportionalErrorDoc = docstring(
    "RootMeanSquaredLogProportionalError(; offset=1)",
    scitype=DOC_INFINITE,
    body=
"""
Specifically, compute the mean of ``(\\log(ŷ_i + δ) - \\log(y_i + δ))^2`` over
all pairs of observations ``(ŷ_i, y_i)`` in `(ŷ, y)`, and return the square root. More
generally, pre-multiply the values averaged by the specified weights. Here
``δ``=`offset`, which is `1` by default. This is the same as
[`RootMeanSquaredLogError`](@ref) but adds an offset.
""",
)

"$(RootMeanSquaredLogProportionalErrorDoc)"
RootMeanSquaredLogProportionalError
"$(RootMeanSquaredLogProportionalErrorDoc)"
const rmslp1 = RootMeanSquaredLogProportionalError()

#  ## MultitargetRootMeanSquaredLogProportionalError

StatisticalMeasures.register(
    MultitargetRootMeanSquaredLogProportionalError,
    "multitarget_rmslp1",
)

const MultitargetRootMeanSquaredLogProportionalErrorDoc = docstring(
    "MultitargetRootMeanSquaredLogProportionalError(; offset=1, atomic_weights=nothing)",
    body=DOC_MULTITARGET(RootMeanSquaredLogProportionalError),
    scitype=DOC_INFINITE,
)

"$(MultitargetRootMeanSquaredLogProportionalErrorDoc)"
MultitargetRootMeanSquaredLogProportionalError
"$(MultitargetRootMeanSquaredLogProportionalErrorDoc)"
const multitarget_rmslp1 = MultitargetRootMeanSquaredLogProportionalError()

# ------------------------------------------------------------------------------
# RootMeanSquaredProportionalError, MultitargetRootMeanSquaredProportionalError

# define constructors:
@combination(
    RootMeanSquaredProportionalError(; tol=eps()) =
        multimeasure(Functions.normalized_absolute_difference, mode=RootMean()),
    kind_of_proxy = LearnAPI.Point(),
    observation_scitype = Infinite,
    orientation = Loss(),
)

# ## RootMeanSquaredProportionalError

register(RootMeanSquaredProportionalError, "rmsp")

const RootMeanSquaredProportionalErrorDoc = docstring(
    "RootMeanSquaredProportionalError(; tol=eps())",
    scitype=DOC_INFINITE,
    body=
"""
Specifically, compute the mean of `((ŷᵢ-yᵢ)/yᵢ)^2}` over all pairs of
observations `(ŷᵢ, yᵢ)` in `(ŷ, y)`, and return the square root of the result. More
generally, pre-multiply the values averaged by the specified weights. Terms for which
`abs(yᵢ) < tol` are dropped in the summation, but counts still
contribute to the mean normalization factor.
""",
)

"$RootMeanSquaredProportionalErrorDoc"
RootMeanSquaredProportionalError
"$RootMeanSquaredProportionalErrorDoc"
const rmsp = RootMeanSquaredProportionalError()


# ## MultitargetRootMeanSquaredProportionalError

StatisticalMeasures.register(
    MultitargetRootMeanSquaredProportionalError,
    "multitarget_rmsp",
)

const MultitargetRootMeanSquaredProportionalErrorDoc = docstring(
    "MultitargetRootMeanSquaredProportionalError(; tol=eps(), atomic_weights=nothing)",
    body=DOC_MULTITARGET(RootMeanSquaredProportionalError),
    scitype=DOC_INFINITE,
)

"$(MultitargetRootMeanSquaredProportionalErrorDoc)"
MultitargetRootMeanSquaredProportionalError
"$(MultitargetRootMeanSquaredProportionalErrorDoc)"
const multitarget_rmsp = MultitargetRootMeanSquaredProportionalError()

# ------------------------------------------------------------------------
# MeanAbsoluteProportionalError, MultitargetMeanAbsoluteProportionalError

# define constructors:
@combination(
    MeanAbsoluteProportionalError(; tol=eps()) =
        multimeasure(Functions.normalized_absolute_difference, mode=Mean()),
    kind_of_proxy = LearnAPI.Point(),
    observation_scitype = Infinite,
    orientation = Loss(),
)

# ## MeanAbsoluteProportionalError

register(MeanAbsoluteProportionalError, "mape")

const MeanAbsoluteProportionalErrorDoc = docstring(
    "MeanAbsoluteProportionalError(; tol=eps())",
    scitype=DOC_INFINITE,
    body=
"""
Specifically, return the mean of ``|ŷ_i-y_i| \\over |y_i|`` over all pairs of observations
``(ŷ_i, y_i)`` in `(ŷ, y)`. More generally, pre-multiply the values averaged by the
specified weights. Terms for which ``|y_i|``<`tol` are dropped in the summation, but
corresponding weights (or counts) still contribute to the mean normalization factor.
""",
)

"$MeanAbsoluteProportionalErrorDoc"
MeanAbsoluteProportionalError
"$MeanAbsoluteProportionalErrorDoc"
const mape = MeanAbsoluteProportionalError()

# ## MultitargetMeanAbsoluteProportionalError

StatisticalMeasures.register(
    MultitargetMeanAbsoluteProportionalError,
    "multitarget_mape",
)

const MultitargetMeanAbsoluteProportionalErrorDoc = docstring(
    "MultitargetMeanAbsoluteProportionalError(; tol=eps(), atomic_weights=nothing)",
    body=DOC_MULTITARGET(MeanAbsoluteProportionalError),
    scitype=DOC_INFINITE,
)

"$(MultitargetMeanAbsoluteProportionalErrorDoc)"
MultitargetMeanAbsoluteProportionalError
"$(MultitargetMeanAbsoluteProportionalErrorDoc)"
const multitarget_mape = MultitargetMeanAbsoluteProportionalError()

# -----------------------------------------------------------------------
# LogCoshLoss, MultitargetLogCoshLoss

# define constructors:
@combination(
    LogCoshLoss() = multimeasure(Functions.log_cosh_difference, mode=Mean()),
    kind_of_proxy = LearnAPI.Point(),
    observation_scitype = Infinite,
    orientation = Loss(),
)

# ## LogCoshLoss

StatisticalMeasures.register(LogCoshLoss, "log_cosh", "log_cosh_loss")

const LogCoshLossDoc = docstring(
    "LogCoshLoss()",
    scitype=DOC_INFINITE,
    body=
"""
Return the mean of ``\\log(\\cosh(ŷ_i-y_i))`` over all pairs of observations
``(ŷ_i, y_i)`` in `(ŷ, y)`. More
generally, pre-multiply the values averaged by the specified weights.
""",
)

"$LogCoshLossDoc"
LogCoshLoss
"$LogCoshLossDoc"
const log_cosh = LogCoshLoss()
"$LogCoshLossDoc"
const log_cosh_loss = log_cosh


# ## MultitargetLogCoshLoss

StatisticalMeasures.register(
    MultitargetLogCoshLoss,
    "multitarget_mape",
)

const MultitargetLogCoshLossDoc = docstring(
    "MultitargetLogCoshLoss(; atomic_weights=nothing)",
    body=DOC_MULTITARGET(LogCoshLoss),
    scitype=DOC_INFINITE,
)

"$(MultitargetLogCoshLossDoc)"
MultitargetLogCoshLoss
"$(MultitargetLogCoshLossDoc)"
const multitarget_log_cosh = MultitargetLogCoshLoss()
"$(MultitargetLogCoshLossDoc)"
const multitarget_log_cosh_loss = multitarget_log_cosh


# -------------------------------------------------------------------------
# R-squared (coefficient of determination)

# type for measure without argument checks:
struct _RSquared  end

function (::_RSquared)(yhat, y)
    numerator = LPSumLoss()(yhat, y) # sum of squared differences, handling `missing`s`
    μ = aggregate(y) # mean, handling `missing`s
    denominator = aggregate(η -> (η - μ)^2, y, mode=Sum())
    return 1 - (numerator / denominator)
end

# constructor for measure with checks:
RSquared() = _RSquared() |> API.robust_measure |> API.fussy_measure

# supertype for measures so-constructed:
const RSquaredType = API.FussyMeasure{<:API.RobustMeasure{<:_RSquared}}

# these traits will be inherited by `RSquared`:
@trait(
    _RSquared,
    consumes_multiple_observations = true,
    kind_of_proxy = LearnAPI.Point(),
    observation_scitype = Union{Missing,Infinite},
    orientation = Score(),
    human_name = "R² coefficient",
)

# get nice show methods measures constructed with `RSquared`:
@fix_show RSquared::RSquaredType

register(RSquared, "rsq", "rsquared")

const RSquaredDoc = docstring(
    "RSquared()",
    scitype=DOC_INFINITE,
    body=
"""
Specifically, return the value of

``1 - \\frac{∑ᵢ (ŷ_i- y_i)^2}{∑ᵢ (ȳ - y_i)^2}``

where ``ȳ`` denote the mean of the ``y_i``. Also known as R-squared or the coefficient
of determination, the `R²` coefficients is suitable for interpreting linear regression
analysis (Chicco et al., [2021](https://doi.org/10.7717/peerj-cs.623)).

""",
)

"$RSquaredDoc"
RSquared
"$RSquaredDoc"
const rsq = RSquared()
"$RSquaredDoc"
const rsquared = rsq

# -------------------------------------------------------------------------
# Willmott index of agreement (d)

# type for measure without argument checks:
struct _WillmottD end

function (::_WillmottD)(yhat, y)
    μ = aggregate(y)  # mean
    # numerator: Σ_i (ŷ_i - y_i)^2
    num = LPSumLoss(p=2)(yhat, y)
    # denominator: Σ_i (|ŷ_i - μ| + |y_i - μ|)^2
    den = multimeasure((yhat, y) -> (abs(yhat - μ) + abs(y - μ))^2; mode=Sum())(yhat, y)
    return den == 0 ? (num == 0 ? 1.0 : 0.0) : 1 - num/den
end

WillmottD() = _WillmottD() |> API.robust_measure |> API.fussy_measure
const WillmottDType = API.FussyMeasure{<:API.RobustMeasure{<:_WillmottD}}

@trait(
    _WillmottD,
    consumes_multiple_observations = true,
    kind_of_proxy = LearnAPI.Point(),
    observation_scitype = Union{Missing,Infinite},
    orientation = Score(),
    human_name = "Willmott index of agreement (d)",
)

@fix_show WillmottD::WillmottDType

register(WillmottD, "willmott_d")

const WillmottDDoc = docstring(
    "WillmottD()",
    scitype=DOC_INFINITE,
    body=
"""
Returns Willmott index of agreement (d)

``d = 1 - \\dfrac{\\sum (ŷ_i - y_i)^2}{\\sum (|ŷ_i - \\bar y| + |y_i - \\bar y|)^2}``,

where ``\\bar y`` is the mean of the targets. The value lies in ``[0,1]`` with higher
being better.

References: Willmott [(1981)](https://doi.org/10.1080/02723646.1981.10642213)
""",
)

"$WillmottDDoc"
WillmottD
"$WillmottDDoc"
const willmott_d = WillmottD()

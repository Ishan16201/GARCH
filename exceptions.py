import warnings

class NonStationaryDataError(Exception):
    """
    Exception raised when financial time-series data fails to reject the 
    null hypothesis of non-stationarity at required significance levels.
    """
    pass

class ArchEffectsNotFoundWarning(UserWarning):
    """
    Warning issued when no significant ARCH (Autoregressive Conditional 
    Heteroskedasticity) effects are detected in the squared returns.
    """
    pass

class ConstraintViolationError(Exception):
    """
    Exception raised when GARCH model parameters violate stationarity 
    or positivity constraints.
    """
    pass

class ConvergenceFailedError(Exception):
    """
    Exception raised when the MLE optimization fails to converge 
    from all starting points.
    """
    pass

class SingularHessianWarning(UserWarning):
    """
    Warning issued when the Hessian matrix is singular or not 
    positive-definite, preventing reliable standard error calculation.
    """
    pass

class ModelMisspecificationWarning(UserWarning):
    """
    Warning issued when diagnostic tests suggest the GARCH model order 
    (p,q) is misspecified (e.g., remaining ARCH effects in residuals).
    """
    pass

class ClusteringRiskWarning(UserWarning):
    """
    Warning issued when VaR exceedances exhibit clustering (non-independence),
    indicating that the volatility model fails to capture variance dynamics.
    """
    pass

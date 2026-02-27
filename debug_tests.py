import traceback
import sys

try:
    print("Testing stationarity_validation...")
    from test_garch_smoke import test_stationarity_validation
    test_stationarity_validation()
    print("SUCCESS")
except Exception:
    traceback.print_exc()

try:
    print("\nTesting likelihood_finite...")
    from test_garch_smoke import test_likelihood_finite
    test_likelihood_finite()
    print("SUCCESS")
except Exception:
    traceback.print_exc()

try:
    print("\nTesting mle_convergence...")
    from test_garch_smoke import test_mle_convergence
    test_mle_convergence()
    print("SUCCESS")
except Exception:
    traceback.print_exc()

try:
    print("\nTesting forecast_convergence...")
    from test_garch_smoke import test_forecast_convergence
    test_forecast_convergence()
    print("SUCCESS")
except Exception:
    traceback.print_exc()

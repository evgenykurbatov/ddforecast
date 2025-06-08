
try:
    import ddforecast
except ImportError:
    import sys
    sys.path.append('..')
    import ddforecast

print("Version:", ddforecast.__version__)

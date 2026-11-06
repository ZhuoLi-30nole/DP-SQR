"""PyCharm: run both public-data quantile grids and reuse housing coefficients."""
import sys
from run import main
if __name__=='__main__':
    sys.argv=[sys.argv[0],'batch','--study','real',*sys.argv[1:]]
    main()

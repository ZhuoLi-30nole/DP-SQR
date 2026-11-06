"""PyCharm: run directly for all three low-dimensional 300-repeat grids."""
import sys
from run import main
if __name__=='__main__':
    sys.argv=[sys.argv[0],'batch','--study','low',*sys.argv[1:]]
    main()

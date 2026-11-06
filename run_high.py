"""PyCharm: run directly for the four non-overlapping sparse simulation grids."""
import sys
from run import main
if __name__=='__main__':
    sys.argv=[sys.argv[0],'batch','--study','high',*sys.argv[1:]]
    main()

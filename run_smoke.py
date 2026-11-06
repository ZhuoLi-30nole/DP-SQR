"""Small actual fits; output is explicitly diagnostic and never a formal grid."""
import sys
from run import main
if __name__=='__main__':
    sys.argv=[sys.argv[0],'batch','--study','smoke','--output-root','results/smoke',*sys.argv[1:]]
    main()

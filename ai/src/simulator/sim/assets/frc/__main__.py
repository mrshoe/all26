import sys

from sim.assets.frc.importer import main

if __name__ == "__main__":  # not again in the importer's worker processes
    sys.exit(main())

"""Legacy entry point — equivalent to ``play.py --platform ios``."""

from play import main

if __name__ == "__main__":
    main(["--platform", "ios"])

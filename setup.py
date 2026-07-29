"""
Setup script for PyStreamAI with post-install hook
"""

import sys
from setuptools import setup
from setuptools.command.install import install


class PostInstallCommand(install):
    """Post-installation command to display welcome message"""

    def run(self):
        install.run(self)
        # Import and run post-install message
        try:
            from pystreamai.scripts.post_install import post_install
            post_install()
        except Exception as e:
            print(f"Note: Could not display welcome message: {e}", file=sys.stderr)


if __name__ == '__main__':
    setup(
        cmdclass={
            'install': PostInstallCommand,
        }
    )

from setuptools import find_packages, setup
import os
from glob import glob

package_name = "madbp_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "configs"), glob("configs/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Siddharth Singh",
    maintainer_email="siddharth@ornl.gov",
    description="ROS 2 integration layer for MA-DBP planner.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "madbp_planner_node = madbp_ros.madbp_planner_node:main",
            "lichtblik_bridge    = madbp_ros.lichtblik_bridge:main",
            "mujoco_bridge       = madbp_ros.mujoco_bridge:main",
        ],
    },
)

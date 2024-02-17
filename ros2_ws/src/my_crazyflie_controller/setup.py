from setuptools import find_packages, setup

package_name = 'my_crazyflie_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='a',
    maintainer_email='a@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "test_node = my_crazyflie_controller.test_node:main",
            "point_cloud_subcriber = my_crazyflie_controller.point_cloud_tracking:main",
            "follower = my_crazyflie_controller.follower:main",
            "circler = my_crazyflie_controller.circler:main",
            "land = my_crazyflie_controller.land:main",
            "pursuer = my_crazyflie_controller.pursuer:main",
            "traj_gen = my_crazyflie_controller.traj_gen:main",
            "plotter = my_crazyflie_controller.plotter:main",
            "bat = my_crazyflie_controller.bat:main",
            "evader = my_crazyflie_controller.evader:main",
            "evader_sim = my_crazyflie_controller.evader_sim:main",
            "pursuer_sim = my_crazyflie_controller.pursuer_sim:main",
            "pursuer_marker = my_crazyflie_controller.pursuer_marker:main"
        ],
    },
)

from setuptools import find_packages, setup

package_name = 'go2_state_estimation'

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
    maintainer='Puneet Dhandale',
    maintainer_email='puneet.dhandale25@kgpian.iitkgp.ac.in',
    description='EKF State Estimation node for Unitree Go2 quadruped robot',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'state_estimator_node = go2_state_estimation.state_estimator_node:main'
        ],
    },
)

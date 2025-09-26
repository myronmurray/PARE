from pathlib import Path
import setuptools


def _read_requirements() -> list[str]:
    requirements_path = Path(__file__).resolve().parent / "requirements.txt"
    if not requirements_path.exists():
        return []

    install_requires: list[str] = []
    with requirements_path.open() as req_file:
        for line in req_file:
            requirement = line.strip()
            if not requirement or requirement.startswith("#"):
                continue
            if requirement.startswith("git+https://github.com/mattloper/chumpy"):
                requirement = "chumpy @ " + requirement
            elif requirement.startswith("git+https://github.com/mkocabas/yolov3-pytorch.git"):
                requirement = "yolov3 @ " + requirement
            elif requirement.startswith("git+https://github.com/mkocabas/multi-person-tracker.git"):
                requirement = "multi-person-tracker @ " + requirement
            elif requirement.startswith("git+https://github.com/giacaglia/pytube.git"):
                requirement = "pytube @ " + requirement
            install_requires.append(requirement)
    return install_requires


setuptools.setup(
    name="pare",
    version="0.1",
    author="PARE team",
    description="PARE: Part Attention Regressor for 3D Human Body Estimation",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"pare": ["data/*", "data/**/*"]},
    python_requires=">=3.11",
    install_requires=_read_requirements(),
    zip_safe=False,
)

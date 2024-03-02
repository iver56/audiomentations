* Check that all unit tests are OK
* Run the demo and listen to the sounds to empirically check the results of any new or changed transforms
* Bump the version number in `audiomentations/__init__.py` in accordance with the [semantic versioning specification](https://semver.org/)
* Write a summary of the changes in the version history section in changelog.md. Remember to add a link to the new version near the bottom of the file.
* Include changelog for only the newest version in README.md
* Commit and push the change with a commit message like this: "Release vx.y.z" (replace x.y.z with the package version)
* Add and push a git tag to the release commit
* Add a release here: https://github.com/iver56/audiomentations/releases/new
* Update the Zenodo badge in README.md. Commit and push.
* Remove any old files inside the dist folder
* `python setup.py sdist bdist_wheel`
* `python -m twine upload dist/*`

For Developers
==============


Installation
------------

Typically, developers will want to fork the repository and install the package in locally editable version as follows:

.. code:: bash

  # start a fresh environment
  conda create -n skerch python==3.10
  conda activate skerch

  # clone e.g. via gh CLI
  gh repo clone andres-fr/skerch
  cd skerch

  # local installation
  pip install -e ".[dev,test,lint,docs]"

You can test the installation by e.g. running one of the examples from the documentation:

.. code:: python

   python docs/materials/examples/example_deep_learning.py


See below (and :ref:`Development` in particular) for further development recommendations.


Testing and Coverage
--------------------

Full tests and coverage can be run via following command in the repo root:

.. code:: bash

  make test

They can take several hours though. A quicker (few minutes), yet still representative subset of the tests can be run via:

.. code:: bash

  make test-light



Documentation and Integration Tests
-----------------------------------

Docs can be locally rendered via ``sphinx`` by running the following in the repo root:

.. code:: bash

  make docs

other targets available are:

* ``docs-prepare``: Creates the docs layout but doesn't render
* ``docs-html``: Renders HTML version
* ``docs-pdf``: Renders PDF version

To work on the documentation, use existing examples as reference.

The gallery examples serve also as integration tests, and they can be run directly by the Python interpreter (assuming it can ``import skerch``). This way, we can conveniently insert `breakpoint()` statements for development and debugging.

Make sure that the docs can be correctly created before submitting any contributions.


Development
-----------


This repository uses `Commitizen <https://commitizen-tools.github.io/commitizen>`_ to standarize and automate git commiting and versioning. We also encourage the use of `pre-commit <https://pre-commit.com/>`_ to ensure contents remain closely aligned to guidelines and releases take less effort (CI/CD).

Setting up ``pre-commit``
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``docs/materials/.pre-commit-config.yaml`` template file can be used and customized to configure pre-commit checks (like linting, versioning and doc checking). Then, pre-commit can be set up as follows:

.. code:: bash

  cd <REPO_ROOT>
  cp docs/materials/.pre-commit-config.yaml .
  # optionally, configure <REPO_ROOT>/.pre-commit-config.yaml
  # as desired since your local changes won't be commited
  # (the file is .gitignored)
  pre-commit install --hook-type commit-msg

You can test if the pre-commit hooks are passing via ``pre-commit run``, which would e.g. look like this::

  debug statements (python)................................................Passed
  check for broken symlinks............................(no files to check)Skipped
  check for added large files..............................................Passed
  check for case conflicts.................................................Passed
  check for merge conflicts................................................Passed
  check docstring is first.................................................Passed
  fix end of files.........................................................Passed
  trim trailing whitespace.................................................Passed
  check json...........................................(no files to check)Skipped
  check yaml...........................................(no files to check)Skipped
  check xml............................................(no files to check)Skipped
  check toml...........................................(no files to check)Skipped
  detect private key.......................................................Passed
  fix python encoding pragma...............................................Passed
  check that executables have shebangs.....................................Passed
  mixed line ending........................................................Passed
  fix requirements.txt.................................(no files to check)Skipped
  isort....................................................................Passed
  black....................................................................Passed
  flake8...................................................................Passed
  pydocstyle...............................................................Passed
  sphinx-html-build........................................................Passed

.. note::

  * If everything is `Skipped`, you may have forgotten to stage your changes via `git add`.
  * Some pre-commit hooks (e.g. `fix end of files`) "fix" (i.e. *modify*) the files that fail to satisfy them. Make sure to run `git add` after the pre-commit, to ensure changes are staged, and they will pass afterwards.


``Commitizen``
^^^^^^^^^^^^^^

Commitizen can then be used to perform commits and version bumps following specific standards. It can be set up as follows (this has been already done for this repo and doesn't need to be done again):

.. code:: bash

  cz init
  # use pyproject.toml with conventional commits
  # store pep440 versions in the .toml
  # create changelog automatically
  # Keep major version zero
  # do not install pre-commit hook via cz init

Check ``pyproject.toml`` for more details. Other ``cz ...`` subcommands like ``ls, example, info, schema, version`` provide also details about the configuration.

To commit, make sure to stage (``git add``) the relevant files, and then call ``cz commit``. After answering the questions, the pre-commit checks will be run, and if all pass, the commit will be successfully logged. Otherwise, make the necessary changes to pass pre-commit checks, stage the new modifications, and call ``cz commit --retry`` until it passes.

Last but not least, major/minor releases and bugfixes are also managed by commitizen. Simply call:

.. code:: bash

  cz bump

Commitizen will read the current version and types of changes from the commit history, and infer the next version. The bump can be pushed via ``git push --tags``.



Releases and CI
---------------

Whenever anything is pushed to ``main|dev``, or a PR is pushed:

* A CI pipeline is triggered and general tests are run (linting, unit/integration tests with coverage, documentation build)
* HTML Documentation is built and published to `ReadTheDocs <https://readthedocs.org>`_
* Coverage report is published to `Coveralls <https://coveralls.io>`_

Furthermore, whenever a tag is pushed, we want this to automatically trigger a release, consisting of the following steps:

1. Run general CI tests and publish status as per above. If anything goes wrong, interrupt the pipeline
2. Publish package to the `Python Package Index (PyPI) <https://pypi.org>`_ so it can be installed via ``pip install <ONLINE PACKAGE>``
3. Also publish package to GitHub itself as a release

This is done in a fully automated way, and the badges in the README inform about the status of these pipelines in (quasi) real-time.

Setting this up requires to configure the GitHub repo together with the PyPI, ReadTheDocs and Coveralls services. Below are the detailed steps that were performed to set this up, plus some explanation (developers don't need to do this again). Here, we expect the GitHub repo to be public.

.. seealso::

  * `Publishing Python packages <https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/>`_
  * `GitHub Actions <https://docs.github.com/en/actions>`_ and the YAML files inside ``.github/workflows`` in the repository
  * `ReadTheDocs advanced configuration <https://docs.readthedocs.io/en/stable/build-customization.html#extend-the-build-process>`_
  * `how ReadTheDocs handles versions and tags <https://docs.readthedocs.io/en/stable/versions.html>`_

Tests pipeline and Coveralls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CI pipeline that performs the tests and reports to Coveralls is defined in the first job of the ``.github/workflows/ci.yaml`` YAML file. The existence of this file is all GitHub needs to run the pipeline.

The `Coveralls service <https://coveralls.io/>`_ reports on code coverage based on the results of running ``pytest``. This was configured by signing in to Coveralls via the GitHub account and "activating" the desired repository. No need to set any Actions evironment variables. Then, the GitHub Action will take care of the rest:

1. Running `pytest` with the ``--cov`` plugin will generate a ``.coverage`` report in the runner's repo root
2. Running ``coveralls`` will then access the service and provide the report
3. The service will then make the report available online

Optionally, grab the Markdown badge from the coveralls website and add it to your repo ``README``.

ReadTheDocs
^^^^^^^^^^^

To render and deploy the documentation online, we add a ``.readthedocs.yaml`` YAML file to the repo root, which works analogously to the GitHub workflow files.

In our case, this is a customized build, requiring extra dependencies and commands to correctly build the documentation). Also, if you want the doc build to be conditioned on some previous command, the current way of doing it is to add said command before ``make docs`` in the ``.readthedocs.yaml`` file.

The `ReadTheDocs <https://readthedocs.org>`_ service needs to be now configured to actually look at this file, build and deploy the docs. Log in e.g. using the GitHub option (if you are doing this for the first time, you may also need to `connect <https://docs.readthedocs.io/en/stable/reference/git-integration.html>`_ both platforms). Then, under "Import a repository", find and add the desired repository.

Optionally, grab the Markdown badge from the ReadTheDocs website and add it to your repo ``README``.


PyPI and GH Releases
^^^^^^^^^^^^^^^^^^^^

Last but not least, whenever a tag is pushed and CI tests went well, we want to publish the package to `PyPI <https://pypi.org>`_ and GitHub (PyPI won't accept packages with 'dev' versions, so only tag pushes can/should be released). The corresponding job in the ``.github/workflows/ci.yaml`` YAML file takes care of this (note the conditional execution based on passing the tests).

Apart from the existence of this file, log into PyPI and authorize the GitHub repository as a "pending publisher" (see `these publishing docs <https://pypi.org/manage/account/publishing/>`_) by providing a package name (must match the name of the package resulting from ``python3 -m build``), as well as the corresponding GitHub user and repo names. Finally, we provide the name of the Actions workflow file, in this case ``ci.yaml``.

The GitHub actions release does not require any further configuration. Now, pushing tags should trigger a release (if all CI tests pass), and the package will be easily accessible online.

Optionally, add badges `for the GitHub actions <https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/adding-a-workflow-status-badge>`_ and `PyPI website <https://stackoverflow.com/a/69223516>`_ and add it to your repo ``README``.


Deploying
^^^^^^^^^

* ``cz commit`` and ``git push``  to push as usual
* To trigger a release, ``cz commit`` and ``cz bump``. If there was actually a bump (depends on the commit history), then ``git push``, and ``git push --tags`` will trigger the release CI.

.ONESHELL: all
.DEFAULT: help

VERSION := $(shell sed -n 's/^version = //gp' pyproject.toml | tr -d '"')
DOCSDIR := "docs/.sphinx"
GITHUB_URL := "https://github.com/andres-fr/skerch"

help:
	@echo "docs-prepare"
	@echo "        Build the documentation"
	@echo "docs-html"
	@echo "        Build the documentation as HTML. Requires docs-prepare"
	@echo "docs-pdf"
	@echo "        Build the documentation as PDF. Requires docs-prepare"
	#
	@echo "docs"
	@echo "        Runs docs-prepare docs-html docs-pdf"
	@echo "isort"
	@echo "        Runs isort, formats Python imports"
	@echo "black"
	@echo "        Runs black, formats Python code"
	@echo "flake8"
	@echo "        Runs flake8, a Python linter"
	@echo "pydocstyle"
	@echo "        Runs pydocstyle, lints and checks style of documentation"
	@echo "lint"
	@echo "        Runs isort black flake8 pydocstyle"
	#
	@echo "test"
	@echo "        Runs all pytests in 'test' plus coverage"
	@echo "test-light"
	@echo "        Runs lightweight (yet fairly comprehensive) subset of"
	@echo "        pytests in 'test' plus coverage"
	@echo "test-ci"
	@echo "        CI-friendly version of tests, subset of test-light"

.PHONY: docs-prepare
docs-prepare:
	# start sphinx dir from scratch and configure
	@rm -rf $(DOCSDIR); mkdir -p $(DOCSDIR)
	@sphinx-quickstart -l en -p skerch -a aferro -v $(VERSION) -r $(VERSION) \
		--ext-githubpages --ext-autodoc --ext-mathjax --ext-viewcode \
		--ext-coverage --ext-intersphinx --ext-githubpages \
		--ext-doctest --no-sep \
		--extensions=sphinx_rtd_theme,sphinx.ext.graphviz,sphinx.ext.autosectionlabel,sphinx_gallery.gen_gallery,sphinx_immaterial.task_lists \
		$(DOCSDIR)

	@echo 'sphinx_gallery_conf = {"examples_dirs": "../materials/examples", \
		"gallery_dirs": "examples", \
		"default_thumb_file": "docs/materials/assets/skerch_logo.svg", \
		"remove_config_comments": True, "filename_pattern": "example", \
		"matplotlib_animations": True}' >> $(DOCSDIR)/conf.py
	# incorporate apidocs and custom materials
	@sphinx-apidoc -M -o $(DOCSDIR) skerch -T \
		-t docs/materials/apidoc_templates
	@cp docs/materials/*.rst $(DOCSDIR)
	# Add metadata to all rst files ("edit on GitHub" link):
	for path in $(shell find $(DOCSDIR) -iname "*.rst");
	do
	  @echo "adding GH link to" $$path
	  @sed -i '1i:github_url: '$(GITHUB_URL)'\n' "$$path"
	done

.PHONY: docs-html
docs-html:
	@sphinx-build -M html $(DOCSDIR) $(DOCSDIR)/_build \
	    -D html_theme="sphinx_rtd_theme" \
	    -D html_favicon=../materials/assets/favicon.ico \
	    -D html_logo=../materials/assets/skerch_horizontal.svg \
	    -D html_theme_options.logo_only=true \
	    -D suppress_warnings="autosectionlabel" \
	    -W

.PHONY: docs-pdf
docs-pdf:
	@sphinx-build -M latexpdf $(DOCSDIR) $(DOCSDIR)/_build \
	    -D html_theme="sphinx_rtd_theme" \
	    -D html_favicon=../materials/assets/favicon.ico \
	    -D html_logo=../materials/assets/skerch_horizontal.svg \
	    -D suppress_warnings="autosectionlabel" \
	    -D html_theme_options.logo_only=true

.PHONY: docs
docs: docs-prepare docs-html docs-pdf

.PHONY: isort
isort:
	@isort . --sp pyproject.toml

.PHONY: black
black:
	@black . --config pyproject.toml

.PHONY: flake8
flake8:
	@flake8 .

.PHONY: pydocstyle
pydocstyle:
	@pydocstyle . --config pyproject.toml # --count

.PHONY: lint
lint: isort black flake8 pydocstyle

.PHONY: test
test:
	@pytest -vx test --cov skerch

.PHONY: test-light
test-light:
	@pytest -vx test --cov skerch --quick --seeds='12345'

.PHONY: test-ci
test-ci:
	@pytest -vx test --cov skerch --quick --seeds='12345' \
	    --skip_toomanyfiles

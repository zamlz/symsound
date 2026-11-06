{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # `md code` runs the code blocks in docs/ and writes their output back
    # (It keeps its own nixpkgs pin: its build lints with `clippy -D warnings`,
    # which a newer nixpkgs would fail.)
    mdutils.url = "github:zamlz/mdutils";
  };

  outputs = { nixpkgs, mdutils, ... }: let
    forAllSystems = nixpkgs.lib.genAttrs [
      "x86_64-linux"
      "x86_64-darwin"
      "aarch64-linux"
      "aarch64-darwin"
    ];

    # FIXME: Is there a better way to get the packages from the
    # pyproject.toml file without redefining them here?
    pythonEnv = pkgs: pkgs.python3.withPackages (ps: with ps; [
      ipdb
      hypothesis
      ipython
      matplotlib
      mypy
      numpy
      pytest
      pytest-cov
      scipy
      sounddevice
      soundfile
      sympy
    ]);
  in
  {
    devShells = forAllSystems (system: let
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      default = pkgs.mkShell {
        packages = [
          (pythonEnv pkgs)
          pkgs.ruff
          mdutils.packages.${system}.default
        ];
        # Lets the scripts in examples/ import the daw package
        shellHook = ''
          export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
        '';
      };
    });

    checks = forAllSystems (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      # Run $script in a writable copy of the repo
      mkCheckWith = extraInputs: name: script: pkgs.runCommand "daw-${name}" {
        nativeBuildInputs = [ (pythonEnv pkgs) pkgs.ruff ] ++ extraInputs;
        # Reproducible property-based tests (see tests/conftest.py)
        HYPOTHESIS_PROFILE = "ci";
      } ''
        cp -r --no-preserve=mode ${./.} repo
        cd repo
        ${script}
        touch $out
      '';
      mkCheck = mkCheckWith [ ];
    in {
      pytest = mkCheck "pytest" "pytest -q -p no:cacheprovider";
      ruff = mkCheck "ruff" "ruff check --no-cache . && ruff format --check --no-cache tests";
      mypy = mkCheck "mypy" "mypy --cache-dir=/dev/null src/daw";
      # Re-runs every code block in docs/ and fails if the committed output or
      # graphs differ (fix with: python tools/build_docs.py)
      docs = mkCheckWith [ mdutils.packages.${system}.default ] "docs" ''
        export HOME=$TMPDIR PYTHONPATH=$PWD/src
        python tools/build_docs.py --check
      '';
    });
  };
}

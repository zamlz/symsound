{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }: let
    forAllSystems = nixpkgs.lib.genAttrs [
      "x86_64-linux"
      "x86_64-darwin"
      "aarch64-linux"
      "aarch64-darwin"
    ];

    # FIXME: Is there a better way to get the packages from the
    # pyproject.toml file without redefining them here?
    pythonEnv = pkgs: pkgs.python311.withPackages (ps: with ps; [
      ipdb
      ipython
      matplotlib
      mypy
      numpy
      pytest
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
        ];
      };
    });

    checks = forAllSystems (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      # Run $script in a writable copy of the repo
      mkCheck = name: script: pkgs.runCommand "daw-${name}" {
        nativeBuildInputs = [ (pythonEnv pkgs) pkgs.ruff ];
      } ''
        cp -r --no-preserve=mode ${./.} repo
        cd repo
        ${script}
        touch $out
      '';
    in {
      pytest = mkCheck "pytest" "pytest -q -p no:cacheprovider";
      ruff = mkCheck "ruff" "ruff check --no-cache .";
      mypy = mkCheck "mypy" "mypy --cache-dir=/dev/null src/daw";
    });
  };
}

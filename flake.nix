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
  in
  {
    devShells = forAllSystems (system: let
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      default = pkgs.mkShell {
        packages = with pkgs; [
          python311Packages.ipdb
          python311Packages.ipython
          python311Packages.matplotlib
          python311Packages.numpy
          python311Packages.sounddevice
          python311Packages.soundfile
          python311Packages.sympy
        ];
      };
    });
  };
}

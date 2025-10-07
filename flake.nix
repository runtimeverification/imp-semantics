{
  description = "IMP Semantics";
  inputs = {
    rv-nix-tools.url = "github:runtimeverification/rv-nix-tools/854d4f05ea78547d46e807b414faad64cea10ae4";
    nixpkgs.follows = "rv-nix-tools/nixpkgs";

    flake-utils.url = "github:numtide/flake-utils";
    k-framework.url = "github:runtimeverification/k/v7.1.288";
    k-framework.inputs.nixpkgs.follows = "nixpkgs";
    uv2nix.url = "github:pyproject-nix/uv2nix/be511633027f67beee87ab499f7b16d0a2f7eceb";
    # uv2nix requires a newer version of nixpkgs
    # therefore, we pin uv2nix specifically to a newer version of nixpkgs
    # until we replaced our stale version of nixpkgs with an upstream one as well
    # but also uv2nix requires us to call it with `callPackage`, so we add stuff
    # from the newer nixpkgs to our stale nixpkgs via an overlay
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
    uv2nix.inputs.nixpkgs.follows = "nixpkgs-unstable";
    # uv2nix.inputs.nixpkgs.follows = "nixpkgs";
    pyproject-build-systems.url = "github:pyproject-nix/build-system-pkgs/dbfc0483b5952c6b86e36f8b3afeb9dde30ea4b5";
    pyproject-build-systems = {
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.uv2nix.follows = "uv2nix";
      inputs.pyproject-nix.follows = "uv2nix/pyproject-nix";
    };
    pyproject-nix.follows = "uv2nix/pyproject-nix";
  };
  outputs = { self, rv-nix-tools, nixpkgs, flake-utils, pyproject-nix, pyproject-build-systems, uv2nix, k-framework, nixpkgs-unstable }:
  let
    pythonVer = "310";
  in flake-utils.lib.eachSystem [
      "x86_64-linux"
      "x86_64-darwin"
      "aarch64-linux"
      "aarch64-darwin"
    ] (system:
    let
      pkgs-unstable = import nixpkgs-unstable {
        inherit system;
      };
      # for uv2nix, remove this once we updated to a newer version of nixpkgs
      staleNixpkgsOverlay = final: prev: {
        inherit (pkgs-unstable) replaceVars;
      };
      # due to the nixpkgs that we use in this flake being outdated, uv is also heavily outdated
      # we can instead use the binary release of uv provided by uv2nix for now
      uvOverlay = final: prev: {
        uv = uv2nix.packages.${final.system}.uv-bin;
      };
      # create custom overlay for k, because the overlay in k-framework currently also includes a lot of other stuff instead of only k
      kOverlay = final: prev: {
        k = k-framework.packages.${final.system}.k;
      };
      kimpOverlay = final: prev:
      let
        kimp-pyk-pyproject = final.callPackage ./nix/kimp-pyk-pyproject {
          inherit uv2nix;
        };
        kimp-pyk = final.callPackage ./nix/kimp-pyk {
          inherit pyproject-nix pyproject-build-systems kimp-pyk-pyproject;
          pyproject-overlays = [
            (k-framework.overlays.pyk-pyproject system)
          ];
          python = final."python${pythonVer}";
        };
        kimp = final.callPackage ./nix/kimp {
          inherit kimp-pyk;
          rev = self.rev or null;
        };
      in {
        inherit kimp kimp-pyk kimp-pyk-pyproject;
      };
      pkgs = import nixpkgs {
        inherit system;
        overlays = [
          staleNixpkgsOverlay
          uvOverlay
          kOverlay
          kimpOverlay
        ];
      };
      python = pkgs."python${pythonVer}";
    in {
      devShells.default = pkgs.mkShell {
        name = "uv develop shell";
        buildInputs = [
          python
          pkgs.uv
        ];
        env = {
          # prevent uv from managing Python downloads and force use of specific 
          UV_PYTHON_DOWNLOADS = "never";
          UV_PYTHON = python.interpreter;
        };
        shellHook = ''
          unset PYTHONPATH
        '';
      };
      packages = rec {
        inherit (pkgs) kimp uv kimp-pyk kimp-pyk-pyproject;
        default = kimp;
      };
    }) // {
      overlays = {
        default = final: prev: {
          inherit (self.packages.${final.system}) kimp;
        };
        # this pyproject-nix overlay allows for overriding the python packages that are otherwise locked in `uv.lock`
        # by using this overlay in dependant nix flakes, you ensure that nix overrides also override the python package     
        pyk-pyproject = system: final: prev: {
          inherit (self.packages.${system}.kimp-pyk-pyproject.lockFileOverlay final prev) kimp-pyk;
        };
      };
    };
}
